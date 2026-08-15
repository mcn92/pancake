#!/usr/bin/env node
// Bench + parity + quantization-accuracy for the fused u8 GEMV spike.
// Shapes are MiniLM-L6's real GEMV shapes; the projection multiplies
// measured throughput into full-forward latency for MiniLM-L6 and the
// tiny-2L design point.

import createModule from './spike.mjs';

const M = await createModule();
const alloc = (bytes) => M._malloc(bytes);
const f32 = (ptr, len) => new Float32Array(M.HEAPF32.buffer, ptr, len);
const u8 = (ptr, len) => new Uint8Array(M.HEAPU8.buffer, ptr, len);

function gaussian(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function mulberry(seed) {
    let s = seed >>> 0;
    return () => {
        s = (s + 0x6d2b79f5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// Quantize a float matrix to block-affine u8 (per-block min/max), returning
// device pointers plus the fp32 original for accuracy comparison.
function quantize(weights, rows, cols, block) {
    const nblocks = cols / block;
    const aPtr = alloc(rows * cols);
    const sPtr = alloc(rows * nblocks * 4);
    const oPtr = alloc(rows * nblocks * 4);
    const a = u8(aPtr, rows * cols);
    const scales = f32(sPtr, rows * nblocks);
    const offsets = f32(oPtr, rows * nblocks);
    for (let r = 0; r < rows; r++) {
        for (let b = 0; b < nblocks; b++) {
            let lo = Infinity, hi = -Infinity;
            for (let c = 0; c < block; c++) {
                const w = weights[r * cols + b * block + c];
                if (w < lo) lo = w;
                if (w > hi) hi = w;
            }
            const scale = (hi - lo) / 255 || 1e-12;
            scales[r * nblocks + b] = scale;
            offsets[r * nblocks + b] = lo;
            for (let c = 0; c < block; c++) {
                const w = weights[r * cols + b * block + c];
                a[r * cols + b * block + c] = Math.max(0, Math.min(255, Math.round((w - lo) / scale)));
            }
        }
    }
    return { aPtr, sPtr, oPtr };
}

const rng = mulberry(20260814);
const BLOCK = 64;
const shapes = [
    ['attn 384x384', 384, 384],
    ['ffn up 1536x384', 1536, 384],
    ['ffn down 384x1536', 384, 1536],
];

let fusedGops = 0;
let f32Gops = 0;
console.log(`block=${BLOCK}, WASM SIMD128\n`);
for (const [label, rows, cols] of shapes) {
    const weights = new Float32Array(rows * cols);
    for (let i = 0; i < weights.length; i++) weights[i] = gaussian(rng) * 0.06;
    const { aPtr, sPtr, oPtr } = quantize(weights, rows, cols, BLOCK);
    const wPtr = alloc(rows * cols * 4);
    f32(wPtr, rows * cols).set(weights);
    const xPtr = alloc(cols * 4);
    const x = f32(xPtr, cols);
    for (let i = 0; i < cols; i++) x[i] = gaussian(rng);
    const xsPtr = alloc((cols / BLOCK) * 4);
    M._block_sums(xPtr, cols, BLOCK, xsPtr);
    const yPtr = alloc(rows * 4);
    const yRefPtr = alloc(rows * 4);

    // Parity: SIMD fused vs scalar fused.
    M._gemv_u8_fused(aPtr, sPtr, oPtr, rows, cols, BLOCK, xPtr, xsPtr, yPtr);
    M._gemv_u8_scalar(aPtr, sPtr, oPtr, rows, cols, BLOCK, xPtr, xsPtr, yRefPtr);
    let maxRel = 0;
    const y = f32(yPtr, rows), yRef = f32(yRefPtr, rows);
    for (let r = 0; r < rows; r++) {
        maxRel = Math.max(maxRel, Math.abs(y[r] - yRef[r]) / (Math.abs(yRef[r]) + 1e-9));
    }

    // Quantization accuracy: fused-u8 output vs exact fp32 GEMV.
    M._gemv_f32(wPtr, rows, cols, xPtr, yRefPtr);
    let dot = 0, na = 0, nb = 0;
    for (let r = 0; r < rows; r++) {
        dot += y[r] * yRef[r];
        na += y[r] * y[r];
        nb += yRef[r] * yRef[r];
    }
    const cosine = dot / Math.sqrt(na * nb);

    const time = (fn, iters) => {
        fn(); fn();
        const t0 = performance.now();
        for (let i = 0; i < iters; i++) fn();
        return (performance.now() - t0) / iters;
    };
    const iters = Math.max(50, Math.round(2e8 / (rows * cols)));
    const fusedMs = time(() => M._gemv_u8_fused(aPtr, sPtr, oPtr, rows, cols, BLOCK, xPtr, xsPtr, yPtr), iters);
    const scalarMs = time(() => M._gemv_u8_scalar(aPtr, sPtr, oPtr, rows, cols, BLOCK, xPtr, xsPtr, yPtr), iters);
    const floatMs = time(() => M._gemv_f32(wPtr, rows, cols, xPtr, yPtr), iters);
    const gflops = (ms) => (2 * rows * cols) / (ms * 1e6);
    fusedGops += gflops(fusedMs);
    f32Gops += gflops(floatMs);
    console.log(`${label.padEnd(18)} fused ${gflops(fusedMs).toFixed(2)} GFLOPs `
        + `| scalar ${gflops(scalarMs).toFixed(2)} | f32 ${gflops(floatMs).toFixed(2)} `
        + `| simd-vs-scalar parity ${maxRel.toExponential(1)} | quant cosine ${cosine.toFixed(6)}`);
}

const meanFused = fusedGops / shapes.length;
const meanF32 = f32Gops / shapes.length;
// Full-forward GEMV FLOPs at seq=32: per token per layer ~4*384^2 (attn
// projections) + 2*384*1536 (FFN), x2 flops.
const minilm = 2 * 32 * 6 * (4 * 384 * 384 + 2 * 384 * 1536);
const tiny = 2 * 32 * 2 * (4 * 192 * 192 + 2 * 192 * 768);
console.log(`\nprojected full-forward GEMV latency at seq=32 (single thread):`);
console.log(`  MiniLM-L6 (${(minilm / 1e6).toFixed(0)} MFLOPs): `
    + `fused-u8 ${(minilm / (meanFused * 1e9) * 1000).toFixed(1)} ms | f32 ${(minilm / (meanF32 * 1e9) * 1000).toFixed(1)} ms`);
console.log(`  tiny-2L   (${(tiny / 1e6).toFixed(0)} MFLOPs): `
    + `fused-u8 ${(tiny / (meanFused * 1e9) * 1000).toFixed(2)} ms | f32 ${(tiny / (meanF32 * 1e9) * 1000).toFixed(2)} ms`);
console.log(`\nweight bytes: MiniLM layers u8 ~11 MB (vs 45 MB fp32 + runtime); attention/softmax/LN excluded (small, unfused here)`);
