#!/usr/bin/env node
// Quantization accuracy of the fused-u8 GEMV on REAL MiniLM layer-0 weights
// and the true activations that feed them (from export_weights.py), swept
// over block sizes. This is the number that decides the per-layer error
// budget for a full inlined forward pass.

import fs from 'node:fs';
import path from 'node:path';
import createModule from './spike.mjs';

const here = path.dirname(new URL(import.meta.url).pathname);
const REAL = path.join(here, 'real');
const meta = JSON.parse(fs.readFileSync(path.join(REAL, 'meta.json'), 'utf8'));

const M = await createModule();
const alloc = (bytes) => M._malloc(bytes);
const f32view = (ptr, len) => new Float32Array(M.HEAPF32.buffer, ptr, len);
const u8view = (ptr, len) => new Uint8Array(M.HEAPU8.buffer, ptr, len);

function quantize(weights, rows, cols, block) {
    const nblocks = cols / block;
    const aPtr = alloc(rows * cols);
    const sPtr = alloc(rows * nblocks * 4);
    const oPtr = alloc(rows * nblocks * 4);
    const a = u8view(aPtr, rows * cols);
    const scales = f32view(sPtr, rows * nblocks);
    const offsets = f32view(oPtr, rows * nblocks);
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

console.log('fused-u8 vs f32 on real MiniLM layer-0 weights + real activations\n');
console.log('matrix     block   mean cosine   worst cosine   mean relRMSE');
for (const [name, { rows, cols, n }] of Object.entries(meta)) {
    const weights = new Float32Array(fs.readFileSync(path.join(REAL, `${name}.w.f32`)).buffer);
    const acts = new Float32Array(fs.readFileSync(path.join(REAL, `${name}.x.f32`)).buffer);
    const wPtr = alloc(rows * cols * 4);
    f32view(wPtr, rows * cols).set(weights);
    for (const block of [32, 64, 128]) {
        if (cols % block !== 0) continue;
        const { aPtr, sPtr, oPtr } = quantize(weights, rows, cols, block);
        const xPtr = alloc(cols * 4);
        const xsPtr = alloc((cols / block) * 4);
        const yPtr = alloc(rows * 4);
        const yRefPtr = alloc(rows * 4);
        let cosSum = 0, cosMin = 1, rmseSum = 0;
        for (let i = 0; i < n; i++) {
            f32view(xPtr, cols).set(acts.subarray(i * cols, (i + 1) * cols));
            M._block_sums(xPtr, cols, block, xsPtr);
            M._gemv_u8_fused(aPtr, sPtr, oPtr, rows, cols, block, xPtr, xsPtr, yPtr);
            M._gemv_f32(wPtr, rows, cols, xPtr, yRefPtr);
            const y = f32view(yPtr, rows), ref = f32view(yRefPtr, rows);
            let dot = 0, na = 0, nb = 0, err = 0;
            for (let r = 0; r < rows; r++) {
                dot += y[r] * ref[r];
                na += y[r] * y[r];
                nb += ref[r] * ref[r];
                err += (y[r] - ref[r]) ** 2;
            }
            const cos = dot / Math.sqrt(na * nb);
            cosSum += cos;
            if (cos < cosMin) cosMin = cos;
            rmseSum += Math.sqrt(err / nb);
        }
        console.log(`${name.padEnd(10)} ${String(block).padStart(4)}    ${(cosSum / n).toFixed(7)}     ${cosMin.toFixed(7)}      ${(rmseSum / n).toExponential(2)}`);
        M._free(aPtr); M._free(sPtr); M._free(oPtr);
        M._free(xPtr); M._free(xsPtr); M._free(yPtr); M._free(yRefPtr);
    }
    M._free(wPtr);
}
