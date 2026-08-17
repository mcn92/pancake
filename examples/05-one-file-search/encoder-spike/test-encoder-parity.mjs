#!/usr/bin/env node
// Stage-by-stage parity: the WASM six-layer forward vs the numpy reference
// computed from the SAME fake-quantized weights the blob encodes. Expected
// agreement is f32-accumulation noise (cosine ~1.0, max abs diff small);
// a stage that diverges hard localizes the kernel bug to that stage.

import fs from 'node:fs';
import path from 'node:path';
import createModule from './encoder.node.mjs';

const here = path.dirname(new URL(import.meta.url).pathname);
const REAL = path.join(here, 'real');
const D = 384;

const meta = JSON.parse(fs.readFileSync(path.join(REAL, 'ref-meta.json'), 'utf8'));
const seq = meta.seq;
const blob = fs.readFileSync(path.join(REAL, 'encoder-weights.bin'));
const asF32 = (buf) => new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const refStages = asF32(fs.readFileSync(path.join(REAL, 'ref-stages.f32')));
const refPooled = asF32(fs.readFileSync(path.join(REAL, 'ref-pooled.f32')));

const M = await createModule();
const blobPtr = M._malloc(blob.length);
M.HEAPU8.set(blob, blobPtr);
const idsPtr = M._malloc(seq * 4);
new Int32Array(M.HEAP32.buffer, idsPtr, seq).set(meta.ids);
const outPtr = M._malloc(seq * D * 4);
const dbgPtr = M._malloc(7 * seq * D * 4);

const t0 = performance.now();
const rc = M._encoder_forward(blobPtr, idsPtr, seq, outPtr, dbgPtr);
const ms = performance.now() - t0;
if (rc !== seq) {
    console.error(`encoder_forward failed: ${rc}`);
    process.exit(1);
}

const stages = new Float32Array(M.HEAPF32.buffer, dbgPtr, 7 * seq * D);
const labels = ['embeddings+LN', ...Array.from({ length: 6 }, (_, i) => `layer ${i}`)];
let failed = 0;
console.log(`forward pass: ${ms.toFixed(1)} ms (seq=${seq}, single thread)\n`);
console.log('stage          cosine        max|diff|');
for (let s = 0; s < 7; s++) {
    let dot = 0, na = 0, nb = 0, maxDiff = 0;
    for (let i = 0; i < seq * D; i++) {
        const a = stages[s * seq * D + i];
        const b = refStages[s * seq * D + i];
        dot += a * b; na += a * a; nb += b * b;
        const diff = Math.abs(a - b);
        if (diff > maxDiff) maxDiff = diff;
    }
    const cos = dot / Math.sqrt(na * nb);
    const ok = cos > 0.99999;
    if (!ok) failed++;
    console.log(`${labels[s].padEnd(14)} ${cos.toFixed(8)}   ${maxDiff.toExponential(2)}  ${ok ? '' : '  <-- DIVERGES'}`);
}

// Pooled output (mean over tokens, L2 normalized) vs reference.
const hidden = new Float32Array(M.HEAPF32.buffer, outPtr, seq * D);
const pooled = new Float32Array(D);
for (let t = 0; t < seq; t++) for (let d = 0; d < D; d++) pooled[d] += hidden[t * D + d];
let norm = 0;
for (let d = 0; d < D; d++) { pooled[d] /= seq; norm += pooled[d] ** 2; }
norm = Math.sqrt(norm);
let dot = 0;
for (let d = 0; d < D; d++) dot += (pooled[d] / norm) * refPooled[d];
console.log(`\npooled+normalized cosine vs reference: ${dot.toFixed(8)}`);
if (dot < 0.99999) failed++;

console.log(failed === 0 ? '\nPARITY OK' : `\n${failed} stage(s) diverged`);
process.exit(failed === 0 ? 0 : 1);
