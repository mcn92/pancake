#!/usr/bin/env node
// Query the wiki pack end to end the way the browser demo will:
// transformers.js MiniLM embeds the query, the sketch artifact selects and
// reranks candidates, and results hydrate from corpus.bin via the offsets
// table — plus a cross-runtime parity check of the JS encoder against the
// Python-embedded corpus vectors.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { pipeline } from '@huggingface/transformers';
import Pancake from '../../pancake.node.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(here, 'data');
const DIM = 384;

const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'fp32' });
async function embed(text) {
    const out = await embedder(text, { pooling: 'mean', normalize: true });
    return Float32Array.from(out.data);
}

// --- Parity: JS encoder vs Python corpus vectors on the same texts -------
const corpusLines = fs.readFileSync(path.join(dataDir, 'corpus.jsonl'), 'utf8').split('\n');
const vecBytes = fs.readFileSync(path.join(dataDir, 'vectors.f32'));
const pyVectors = new Float32Array(vecBytes.buffer, vecBytes.byteOffset, vecBytes.byteLength / 4);
let worst = 1;
for (const id of [0, 100, 1000, 3000, 5000]) {
    const row = JSON.parse(corpusLines[id]);
    const js = await embed(row.text);
    const py = pyVectors.subarray(id * DIM, (id + 1) * DIM);
    let dot = 0;
    for (let d = 0; d < DIM; d++) dot += js[d] * py[d];
    worst = Math.min(worst, dot);
}
console.log(`encoder parity (JS fp32 vs Python fp32): worst cosine ${worst.toFixed(5)} over 5 chunks`);

// --- Search + hydrate ----------------------------------------------------
const artifact = await Pancake.openSketchArtifactFile(path.join(dataDir, 'wiki.pancake-sketch'));
const scanner = await Pancake.createSketchScanner(artifact);
const offsets = new Uint32Array(fs.readFileSync(path.join(dataDir, 'corpus-offsets.u32')).buffer);
const corpusFd = fs.openSync(path.join(dataDir, 'corpus.bin'), 'r');

function hydrate(id) {
    const start = offsets[id];
    const end = offsets[id + 1];
    const buf = Buffer.alloc(end - start);
    fs.readSync(corpusFd, buf, 0, buf.length, start);
    return JSON.parse(buf.toString('utf8'));
}

const queries = process.argv.length > 2 ? process.argv.slice(2) : [
    'how do plants turn sunlight into energy',
    'what causes earthquakes',
    'who wrote romeo and juliet',
];
for (const q of queries) {
    const t0 = performance.now();
    const qv = await embed(q);
    const tEmbed = performance.now();
    const { results } = await artifact.search(qv, 5, { rerank: 200, scanner });
    const tSearch = performance.now();
    console.log(`\nQ: ${q}   (embed ${(tEmbed - t0).toFixed(0)} ms, search ${(tSearch - tEmbed).toFixed(1)} ms)`);
    for (const r of results.slice(0, 3)) {
        const row = hydrate(r.id);
        console.log(`  [${r.distance.toFixed(3)}] ${row.title}: ${row.text.slice(row.title.length + 2, row.title.length + 122)}...`);
    }
}
scanner.dispose();
await artifact.close();
fs.closeSync(corpusFd);
