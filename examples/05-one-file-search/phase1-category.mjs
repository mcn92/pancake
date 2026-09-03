#!/usr/bin/env node
// Phase 1, step 2: pick the category with the densest slice of eval-query
// ground truth, extract its chunks as a training corpus, and launch the
// specialist student's training.
//
//   node phase1-category.mjs            # select + report only
//   node phase1-category.mjs --train    # select, then launch the trainer
//
// A query "belongs" to a cluster when >= 8 of its 10 ground-truth ids fall
// inside — the phase-1 metric only makes sense for queries the category can
// actually answer alone.

import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { spawn } from 'node:child_process';

const here = path.dirname(new URL(import.meta.url).pathname);
const DATA = path.join(here, '..', '04-static-wiki-pack', 'data-full');
const OUT = path.join(here, 'student-pilot');
const TRAINER = path.join(here, '..', '..', 'pikelet', 'tools', 'train_student.py');

const raw = fs.readFileSync(path.join(OUT, 'cluster-assignments.u16'));
const assignments = new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2);
const groundTruth = JSON.parse(fs.readFileSync(path.join(DATA, 'eval-gt.json'), 'utf8'));
const evalQueries = JSON.parse(fs.readFileSync(path.join(DATA, 'eval-queries.json'), 'utf8'));

let k = 0;
for (const c of assignments) if (c >= k) k = c + 1;
const owned = new Map();
for (let q = 0; q < groundTruth.length; q++) {
    const tally = new Map();
    for (const id of groundTruth[q]) {
        const c = assignments[id];
        tally.set(c, (tally.get(c) || 0) + 1);
    }
    for (const [c, hits] of tally) {
        if (hits >= 8) {
            if (!owned.get(c)) owned.set(c, []);
            owned.get(c).push(q);
        }
    }
}

const ranked = [...owned.entries()]
    .map(([cluster, queries]) => ({
        cluster,
        queries,
        chunks: assignments.reduce((sum, c) => sum + (c === cluster ? 1 : 0), 0),
    }))
    .sort((a, b) => b.queries.length - a.queries.length);

console.log('clusters ranked by owned eval queries (>=8/10 GT inside):');
for (const row of ranked.slice(0, 6)) {
    console.log(`  cluster ${String(row.cluster).padStart(2)}: ${String(row.queries.length).padStart(3)} queries, `
        + `${row.chunks.toLocaleString()} chunks`);
}
const chosen = ranked[0];
if (!chosen || chosen.queries.length < 10) {
    console.error('no cluster owns enough eval queries for a meaningful phase-1 measurement');
    process.exit(1);
}
console.log(`\nchosen: cluster ${chosen.cluster} — ${chosen.queries.length} owned queries, ${chosen.chunks.toLocaleString()} chunks`);

// Extract the category corpus in row order.
const wanted = new Set();
for (let row = 0; row < assignments.length; row++) {
    if (assignments[row] === chosen.cluster) wanted.add(row);
}
const category = [];
let row = 0;
const reader = readline.createInterface({
    input: fs.createReadStream(path.join(DATA, 'corpus.jsonl')),
    crlfDelay: Infinity,
});
for await (const line of reader) {
    if (!line.trim()) continue;
    if (wanted.has(row)) {
        const record = JSON.parse(line);
        category.push({ title: record.title, text: record.text });
    }
    row++;
}
fs.writeFileSync(path.join(OUT, 'category-corpus.json'), JSON.stringify(category));
fs.writeFileSync(path.join(OUT, 'category-meta.json'), JSON.stringify({
    cluster: chosen.cluster,
    chunks: chosen.chunks,
    evalQueryIndices: chosen.queries,
    sampleQueries: chosen.queries.slice(0, 8).map((q) => evalQueries[q].text),
}, null, 1));
console.log(`wrote category-corpus.json (${category.length.toLocaleString()} chunks) + category-meta.json`);
console.log(`sample owned queries: ${chosen.queries.slice(0, 5).map((q) => JSON.stringify(evalQueries[q].text)).join(', ')}`);

if (process.argv.includes('--train')) {
    const args = [TRAINER,
        '--corpus', path.join(OUT, 'category-corpus.json'),
        '--out', path.join(OUT, 'category'),
        '--skip-abstention'];
    fs.mkdirSync(path.join(OUT, 'category'), { recursive: true });
    console.log('\nlaunching category student training...');
    const child = spawn('python3', ['-u', ...args], { stdio: 'inherit' });
    child.on('exit', (code) => process.exit(code ?? 1));
}
