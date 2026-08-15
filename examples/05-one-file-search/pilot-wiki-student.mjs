#!/usr/bin/env node
// Wiki student pilot, step 1: sample chunks from the wiki corpus into the
// trainer's expected JSON-list format and launch the PSTU distillation
// (create-pancake-search/tools/train_student.py, CPU) with abstention
// skipped — the pilot measures ENCODER viability on an open-domain corpus;
// calibration comes later only if the encoder clears the bar.
//
//   node pilot-wiki-student.mjs [chunks=10000] [epochs=100]
//
// Outputs to student-pilot/ (gitignored): sample-corpus.json plus the
// trainer's student-model.bin / student-manifest.json / docs-vectors.f32 /
// student-evaluation.json. Evaluate with eval-wiki-student.mjs.

import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { spawn } from 'node:child_process';

const here = path.dirname(new URL(import.meta.url).pathname);
const DATA = path.join(here, '..', '04-static-wiki-pack', 'data-full');
const OUT = path.join(here, 'student-pilot');
const TRAINER = path.join(here, '..', '..', 'create-pancake-search', 'tools', 'train_student.py');

const sampleSize = Number(process.argv[2]) || 10000;
const epochs = Number(process.argv[3]) || 100;
const SEED = 20260814;

// Seeded reservoir sample over the JSONL corpus, keeping row order stable
// so runs are reproducible.
function mulberry(seed) {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const rand = mulberry(SEED);
const reservoir = [];
let row = 0;
const reader = readline.createInterface({
    input: fs.createReadStream(path.join(DATA, 'corpus.jsonl')),
    crlfDelay: Infinity,
});
for await (const line of reader) {
    if (!line.trim()) continue;
    if (reservoir.length < sampleSize) {
        reservoir.push({ row, line });
    } else {
        const j = Math.floor(rand() * (row + 1));
        if (j < sampleSize) reservoir[j] = { row, line };
    }
    row++;
}
reservoir.sort((a, b) => a.row - b.row);
const sample = reservoir.map(({ line }) => {
    const record = JSON.parse(line);
    return { title: record.title, text: record.text };
});
fs.mkdirSync(OUT, { recursive: true });
fs.writeFileSync(path.join(OUT, 'sample-corpus.json'), JSON.stringify(sample));
console.log(`sampled ${sample.length}/${row} chunks (seed ${SEED}) -> student-pilot/sample-corpus.json`);

const args = [
    TRAINER,
    '--corpus', path.join(OUT, 'sample-corpus.json'),
    '--out', OUT,
    '--epochs', String(epochs),
    '--skip-abstention',
];
console.log(`launching: python3 ${args.map((a) => path.basename(a)).join(' ')}`);
const child = spawn('python3', args, { stdio: 'inherit' });
child.on('exit', (code) => process.exit(code ?? 1));
