#!/usr/bin/env node
// Acceptance for the kind-3 artifact: open pancake-wiki-inline.pancake with
// NO options — no host encoder, no ML runtime — and reproduce the verified
// numbers: recall@10 = 82.4% over the 200 eval queries, abstention on
// out-of-domain text, provenance in the declaration.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { openPancakeFile } from './pancake-file-reader.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.join(here, '..', '04-static-wiki-pack', 'data-full');
const pancakePath = path.join(here, 'pancake-wiki-inline.pancake');

let passed = 0;
let failed = 0;
const check = (name, ok, detail) => {
    if (ok) { passed++; console.log(`  ok: ${name}`); }
    else { failed++; console.log(`  FAIL: ${name}${detail ? ` — ${detail}` : ''}`); }
};

const openStart = performance.now();
const search = await openPancakeFile(pancakePath); // no options: the file is complete
const openMs = performance.now() - openStart;
const info = search.info();
console.log(`opened ${path.basename(pancakePath)} in ${(openMs / 1000).toFixed(1)}s: `
    + `${(info.fileBytes / 1048576).toFixed(0)} MiB, encoder ${info.encoder.kind}`);

check('opens with zero options (self-contained)', info.encoder.kind === 'inline-transformer-v1');
check('declaration carries provenance', info.encoder.model === 'sentence-transformers/all-MiniLM-L6-v2'
    && info.encoder.license === 'apache-2.0' && !!info.encoder.attribution);

const probe = await search.query('how do volcanoes form', { k: 5 });
check('natural-language query answers with hydrated results',
    probe.matchQuality === 'strong' && probe.results.length === 5
    && probe.results.every((r) => r.title && r.text && r.url),
    `${probe.matchQuality}, ${probe.results.length} results`);

const nonsense = await search.query('qzxv blorpt nym vex', { k: 5 });
check('nonsense abstains or downgrades', nonsense.matchQuality !== 'strong',
    `${nonsense.matchQuality} (${nonsense.confidence?.toFixed(3)})`);

const evalQueries = JSON.parse(fs.readFileSync(path.join(DATA, 'eval-queries.json'), 'utf8'));
const groundTruth = JSON.parse(fs.readFileSync(path.join(DATA, 'eval-gt.json'), 'utf8'));
const t0 = performance.now();
let hits = 0;
for (let i = 0; i < evalQueries.length; i++) {
    const out = await search.query(evalQueries[i].text, { k: 10 });
    const truth = new Set(groundTruth[i]);
    hits += out.results.filter((r) => truth.has(r.id)).length;
}
const perQuery = (performance.now() - t0) / evalQueries.length;
const recall = hits / (evalQueries.length * 10);
console.log(`  recall@10 over ${evalQueries.length} queries: ${(recall * 100).toFixed(1)}% `
    + `(kernel harness measured 82.4%; fp32 teacher 82.8%); ${perQuery.toFixed(0)} ms/query end to end`);
check('recall matches the verified inline-encoder number (>= 82%)', recall >= 0.82, recall.toFixed(4));

await search.close();
console.log(`\nkind-3 artifact acceptance: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
