#!/usr/bin/env node
// Calibrate abstention for the wiki pack. Signals are retrieval-only (d0,
// margin, mean10) so the client can score any query from the results it
// already has — MiniLM has no student-style feature buckets or hidden probe.
//
// Labels:
//   answerable  — hand-written in-domain questions + templated questions from
//                 sampled article titles (the corpus answers these by
//                 construction)
//   unanswerable— curated queries the 2023-11 Simple English Wikipedia dump
//                 cannot answer (recent events, tech support, local/personal,
//                 code errors, gibberish)
//   weak        — tangential: related topic exists but the specific question
//                 is not covered; used only for threshold placement, not fit
//
// Fit: logistic regression P(answerable) on standardized signals. Thresholds:
//   hard — highest cut that abstains on NO answerable query (then backed off
//          by a safety factor)
//   weak — placed between the weak set's median and the answerable mass
//
// Output: data-perm/wiki-abstention.json + wiki-abstention-probes.json
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { pipeline } from '@huggingface/transformers';
import Pancake from '../../pancake.node.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(here, process.argv[2] || 'data-perm');
const K = 10;
const RERANK = 200; // must match the demo's search config

const HAND_ANSWERABLE = [
    'how do plants turn sunlight into energy',
    'what causes earthquakes',
    'who wrote romeo and juliet',
    'why is the sky blue',
    'how does the human heart pump blood',
    'what is the largest planet in the solar system',
    'history of the roman empire',
    'how do vaccines protect against disease',
    'what language is spoken in brazil',
    'how are mountains formed',
    'what is the speed of light',
    'who painted the mona lisa',
    'how do bees make honey',
    'what caused the first world war',
    'what is the capital of australia',
    'how do computers store information',
    'what animals live in antarctica',
    'why do seasons change',
    'how is glass made',
    'what is photosynthesis',
    'who was albert einstein',
    'what is the difference between weather and climate',
    'how do volcanoes erupt',
    'what is dna made of',
    'why do we dream when we sleep',
];

// Class rule (product semantics, decided from the 2026-08-03 signal dump):
// UNANSWERABLE = the top chunks would be USELESS to an LLM answering this —
// tech support steps, code errors, personal/commerce intent, gibberish. The
// corpus may contain the subject's background article, but it cannot help.
// WEAK = an adjacent article exists and gives real partial background, but
// the specific question (often temporal or intent-gapped: "2026 winner",
// "near me") is not answered. Retrieval signals CANNOT separate weak from
// answerable intent — the weak band exists to caveat, not to block.
const UNANSWERABLE = [
    // tech support / product help
    'iphone 17 battery draining fast fix',
    'reset netgear router admin password',
    'macbook pro m4 overheating solution',
    'my wifi keeps disconnecting every few minutes',
    // code / developer
    'react usestate not updating immediately',
    'kubernetes pod crashloopbackoff how to debug',
    'python typeerror nonetype object is not subscriptable',
    'css flexbox center vertically not working',
    'git undo last commit but keep changes',
    'excel vlookup returning n a error',
    // personal / commerce intent
    'coupon code for free shipping today',
    'what should i text my crush back',
    'chase sapphire preferred annual fee waived',
    'uber driver cancelled my ride refund',
    'is the dmv open on saturday',
    'transfer news deadline day signings this january',
    // niche academic beyond the corpus
    'bounded treewidth fpt algorithms for steiner tree',
    'surface code logical qubit error threshold',
    'batch normalization internal covariate shift debate',
    // gibberish / degenerate
    'asdkjh qwerty zxcvb',
    'zzyzx qwfp jkl glorp',
    'aaaaaa bbbbbb cccccc',
    'flurbo zanzim quxlet prandle',
    'qqqq wwww eeee rrrr tttt',
    'blorp snigglet vexumal crandow',
    'jjjj kkkk llll mmmm nnnn oooo',
];

const WEAK = [
    'transformer neural network attention mechanism',
    'docker container networking basics',
    'quantum error correction',
    'photosynthesis in deep sea hydrothermal vents',
    'medieval japanese tax collection system',
    'reinforcement learning reward shaping',
    'how do noise cancelling headphones work',
    'etymology of the word quarantine',
    // adjacent-article traps: a related article exists (often pre-event),
    // the specific question is not answered
    'results of the 2024 united states presidential election',
    'taylor swift eras tour 2026 setlist',
    'who won the 2026 world cup',
    'super bowl 2026 final score',
    'oscar best picture winner 2026',
    'what time does the game start tonight',
    'crispr prime editing off target rates',
    'homotopy type theory univalence axiom explained',
    'best pizza restaurant near me',
    'cheap flights from denver to chicago next weekend',
    'best dentist in austin texas',
    'nike air max 270 current price',
    'spotify shuffle not random enough settings',
    'why does my printer say offline',
    'why is my sourdough starter not rising in my kitchen',
    'property tax deadline extension this year',
    // the dump has a real "Stop error screen" article, so this is adjacent
    // content, not a useless-content negative
    'how to fix blue screen error on windows 11',
];

function titleQuestions(corpusPath, n, seed) {
    // Deterministic sample of article titles -> templated questions, keeping
    // the source title so the label can be VERIFIED: a template query is an
    // "answerable" positive only if retrieval actually returns its source
    // article in the top results — the "answer" verdict means answerable AND
    // retrieved, so unretrievable-title questions must not drag the
    // calibration floor down.
    const titles = [];
    const seen = new Set();
    for (const line of fs.readFileSync(corpusPath, 'utf8').split('\n')) {
        if (!line) continue;
        const t = JSON.parse(line).title;
        if (!seen.has(t)) { seen.add(t); titles.push(t); }
    }
    let state = seed >>> 0;
    const next = () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 0xffffffff);
    const picked = [];
    const used = new Set();
    while (picked.length < n) {
        const i = Math.floor(next() * titles.length);
        if (used.has(i)) continue;
        used.add(i);
        picked.push(titles[i]);
    }
    const templates = [(t) => `what is ${t}`, (t) => `tell me about ${t}`, (t) => `${t} explained`, (t) => `facts about ${t}`];
    return picked.map((t, i) => ({ text: templates[i % templates.length](t.toLowerCase()), sourceTitle: t }));
}

// --- Corpus vocabulary bloom: the known-token fraction separates gibberish
// (d0 ~0.49, inside the answerable band — distance alone cannot catch it)
// from real queries. FNV-1a with two seeds over lowercase word tokens;
// the client computes the identical hash from the shipped bitset.
const BLOOM_BITS = 1 << 21;
function fnv1a(str, seed) {
    let h = 0x811c9dc5 ^ seed;
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 0x01000193);
    }
    return (h >>> 0) % BLOOM_BITS;
}
const tokenize = (text) => (text.toLowerCase().match(/[a-z0-9']+/g) || []);

function buildVocabBloom(corpusPath) {
    const counts = new Map();
    for (const line of fs.readFileSync(corpusPath, 'utf8').split('\n')) {
        if (!line) continue;
        for (const w of tokenize(JSON.parse(line).text)) {
            counts.set(w, (counts.get(w) || 0) + 1);
        }
    }
    const bloom = new Uint8Array(BLOOM_BITS / 8);
    let kept = 0;
    for (const [w, c] of counts) {
        if (c < 3) continue;
        kept++;
        for (const seed of [0, 0x9e3779b9]) {
            const bit = fnv1a(w, seed);
            bloom[bit >> 3] |= 1 << (bit & 7);
        }
    }
    console.log(`vocab bloom: ${kept} words (of ${counts.size} unique) -> ${bloom.length / 1024} KiB`);
    return bloom;
}

const bloomPath = path.join(dataDir, 'wiki-vocab.bloom');
let bloom;
if (fs.existsSync(bloomPath)) {
    bloom = new Uint8Array(fs.readFileSync(bloomPath).buffer.slice(0));
    console.log('vocab bloom: loaded existing');
} else {
    bloom = buildVocabBloom(path.join(dataDir, 'corpus.jsonl'));
    fs.writeFileSync(bloomPath, Buffer.from(bloom));
}
function knownFrac(text) {
    const words = tokenize(text);
    if (!words.length) return 0;
    let known = 0;
    for (const w of words) {
        const hit = [0, 0x9e3779b9].every((seed) => {
            const bit = fnv1a(w, seed);
            return (bloom[bit >> 3] >> (bit & 7)) & 1;
        });
        if (hit) known++;
    }
    return known / words.length;
}

const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'fp16' });
const artifact = await Pancake.openSketchArtifactFile(path.join(dataDir, 'wiki.pancake-sketch'));
const scanner = await Pancake.createSketchScanner(artifact);
const offsets = new Uint32Array(fs.readFileSync(path.join(dataDir, 'corpus-offsets.u32')).buffer);
const corpusFd = fs.openSync(path.join(dataDir, 'corpus.bin'), 'r');
function chunkTitle(id) {
    const buf = Buffer.alloc(offsets[id + 1] - offsets[id]);
    fs.readSync(corpusFd, buf, 0, buf.length, offsets[id]);
    return JSON.parse(buf.toString('utf8')).title;
}

async function signalsFor(text) {
    const out = await embedder(text, { pooling: 'mean', normalize: true });
    const { results } = await artifact.search(Float32Array.from(out.data), K, { rerank: RERANK, scanner });
    const d0 = results[0].distance;
    const margin = results[Math.min(4, results.length - 1)].distance - d0;
    const mean10 = results.reduce((s, r) => s + r.distance, 0) / results.length;
    return { d0, margin, mean10, known_frac: knownFrac(text), topIds: results.map((r) => r.id) };
}

const templates = titleQuestions(path.join(dataDir, 'corpus.jsonl'), 75, 424242);
const rows = [];
for (const text of HAND_ANSWERABLE) rows.push({ text, label: 1, ...await signalsFor(text) });

// Template positives are kept only when retrieval verifiably returns the
// source article — otherwise they are silently unanswerable-in-practice and
// would drag the no-false-abstain floor toward zero.
let dropped = 0;
for (const { text, sourceTitle } of templates) {
    const sig = await signalsFor(text);
    const found = sig.topIds.some((id) => chunkTitle(id) === sourceTitle);
    if (found) rows.push({ text, label: 1, ...sig });
    else dropped++;
}
console.log(`template positives: ${templates.length - dropped} verified, ${dropped} dropped (source article not retrieved)`);

for (const text of UNANSWERABLE) rows.push({ text, label: 0, ...await signalsFor(text) });
for (const text of WEAK) rows.push({ text, label: -1, ...await signalsFor(text) });
console.log(`scoring ${rows.filter((r) => r.label === 1).length} answerable, ${UNANSWERABLE.length} unanswerable, ${WEAK.length} weak`);

// Flag remaining suspicious labels for manual review.
for (const r of rows) {
    if (r.label === 0 && r.d0 < 0.30) console.log(`  REVIEW negative with strong hit: "${r.text}" d0=${r.d0.toFixed(3)} top="${chunkTitle(r.topIds[0])}"`);
}

// Standardize features, fit logistic regression by gradient descent.
const FEATS = ['d0', 'margin', 'mean10', 'known_frac'];
const fit = rows.filter((r) => r.label >= 0);
const mean = {}, std = {};
for (const f of FEATS) {
    const vals = fit.map((r) => r[f]);
    mean[f] = vals.reduce((a, b) => a + b, 0) / vals.length;
    std[f] = Math.sqrt(vals.reduce((a, b) => a + (b - mean[f]) ** 2, 0) / vals.length) || 1;
}
const xs = fit.map((r) => FEATS.map((f) => (r[f] - mean[f]) / std[f]));
const ys = fit.map((r) => r.label);
let w = FEATS.map(() => 0);
let b = 0;
for (let epoch = 0; epoch < 4000; epoch++) {
    const gw = FEATS.map(() => 0);
    let gb = 0;
    for (let i = 0; i < xs.length; i++) {
        const z = xs[i].reduce((s, v, j) => s + v * w[j], b);
        const p = 1 / (1 + Math.exp(-z));
        const err = p - ys[i];
        xs[i].forEach((v, j) => { gw[j] += err * v; });
        gb += err;
    }
    w = w.map((wj, j) => wj - 0.1 * (gw[j] / xs.length + 1e-3 * wj));
    b -= 0.1 * (gb / xs.length);
}
const prob = (r) => 1 / (1 + Math.exp(-(FEATS.reduce((s, f, j) => s + ((r[f] - mean[f]) / std[f]) * w[j], b))));
for (const r of rows) r.p = prob(r);

// AUC as sanity, then thresholds.
const pos = rows.filter((r) => r.label === 1).map((r) => r.p);
const neg = rows.filter((r) => r.label === 0).map((r) => r.p);
let auc = 0;
for (const a of pos) for (const c of neg) auc += a > c ? 1 : a === c ? 0.5 : 0;
auc /= pos.length * neg.length;

// hard: every answerable query must stay above it (no false abstain), and
// ideally every weak query too (weak = real partial content — caveat it,
// don't hide it). With a clean gap, place it at the geometric midpoint of
// the NEG ceiling and the POS/WEAK floor; on overlap sit at the NEG ceiling
// and accept that the overlapping weak queries abstain.
// weak: just under the weakest answerable query, so every labeled
// answerable keeps a full 'answer' verdict.
const minPos = Math.min(...pos);
const maxNeg = Math.max(...neg);
const weakP = rows.filter((r) => r.label === -1).map((r) => r.p);
const floor = Math.min(minPos, Math.min(...weakP));
let hard = floor > maxNeg ? Math.sqrt(maxNeg * floor) : maxNeg;
const weak = minPos * 0.9;
hard = Math.min(hard, weak * 0.6);

const verdict = (p) => (p < hard ? 'abstain' : p < weak ? 'weak' : 'answer');
if (process.env.DUMP) {
    for (const r of [...rows].sort((a, b) => a.p - b.p)) {
        const lab = r.label === 1 ? 'POS ' : r.label === 0 ? 'NEG ' : 'WEAK';
        console.log(`${lab} p=${r.p.toFixed(3)} d0=${r.d0.toFixed(3)} margin=${r.margin.toFixed(3)} mean=${r.mean10.toFixed(3)} top="${chunkTitle(r.topIds[0])}"  ${r.text.slice(0, 48)}`);
    }
}
const confusion = {};
for (const r of rows) {
    const key = `${r.label === 1 ? 'answerable' : r.label === 0 ? 'unanswerable' : 'weak'}:${verdict(r.p)}`;
    confusion[key] = (confusion[key] || 0) + 1;
}
console.log(`\nAUC ${auc.toFixed(4)}  minPos ${minPos.toFixed(4)}  maxNeg ${Math.max(...neg).toFixed(4)}`);
console.log('thresholds:', { hard: +hard.toFixed(4), weak: +weak.toFixed(4) });
console.log('confusion:', JSON.stringify(confusion, null, 1));

const asset = {
    version: 1,
    corpus: 'simple-wikipedia-20231101',
    searchConfig: { rerank: RERANK, k: K },
    features: FEATS,
    standardize: { mean, std },
    weights: w,
    bias: b,
    thresholds: { hard: +hard.toFixed(6), weak: +weak.toFixed(6) },
    vocabBloom: { file: 'wiki-vocab.bloom', bits: BLOOM_BITS, hashes: ['fnv1a:0', 'fnv1a:0x9e3779b9'], minCount: 3 },
};
fs.writeFileSync(path.join(dataDir, 'wiki-abstention.json'), JSON.stringify(asset, null, 2) + '\n');

// Golden probes: fixed queries + expected verdicts, for CI and in-browser checks.
const probes = [
    ...HAND_ANSWERABLE.slice(0, 6).map((text) => ({ text, expect: 'answer' })),
    // useless-content territory must abstain
    { text: 'reset netgear router admin password', expect: 'abstain' },
    { text: 'excel vlookup returning n a error', expect: 'abstain' },
    { text: 'flurbo zanzim quxlet prandle', expect: 'abstain' },
    { text: 'react usestate not updating immediately', expect: 'abstain' },
    // adjacent-content traps must NOT be hidden — weak (or better)
    { text: 'who won the 2026 world cup', expect: ['weak', 'answer'] },
    // retrieves a real restaurant-chain article ("Pizza Pizza") — correct
    // retrieval; the intent gap ("near me") is invisible to these signals
    { text: 'best pizza restaurant near me', expect: ['answer', 'weak'] },
    { text: 'quantum error correction', expect: verdict(rows.find((r) => r.text === 'quantum error correction').p) },
    { text: 'docker container networking basics', expect: verdict(rows.find((r) => r.text === 'docker container networking basics').p) },
];
let probeFails = 0;
for (const probe of probes) {
    const r = rows.find((x) => x.text === probe.text) || { ...await signalsFor(probe.text) };
    const v = verdict(r.p ?? prob(r));
    const expected = Array.isArray(probe.expect) ? probe.expect : [probe.expect];
    if (!expected.includes(v)) { probeFails++; console.log(`  PROBE FAIL: "${probe.text}" expected ${expected.join('|')} got ${v}`); }
}
fs.writeFileSync(path.join(dataDir, 'wiki-abstention-probes.json'), JSON.stringify(probes, null, 2) + '\n');
console.log(`golden probes: ${probes.length - probeFails}/${probes.length} pass`);
console.log('wrote wiki-abstention.json + wiki-abstention-probes.json');

scanner.dispose();
await artifact.close();
