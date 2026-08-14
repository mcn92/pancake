#!/usr/bin/env node
// Compile the Simple English Wikipedia pack (examples/04-static-wiki-pack,
// 456k chunks) into one .pancake — the complete profile's scale test, and
// its first kind-2 (pinned-external encoder) artifact: the MiniLM encoder
// is declared and verifiable, not embedded (contract section 4.4 mode 2);
// the host executes it, exactly as the deployed wiki demo already does.
//
//   node compile-wiki.mjs                      # -> pancake-wiki.pancake (~535 MB)
//   node compile.mjs --inspect pancake-wiki.pancake

import fs from 'node:fs';
import path from 'node:path';
import { buildQueryInterpSegment, assemblePancakeFile } from './container.mjs';
import { inspect } from './compile.mjs';

const here = path.dirname(new URL(import.meta.url).pathname);
const DATA = path.join(here, '..', '04-static-wiki-pack', 'data-full');

function buildWikiCorpusSegment() {
    // The pack already stores the corpus as JSONL bytes plus a u32
    // count+1 offsets index; the container form is the same records behind
    // u64 offsets (spec 3.5). Records pass through untouched.
    const corpusBytes = fs.readFileSync(path.join(DATA, 'corpus.bin'));
    const u32 = fs.readFileSync(path.join(DATA, 'corpus-offsets.u32'));
    const packOffsets = new Uint32Array(u32.buffer, u32.byteOffset, u32.byteLength / 4);
    const count = packOffsets.length - 1;
    if (packOffsets[count] !== corpusBytes.length) {
        throw new Error(`corpus offsets end ${packOffsets[count]} != corpus bytes ${corpusBytes.length}`);
    }
    const prefix = 4 + 8 * (count + 1);
    const out = Buffer.alloc(prefix + corpusBytes.length);
    out.writeUInt32LE(count, 0);
    for (let i = 0; i <= count; i++) {
        out.writeBigUInt64LE(BigInt(prefix + packOffsets[i]), 4 + 8 * i);
    }
    corpusBytes.copy(out, prefix);
    return { segment: out, count };
}

function buildWikiQueryInterp(packManifest) {
    // Encoder declaration (kind 2): pin the external model and carry
    // verification vectors — the first hand-written eval queries with their
    // exact MiniLM embeddings, so a host encoder is checkable before serving.
    const evalQueries = JSON.parse(fs.readFileSync(path.join(DATA, 'eval-queries.json'), 'utf8'));
    const vectors = fs.readFileSync(path.join(DATA, 'eval-queries.f32'));
    const dim = packManifest.dim;
    const queryTexts = evalQueries.queries || evalQueries;
    const handWritten = queryTexts
        .map((q, i) => ({ q, i }))
        .filter(({ q }) => (typeof q === 'object' ? q.kind === 'hand' || q.source === 'hand' : false));
    const testPool = handWritten.length >= 3 ? handWritten : queryTexts.map((q, i) => ({ q, i })).slice(0, 3);
    const testVectors = testPool.slice(0, 3).map(({ q, i }) => ({
        text: typeof q === 'object' ? q.text : q,
        embedding: Array.from(new Float32Array(vectors.buffer, vectors.byteOffset + i * dim * 4, dim))
            .map((v) => Number(v.toFixed(6))),
        tolerance: 1e-3,
    }));
    const encoderDeclaration = Buffer.from(JSON.stringify({
        kind: 'external-transformers-v1',
        model: packManifest.model,
        dim,
        pooling: packManifest.pooling,
        normalized: packManifest.normalized,
        maxTokens: packManifest.maxTokens,
        testVectors,
    }), 'utf8');

    const calibration = Buffer.from(JSON.stringify({
        kind: 'retrieval-signals-v1',
        asset: JSON.parse(fs.readFileSync(path.join(DATA, 'wiki-abstention.json'), 'utf8')),
        vocabBloomBase64: fs.readFileSync(path.join(DATA, 'wiki-vocab.bloom')).toString('base64'),
    }), 'utf8');

    return buildQueryInterpSegment(2, encoderDeclaration, calibration);
}

const packManifest = JSON.parse(fs.readFileSync(path.join(DATA, 'pack-manifest.json'), 'utf8'));
const { segment: corpusSegment, count } = buildWikiCorpusSegment();
if (count !== packManifest.chunks) {
    throw new Error(`corpus count ${count} != pack manifest chunks ${packManifest.chunks}`);
}

const evaluationSegment = Buffer.from(JSON.stringify({
    packEvaluation: packManifest.evaluation,
    groundTruth: JSON.parse(fs.readFileSync(path.join(DATA, 'eval-gt.json'), 'utf8')),
    abstentionProbes: JSON.parse(fs.readFileSync(path.join(DATA, 'wiki-abstention-probes.json'), 'utf8')),
}), 'utf8');

const segments = [
    { kind: 'index', bytes: fs.readFileSync(path.join(DATA, 'wiki.pancake-sketch')) },
    { kind: 'corpus', bytes: corpusSegment },
    { kind: 'query-interp', bytes: buildWikiQueryInterp(packManifest) },
    { kind: 'evaluation', bytes: evaluationSegment },
];

const outPath = process.argv[2] || path.join(here, 'pancake-wiki.pancake');
const result = assemblePancakeFile({
    profile: 'pancake-complete-v1',
    corpus: {
        records: count,
        provenance: { dataset: packManifest.dataset, articles: packManifest.articles },
    },
    dim: packManifest.dim,
    metric: packManifest.metric,
    encoder: {
        kind: 'external-transformers-v1',
        model: packManifest.model,
        pooling: packManifest.pooling,
        normalized: packManifest.normalized,
        maxTokens: packManifest.maxTokens,
    },
    recommendedRerank: packManifest.recommendedRerank,
    sampleQueries: ['who was the first person on the moon', 'how do volcanoes form'],
}, segments, outPath);

console.log(`compiled ${result.outPath}`);
console.log(`  ${(result.fileBytes / 1048576).toFixed(2)} MiB, identity ${result.identity.slice(0, 16)}...`);
for (const s of result.manifest.segments) {
    console.log(`  ${s.kind.padEnd(13)} ${(s.bytes / 1048576).toFixed(2).padStart(8)} MiB  ${s.sha256.slice(0, 12)}...`);
}
