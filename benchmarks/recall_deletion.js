#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const Pancake = require('../dist/engine.js');
const { parseBenchmarkArgs, resolveSingleValue } = require('./bench_args');

const parsedArgs = parseBenchmarkArgs();

const DIMS = 1536;
const K = 10;
const MAX_ELEM = 11_000;
const QUERY_COUNT = 100;
const FVECS_PATH = path.join(__dirname, '..', 'dbpedia', 'dbpedia_base_5k.fvecs');
const M = resolveSingleValue(parsedArgs.m, 12);
const EF_CONSTRUCTION = resolveSingleValue(parsedArgs.efConstruction, 150);
const EF_SEARCH = resolveSingleValue(parsedArgs.efSearch, 250);

function loadFvecs(filePath) {
    const buf = fs.readFileSync(filePath);
    const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    const recSize = 4 + DIMS * 4;
    const n = Math.floor(buf.byteLength / recSize);
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
        const off = i * recSize;
        const d = view.getInt32(off, true);
        if (d !== DIMS) throw new Error(`vec ${i} dim=${d} expected ${DIMS}`);
        const v = new Float32Array(DIMS);
        for (let j = 0; j < DIMS; j++) v[j] = view.getFloat32(off + 4 + j * 4, true);
        out[i] = v;
    }
    return out;
}

// dbpedia uses cosine; vectors may not be normalized
function normalize(v) {
    let n = 0;
    for (let i = 0; i < DIMS; i++) n += v[i] * v[i];
    n = Math.sqrt(n) || 1;
    const out = new Float32Array(DIMS);
    for (let i = 0; i < DIMS; i++) out[i] = v[i] / n;
    return out;
}

function cosineDist(a, b) {
    let dot = 0;
    for (let d = 0; d < DIMS; d++) dot += a[d] * b[d];
    // assumes normalized
    return 1 - dot;
}

(async () => {
    console.log('Loading DBpedia 1536D vectors...');
    const raw = loadFvecs(FVECS_PATH);
    const all = raw.map(normalize);
    console.log(`Loaded ${all.length} vectors`);

    const N = all.length - QUERY_COUNT;
    const queryStart = N;

    console.log('Loading engine...');
    const wasmBinary = fs.readFileSync(path.join(__dirname, '..', 'dist', 'engine.wasm'));
    const engine = await Pancake({ wasmBinary });

    const queryPtr = engine._emsc_malloc(DIMS * 4);
    const resultIdPtr = engine._emsc_malloc(K * 8);
    const resultDistPtr = engine._emsc_malloc(K * 4);

    const handle = engine._pancake_init(DIMS, MAX_ELEM, 1, 1, M, EF_CONSTRUCTION, EF_SEARCH);

    console.log(`Building index with ${N} vectors at 1536D (cosine, int8, M=${M}, ef_c=${EF_CONSTRUCTION}, ef_s=${EF_SEARCH})...`);
    const batchPtr = engine._emsc_malloc(500 * DIMS * 4);
    const t0 = Date.now();
    for (let i = 0; i < N; i += 500) {
        const n = Math.min(500, N - i);
        const off = batchPtr >> 2;
        for (let j = 0; j < n; j++) engine.HEAPF32.set(all[i + j], off + j * DIMS);
        engine._pancake_bulk_insert(handle, batchPtr, n);
    }
    engine._emsc_free(batchPtr);
    const buildSec = (Date.now() - t0) / 1000;
    console.log(`Built in ${buildSec.toFixed(2)}s, count=${engine._pancake_count(handle)}`);

    const liveSet = new Set();
    for (let i = 0; i < N; i++) liveSet.add(i);

    const queryIndices = [];
    for (let i = 0; i < QUERY_COUNT; i++) queryIndices.push(queryStart + i);

    function measureRecall() {
        const liveArr = Array.from(liveSet);
        let totalRecall = 0;
        const latencies = [];

        for (const qIdx of queryIndices) {
            const qVec = all[qIdx];

            // Brute-force top-K among live
            const scored = new Array(liveArr.length);
            for (let i = 0; i < liveArr.length; i++) {
                scored[i] = { id: liveArr[i], dist: cosineDist(qVec, all[liveArr[i]]) };
            }
            scored.sort((a, b) => a.dist - b.dist);
            const trueK = Math.min(K, scored.length);
            const trueTopK = new Set();
            for (let i = 0; i < trueK; i++) trueTopK.add(scored[i].id);

            engine.HEAPF32.set(qVec, queryPtr >> 2);
            const t = performance.now();
            const found = engine._pancake_query(handle, queryPtr, K, resultIdPtr, resultDistPtr);
            latencies.push(performance.now() - t);

            let hits = 0;
            for (let i = 0; i < found; i++) {
                const id = engine.HEAPU32[(resultIdPtr >> 2) + i * 2];
                if (trueTopK.has(id)) hits++;
            }
            totalRecall += hits / trueK;
        }
        latencies.sort((a, b) => a - b);
        return {
            recall: totalRecall / QUERY_COUNT,
            p50: latencies[Math.floor(latencies.length * 0.5)],
            p99: latencies[Math.floor(latencies.length * 0.99)],
        };
    }

    function deleteRandomLive(n) {
        const liveArr = Array.from(liveSet);
        const actual = Math.min(n, liveArr.length - K);
        if (actual <= 0) return 0;
        for (let i = 0; i < actual; i++) {
            const j = i + Math.floor(Math.random() * (liveArr.length - i));
            [liveArr[i], liveArr[j]] = [liveArr[j], liveArr[i]];
        }
        for (let i = 0; i < actual; i++) {
            engine._pancake_delete(handle, liveArr[i]);
            liveSet.delete(liveArr[i]);
        }
        return actual;
    }

    const ghostTargets = [0, 5, 10, 20, 30, 50, 70, 80, 85, 90, 93, 95];
    const results = [];

    for (const targetPct of ghostTargets) {
        const targetGhosts = Math.floor(N * targetPct / 100);
        const currentGhosts = N - liveSet.size;
        const toDelete = targetGhosts - currentGhosts;
        if (toDelete > 0) deleteRandomLive(toDelete);

        const m = measureRecall();
        const realGhostPct = (1 - liveSet.size / N) * 100;
        results.push({
            targetPct,
            ghostPct: realGhostPct,
            live: liveSet.size,
            recall: m.recall,
            p50: m.p50,
            p99: m.p99,
        });
        console.log(`target=${targetPct}%  ghosts=${realGhostPct.toFixed(1)}%  live=${liveSet.size}  recall=${(m.recall*100).toFixed(2)}%  p50=${m.p50.toFixed(3)}ms  p99=${m.p99.toFixed(3)}ms`);
    }

    fs.writeFileSync('/tmp/recall_dbpedia_1536d.json', JSON.stringify({ buildSec, results }, null, 2));
    process.exit(0);
})().catch(err => { console.error(err); process.exit(1); });
