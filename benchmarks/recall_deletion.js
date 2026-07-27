#!/usr/bin/env node
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { pathToFileURL } = require('url');
const { parseBenchmarkArgs, resolveSingleValue } = require('./bench_args');

let HierarchicalNSW;
try {
    HierarchicalNSW = require('hnswlib-node').HierarchicalNSW;
} catch (e) {}

const parsedArgs = parseBenchmarkArgs();

const DIMS = 1536;
const K = 10;
const MAX_ELEM = 11_000;
const QUERY_COUNT = 100;
const FVECS_PATH = path.join(__dirname, '..', 'dbpedia', 'dbpedia_base_5k.fvecs');
const M = resolveSingleValue(parsedArgs.m, 12);
const EF_CONSTRUCTION = resolveSingleValue(parsedArgs.efConstruction, 150);
const EF_SEARCH = resolveSingleValue(parsedArgs.efSearch, 250);

async function loadPancake() {
    const mod = await import(pathToFileURL(path.join(__dirname, '..', 'pancake.node.mjs')).href);
    return mod.default;
}

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

    console.log('Loading Pancake public API...');
    const Pancake = await loadPancake();
    const index = await Pancake.create({
        dim: DIMS,
        maxElements: MAX_ELEM,
        metric: 'cosine',
        quantized: true,
        M,
        efConstruction: EF_CONSTRUCTION,
        efSearch: EF_SEARCH
    });

    console.log(`Building index with ${N} vectors at 1536D (cosine, uint8, M=${M}, ef_c=${EF_CONSTRUCTION}, ef_s=${EF_SEARCH})...`);
    const t0 = Date.now();
    index.addBatch(all.slice(0, N));
    const buildSec = (Date.now() - t0) / 1000;
    console.log(`Built in ${buildSec.toFixed(2)}s, count=${index.count}`);

    const queryIndices = [];
    for (let i = 0; i < QUERY_COUNT; i++) queryIndices.push(queryStart + i);

    // --- Build hnswlib index (if available) ---
    let hnswIndex = null;
    let hnswBuildSec = null;
    if (HierarchicalNSW) {
        hnswIndex = new HierarchicalNSW('cosine', DIMS);
        hnswIndex.initIndex(MAX_ELEM, M, EF_CONSTRUCTION, 100);
        hnswIndex.setEf(EF_SEARCH);

        const ht0 = Date.now();
        for (let i = 0; i < N; i++) {
            hnswIndex.addPoint(Array.from(all[i]), i);
        }
        hnswBuildSec = (Date.now() - ht0) / 1000;
        console.log(`Built hnswlib in ${hnswBuildSec.toFixed(2)}s, count=${N}`);
    } else {
        console.log('hnswlib-node not installed; skipping comparison');
    }

    // --- Shared live set so both engines delete the same vectors ---
    const liveSet = new Set();
    for (let i = 0; i < N; i++) liveSet.add(i);

    function computeGroundTruth(qVec) {
        const liveArr = Array.from(liveSet);
        const scored = new Array(liveArr.length);
        for (let i = 0; i < liveArr.length; i++) {
            scored[i] = { id: liveArr[i], dist: cosineDist(qVec, all[liveArr[i]]) };
        }
        scored.sort((a, b) => a.dist - b.dist);
        const trueK = Math.min(K, scored.length);
        const trueTopK = new Set();
        for (let i = 0; i < trueK; i++) trueTopK.add(scored[i].id);
        return { trueTopK, trueK };
    }

    function measurePancakeRecall() {
        let totalRecall = 0;
        const latencies = [];
        for (const qIdx of queryIndices) {
            const qVec = all[qIdx];
            const { trueTopK, trueK } = computeGroundTruth(qVec);

            const t = performance.now();
            const found = index.search(qVec, K);
            latencies.push(performance.now() - t);

            let hits = 0;
            for (const result of found) {
                if (trueTopK.has(result.id)) hits++;
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

    function measureHnswlibRecall(ef) {
        hnswIndex.setEf(ef);
        let totalRecall = 0;
        const latencies = [];
        for (const qIdx of queryIndices) {
            const qVec = all[qIdx];
            const { trueTopK, trueK } = computeGroundTruth(qVec);

            const t = performance.now();
            const res = hnswIndex.searchKnn(Array.from(qVec), K);
            latencies.push(performance.now() - t);

            let hits = 0;
            for (let i = 0; i < res.neighbors.length; i++) {
                if (trueTopK.has(res.neighbors[i])) hits++;
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

    const HNSW_EF_SWEEP = [20, 40, 60, 100, EF_SEARCH];

    function deleteRandomLive(n) {
        const liveArr = Array.from(liveSet);
        const actual = Math.min(n, liveArr.length - K);
        if (actual <= 0) return 0;
        // Fisher-Yates partial shuffle to pick `actual` random live IDs
        for (let i = 0; i < actual; i++) {
            const j = i + Math.floor(Math.random() * (liveArr.length - i));
            [liveArr[i], liveArr[j]] = [liveArr[j], liveArr[i]];
        }
        for (let i = 0; i < actual; i++) {
            index.delete(liveArr[i]);
            if (hnswIndex) hnswIndex.markDelete(liveArr[i]);
            liveSet.delete(liveArr[i]);
        }
        return actual;
    }

    const ghostTargets = [0, 5, 10, 20, 30, 50, 70, 80, 85, 90, 93, 95];
    const results = [];
    // hnswResults keyed by ef: { 20: [...], 40: [...], ... }
    const hnswResults = hnswIndex ? {} : null;
    if (hnswResults) for (const ef of HNSW_EF_SWEEP) hnswResults[ef] = [];

    for (const targetPct of ghostTargets) {
        const targetGhosts = Math.floor(N * targetPct / 100);
        const currentGhosts = N - liveSet.size;
        const toDelete = targetGhosts - currentGhosts;
        if (toDelete > 0) deleteRandomLive(toDelete);

        const realGhostPct = (1 - liveSet.size / N) * 100;

        const pm = measurePancakeRecall();
        results.push({
            targetPct, ghostPct: realGhostPct, live: liveSet.size,
            recall: pm.recall, p50: pm.p50, p99: pm.p99,
        });

        if (hnswIndex) {
            const parts = [];
            for (const ef of HNSW_EF_SWEEP) {
                const hm = measureHnswlibRecall(ef);
                hnswResults[ef].push({
                    targetPct, ghostPct: realGhostPct, live: liveSet.size,
                    recall: hm.recall, p50: hm.p50, p99: hm.p99,
                });
                parts.push(`ef${ef}=${(hm.recall*100).toFixed(1)}%`);
            }
            console.log(
                `target=${targetPct}%  ghosts=${realGhostPct.toFixed(1)}%  live=${liveSet.size}  `
                + `pancake(ef${EF_SEARCH})=${(pm.recall*100).toFixed(1)}%  `
                + `hnswlib: ${parts.join('  ')}`
            );
        } else {
            console.log(`target=${targetPct}%  ghosts=${realGhostPct.toFixed(1)}%  live=${liveSet.size}  recall=${(pm.recall*100).toFixed(2)}%  p50=${pm.p50.toFixed(3)}ms  p99=${pm.p99.toFixed(3)}ms`);
        }
    }

    // --- Summary table ---
    if (hnswResults) {
        const efCols = HNSW_EF_SWEEP.map(ef => `hnsw ef=${ef}`);
        console.log('\n--- Side-by-side (recall) ---');
        console.log(`Ghost %  | Pancake ef=${EF_SEARCH} | ${efCols.join(' | ')}`);
        console.log(`---------|${'-'.repeat(15)}|${efCols.map(() => '-'.repeat(14)).join('|')}`);
        for (let i = 0; i < results.length; i++) {
            const p = results[i];
            const hCols = HNSW_EF_SWEEP.map(ef =>
                `${(hnswResults[ef][i].recall*100).toFixed(2).padStart(12)}%`
            );
            console.log(
                `${String(p.targetPct).padStart(6)}%  | `
                + `${(p.recall*100).toFixed(2).padStart(12)}% | `
                + hCols.join(' | ')
            );
        }

        console.log('\n--- Side-by-side (p50 latency) ---');
        console.log(`Ghost %  | Pancake ef=${EF_SEARCH} | ${efCols.join(' | ')}`);
        console.log(`---------|${'-'.repeat(15)}|${efCols.map(() => '-'.repeat(14)).join('|')}`);
        for (let i = 0; i < results.length; i++) {
            const p = results[i];
            const hCols = HNSW_EF_SWEEP.map(ef =>
                `${hnswResults[ef][i].p50.toFixed(3).padStart(11)}ms`
            );
            console.log(
                `${String(p.targetPct).padStart(6)}%  | `
                + `${p.p50.toFixed(3).padStart(11)}ms | `
                + hCols.join(' | ')
            );
        }
    }

    fs.writeFileSync(path.join(os.tmpdir(), 'recall_dbpedia_1536d.json'), JSON.stringify({
        pancakeBuildSec: buildSec,
        hnswlibBuildSec: hnswBuildSec,
        pancake: results,
        hnswlib: hnswResults,
    }, null, 2));
    index.dispose();
    process.exit(0);
})().catch(err => { console.error(err); process.exit(1); });
