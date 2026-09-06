#!/usr/bin/env node
'use strict';

/**
 * Faiss Comparison Benchmark: Pikelet u8 vs Pikelet FP32 vs Faiss HNSW vs Faiss Flat
 *
 * Uses the DBpedia-OpenAI-1536D dataset with L2 distance.
 * Faiss HNSW uses default efConstruction=40 and efSearch=16 (not tunable
 * via faiss-node), so this is a fixed-point comparison rather than a sweep.
 * Pikelet and hnswlib are included at matched parameters for context.
 *
 * Expected files in dbpedia/:
 *   dbpedia_base_100k.fvecs - base vectors (float32, 1536D)
 *   dbpedia_base_5k.fvecs   - smaller subset for quick runs
 *   dbpedia_query.fvecs     - query vectors
 *
 * Usage:
 *   node benchmarks/benchmark_faiss.js
 *   node benchmarks/benchmark_faiss.js --count 5000
 *   node benchmarks/benchmark_faiss.js --m 16 --ef-construction 100
 */

const fs = require('fs');
const path = require('path');
const Pikelet = require('../pikelet.js');
const { parseBenchmarkArgs, resolveSingleValue } = require('./bench_args');

const parsedArgs = parseBenchmarkArgs();
const DBPEDIA_DIR = path.join(__dirname, '..', 'dbpedia');

// --- Config ---
const args = process.argv.slice(2);
function getArg(name, defaultVal) {
    const idx = args.indexOf('--' + name);
    return idx >= 0 && idx + 1 < args.length ? parseInt(args[idx + 1], 10) : defaultVal;
}

const N_BASE = getArg('count', 50_000);
const N_QUERIES = 1_000;
const K = 10;
const M = resolveSingleValue(parsedArgs.m, 16);
const EF_CONSTRUCTION = resolveSingleValue(parsedArgs.efConstruction, 100);
const EF_SEARCH = resolveSingleValue(parsedArgs.efSearch, 100);
const REPETITIONS = 3;
const WARMUP_QUERIES = 200;

// --- Optional libraries ---
let faiss;
try { faiss = require('faiss-node'); } catch (e) {}

let HierarchicalNSW;
try { HierarchicalNSW = require('hnswlib-node').HierarchicalNSW; } catch (e) {}

// --- .fvecs reader ---
function readFvecs(filePath, maxVectors) {
    console.log(`  Loading ${filePath}${maxVectors ? ` (first ${maxVectors.toLocaleString()})` : ''}...`);
    const buf = fs.readFileSync(filePath);
    const vectors = [];
    let offset = 0;
    while (offset < buf.length) {
        const dim = buf.readInt32LE(offset); offset += 4;
        const vec = new Float32Array(dim);
        for (let d = 0; d < dim; d++) {
            vec[d] = buf.readFloatLE(offset); offset += 4;
        }
        vectors.push(vec);
        if (maxVectors && vectors.length >= maxVectors) break;
    }
    return { vectors, dim: vectors[0].length };
}

// --- Ground truth (brute-force L2) ---
function computeGroundTruth(train, queries, dim) {
    console.log(`  Computing brute-force L2 ground truth (${queries.length} × ${train.length})...`);
    const t0 = performance.now();
    const gt = new Array(queries.length);
    for (let q = 0; q < queries.length; q++) {
        const query = queries[q];
        const dists = new Float32Array(train.length);
        for (let i = 0; i < train.length; i++) {
            let sum = 0;
            for (let d = 0; d < dim; d++) {
                const diff = query[d] - train[i][d];
                sum += diff * diff;
            }
            dists[i] = sum;
        }
        const indices = new Array(train.length);
        for (let i = 0; i < train.length; i++) indices[i] = i;
        indices.sort((a, b) => dists[a] - dists[b]);
        gt[q] = indices.slice(0, K);

        if ((q + 1) % 100 === 0) {
            const elapsed = (performance.now() - t0) / 1000;
            const eta = (elapsed / (q + 1)) * (queries.length - q - 1);
            process.stdout.write(`    ${q + 1}/${queries.length} (${elapsed.toFixed(0)}s, ~${eta.toFixed(0)}s left)\r`);
        }
    }
    console.log(`  Ground truth computed in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
    return gt;
}

// --- Recall helper ---
function recall(predicted, truth) {
    const truthSet = new Set(truth);
    let hits = 0;
    for (const id of predicted) if (truthSet.has(id)) hits++;
    return hits / truth.length;
}

function percentile(sorted, p) {
    return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
}

// --- Benchmark runners ---
async function benchPikelet(train, queries, groundTruth, dim, quantized) {
    const label = quantized ? 'pikelet-u8' : 'pikelet-f32';
    const index = await Pikelet.create({
        dim, maxElements: train.length, quantized,
        metric: 'l2', M, efConstruction: EF_CONSTRUCTION, efSearch: EF_SEARCH,
    });

    const t0 = performance.now();
    const batchSize = 500;
    for (let start = 0; start < train.length; start += batchSize) {
        index.addBatch(train.slice(start, Math.min(start + batchSize, train.length)));
    }
    const buildMs = performance.now() - t0;

    // Warmup
    for (let i = 0; i < WARMUP_QUERIES && i < queries.length; i++) {
        index.search(queries[i], K);
    }

    // Timed search
    const latencies = [];
    let totalRecall = 0;
    for (let rep = 0; rep < REPETITIONS; rep++) {
        for (let i = 0; i < queries.length; i++) {
            const t = performance.now();
            const results = index.search(queries[i], K);
            latencies.push(performance.now() - t);
            if (rep === 0) totalRecall += recall(results.map(r => r.id), groundTruth[i]);
        }
    }
    latencies.sort((a, b) => a - b);
    const avgRecall = totalRecall / queries.length;
    const qps = 1000 / (latencies.reduce((a, b) => a + b, 0) / latencies.length);

    index.dispose();
    return {
        label, buildMs, recall: avgRecall, qps,
        p50: percentile(latencies, 0.5),
        p99: percentile(latencies, 0.99),
        memory: null,
        params: `M=${M} efC=${EF_CONSTRUCTION} efS=${EF_SEARCH}`,
    };
}

function benchHnswlib(train, queries, groundTruth, dim) {
    if (!HierarchicalNSW) return null;
    const index = new HierarchicalNSW('l2', dim);
    index.initIndex(train.length, M, EF_CONSTRUCTION, 100);
    index.setEf(EF_SEARCH);

    const t0 = performance.now();
    for (let i = 0; i < train.length; i++) {
        index.addPoint(Array.from(train[i]), i);
    }
    const buildMs = performance.now() - t0;

    // Warmup
    for (let i = 0; i < WARMUP_QUERIES && i < queries.length; i++) {
        index.searchKnn(Array.from(queries[i]), K);
    }

    // Timed search
    const latencies = [];
    let totalRecall = 0;
    for (let rep = 0; rep < REPETITIONS; rep++) {
        for (let i = 0; i < queries.length; i++) {
            const query = Array.from(queries[i]);
            const t = performance.now();
            const results = index.searchKnn(query, K);
            latencies.push(performance.now() - t);
            if (rep === 0) totalRecall += recall(results.neighbors, groundTruth[i]);
        }
    }
    latencies.sort((a, b) => a - b);
    const avgRecall = totalRecall / queries.length;
    const qps = 1000 / (latencies.reduce((a, b) => a + b, 0) / latencies.length);

    return {
        label: 'hnswlib', buildMs, recall: avgRecall, qps,
        p50: percentile(latencies, 0.5),
        p99: percentile(latencies, 0.99),
        params: `M=${M} efC=${EF_CONSTRUCTION} efS=${EF_SEARCH}`,
    };
}

function benchFaissHNSW(train, queries, groundTruth, dim) {
    if (!faiss) return null;

    // faiss-node HNSW: M is set via factory string, efConstruction=40 and
    // efSearch=16 are faiss defaults (not tunable via faiss-node).
    const factoryM = M;
    const index = faiss.Index.fromFactory(dim, `HNSW${factoryM},Flat`, faiss.MetricType.METRIC_L2);

    const t0 = performance.now();
    // Batch add as flat Float32Array for speed
    const flat = new Float32Array(train.length * dim);
    for (let i = 0; i < train.length; i++) {
        flat.set(train[i], i * dim);
    }
    index.add(Array.from(flat));
    const buildMs = performance.now() - t0;

    // Warmup
    for (let i = 0; i < WARMUP_QUERIES && i < queries.length; i++) {
        index.search(Array.from(queries[i]), K);
    }

    // Timed search
    const latencies = [];
    let totalRecall = 0;
    for (let rep = 0; rep < REPETITIONS; rep++) {
        for (let i = 0; i < queries.length; i++) {
            const query = Array.from(queries[i]);
            const t = performance.now();
            const results = index.search(query, K);
            latencies.push(performance.now() - t);
            if (rep === 0) totalRecall += recall(results.labels, groundTruth[i]);
        }
    }
    latencies.sort((a, b) => a - b);
    const avgRecall = totalRecall / queries.length;
    const qps = 1000 / (latencies.reduce((a, b) => a + b, 0) / latencies.length);

    return {
        label: 'faiss-hnsw', buildMs, recall: avgRecall, qps,
        p50: percentile(latencies, 0.5),
        p99: percentile(latencies, 0.99),
        params: `M=${factoryM} efC=40(default) efS=16(default)`,
    };
}

function benchFaissFlat(train, queries, groundTruth, dim) {
    if (!faiss) return null;

    const index = new faiss.IndexFlatL2(dim);

    const t0 = performance.now();
    const flat = new Float32Array(train.length * dim);
    for (let i = 0; i < train.length; i++) {
        flat.set(train[i], i * dim);
    }
    index.add(Array.from(flat));
    const buildMs = performance.now() - t0;

    // Warmup
    for (let i = 0; i < WARMUP_QUERIES && i < queries.length; i++) {
        index.search(Array.from(queries[i]), K);
    }

    // Timed search
    const latencies = [];
    let totalRecall = 0;
    for (let rep = 0; rep < REPETITIONS; rep++) {
        for (let i = 0; i < queries.length; i++) {
            const query = Array.from(queries[i]);
            const t = performance.now();
            const results = index.search(query, K);
            latencies.push(performance.now() - t);
            if (rep === 0) totalRecall += recall(results.labels, groundTruth[i]);
        }
    }
    latencies.sort((a, b) => a - b);
    const avgRecall = totalRecall / queries.length;
    const qps = 1000 / (latencies.reduce((a, b) => a + b, 0) / latencies.length);

    return {
        label: 'faiss-flat', buildMs, recall: avgRecall, qps,
        p50: percentile(latencies, 0.5),
        p99: percentile(latencies, 0.99),
        params: 'exact brute-force',
    };
}

// --- Main ---
async function main() {
    // Pick dataset file based on count
    let basePath;
    if (N_BASE <= 5000) {
        basePath = path.join(DBPEDIA_DIR, 'dbpedia_base_5k.fvecs');
    } else {
        basePath = path.join(DBPEDIA_DIR, 'dbpedia_base_100k.fvecs');
    }
    const queryPath = path.join(DBPEDIA_DIR, 'dbpedia_query.fvecs');

    console.log('='.repeat(70));
    console.log('Faiss Comparison Benchmark (L2)');
    console.log('='.repeat(70));
    console.log();

    const { vectors: train, dim } = readFvecs(basePath, N_BASE);
    const { vectors: queries } = readFvecs(queryPath, N_QUERIES);

    console.log(`  Base:    ${train.length.toLocaleString()} vectors, ${dim}D`);
    console.log(`  Queries: ${queries.length.toLocaleString()}`);
    console.log(`  k=${K}, M=${M}, efC=${EF_CONSTRUCTION}, efS=${EF_SEARCH}`);
    console.log(`  Libraries: pikelet${HierarchicalNSW ? ', hnswlib' : ''}${faiss ? ', faiss' : ''}`);
    console.log();

    const groundTruth = computeGroundTruth(train, queries, dim);
    console.log();

    // Run all benchmarks
    const results = [];

    console.log('--- Pikelet u8 ---');
    results.push(await benchPikelet(train, queries, groundTruth, dim, true));

    console.log('--- Pikelet FP32 ---');
    results.push(await benchPikelet(train, queries, groundTruth, dim, false));

    if (HierarchicalNSW) {
        console.log('--- hnswlib ---');
        results.push(benchHnswlib(train, queries, groundTruth, dim));
    }

    if (faiss) {
        console.log('--- Faiss HNSW ---');
        results.push(benchFaissHNSW(train, queries, groundTruth, dim));

        console.log('--- Faiss Flat (brute-force) ---');
        results.push(benchFaissFlat(train, queries, groundTruth, dim));
    }

    // Summary table
    console.log();
    console.log('='.repeat(70));
    console.log('Results');
    console.log('='.repeat(70));
    console.log();
    console.log(
        'Library'.padEnd(16)
        + 'Build(s)'.padStart(10)
        + 'Recall'.padStart(10)
        + 'QPS'.padStart(10)
        + 'p50(ms)'.padStart(10)
        + 'p99(ms)'.padStart(10)
        + '  Params'
    );
    console.log('-'.repeat(70));
    for (const r of results) {
        if (!r) continue;
        console.log(
            r.label.padEnd(16)
            + (r.buildMs / 1000).toFixed(2).padStart(10)
            + `${(r.recall * 100).toFixed(1)}%`.padStart(10)
            + r.qps.toFixed(0).padStart(10)
            + r.p50.toFixed(3).padStart(10)
            + r.p99.toFixed(3).padStart(10)
            + `  ${r.params}`
        );
    }
}

main().catch(e => { console.error(e); process.exit(1); });
