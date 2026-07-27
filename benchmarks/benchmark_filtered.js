#!/usr/bin/env node
/**
 * Benchmark: filtered vs unfiltered search
 *
 * Measures latency and recall across different filter selectivities.
 * Selectivity = fraction of vectors that pass the filter.
 *
 * Usage: node benchmarks/benchmark_filtered.js
 */

'use strict';

const Pancake = require('../pancake.js');
const { parseBenchmarkArgs, resolveSingleValue } = require('./bench_args');

const parsedArgs = parseBenchmarkArgs();

const DIM = 128;
const COUNT = 50000;
const QUERIES = 200;
const K = 10;
const WARMUP = 20;
const M = resolveSingleValue(parsedArgs.m, 16);
const EF_CONSTRUCTION = resolveSingleValue(parsedArgs.efConstruction, 200);
const EF_SEARCH = resolveSingleValue(parsedArgs.efSearch, 128);

function normalizedVec(dim, seed) {
    const v = new Float32Array(dim);
    for (let i = 0; i < dim; i++) v[i] = Math.sin(seed * 17.1 + i * 7.3);
    let n = 0;
    for (let i = 0; i < dim; i++) n += v[i] * v[i];
    n = Math.sqrt(n);
    for (let i = 0; i < dim; i++) v[i] /= n;
    return v;
}

function buildFilterSet(ids, selectivity) {
    const target = Math.max(1, Math.round(ids.length * selectivity));
    // Deterministic selection: every N-th ID
    const step = Math.max(1, Math.floor(ids.length / target));
    const set = new Set();
    for (let i = 0; i < ids.length && set.size < target; i += step) {
        set.add(ids[i]);
    }
    return set;
}

function cosineDist(a, b) {
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return 1 - dot;
}

// Brute-force ground truth: compute distance to every allowed vector, sort, take top k
function bruteForceFiltered(query, vecs, ids, allowedSet, k) {
    const candidates = [];
    for (let i = 0; i < ids.length; i++) {
        if (!allowedSet.has(ids[i])) continue;
        candidates.push({ id: ids[i], distance: cosineDist(query, vecs[i]) });
    }
    candidates.sort((a, b) => a.distance - b.distance);
    return candidates.slice(0, k);
}

function computeRecall(results, groundTruth) {
    if (groundTruth.length === 0) return 1.0;
    const truth = new Set(groundTruth.map(r => r.id));
    let hits = 0;
    for (const r of results) {
        if (truth.has(r.id)) hits++;
    }
    return hits / groundTruth.length;
}

function percentile(arr, p) {
    const sorted = arr.slice().sort((a, b) => a - b);
    const idx = Math.ceil(sorted.length * p / 100) - 1;
    return sorted[Math.max(0, idx)];
}

async function run() {
    console.log('Filtered Search Benchmark');
    console.log('='.repeat(70));
    console.log(`${COUNT.toLocaleString()} vectors, ${DIM}D, ${QUERIES} queries, k=${K}`);
    console.log('');

    for (const quantized of [true, false]) {
        const label = quantized ? 'uint8' : 'float32';
        console.log(`--- ${label} ---`);

        const idx = await Pancake.create({
            dim: DIM, maxElements: COUNT + 100, metric: 'cosine', quantized,
            M, efConstruction: EF_CONSTRUCTION, efSearch: EF_SEARCH,
        });

        // Insert vectors
        const ids = [];
        const vecs = [];
        process.stdout.write(`  Inserting ${COUNT.toLocaleString()} vectors...`);
        for (let i = 0; i < COUNT; i++) {
            const v = normalizedVec(DIM, i);
            ids.push(idx.add(v));
            vecs.push(v);
        }
        console.log(` done (${(idx.memory / 1024 / 1024).toFixed(1)} MB)`);

        // Generate query vectors (use vectors not in the index)
        const queryVecs = [];
        for (let i = 0; i < QUERIES + WARMUP; i++) {
            queryVecs.push(normalizedVec(DIM, COUNT + i + 1));
        }

        // Warmup
        for (let i = 0; i < WARMUP; i++) {
            idx.search(queryVecs[i], K);
        }

        // Benchmark unfiltered baseline
        const unfilteredLatencies = [];
        const unfilteredResults = [];
        for (let i = 0; i < QUERIES; i++) {
            const q = queryVecs[WARMUP + i];
            const t0 = performance.now();
            const results = idx.search(q, K);
            unfilteredLatencies.push(performance.now() - t0);
            unfilteredResults.push(results);
        }

        console.log(`  Unfiltered:    p50=${percentile(unfilteredLatencies, 50).toFixed(3)}ms  p99=${percentile(unfilteredLatencies, 99).toFixed(3)}ms`);

        // Benchmark filtered at different selectivities
        const selectivities = [0.5, 0.1, 0.01, 0.001];

        for (const sel of selectivities) {
            const filterSet = buildFilterSet(ids, sel);
            const pct = (sel * 100).toFixed(1);

            // Warmup filtered
            for (let i = 0; i < WARMUP; i++) {
                idx.searchFiltered(queryVecs[i], K, filterSet);
            }

            const latencies = [];
            let totalRecall = 0;
            let totalReturned = 0;

            let recallCount = 0;

            for (let i = 0; i < QUERIES; i++) {
                const q = queryVecs[WARMUP + i];
                const t0 = performance.now();
                const results = idx.searchFiltered(q, K, filterSet);
                latencies.push(performance.now() - t0);
                totalReturned += results.length;

                // Recall vs exhaustive filtered search (search all, then filter)
                // This measures how well iterative deepening finds filtered
                // results compared to searching the full graph and filtering after.
                const exhaustive = idx.search(q, Math.min(COUNT, 1000));
                const gt = exhaustive.filter(r => filterSet.has(r.id)).slice(0, K);
                if (gt.length > 0) {
                    totalRecall += computeRecall(results, gt);
                    recallCount++;
                }
            }

            const avgReturned = (totalReturned / QUERIES).toFixed(1);
            const recall = recallCount > 0 ? (totalRecall / recallCount * 100).toFixed(1) : 'n/a';
            const p50 = percentile(latencies, 50).toFixed(3);
            const p99 = percentile(latencies, 99).toFixed(3);
            const overhead = (percentile(latencies, 50) / percentile(unfilteredLatencies, 50)).toFixed(2);

            console.log(`  ${pct}% filter:  p50=${p50}ms  p99=${p99}ms  overhead=${overhead}x  recall=${recall}%  avg_k=${avgReturned}`);
        }

        idx.dispose();
        console.log('');
    }
}

run().catch(err => { console.error(err); process.exit(1); });
