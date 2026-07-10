#!/usr/bin/env node
/**
 * Pancake Worker Integration Test
 *
 * Tests the worker HTTP API with synthetic 1536D unit-normalized vectors.
 * This exercises the dedicated 1536D worker backend over HTTP without
 * depending on external embedding files.
 *
 * Works against local wrangler dev or a deployed worker.
 *
 * Usage:
 *   node test_worker.js                          # local (default)
 *   node test_worker.js http://localhost:8787    # local explicit
 *   node test_worker.js https://pancake-search.yourname.workers.dev
 */

'use strict';

const BASE_URL = process.argv[2] || 'http://localhost:8787';
const DIMS = 1536;
const BUILD_COUNT = 500;   // vectors to insert (keep small for HTTP overhead)
const SEARCH_COUNT = 20;   // searches to run
const K = 10;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function request(method, endpoint, body) {
    const url = `${BASE_URL}${endpoint}`;
    const opts = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body !== undefined) opts.body = JSON.stringify(body);

    const res = await fetch(url, opts);
    const json = await res.json();
    if (!res.ok) throw new Error(`${method} ${endpoint} → ${res.status}: ${JSON.stringify(json)}`);
    return json;
}

function loadVectors() {
    const count = 600;
    const vecs = [];
    for (let i = 0; i < count; i++) {
        const v = new Array(DIMS);
        let norm = 0;
        for (let d = 0; d < DIMS; d++) {
            v[d] = Math.random() * 2 - 1;
            norm += v[d] * v[d];
        }
        norm = Math.sqrt(norm);
        for (let d = 0; d < DIMS; d++) v[d] /= norm;
        vecs.push(v);
    }
    return vecs;
}

function cosineSim(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function bruteForcTopK(query, vecs, k) {
    return vecs
        .map((v, i) => ({ id: i, sim: cosineSim(query, v) }))
        .sort((a, b) => b.sim - a.sim)
        .slice(0, k)
        .map(r => r.id);
}

function pass(msg) { console.log(`  ✓ ${msg}`); }
function fail(msg) { console.log(`  ✗ ${msg}`); process.exitCode = 1; }

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

async function testHealth() {
    console.log('\n── Health check');
    const res = await request('GET', '/health');
    if (res.status === 'ok') pass(`status=ok`);
    else fail(`unexpected status: ${res.status}`);
    return res;
}

async function testInit(vectors) {
    console.log(`\n── Init with ${BUILD_COUNT} vectors (dims=${DIMS})`);
    const subset = vectors.slice(0, BUILD_COUNT);
    const t0 = Date.now();
    const res = await request('POST', '/init', {
        dims: DIMS,
        maxElements: BUILD_COUNT + 1000,
        M: 8,
        efConstruction: 100,
        vectors: subset
    });
    const elapsed = Date.now() - t0;

    if (res.inserted === BUILD_COUNT) pass(`inserted=${res.inserted}`);
    else fail(`inserted=${res.inserted}, expected ${BUILD_COUNT}`);

    if (res.memory_bytes > 0) pass(`memory=${(res.memory_bytes / 1024).toFixed(1)} KB`);
    else fail(`memory=0`);

    pass(`build time=${elapsed}ms`);
    return res;
}

async function testHealth2() {
    console.log('\n── Health after init');
    const res = await request('GET', '/health');
    if (res.initialized) pass(`initialized=true`);
    else fail(`initialized=false`);
    if (res.count === BUILD_COUNT) pass(`count=${res.count}`);
    else fail(`count=${res.count}, expected ${BUILD_COUNT}`);
}

async function testSearch(vectors) {
    console.log(`\n── Search (${SEARCH_COUNT} queries, k=${K})`);
    const latencies = [];
    let totalRecall = 0;

    for (let i = 0; i < SEARCH_COUNT; i++) {
        const qIdx = Math.floor(Math.random() * BUILD_COUNT);
        const query = vectors[qIdx];
        const trueTopK = new Set(bruteForcTopK(query, vectors.slice(0, BUILD_COUNT), K));

        const t0 = performance.now();
        const res = await request('POST', '/search', { query, k: K, ef: 100 });
        const latency = performance.now() - t0;
        latencies.push(latency);

        const hits = res.neighbors.filter(id => trueTopK.has(id)).length;
        totalRecall += hits / K;
    }

    latencies.sort((a, b) => a - b);
    const p50 = latencies[Math.floor(latencies.length * 0.5)];
    const p99 = latencies[Math.floor(latencies.length * 0.99)];
    const recall = (totalRecall / SEARCH_COUNT * 100).toFixed(1);

    pass(`p50=${p50.toFixed(0)}ms p99=${p99.toFixed(0)}ms (includes HTTP round-trip)`);

    if (parseFloat(recall) >= 90) pass(`recall@${K}=${recall}%`);
    else fail(`recall@${K}=${recall}% (expected ≥90%)`);
}

async function testAdd(vectors) {
    console.log('\n── Add single vector');
    const vec = vectors[BUILD_COUNT]; // one beyond what was inserted
    const res = await request('POST', '/add', { vector: vec });
    if (res.count === BUILD_COUNT + 1) pass(`count after add=${res.count}`);
    else fail(`count after add=${res.count}, expected ${BUILD_COUNT + 1}`);
    return res.id;
}

async function testExport() {
    console.log('\n── Export');
    const res = await fetch(`${BASE_URL}/export`);
    if (!res.ok) { fail(`export status=${res.status}`); return null; }

    const dims = res.headers.get('X-Pancake-Dims');
    const count = res.headers.get('X-Pancake-Count');
    const bytes = await res.arrayBuffer();

    if (bytes.byteLength > 0) pass(`exported ${bytes.byteLength} bytes`);
    else { fail('export produced 0 bytes'); return null; }

    if (dims) pass(`X-Pancake-Dims=${dims}`);
    if (count) pass(`X-Pancake-Count=${count}`);

    return { bytes, dims: parseInt(dims) };
}

async function testImport(exportData) {
    console.log('\n── Import round-trip');
    if (!exportData) { fail('skipped — export failed'); return; }

    const res = await fetch(`${BASE_URL}/import?dims=${exportData.dims}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: exportData.bytes
    });
    const json = await res.json();
    if (!res.ok) { fail(`import status=${res.status}: ${JSON.stringify(json)}`); return; }

    if (json.status === 'imported') pass(`imported successfully`);
    else fail(`unexpected status: ${json.status}`);

    if (json.count > 0) pass(`count after import=${json.count}`);
    else fail(`count after import=${json.count}`);
}

async function testSearchAfterImport(vectors) {
    console.log('\n── Search after import');
    const query = vectors[0];
    const res = await request('POST', '/search', { query, k: 5 });
    if (res.neighbors.length > 0) pass(`got ${res.neighbors.length} results`);
    else fail('no results after import');
    pass(`latency=${res.latency_ms.toFixed(2)}ms`);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
    console.log(`\nPancake Worker Integration Test`);
    console.log(`Target: ${BASE_URL}`);
    console.log(`Mode: synthetic ${DIMS}D vectors`);

    console.log('\nGenerating synthetic vectors...');
    const vectors = loadVectors();
    console.log(`Generated ${vectors.length} vectors (${DIMS}D)`);

    try {
        await testHealth();
        await testInit(vectors);
        await testHealth2();
        await testSearch(vectors);
        await testAdd(vectors);
        const exportData = await testExport();
        await testImport(exportData);
        await testSearchAfterImport(vectors);

        console.log('\n────────────────────────────────');
        if (process.exitCode === 1) {
            console.log('RESULT: SOME TESTS FAILED');
        } else {
            console.log('RESULT: ALL TESTS PASSED ✓');
        }
        console.log('────────────────────────────────\n');

    } catch (err) {
        console.error(`\nFATAL: ${err.message}`);
        process.exit(1);
    }
}

main();
