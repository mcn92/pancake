#!/usr/bin/env node
'use strict';

/**
 * NYTimes-256 Benchmark via Worker API
 *
 * Builds the index on a running Pancake Worker, then measures search
 * recall and latency over the wire. This benchmarks the full deployment
 * path: HTTP → Worker → WASM → HNSW → response.
 *
 * Usage:
 *   node benchmark_nytimes_worker.js [worker-url] [nytimes-dir]
 *
 * Prerequisites:
 *   - Worker running: cd examples/worker && npx wrangler dev --port 8787
 *   - NYTimes fvecs/ivecs in nytimes/ directory
 */

const fs = require('fs');
const path = require('path');

const WORKER_URL = process.argv[2] || 'http://localhost:8787';
const NYTIMES_DIR = process.argv[3] || path.join(__dirname, 'nytimes');
const K = 10;
const MAX_VECTORS = parseInt(process.env.N || '30000', 10);
const BATCH_SIZE = 500;

// HNSW params
const M = 12;
const EF_CONSTRUCTION = 100;
const EF_SEARCH = 100;

function log(msg = '') { console.log(msg); }

// =============================================================================
// fvecs / ivecs readers
// =============================================================================

function readFvecs(filePath) {
  log(`  Loading ${filePath}...`);
  const buf = fs.readFileSync(filePath);
  const vectors = [];
  let offset = 0;
  while (offset < buf.length) {
    const dim = buf.readInt32LE(offset); offset += 4;
    const vec = [];
    for (let d = 0; d < dim; d++) {
      vec.push(buf.readFloatLE(offset)); offset += 4;
    }
    vectors.push(vec);
  }
  return { vectors, dim: vectors[0].length };
}

function readIvecs(filePath) {
  log(`  Loading ${filePath}...`);
  const buf = fs.readFileSync(filePath);
  const rows = [];
  let offset = 0;
  while (offset < buf.length) {
    const dim = buf.readInt32LE(offset); offset += 4;
    const row = [];
    for (let d = 0; d < dim; d++) {
      const val = buf.readInt32LE(offset); offset += 4;
      if (d < K) row.push(val);
    }
    rows.push(row);
  }
  return rows;
}

// =============================================================================
// HTTP helpers
// =============================================================================

async function post(path, body) {
  const res = await fetch(`${WORKER_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  return { status: res.status, data: await res.json() };
}

async function get(path) {
  const res = await fetch(`${WORKER_URL}${path}`);
  return { status: res.status, data: await res.json() };
}

// =============================================================================
// Benchmark
// =============================================================================

function recall(predicted, truth) {
  const truthSet = new Set(truth.slice(0, K));
  let hits = 0;
  for (const id of predicted) if (truthSet.has(id)) hits++;
  return hits / K;
}

function percentile(sorted, p) {
  return sorted[Math.floor(sorted.length * p)];
}

async function main() {
  // Check worker is up
  const health = await get('/health').catch(() => null);
  if (!health || health.status !== 200) {
    log(`Cannot reach worker at ${WORKER_URL}`);
    log('Start it with: cd examples/worker && npx wrangler dev --port 8787');
    process.exit(1);
  }

  // Load dataset
  log('Loading dataset...');
  const { vectors: baseVecs, dim } = readFvecs(path.join(NYTIMES_DIR, 'nytimes_base.fvecs'));
  const { vectors: queryVecs } = readFvecs(path.join(NYTIMES_DIR, 'nytimes_query.fvecs'));
  const groundTruth = readIvecs(path.join(NYTIMES_DIR, 'nytimes_groundtruth.ivecs'));

  const n = Math.min(baseVecs.length, MAX_VECTORS);
  const nQueries = queryVecs.length;
  log(`  Base:    ${baseVecs.length} vectors, ${dim}D (using ${n})`);
  log(`  Queries: ${nQueries}`);

  // Compute brute-force ground truth before building.
  // Ping the Worker periodically to prevent wrangler dev from recycling it.
  log(`\nComputing ground truth over ${n} vectors...`);
  const indexed = baseVecs.slice(0, n);
  // Pre-normalize base and query vectors (engine normalizes on insert for cosine)
  function normalize(v) {
    let norm = 0;
    for (let d = 0; d < v.length; d++) norm += v[d] * v[d];
    norm = Math.sqrt(norm) || 1;
    const out = new Array(v.length);
    for (let d = 0; d < v.length; d++) out[d] = v[d] / norm;
    return out;
  }

  const normBase = indexed.map(normalize);
  const normQueries = queryVecs.map(normalize);

  const localGT = new Array(nQueries);
  for (let q = 0; q < nQueries; q++) {
    const qv = normQueries[q];
    const scored = new Array(n);
    for (let i = 0; i < n; i++) {
      const bv = normBase[i];
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += qv[d] * bv[d];
      scored[i] = { id: i, dist: 1 - dot };
    }
    scored.sort((a, b) => a.dist - b.dist);
    localGT[q] = scored.slice(0, K).map(s => s.id);
    if ((q + 1) % 2000 === 0) log(`    ${q + 1}/${nQueries}`);
  }
  log('  Done.');

  // Verify worker is reachable before building
  log('\nConnecting to worker...');
  for (let attempt = 0; attempt < 5; attempt++) {
    const h = await get('/health').catch(() => null);
    if (h && h.status === 200) break;
    if (attempt === 4) { log('Worker unreachable. Restart it and try again.'); process.exit(1); }
    log(`  Retrying... (${attempt + 1}/5)`);
    await new Promise(r => setTimeout(r, 2000));
  }

  // Init index
  log(`Initializing index (M=${M}, efC=${EF_CONSTRUCTION}, efS=${EF_SEARCH})...`);
  const initRes = await post('/init', {
    dims: dim,
    maxElements: n + 1000,
    M, efConstruction: EF_CONSTRUCTION, efSearch: EF_SEARCH
  });
  if (initRes.status !== 200) {
    log(`/init failed: ${JSON.stringify(initRes.data)}`);
    process.exit(1);
  }

  // Insert vectors in batches
  log(`Building index (${n} vectors, batch=${BATCH_SIZE})...`);
  const buildStart = performance.now();
  for (let i = 0; i < n; i += BATCH_SIZE) {
    const end = Math.min(i + BATCH_SIZE, n);
    const batch = baseVecs.slice(i, end);
    const res = await post('/add_batch', { vectors: batch });
    if (res.status !== 200) {
      log(`/add_batch failed at ${i}: ${JSON.stringify(res.data)}`);
      process.exit(1);
    }
    if (end % 5000 < BATCH_SIZE) {
      const rate = (end / ((performance.now() - buildStart) / 1000)).toFixed(0);
      log(`    ${end}/${n} (${rate} vec/s)`);
    }
  }
  const buildMs = performance.now() - buildStart;
  log(`  Build: ${(buildMs / 1000).toFixed(1)}s (${(n / (buildMs / 1000)).toFixed(0)} vec/s)`);

  // Stats
  const stats = await get('/stats');
  log(`  Count: ${stats.data.count}, Memory: ${(stats.data.memory_bytes / 1024 / 1024).toFixed(1)} MB`);

  // Quick sanity check
  const probe = await post('/search', { query: queryVecs[0], k: 5, ef: EF_SEARCH });
  log(`\n  Sanity: query[0] → neighbors=${JSON.stringify(probe.data.neighbors)} dists=${JSON.stringify(probe.data.distances?.slice(0,3).map(d => d.toFixed(4)))}`);
  log(`  Ground truth[0] → ${JSON.stringify(localGT[0]?.slice(0, 5))}`);

  // Warmup
  log('\nWarming up...');
  for (let i = 0; i < 10; i++) {
    await post('/search', { query: queryVecs[i], k: K, ef: EF_SEARCH });
  }

  // Search benchmark
  log(`Searching (${nQueries} queries, k=${K}, ef=${EF_SEARCH})...`);
  const latencies = [];
  let totalRecall = 0;

  for (let i = 0; i < nQueries; i++) {
    const t0 = performance.now();
    const res = await post('/search', { query: queryVecs[i], k: K, ef: EF_SEARCH });
    const latency = performance.now() - t0;
    latencies.push(latency);

    if (res.status === 200 && res.data.neighbors) {
      totalRecall += recall(res.data.neighbors, localGT[i]);
    }

    if ((i + 1) % 2000 === 0) {
      log(`    ${i + 1}/${nQueries}`);
    }
  }

  latencies.sort((a, b) => a - b);
  const avgRecall = totalRecall / nQueries;

  // Results
  log(`\n${'='.repeat(60)}`);
  log(`NYTimes-${dim} Worker Benchmark`);
  log(`${n} vectors, ${dim}D, ${nQueries} queries, k=${K}`);
  log(`M=${M}, efConstruction=${EF_CONSTRUCTION}, efSearch=${EF_SEARCH}`);
  log(`Worker: ${WORKER_URL}`);
  log('='.repeat(60));
  log(`  Build time:    ${(buildMs / 1000).toFixed(1)}s (${(n / (buildMs / 1000)).toFixed(0)} vec/s via API)`);
  log(`  Memory:        ${(stats.data.memory_bytes / 1024 / 1024).toFixed(1)} MB`);
  log(`  Recall@${K}:    ${(avgRecall * 100).toFixed(1)}%`);
  log(`  QPS:           ${(1000 / (latencies.reduce((a, b) => a + b) / latencies.length)).toFixed(0)} (includes HTTP)`);
  log(`  p50 latency:   ${percentile(latencies, 0.5).toFixed(1)} ms`);
  log(`  p99 latency:   ${percentile(latencies, 0.99).toFixed(1)} ms`);
  log(`  max latency:   ${latencies[latencies.length - 1].toFixed(1)} ms`);
  log('='.repeat(60));
}

main().catch(err => { console.error(err); process.exit(1); });
