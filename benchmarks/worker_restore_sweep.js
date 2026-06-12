#!/usr/bin/env node
'use strict';

/**
 * Worker restore sweep
 *
 * Builds synthetic quantized Pancake indexes through the Worker API,
 * persists each snapshot to the Worker's backing R2 bucket, forces the
 * in-memory cache cold, then measures:
 *   - R2 fetch time
 *   - deserialize/import time
 *   - total restore time
 *   - materialized index memory
 *   - snapshot size
 *   - cold and warm query latency
 *
 * Usage:
 *   node benchmarks/worker_restore_sweep.js [worker-url]
 *   node benchmarks/worker_restore_sweep.js --counts 1000,5000,10000
 *
 * Prerequisites:
 *   - Worker running: cd examples/worker && npx wrangler dev --port 8787
 *   - Set MAX_ELEMENTS_LIMIT in examples/worker/wrangler.toml high enough
 *     for the largest requested count.
 */

const { parseBenchmarkArgs, resolveSingleValue } = require('./bench_args');

const parsedArgs = parseBenchmarkArgs();
const POSITIONAL_ARGS = parsedArgs.args.filter((arg, idx, arr) => {
  if (!arg) return false;
  if (arg.startsWith('--')) return false;
  if (idx > 0 && arr[idx - 1] && arr[idx - 1].startsWith('--')) return false;
  return true;
});
const WORKER_URL = POSITIONAL_ARGS[0] || 'http://localhost:8787';
const COUNTS = (() => {
  const idx = process.argv.indexOf('--counts');
  if (idx !== -1 && process.argv[idx + 1]) {
    return process.argv[idx + 1]
      .split(',')
      .map((v) => parseInt(v, 10))
      .filter((v) => Number.isInteger(v) && v > 0);
  }
  return [1000, 2500, 5000];
})();
const dimArgIndex = process.argv.indexOf('--dim');
const DIM = resolveSingleValue(dimArgIndex !== -1 ? parseInt(process.argv[dimArgIndex + 1], 10) : undefined, 256);
const M = resolveSingleValue(parsedArgs.m, 12);
const EF_CONSTRUCTION = resolveSingleValue(parsedArgs.efConstruction, 100);
const EF_SEARCH = resolveSingleValue(parsedArgs.efSearch, 120);
const BATCH_SIZE = 500;
const QUERY_REPS = 10;

function log(msg = '') { console.log(msg); }

async function post(path, body = null) {
  const opts = { method: 'POST' };
  if (body !== null) {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(`${WORKER_URL}${path}`, opts);
  let data = null;
  try { data = await res.json(); } catch {}
  return { status: res.status, data };
}

async function get(path) {
  const res = await fetch(`${WORKER_URL}${path}`);
  let data = null;
  try { data = await res.json(); } catch {}
  return { status: res.status, data };
}

function normalize(vec) {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  return vec;
}

function makeRng(seed) {
  let x = seed >>> 0;
  return () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return (x >>> 0) / 4294967296;
  };
}

function makeVector(seed, dim) {
  const rand = makeRng(seed);
  const vec = new Array(dim);
  for (let i = 0; i < dim; i++) vec[i] = rand() * 2 - 1;
  return normalize(vec);
}

function percentile(sorted, p) {
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
}

function fmtMs(value) {
  return Number.isFinite(value) ? value.toFixed(2) : '-';
}

function fmtMb(value) {
  return Number.isFinite(value) ? (value / 1024 / 1024).toFixed(2) : '-';
}

async function ensureWorker() {
  const health = await get('/health').catch(() => null);
  if (!health || health.status !== 200) {
    throw new Error(`Cannot reach worker at ${WORKER_URL}`);
  }
}

async function buildIndex(count, queryVec) {
  log(`\n[count=${count}] init`);
  const initRes = await post('/init', {
    dims: DIM,
    maxElements: count + 64,
    M,
    efConstruction: EF_CONSTRUCTION,
    efSearch: EF_SEARCH
  });
  if (initRes.status !== 200) {
    throw new Error(`/init failed for count=${count}: ${JSON.stringify(initRes.data)}`);
  }
  const emptyMemoryBytes = initRes.data?.memory_bytes ?? null;

  log(`[count=${count}] insert (${BATCH_SIZE}/batch)`);
  for (let start = 0; start < count; start += BATCH_SIZE) {
    const end = Math.min(start + BATCH_SIZE, count);
    const batch = new Array(end - start);
    for (let i = start; i < end; i++) batch[i - start] = makeVector(i + 1, DIM);
    const res = await post('/add_batch', { vectors: batch });
    if (res.status !== 200) {
      throw new Error(`/add_batch failed at ${start}: ${JSON.stringify(res.data)}`);
    }
  }

  const exportRes = await fetch(`${WORKER_URL}/export`);
  if (!exportRes.ok) {
    throw new Error(`/export failed: ${exportRes.status}`);
  }
  const snapshotBytes = parseInt(exportRes.headers.get('content-length') || '0', 10);

  const resetRes = await post('/reset_cache');
  if (resetRes.status !== 200) {
    throw new Error(`/reset_cache failed: ${JSON.stringify(resetRes.data)}`);
  }

  const coldT0 = performance.now();
  const coldSearch = await post('/search', { query: queryVec, k: 10, ef: EF_SEARCH });
  const coldWallMs = performance.now() - coldT0;
  if (coldSearch.status !== 200) {
    throw new Error(`/search after reset failed: ${JSON.stringify(coldSearch.data)}`);
  }

  const health = await get('/health');
  if (health.status !== 200) {
    throw new Error(`/health failed after restore`);
  }
  const restoredMemoryBytes = health.data.memory_bytes;

  const warmLatencies = [];
  for (let i = 0; i < QUERY_REPS; i++) {
    const t0 = performance.now();
    const res = await post('/search', { query: queryVec, k: 10, ef: EF_SEARCH });
    if (res.status !== 200) throw new Error(`/search warm failed`);
    warmLatencies.push(performance.now() - t0);
  }
  warmLatencies.sort((a, b) => a - b);

  return {
    count,
    snapshotBytes,
    fetchMs: health.data.last_fetch_ms,
    deserializeMs: health.data.last_deserialize_ms,
    restoreMs: health.data.last_restore_ms,
    emptyMemoryBytes,
    restoredMemoryBytes,
    materializedDeltaBytes:
      Number.isFinite(restoredMemoryBytes) && Number.isFinite(emptyMemoryBytes)
        ? restoredMemoryBytes - emptyMemoryBytes
        : null,
    coldWallMs,
    coldSearchMs: coldSearch.data.latency_ms,
    warmP50Ms: percentile(warmLatencies, 0.5),
    warmP95Ms: percentile(warmLatencies, 0.95)
  };
}

async function main() {
  await ensureWorker();
  const queryVec = makeVector(999999, DIM);

  log('Worker restore sweep');
  log(`Worker: ${WORKER_URL}`);
  log(`dim=${DIM} M=${M} efConstruction=${EF_CONSTRUCTION} efSearch=${EF_SEARCH}`);
  log(`counts=[${COUNTS.join(', ')}]`);

  const rows = [];
  for (const count of COUNTS) rows.push(await buildIndex(count, queryVec));

  log('\nResults');
  log('count\tsnapshot_kb\tfetch_ms\tdeserialize_ms\trestore_ms\tempty_mb\trestored_mb\tdelta_mb\tcold_wall_ms\tcold_search_ms\twarm_p50_ms\twarm_p95_ms');
  for (const row of rows) {
    log(
      `${row.count}\t${(row.snapshotBytes / 1024).toFixed(1)}\t${fmtMs(row.fetchMs)}\t${fmtMs(row.deserializeMs)}\t${fmtMs(row.restoreMs)}\t${fmtMb(row.emptyMemoryBytes)}\t${fmtMb(row.restoredMemoryBytes)}\t${fmtMb(row.materializedDeltaBytes)}\t${fmtMs(row.coldWallMs)}\t${fmtMs(row.coldSearchMs)}\t${fmtMs(row.warmP50Ms)}\t${fmtMs(row.warmP95Ms)}`
    );
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
