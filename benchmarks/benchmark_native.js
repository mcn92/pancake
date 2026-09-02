#!/usr/bin/env node
'use strict';

/**
 * Pancake WASM vs Pancake Native Benchmark (DBpedia 100K, L2, 1536D)
 *
 * Same engine, same graph construction, same distance functions —
 * only difference is WASM vs native compilation. Measures the pure
 * WASM overhead on both build and search.
 *
 * Configs:
 *   1. Pancake u8  WASM    (WASM SIMD 128-bit)
 *   2. Pancake u8  Native  (SSE2 128-bit)
 *   3. Pancake FP32  WASM    (WASM SIMD 128-bit)
 *   4. Pancake FP32  Native  (SSE2 128-bit)
 *
 * Usage:
 *   node benchmarks/benchmark_native.js
 *   node benchmarks/benchmark_native.js --m 16 --ef-construction 50
 */

const fs = require('fs');
const path = require('path');
const Pancake = require('../pancake.js');
const { parseBenchmarkArgs, resolveSingleValue, resolveSweepValues } = require('./bench_args');

let native;
try {
  native = require('../native');
} catch (e) {
  console.error('ERROR: native binding not built. Run: cd native && npm install');
  process.exit(1);
}

const args = process.argv.slice(2);
function getArg(name, defaultVal) {
  const idx = args.indexOf('--' + name);
  return idx >= 0 && idx + 1 < args.length ? parseInt(args[idx + 1], 10) : defaultVal;
}

const parsedArgs = parseBenchmarkArgs();
const POSITIONAL_ARGS = parsedArgs.args.filter((arg, idx, arr) => {
  if (!arg) return false;
  if (arg.startsWith('-')) return false;
  if (idx > 0 && arr[idx - 1] === '--count') return false;
  return true;
});
const DBPEDIA_DIR = POSITIONAL_ARGS[0] || path.join(__dirname, '..', 'dbpedia');
const REGENERATE_GT = parsedArgs.args.includes('--regenerate-gt');

// --- Config ---
const N_BASE = getArg('count', 100_000);
const N_QUERIES = 1_000;
const K = 10;
const M = resolveSingleValue(parsedArgs.m, 16);
const EF_CONSTRUCTION = resolveSingleValue(parsedArgs.efConstruction, 50);
const EF_SEARCH_VALUES = resolveSweepValues(parsedArgs, [10, 20, 40, 60, 80, 100, 150, 200, 300, 500, 800]);
const REPETITIONS = 3;
const WARMUP_QUERIES = 200;

const CONFIGS = [
  { label: 'pancake-u8-wasm',   runtime: 'wasm',   dtype: 'u8'  },
  { label: 'pancake-u8-native', runtime: 'native', dtype: 'u8'  },
  { label: 'pancake-f32-wasm',    runtime: 'wasm',   dtype: 'f32' },
  { label: 'pancake-f32-native',  runtime: 'native', dtype: 'f32' },
];

// --- Logging / output ---
const RESULTS_DIR = path.join(__dirname, '..', 'benchmark_results');
if (!fs.existsSync(RESULTS_DIR)) fs.mkdirSync(RESULTS_DIR);
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const LOG_PATH  = path.join(RESULTS_DIR, `native_${timestamp}.log`);
const JSON_PATH = path.join(RESULTS_DIR, `native_${timestamp}.json`);
const CSV_PATH  = path.join(RESULTS_DIR, `native_${timestamp}.csv`);
const logStream = fs.createWriteStream(LOG_PATH);

function log(msg = '') {
  console.log(msg);
  logStream.write(msg + '\n');
}

// --- .fvecs reader ---
function readFvecs(filePath, maxVectors) {
  log(`  Loading ${filePath}${maxVectors ? ` (first ${maxVectors.toLocaleString()})` : ''}...`);
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

// --- Ground truth ---
const CACHE_DIR = path.join(RESULTS_DIR, 'cache');
if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR);
const GT_CACHE_PATH = path.join(CACHE_DIR, `gt_dbpedia_l2_n${N_BASE}_k${K}.bin`);

function saveGroundTruth(gt) {
  const buf = Buffer.alloc(8 + gt.length * K * 4);
  buf.writeUInt32LE(gt.length, 0);
  buf.writeUInt32LE(K, 4);
  let offset = 8;
  for (const row of gt) {
    for (let j = 0; j < K; j++) {
      buf.writeInt32LE(row[j], offset);
      offset += 4;
    }
  }
  fs.writeFileSync(GT_CACHE_PATH, buf);
  log(`  Cached ground truth to ${GT_CACHE_PATH}`);
}

function loadGroundTruth() {
  if (!fs.existsSync(GT_CACHE_PATH)) return null;
  const buf = fs.readFileSync(GT_CACHE_PATH);
  const nq = buf.readUInt32LE(0);
  const k = buf.readUInt32LE(4);
  if (k !== K) return null;
  const gt = new Array(nq);
  let offset = 8;
  for (let i = 0; i < nq; i++) {
    gt[i] = new Array(k);
    for (let j = 0; j < k; j++) {
      gt[i][j] = buf.readInt32LE(offset);
      offset += 4;
    }
  }
  return gt;
}

function computeGroundTruth(train, queries, dim) {
  const nq = queries.length;
  log(`Computing brute-force L2 ground truth (${nq} x ${train.length} x ${dim}D)...`);
  const t0 = performance.now();
  const gt = new Array(nq);
  for (let q = 0; q < nq; q++) {
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
    if ((q + 1) % 25 === 0) {
      const elapsed = (performance.now() - t0) / 1000;
      const eta = (elapsed / (q + 1)) * (nq - q - 1);
      process.stdout.write(`  ${q + 1}/${nq} (${elapsed.toFixed(0)}s elapsed, ~${eta.toFixed(0)}s remaining)\r`);
    }
  }
  log(`  Ground truth computed in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  return gt;
}

// --- Metric helpers ---
function recall(predicted, truth) {
  const truthSet = new Set(truth);
  let hits = 0;
  for (const id of predicted) if (truthSet.has(id)) hits++;
  return hits / truth.length;
}

function percentile(sorted, p) {
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
}

function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function stddev(arr) {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
}

// --- Pancake WASM: build + query ---
async function buildWasm({ train, dim, dtype }) {
  const quantized = dtype === 'u8';
  log(`  [wasm-${dtype}] building index (M=${M}, ef_c=${EF_CONSTRUCTION})...`);
  const index = await Pancake.create({
    dim, maxElements: train.length, quantized,
    metric: 'l2', M, efConstruction: EF_CONSTRUCTION, efSearch: EF_SEARCH_VALUES[0],
  });

  const t0 = performance.now();
  const batchSize = 500;
  for (let start = 0; start < train.length; start += batchSize) {
    const end = Math.min(start + batchSize, train.length);
    index.addBatch(train.slice(start, end));
    if (end % 10000 < batchSize) log(`    ${end.toLocaleString()}/${train.length.toLocaleString()}`);
  }
  const buildMs = performance.now() - t0;
  log(`  [wasm-${dtype}] build: ${(buildMs / 1000).toFixed(1)}s, memory: ${(index.memory / 1024 / 1024).toFixed(1)} MB`);
  return { index, buildMs, memBytes: index.memory };
}

function queryWasm(index, test, groundTruth, efSearch) {
  index.setEfSearch(efSearch);
  for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) index.search(test[i], K);
  const latencies = new Array(test.length);
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const st = performance.now();
    const results = index.search(test[i], K);
    latencies[i] = performance.now() - st;
    totalRecall += recall(results.map(r => r.id), groundTruth[i]);
  }
  return { latencies, meanRecall: totalRecall / test.length };
}

// --- Pancake Native: build + query ---
function buildNative({ train, dim, dtype }) {
  const quantized = dtype === 'u8' ? 1 : 0;
  log(`  [native-${dtype}] building index (M=${M}, ef_c=${EF_CONSTRUCTION})...`);
  const h = native.pancake_init(dim, train.length, quantized, 0 /* L2 */, M, EF_CONSTRUCTION, EF_SEARCH_VALUES[0], 108);
  if (h === 0xFFFFFFFF) throw new Error('Failed to init native index');

  const t0 = performance.now();
  // Flatten vectors into a single Float32Array for bulk_insert
  const flat = new Float32Array(train.length * dim);
  for (let i = 0; i < train.length; i++) flat.set(train[i], i * dim);

  const batchSize = 10000;
  let inserted = 0;
  for (let start = 0; start < train.length; start += batchSize) {
    const end = Math.min(start + batchSize, train.length);
    const batch = flat.subarray(start * dim, end * dim);
    inserted += native.pancake_bulk_insert(h, batch, end - start);
    if (end % 10000 === 0) log(`    ${end.toLocaleString()}/${train.length.toLocaleString()}`);
  }
  const buildMs = performance.now() - t0;
  const memBytes = native.pancake_memory(h);
  log(`  [native-${dtype}] build: ${(buildMs / 1000).toFixed(1)}s, memory: ${(memBytes / 1024 / 1024).toFixed(1)} MB`);
  return { handle: h, buildMs, memBytes };
}

function queryNative(handle, test, groundTruth, efSearch) {
  native.pancake_set_ef(handle, efSearch);
  for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) native.pancake_query(handle, test[i], K);
  const latencies = new Array(test.length);
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const st = performance.now();
    const result = native.pancake_query(handle, test[i], K);
    latencies[i] = performance.now() - st;
    totalRecall += recall(Array.from(result.ids), groundTruth[i]);
  }
  return { latencies, meanRecall: totalRecall / test.length };
}

// --- Sweep driver ---
async function sweepOne(config, dataset) {
  const { train, test, groundTruth, dim } = dataset;
  log(`\n${'='.repeat(70)}`);
  log(`Config: ${config.label}`);
  log('='.repeat(70));

  let built;
  if (config.runtime === 'wasm') {
    built = await buildWasm({ train, dim, dtype: config.dtype });
  } else {
    built = buildNative({ train, dim, dtype: config.dtype });
  }

  const points = [];

  for (const efSearch of EF_SEARCH_VALUES) {
    log(`\n  ef_search=${efSearch}`);
    const reps = [];
    for (let rep = 0; rep < REPETITIONS; rep++) {
      const { latencies, meanRecall } = config.runtime === 'wasm'
        ? queryWasm(built.index, test, groundTruth, efSearch)
        : queryNative(built.handle, test, groundTruth, efSearch);
      const sorted = [...latencies].sort((a, b) => a - b);
      const avgLatency = mean(latencies);
      reps.push({
        recall: meanRecall,
        qps: 1000 / avgLatency,
        meanLatencyMs: avgLatency,
        p50: percentile(sorted, 0.5),
        p95: percentile(sorted, 0.95),
        p99: percentile(sorted, 0.99),
      });
      log(`    rep ${rep + 1}: recall=${(meanRecall * 100).toFixed(2)}%  `
        + `qps=${(1000 / avgLatency).toFixed(0)}  `
        + `p50=${percentile(sorted, 0.5).toFixed(3)}ms  `
        + `p99=${percentile(sorted, 0.99).toFixed(3)}ms`);
    }

    const summary = {
      ef_search: efSearch,
      recall_mean: mean(reps.map(r => r.recall)),
      recall_std:  stddev(reps.map(r => r.recall)),
      qps_mean:    mean(reps.map(r => r.qps)),
      qps_std:     stddev(reps.map(r => r.qps)),
      p50_mean:    mean(reps.map(r => r.p50)),
      p95_mean:    mean(reps.map(r => r.p95)),
      p99_mean:    mean(reps.map(r => r.p99)),
      reps,
    };
    log(`    summary: recall=${(summary.recall_mean * 100).toFixed(2)}% `
      + `(+/-${(summary.recall_std * 100).toFixed(2)}) `
      + `qps=${summary.qps_mean.toFixed(0)} (+/-${summary.qps_std.toFixed(0)})`);
    points.push(summary);
  }

  // Cleanup
  if (config.runtime === 'wasm') {
    if (typeof built.index.dispose === 'function') built.index.dispose();
  } else {
    native.pancake_dispose(built.handle);
  }

  return {
    label: config.label,
    runtime: config.runtime,
    dtype: config.dtype,
    buildMs: built.buildMs,
    memMB: built.memBytes ? built.memBytes / 1024 / 1024 : null,
    params: { M, ef_construction: EF_CONSTRUCTION, K },
    points,
  };
}

// --- CSV output ---
function writeCsv(allResults, csvPath) {
  const rows = [['label', 'runtime', 'dtype', 'ef_search', 'recall', 'recall_std',
                 'qps', 'qps_std', 'p50_ms', 'p95_ms', 'p99_ms']];
  for (const r of allResults) {
    for (const p of r.points) {
      rows.push([
        r.label, r.runtime, r.dtype, p.ef_search,
        p.recall_mean.toFixed(5), p.recall_std.toFixed(5),
        p.qps_mean.toFixed(2), p.qps_std.toFixed(2),
        p.p50_mean.toFixed(4), p.p95_mean.toFixed(4), p.p99_mean.toFixed(4),
      ]);
    }
  }
  fs.writeFileSync(csvPath, rows.map(r => r.join(',')).join('\n') + '\n');
}

// --- WASM overhead summary ---
function wasmOverheadSummary(results) {
  log(`\n${'='.repeat(70)}`);
  log('WASM Overhead Analysis');
  log('='.repeat(70));

  for (const dtype of ['u8', 'f32']) {
    const wasm = results.find(r => r.runtime === 'wasm' && r.dtype === dtype);
    const nat  = results.find(r => r.runtime === 'native' && r.dtype === dtype);
    if (!wasm || !nat) continue;

    log(`\n  ${dtype.toUpperCase()} backend:`);
    log(`  Build: wasm=${(wasm.buildMs / 1000).toFixed(1)}s, native=${(nat.buildMs / 1000).toFixed(1)}s, `
      + `ratio=${(wasm.buildMs / nat.buildMs).toFixed(2)}x`);
    if (wasm.memMB && nat.memMB)
      log(`  Memory: wasm=${wasm.memMB.toFixed(1)}MB, native=${nat.memMB.toFixed(1)}MB`);

    log(`\n  ${'ef'.padEnd(6)} ${'WASM QPS'.padStart(10)} ${'Native QPS'.padStart(11)} ${'Ratio'.padStart(7)} ${'WASM R@10'.padStart(10)} ${'Native R@10'.padStart(12)}`);
    log('  ' + '-'.repeat(58));
    for (let i = 0; i < wasm.points.length; i++) {
      const w = wasm.points[i], n = nat.points[i];
      log(`  ${String(w.ef_search).padEnd(6)} ${w.qps_mean.toFixed(0).padStart(10)} ${n.qps_mean.toFixed(0).padStart(11)} `
        + `${(w.qps_mean / n.qps_mean).toFixed(2).padStart(7)}x `
        + `${(w.recall_mean * 100).toFixed(2).padStart(9)}% ${(n.recall_mean * 100).toFixed(2).padStart(11)}%`);
    }
  }
}

// --- Main ---
async function main() {
  const basePath = N_BASE <= 5_000
    ? path.join(DBPEDIA_DIR, 'dbpedia_base_5k.fvecs')
    : path.join(DBPEDIA_DIR, 'dbpedia_base_100k.fvecs');
  const queryPath = path.join(DBPEDIA_DIR, 'dbpedia_query.fvecs');
  for (const f of [basePath, queryPath]) {
    if (!fs.existsSync(f)) {
      log(`Missing file: ${f}`);
      process.exit(1);
    }
  }

  log('='.repeat(70));
  log('Pancake WASM vs Native Benchmark (DBpedia 100K, L2, 1536D)');
  log('='.repeat(70));

  log('\nLoading dataset...');
  const { vectors: train, dim } = readFvecs(basePath, N_BASE);
  const { vectors: test } = readFvecs(queryPath, N_QUERIES);

  log(`  Base:    ${train.length.toLocaleString()} vectors, ${dim}D`);
  log(`  Queries: ${test.length.toLocaleString()}`);
  log(`  Metric:  L2`);
  log(`  k=${K}, M=${M}, ef_construction=${EF_CONSTRUCTION}`);
  log(`  ef_search sweep: [${EF_SEARCH_VALUES.join(', ')}]`);

  let groundTruth;
  if (!REGENERATE_GT) groundTruth = loadGroundTruth();
  if (groundTruth) {
    log(`\nLoaded cached ground truth (${groundTruth.length} queries)`);
  } else {
    groundTruth = computeGroundTruth(train, test, dim);
    saveGroundTruth(groundTruth);
  }

  const dataset = { train, test, groundTruth, dim };

  const allResults = [];
  for (const config of CONFIGS) {
    const result = await sweepOne(config, dataset);
    allResults.push(result);
  }

  fs.writeFileSync(JSON_PATH, JSON.stringify({
    benchmark: 'pikelet-wasm-vs-native-dbpedia-100k',
    timestamp: new Date().toISOString(),
    dataset: { vectors: train.length, queries: test.length, dim, metric: 'l2' },
    params: { K, M, EF_CONSTRUCTION, EF_SEARCH_VALUES, REPETITIONS, WARMUP_QUERIES },
    results: allResults,
  }, null, 2) + '\n');
  writeCsv(allResults, CSV_PATH);

  wasmOverheadSummary(allResults);

  log(`\n${'='.repeat(70)}`);
  log('Outputs:');
  log(`  log:  ${LOG_PATH}`);
  log(`  json: ${JSON_PATH}`);
  log(`  csv:  ${CSV_PATH}`);
  logStream.end();
}

main().catch(e => { console.error(e); process.exit(1); });
