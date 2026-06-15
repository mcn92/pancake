#!/usr/bin/env node
'use strict';

/**
 * Pareto-Frontier Benchmark Harness (QPS-Recall) — DBpedia-50k, L2, 1536D
 *
 * Produces, for a fixed graph (M=16, ef_construction=50):
 *   1. A QPS-recall sweep for every library that exposes a query-time knob
 *      (ef_search swept across a standard range), and
 *   2. The Pareto frontier over each library's sweep (non-dominated points), and
 *   3. Interpolated equal-recall curves: QPS for every library at a common grid
 *      of recall targets, log-linearly interpolated along each frontier.
 *
 * Configs (8):
 *   pancake-wasm   int8 / fp32   (Pancake.create, _setEfSearch per ef)
 *   pancake-native int8 / fp32   (native.pancake_*, pancake_set_ef per ef)
 *   usearch        i8 / f32      (build-once-save-view per ef — JS binding
 *                                 only honors expansion_search at construction)
 *   hnswlib        f32           (HierarchicalNSW.setEf per ef)
 *   faiss-node                   (HNSW16,Flat — SINGLE POINT, see note below)
 *
 * faiss-node NOTE: faiss-node does not expose efConstruction or efSearch.
 *   Its HNSW uses the faiss defaults (efConstruction=40, efSearch=16) and only
 *   M is settable (via the "HNSW<M>,Flat" factory string). It therefore cannot
 *   be swept — it contributes a SINGLE (recall, qps) point and is explicitly
 *   excluded from the ef_search sweep and from equal-recall interpolation
 *   (a single point has no curve to interpolate along). This is flagged in
 *   every output. All other libraries use M=16, ef_construction=50, and the
 *   ef_search sweep below (which includes ef_search=100).
 *
 * Usage:
 *   node benchmarks/pareto_frontier.js
 *   node benchmarks/pareto_frontier.js --count 50000
 *   node benchmarks/pareto_frontier.js --ef-search-values 10,50,100,200
 *   node benchmarks/pareto_frontier.js --regenerate-gt
 *
 * Plot with:
 *   python3 benchmarks/plot_pareto.py benchmark_results/pareto_<ts>.csv
 */

const fs = require('fs');
const path = require('path');
const Pancake = require('../pancake.js');
const { parseBenchmarkArgs, resolveSingleValue, resolveSweepValues } = require('./bench_args');

// --- Optional libraries (each is independently optional; missing => skipped) ---
let native;
try { native = require('../native'); }
catch (e) { console.warn('WARN: pancake native binding not built (cd native && npm install) — skipping pancake-native configs.'); }

let usearch;
try { usearch = require('usearch'); }
catch (e) { console.warn('WARN: usearch not installed — skipping usearch configs.'); }

let HierarchicalNSW;
try { HierarchicalNSW = require('hnswlib-node').HierarchicalNSW; }
catch (e) { console.warn('WARN: hnswlib-node not installed — skipping hnswlib config.'); }

let faiss;
try { faiss = require('faiss-node'); }
catch (e) { console.warn('WARN: faiss-node not installed — skipping faiss config.'); }

// --- Args / config ---
const parsedArgs = parseBenchmarkArgs();
const rawArgs = process.argv.slice(2);
function getIntArg(name, defaultVal) {
  const idx = rawArgs.indexOf('--' + name);
  return idx >= 0 && idx + 1 < rawArgs.length ? parseInt(rawArgs[idx + 1], 10) : defaultVal;
}
function getStrArg(name, defaultVal) {
  const idx = rawArgs.indexOf('--' + name);
  return idx >= 0 && idx + 1 < rawArgs.length ? rawArgs[idx + 1] : defaultVal;
}
const REGENERATE_GT = rawArgs.includes('--regenerate-gt');

// --- Dataset selection ---------------------------------------------------
// --dataset dbpedia (default) or sift. SIFT ships a precomputed .ivecs ground
// truth (per-query, order-aligned), so it is read from disk rather than
// brute-forced; dbpedia has no shipped GT, so it is computed and cached.
const DATASET = getStrArg('dataset', 'dbpedia').toLowerCase();
const DATASETS = {
  dbpedia: {
    dir: path.join(__dirname, '..', 'dbpedia'),
    baseFile: (n) => (n <= 5000 ? 'dbpedia_base_5k.fvecs' : 'dbpedia_base_100k.fvecs'),
    queryFile: 'dbpedia_query.fvecs',
    gtFile: null,                 // computed
    defaultCount: 50_000,
    metric: 'l2',
  },
  sift: {
    dir: path.join(__dirname, '..', 'sift'),
    baseFile: () => 'sift_base.fvecs',
    queryFile: 'sift_query.fvecs',
    gtFile: 'sift_groundtruth.ivecs',  // precomputed, read from disk
    defaultCount: 1_000_000,
    metric: 'l2',
  },
};
if (!DATASETS[DATASET]) {
  console.error(`Unknown --dataset ${DATASET}. Use 'dbpedia' or 'sift'.`);
  process.exit(1);
}
const DS = DATASETS[DATASET];
// Optional positional override for the dataset directory.
const DATA_DIR =
  parsedArgs.args.find((a, i, arr) => a && !a.startsWith('-') && !(i > 0 && arr[i - 1] === '--count'))
  || DS.dir;

const N_BASE = getIntArg('count', DS.defaultCount);
const N_QUERIES = 1_000;
const K = 10;
const METRIC = DS.metric;

// Fixed graph parameters, per request.
const M = resolveSingleValue(parsedArgs.m, 16);
const EF_CONSTRUCTION = resolveSingleValue(parsedArgs.efConstruction, 50);
// ef_search sweep — includes 100 (the requested operating point) plus the
// surrounding range needed to trace a QPS-recall frontier.
const EF_SEARCH_VALUES = resolveSweepValues(parsedArgs, [10, 20, 40, 60, 80, 100, 150, 200, 300, 500, 800]);

const REPETITIONS = 3;
const WARMUP_QUERIES = 200;

// faiss is not tunable; record the defaults it actually uses for the record.
const FAISS_EF_CONSTRUCTION_DEFAULT = 40;
const FAISS_EF_SEARCH_DEFAULT = 16;

// --- Config table. `sweep: false` => single fixed point (faiss only). ---
const CONFIGS = [];
CONFIGS.push({ label: 'pancake-wasm-int8',   library: 'pancake', runtime: 'wasm',   dtype: 'i8',  sweep: true });
CONFIGS.push({ label: 'pancake-wasm-f32',    library: 'pancake', runtime: 'wasm',   dtype: 'f32', sweep: true });
if (native) {
  CONFIGS.push({ label: 'pancake-native-int8', library: 'pancake', runtime: 'native', dtype: 'i8',  sweep: true });
  CONFIGS.push({ label: 'pancake-native-f32',  library: 'pancake', runtime: 'native', dtype: 'f32', sweep: true });
}
if (usearch) {
  CONFIGS.push({ label: 'usearch-i8',  library: 'usearch', dtype: 'i8',  sweep: true });
  CONFIGS.push({ label: 'usearch-f32', library: 'usearch', dtype: 'f32', sweep: true });
}
if (HierarchicalNSW) {
  CONFIGS.push({ label: 'hnswlib-f32', library: 'hnswlib', dtype: 'f32', sweep: true });
}
if (faiss) {
  CONFIGS.push({ label: 'faiss-node', library: 'faiss', dtype: 'f32', sweep: false });
}

// --- Output paths ---
const RESULTS_DIR = path.join(__dirname, '..', 'benchmark_results');
if (!fs.existsSync(RESULTS_DIR)) fs.mkdirSync(RESULTS_DIR);
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const LOG_PATH      = path.join(RESULTS_DIR, `pareto_${timestamp}.log`);
const JSON_PATH     = path.join(RESULTS_DIR, `pareto_${timestamp}.json`);
const CSV_PATH      = path.join(RESULTS_DIR, `pareto_${timestamp}.csv`);
const FRONTIER_CSV  = path.join(RESULTS_DIR, `pareto_${timestamp}_frontier.csv`);
const EQRECALL_CSV  = path.join(RESULTS_DIR, `pareto_${timestamp}_equalrecall.csv`);
const logStream = fs.createWriteStream(LOG_PATH);
function log(msg = '') { console.log(msg); logStream.write(msg + '\n'); }

// --- .fvecs reader ---
function readFvecs(filePath, maxVectors) {
  log(`  Loading ${filePath}${maxVectors ? ` (first ${maxVectors.toLocaleString()})` : ''}...`);
  const buf = fs.readFileSync(filePath);
  const vectors = [];
  let offset = 0;
  while (offset < buf.length) {
    const dim = buf.readInt32LE(offset); offset += 4;
    const vec = new Float32Array(dim);
    for (let d = 0; d < dim; d++) { vec[d] = buf.readFloatLE(offset); offset += 4; }
    vectors.push(vec);
    if (maxVectors && vectors.length >= maxVectors) break;
  }
  return { vectors, dim: vectors[0].length };
}

// --- .ivecs reader (int32 ground-truth neighbor IDs) ---
// Reads the first `maxRows` rows, keeping the first K neighbor IDs per row.
function readIvecs(filePath, maxRows) {
  log(`  Loading ${filePath} (ground truth)...`);
  const buf = fs.readFileSync(filePath);
  const rows = [];
  let offset = 0;
  while (offset < buf.length) {
    const dim = buf.readInt32LE(offset); offset += 4;
    const row = new Array(Math.min(dim, K));
    for (let d = 0; d < dim; d++) {
      const v = buf.readInt32LE(offset); offset += 4;
      if (d < K) row[d] = v;
    }
    rows.push(row);
    if (maxRows && rows.length >= maxRows) break;
  }
  return rows;
}

// --- Ground truth (brute-force L2, cached) ---
const CACHE_DIR = path.join(RESULTS_DIR, 'cache');
if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR);
const GT_CACHE_PATH = path.join(CACHE_DIR, `gt_${DATASET}_l2_n${N_BASE}_q${N_QUERIES}_k${K}.bin`);

function saveGroundTruth(gt) {
  const buf = Buffer.alloc(8 + gt.length * K * 4);
  buf.writeUInt32LE(gt.length, 0);
  buf.writeUInt32LE(K, 4);
  let offset = 8;
  for (const row of gt) for (let j = 0; j < K; j++) { buf.writeInt32LE(row[j], offset); offset += 4; }
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
    for (let j = 0; j < k; j++) { gt[i][j] = buf.readInt32LE(offset); offset += 4; }
  }
  return gt;
}
function computeGroundTruth(train, queries, dim) {
  const nq = queries.length;
  log(`Computing brute-force L2 ground truth (${nq} x ${train.length} x ${dim}D)... this can take minutes.`);
  const t0 = performance.now();
  const gt = new Array(nq);
  for (let q = 0; q < nq; q++) {
    const query = queries[q];
    const dists = new Float32Array(train.length);
    for (let i = 0; i < train.length; i++) {
      let sum = 0;
      for (let d = 0; d < dim; d++) { const diff = query[d] - train[i][d]; sum += diff * diff; }
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
function percentile(sorted, p) { return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))]; }
function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function stddev(arr) { const m = mean(arr); return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length); }

// =====================================================================
// Per-library build + query adapters.
// Each build* returns a handle object; each query* returns {latencies, meanRecall}.
// =====================================================================

// --- Pancake WASM ---
async function buildPancakeWasm({ train, dim, dtype }) {
  const quantized = dtype === 'i8';
  log(`  [${dtype} wasm] build (M=${M}, ef_c=${EF_CONSTRUCTION}, metric=${METRIC})...`);
  const index = await Pancake.create({
    dim, maxElements: train.length, quantized, metric: METRIC,
    M, efConstruction: EF_CONSTRUCTION, efSearch: EF_SEARCH_VALUES[0],
  });
  const t0 = performance.now();
  const batchSize = 500;
  for (let start = 0; start < train.length; start += batchSize) {
    index.addBatch(train.slice(start, Math.min(start + batchSize, train.length)));
  }
  const buildMs = performance.now() - t0;
  log(`  [${dtype} wasm] build: ${(buildMs / 1000).toFixed(1)}s, memory: ${(index.memory / 1024 / 1024).toFixed(1)} MB`);
  return { index, buildMs, memBytes: index.memory };
}
function queryPancakeWasm(built, test, gt, ef) {
  built.index._setEfSearch(ef);
  for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) built.index.search(test[i], K);
  const latencies = new Array(test.length);
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const st = performance.now();
    const results = built.index.search(test[i], K);
    latencies[i] = performance.now() - st;
    totalRecall += recall(results.map(r => r.id), gt[i]);
  }
  return { latencies, meanRecall: totalRecall / test.length };
}

// --- Pancake native ---
function buildPancakeNative({ train, dim, dtype }) {
  const quantized = dtype === 'i8' ? 1 : 0;
  log(`  [${dtype} native] build (M=${M}, ef_c=${EF_CONSTRUCTION}, metric=${METRIC})...`);
  const h = native.pancake_init(dim, train.length, quantized, 0 /* 0=L2, 1=cosine */, M, EF_CONSTRUCTION, EF_SEARCH_VALUES[0]);
  if (h === 0xFFFFFFFF) throw new Error('Failed to init native index');
  const t0 = performance.now();
  const flat = new Float32Array(train.length * dim);
  for (let i = 0; i < train.length; i++) flat.set(train[i], i * dim);
  const batchSize = 10000;
  for (let start = 0; start < train.length; start += batchSize) {
    const end = Math.min(start + batchSize, train.length);
    native.pancake_bulk_insert(h, flat.subarray(start * dim, end * dim), end - start);
  }
  const buildMs = performance.now() - t0;
  const memBytes = native.pancake_memory(h);
  log(`  [${dtype} native] build: ${(buildMs / 1000).toFixed(1)}s, memory: ${(memBytes / 1024 / 1024).toFixed(1)} MB`);
  return { handle: h, buildMs, memBytes };
}
function queryPancakeNative(built, test, gt, ef) {
  native.pancake_set_ef(built.handle, ef);
  for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) native.pancake_query(built.handle, test[i], K);
  const latencies = new Array(test.length);
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const st = performance.now();
    const result = native.pancake_query(built.handle, test[i], K);
    latencies[i] = performance.now() - st;
    totalRecall += recall(Array.from(result.ids), gt[i]);
  }
  return { latencies, meanRecall: totalRecall / test.length };
}

// --- USearch ---
// The JS binding only honors expansion_search at construction. We build once,
// save to disk, then view() with a fresh index per ef_search value.
function buildUsearch({ train, dim, dtype }) {
  const quantization = dtype === 'i8' ? 'i8' : 'f32';
  log(`  [usearch-${dtype}] build (connectivity=${M}, expansion_add=${EF_CONSTRUCTION}, metric=l2sq, quantization=${quantization})...`);
  const index = new usearch.Index({
    metric: 'l2sq', connectivity: M, dimensions: dim,
    quantization, expansion_add: EF_CONSTRUCTION, expansion_search: EF_SEARCH_VALUES[0],
  });
  const t0 = performance.now();
  for (let i = 0; i < train.length; i++) index.add(BigInt(i), train[i]);
  const buildMs = performance.now() - t0;
  const savePath = path.join(RESULTS_DIR, `_usearch_${dtype}_${timestamp}.bin`);
  index.save(savePath);
  log(`  [usearch-${dtype}] build: ${(buildMs / 1000).toFixed(1)}s (saved to ${savePath})`);
  return { buildMs, memBytes: null, savePath, quantization, dim };
}
function queryUsearch(built, test, gt, ef) {
  const view = new usearch.Index({
    metric: 'l2sq', connectivity: M, dimensions: built.dim,
    quantization: built.quantization, expansion_search: ef,
  });
  view.view(built.savePath);
  for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) view.search(test[i], K);
  const latencies = new Array(test.length);
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const st = performance.now();
    const results = view.search(test[i], K);
    latencies[i] = performance.now() - st;
    totalRecall += recall(Array.from(results.keys).map(Number), gt[i]);
  }
  return { latencies, meanRecall: totalRecall / test.length };
}

// --- hnswlib-node ---
function buildHnswlib({ train, dim }) {
  log(`  [hnswlib-f32] build (M=${M}, ef_c=${EF_CONSTRUCTION}, metric=l2)...`);
  const index = new HierarchicalNSW('l2', dim);
  index.initIndex(train.length, M, EF_CONSTRUCTION, 100);
  const t0 = performance.now();
  for (let i = 0; i < train.length; i++) index.addPoint(Array.from(train[i]), i);
  const buildMs = performance.now() - t0;
  log(`  [hnswlib-f32] build: ${(buildMs / 1000).toFixed(1)}s`);
  return { index, buildMs, memBytes: null };
}
function queryHnswlib(built, test, gt, ef) {
  built.index.setEf(ef);
  for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) built.index.searchKnn(Array.from(test[i]), K);
  const latencies = new Array(test.length);
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const query = Array.from(test[i]);
    const st = performance.now();
    const results = built.index.searchKnn(query, K);
    latencies[i] = performance.now() - st;
    totalRecall += recall(results.neighbors, gt[i]);
  }
  return { latencies, meanRecall: totalRecall / test.length };
}

// --- faiss-node (single fixed point; ef NOT tunable) ---
function buildFaiss({ train, dim }) {
  log(`  [faiss-node] build (HNSW${M},Flat — efC=${FAISS_EF_CONSTRUCTION_DEFAULT}/efS=${FAISS_EF_SEARCH_DEFAULT} are faiss defaults, NOT tunable)...`);
  const index = faiss.Index.fromFactory(dim, `HNSW${M},Flat`, faiss.MetricType.METRIC_L2);
  const t0 = performance.now();
  const flat = new Float32Array(train.length * dim);
  for (let i = 0; i < train.length; i++) flat.set(train[i], i * dim);
  index.add(Array.from(flat));
  const buildMs = performance.now() - t0;
  log(`  [faiss-node] build: ${(buildMs / 1000).toFixed(1)}s`);
  return { index, buildMs, memBytes: null };
}
function queryFaiss(built, test, gt) {
  for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) built.index.search(Array.from(test[i]), K);
  const latencies = new Array(test.length);
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const query = Array.from(test[i]);
    const st = performance.now();
    const results = built.index.search(query, K);
    latencies[i] = performance.now() - st;
    totalRecall += recall(results.labels, gt[i]);
  }
  return { latencies, meanRecall: totalRecall / test.length };
}

// --- Dispatch ---
async function build(config, dataset) {
  const { train, dim } = dataset;
  switch (config.library) {
    case 'pancake':
      return config.runtime === 'wasm'
        ? await buildPancakeWasm({ train, dim, dtype: config.dtype })
        : buildPancakeNative({ train, dim, dtype: config.dtype });
    case 'usearch': return buildUsearch({ train, dim, dtype: config.dtype });
    case 'hnswlib': return buildHnswlib({ train, dim });
    case 'faiss':   return buildFaiss({ train, dim });
  }
}
function query(config, built, dataset, ef) {
  const { test, groundTruth } = dataset;
  switch (config.library) {
    case 'pancake':
      return config.runtime === 'wasm'
        ? queryPancakeWasm(built, test, groundTruth, ef)
        : queryPancakeNative(built, test, groundTruth, ef);
    case 'usearch': return queryUsearch(built, test, groundTruth, ef);
    case 'hnswlib': return queryHnswlib(built, test, groundTruth, ef);
    case 'faiss':   return queryFaiss(built, test, groundTruth);
  }
}
function cleanup(config, built) {
  if (config.library === 'pancake' && config.runtime === 'native') native.pancake_dispose(built.handle);
  else if (built.index && typeof built.index.dispose === 'function') built.index.dispose();
  if (built.savePath && fs.existsSync(built.savePath)) fs.unlinkSync(built.savePath);
}

// =====================================================================
// Sweep driver
// =====================================================================
async function sweepOne(config, dataset) {
  log(`\n${'='.repeat(70)}`);
  log(`Config: ${config.label}${config.sweep ? '' : '  (single fixed point — not tunable)'}`);
  log('='.repeat(70));

  const built = await build(config, dataset);
  const efValues = config.sweep ? EF_SEARCH_VALUES : [FAISS_EF_SEARCH_DEFAULT];
  const points = [];

  for (const ef of efValues) {
    log(`\n  ef_search=${ef}${config.sweep ? '' : ' (faiss default, fixed)'}`);
    const reps = [];
    for (let rep = 0; rep < REPETITIONS; rep++) {
      const { latencies, meanRecall } = query(config, built, dataset, ef);
      const sorted = [...latencies].sort((a, b) => a - b);
      const avgLatency = mean(latencies);
      reps.push({
        recall: meanRecall, qps: 1000 / avgLatency, meanLatencyMs: avgLatency,
        p50: percentile(sorted, 0.5), p95: percentile(sorted, 0.95), p99: percentile(sorted, 0.99),
      });
      log(`    rep ${rep + 1}: recall=${(meanRecall * 100).toFixed(2)}%  qps=${(1000 / avgLatency).toFixed(0)}  `
        + `p50=${percentile(sorted, 0.5).toFixed(3)}ms  p99=${percentile(sorted, 0.99).toFixed(3)}ms`);
    }
    const summary = {
      ef_search: ef,
      recall_mean: mean(reps.map(r => r.recall)), recall_std: stddev(reps.map(r => r.recall)),
      qps_mean: mean(reps.map(r => r.qps)),       qps_std: stddev(reps.map(r => r.qps)),
      p50_mean: mean(reps.map(r => r.p50)), p95_mean: mean(reps.map(r => r.p95)), p99_mean: mean(reps.map(r => r.p99)),
      reps,
    };
    log(`    summary: recall=${(summary.recall_mean * 100).toFixed(2)}% (+/-${(summary.recall_std * 100).toFixed(2)}) `
      + `qps=${summary.qps_mean.toFixed(0)} (+/-${summary.qps_std.toFixed(0)})`);
    points.push(summary);
  }

  cleanup(config, built);

  return {
    label: config.label, library: config.library, runtime: config.runtime || null,
    dtype: config.dtype, tunable: config.sweep,
    buildMs: built.buildMs, memMB: built.memBytes ? built.memBytes / 1024 / 1024 : null,
    params: config.sweep
      ? { M, ef_construction: EF_CONSTRUCTION, K }
      : { M, ef_construction: FAISS_EF_CONSTRUCTION_DEFAULT, ef_search: FAISS_EF_SEARCH_DEFAULT, K, note: 'faiss-node defaults; not tunable' },
    points,
  };
}

// =====================================================================
// Analysis: Pareto frontier + interpolated equal-recall curves
// =====================================================================

// A point dominates another if it has >= recall AND >= qps (and strictly better
// on at least one). The frontier is the set of non-dominated points.
function paretoFrontier(points) {
  const pts = points.map(p => ({ ef_search: p.ef_search, recall: p.recall_mean, qps: p.qps_mean }));
  const frontier = pts.filter(a =>
    !pts.some(b => b !== a && b.recall >= a.recall && b.qps >= a.qps && (b.recall > a.recall || b.qps > a.qps)));
  // Sort along recall for a clean curve.
  return frontier.sort((x, y) => x.recall - y.recall);
}

// Log-linear interpolation of QPS at a target recall along a frontier.
// Returns null if the target falls outside the frontier's recall span.
function qpsAtRecall(frontier, targetRecall) {
  if (frontier.length === 0) return null;
  if (frontier.length === 1) {
    return Math.abs(frontier[0].recall - targetRecall) < 1e-9 ? frontier[0].qps : null;
  }
  const lo = frontier[0].recall, hi = frontier[frontier.length - 1].recall;
  if (targetRecall < lo - 1e-9 || targetRecall > hi + 1e-9) return null;
  for (let i = 1; i < frontier.length; i++) {
    if (frontier[i].recall >= targetRecall) {
      const r0 = frontier[i - 1].recall, r1 = frontier[i].recall;
      const q0 = frontier[i - 1].qps, q1 = frontier[i].qps;
      if (r1 === r0) return Math.max(q0, q1);
      const t = (targetRecall - r0) / (r1 - r0);
      return Math.exp(Math.log(q0) + t * (Math.log(q1) - Math.log(q0))); // log-linear in QPS
    }
  }
  return frontier[frontier.length - 1].qps;
}

function buildAnalysis(allResults) {
  const sweepable = allResults.filter(r => r.tunable && r.points.length >= 2);
  const frontiers = {};
  for (const r of allResults) frontiers[r.label] = paretoFrontier(r.points);

  // Common recall grid for equal-recall curves: span the overlap of all
  // sweepable frontiers, sampled at a fixed set of targets.
  const candidateTargets = [0.80, 0.85, 0.90, 0.925, 0.95, 0.965, 0.98, 0.99, 0.995];
  const equalRecall = candidateTargets.map(target => {
    const row = { recall: target };
    for (const r of sweepable) row[r.label] = qpsAtRecall(frontiers[r.label], target);
    return row;
  });

  return { frontiers, equalRecall, sweepableLabels: sweepable.map(r => r.label) };
}

// =====================================================================
// Output writers
// =====================================================================
function writeRawCsv(allResults, p) {
  const rows = [['label', 'library', 'runtime', 'dtype', 'tunable', 'ef_search',
                 'recall', 'recall_std', 'qps', 'qps_std', 'p50_ms', 'p95_ms', 'p99_ms']];
  for (const r of allResults) for (const pt of r.points) {
    rows.push([
      r.label, r.library, r.runtime || '', r.dtype, r.tunable, pt.ef_search,
      pt.recall_mean.toFixed(5), pt.recall_std.toFixed(5),
      pt.qps_mean.toFixed(2), pt.qps_std.toFixed(2),
      pt.p50_mean.toFixed(4), pt.p95_mean.toFixed(4), pt.p99_mean.toFixed(4),
    ]);
  }
  fs.writeFileSync(p, rows.map(r => r.join(',')).join('\n') + '\n');
}

function writeFrontierCsv(frontiers, p) {
  const rows = [['label', 'ef_search', 'recall', 'qps']];
  for (const [label, fr] of Object.entries(frontiers))
    for (const pt of fr) rows.push([label, pt.ef_search, pt.recall.toFixed(5), pt.qps.toFixed(2)]);
  fs.writeFileSync(p, rows.map(r => r.join(',')).join('\n') + '\n');
}

function writeEqualRecallCsv(equalRecall, labels, p) {
  const rows = [['recall', ...labels]];
  for (const row of equalRecall)
    rows.push([row.recall.toFixed(3), ...labels.map(l => (row[l] == null ? '' : row[l].toFixed(2)))]);
  fs.writeFileSync(p, rows.map(r => r.join(',')).join('\n') + '\n');
}

function printFrontierTables(allResults, analysis) {
  log(`\n${'='.repeat(70)}`);
  log('Pareto frontiers (non-dominated QPS-recall points)');
  log('='.repeat(70));
  for (const r of allResults) {
    const fr = analysis.frontiers[r.label];
    log(`\n  ${r.label}${r.tunable ? '' : '  (single point — faiss not tunable)'}`);
    log(`    ${'ef'.padStart(5)}  ${'recall'.padStart(8)}  ${'qps'.padStart(9)}`);
    for (const pt of fr)
      log(`    ${String(pt.ef_search).padStart(5)}  ${(pt.recall * 100).toFixed(2).padStart(7)}%  ${pt.qps.toFixed(0).padStart(9)}`);
  }

  log(`\n${'='.repeat(70)}`);
  log('Interpolated equal-recall QPS (log-linear along each frontier)');
  log('  Blank = recall target outside that library\'s frontier span.');
  log('  faiss-node excluded (single point, no curve to interpolate).');
  log('='.repeat(70));
  const labels = analysis.sweepableLabels;
  log('\n  ' + 'recall'.padStart(7) + labels.map(l => l.padStart(20)).join(''));
  log('  ' + '-'.repeat(7 + labels.length * 20));
  for (const row of analysis.equalRecall) {
    let line = '  ' + `${(row.recall * 100).toFixed(1)}%`.padStart(7);
    for (const l of labels) line += (row[l] == null ? 'n/a' : row[l].toFixed(0)).padStart(20);
    log(line);
  }
}

// =====================================================================
// Main
// =====================================================================
async function main() {
  const basePath = path.join(DATA_DIR, DS.baseFile(N_BASE));
  const queryPath = path.join(DATA_DIR, DS.queryFile);
  const gtPath = DS.gtFile ? path.join(DATA_DIR, DS.gtFile) : null;
  for (const f of [basePath, queryPath, ...(gtPath ? [gtPath] : [])]) {
    if (!fs.existsSync(f)) { log(`Missing file: ${f}`); process.exit(1); }
  }

  log('='.repeat(70));
  log(`Pareto-Frontier Benchmark (${DATASET.toUpperCase()}, L2)`);
  log('='.repeat(70));
  log(`Configs: ${CONFIGS.map(c => c.label).join(', ')}`);
  log('\nLoading dataset...');
  const { vectors: train, dim } = readFvecs(basePath, N_BASE);
  const { vectors: test } = readFvecs(queryPath, N_QUERIES);
  log(`  Base:    ${train.length.toLocaleString()} vectors, ${dim}D`);
  log(`  Queries: ${test.length.toLocaleString()}`);
  log(`  Metric:  L2`);
  log(`  k=${K}, M=${M}, ef_construction=${EF_CONSTRUCTION}`);
  log(`  ef_search sweep: [${EF_SEARCH_VALUES.join(', ')}]  (includes 100)`);
  log(`  Repetitions: ${REPETITIONS}, Warmup: ${WARMUP_QUERIES}`);
  log(`  NOTE: faiss-node is NOT tunable — it runs once at its built-in HNSW${M}`);
  log(`        defaults (ef_construction=${FAISS_EF_CONSTRUCTION_DEFAULT}, ef_search=${FAISS_EF_SEARCH_DEFAULT}) and contributes a single point.`);

  // Ground truth: SIFT ships a precomputed .ivecs aligned to the full base set,
  // so it is valid only when the whole base is used. dbpedia computes + caches.
  let groundTruth;
  if (gtPath) {
    if (N_BASE < DS.defaultCount) {
      log(`\nERROR: ${DATASET} ships a precomputed ground truth for the full`);
      log(`  ${DS.defaultCount.toLocaleString()}-vector base. Running a --count subset would`);
      log(`  invalidate it (neighbors would point outside the subset). Re-run with the`);
      log(`  full base, or extend the harness to recompute GT for subsets.`);
      process.exit(1);
    }
    groundTruth = readIvecs(gtPath, N_QUERIES);
    log(`  Loaded precomputed ground truth: ${groundTruth.length} rows, first ${K} neighbors each`);
  } else {
    groundTruth = REGENERATE_GT ? null : loadGroundTruth();
    if (groundTruth) log(`\nLoaded cached ground truth (${groundTruth.length} queries) from ${GT_CACHE_PATH}`);
    else { groundTruth = computeGroundTruth(train, test, dim); saveGroundTruth(groundTruth); }
  }

  const dataset = { train, test, groundTruth, dim };

  const allResults = [];
  for (const config of CONFIGS) allResults.push(await sweepOne(config, dataset));

  const analysis = buildAnalysis(allResults);

  // Outputs
  fs.writeFileSync(JSON_PATH, JSON.stringify({
    benchmark: `pancake-pareto-frontier-${DATASET}-l2`,
    timestamp: new Date().toISOString(),
    dataset: { name: DATASET, vectors: train.length, queries: test.length, dim, metric: 'l2', source: DATA_DIR },
    params: { K, M, EF_CONSTRUCTION, EF_SEARCH_VALUES, REPETITIONS, WARMUP_QUERIES },
    faiss_note: `faiss-node is not tunable; runs once at HNSW${M},Flat with faiss defaults efConstruction=${FAISS_EF_CONSTRUCTION_DEFAULT}, efSearch=${FAISS_EF_SEARCH_DEFAULT}. Excluded from sweep and equal-recall interpolation.`,
    results: allResults,
    frontiers: analysis.frontiers,
    equal_recall: analysis.equalRecall,
  }, null, 2) + '\n');
  writeRawCsv(allResults, CSV_PATH);
  writeFrontierCsv(analysis.frontiers, FRONTIER_CSV);
  writeEqualRecallCsv(analysis.equalRecall, analysis.sweepableLabels, EQRECALL_CSV);

  printFrontierTables(allResults, analysis);

  log(`\n${'='.repeat(70)}`);
  log('Outputs:');
  log(`  log:            ${LOG_PATH}`);
  log(`  json:           ${JSON_PATH}`);
  log(`  raw sweep csv:  ${CSV_PATH}`);
  log(`  frontier csv:   ${FRONTIER_CSV}`);
  log(`  equal-recall:   ${EQRECALL_CSV}`);
  log('\nPlot with:');
  log(`  python3 benchmarks/plot_pareto.py ${CSV_PATH}`);
  logStream.end();
}

main().catch(e => { console.error(e); process.exit(1); });
