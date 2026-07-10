#!/usr/bin/env node
'use strict';

/**
 * Full DBpedia 50K comparison benchmark (L2, 1536D)
 *
 * Compares:
 *   - Pancake Int8 WASM
 *   - Pancake Int8 Native
 *   - Pancake FP32 WASM
 *   - Pancake FP32 Native
 *   - hnswlib-node FP32
 *   - USearch Int8
 *   - USearch FP32
 *   - Faiss HNSW
 *
 * Notes:
 * - Pancake, hnswlib, and USearch are run at the requested M / efConstruction /
 *   efSearch settings.
 * - faiss-node HNSW does not expose efConstruction / efSearch control through
 *   its JS binding. It is included with those defaults clearly labeled.
 *
 * Usage:
 *   node benchmarks/benchmark_dbpedia_50k_full.js
 *   node benchmarks/benchmark_dbpedia_50k_full.js --m 16 --ef-construction 50 --ef-search 100
 *   node benchmarks/benchmark_dbpedia_50k_full.js --count 50000
 */

const fs = require('fs');
const path = require('path');
const Pancake = require('../pancake.js');
const { parseBenchmarkArgs, resolveSingleValue } = require('./bench_args');

let native;
try {
  native = require('../native');
} catch (e) {
  console.error('ERROR: native binding not built. Run: cd native && npm install');
  process.exit(1);
}

let usearch;
try {
  usearch = require('usearch');
} catch (e) {
  console.error('ERROR: usearch not installed. Run: npm install usearch');
  process.exit(1);
}

let HierarchicalNSW;
try {
  HierarchicalNSW = require('hnswlib-node').HierarchicalNSW;
} catch (e) {
  console.error('ERROR: hnswlib-node not installed. Run: npm install hnswlib-node');
  process.exit(1);
}

let faiss;
try {
  faiss = require('faiss-node');
} catch (e) {
  console.error('ERROR: faiss-node not installed. Run: npm install faiss-node');
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
const N_BASE = getArg('count', 50_000);
const N_QUERIES = 1_000;
const K = 10;
const M = resolveSingleValue(parsedArgs.m, 16);
const EF_CONSTRUCTION = resolveSingleValue(parsedArgs.efConstruction, 50);
const EF_SEARCH = resolveSingleValue(parsedArgs.efSearch, 100);
const REPETITIONS = 3;
const WARMUP_QUERIES = 200;

const RESULTS_DIR = path.join(__dirname, '..', 'benchmark_results');
if (!fs.existsSync(RESULTS_DIR)) fs.mkdirSync(RESULTS_DIR);
const CACHE_DIR = path.join(RESULTS_DIR, 'cache');
if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR);
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const LOG_PATH = path.join(RESULTS_DIR, `dbpedia_50k_full_${timestamp}.log`);
const JSON_PATH = path.join(RESULTS_DIR, `dbpedia_50k_full_${timestamp}.json`);
const CSV_PATH = path.join(RESULTS_DIR, `dbpedia_50k_full_${timestamp}.csv`);
const GT_CACHE_PATH = path.join(CACHE_DIR, `gt_dbpedia_l2_n${N_BASE}_k${K}.bin`);
const logStream = fs.createWriteStream(LOG_PATH);

function log(msg = '') {
  console.log(msg);
  logStream.write(msg + '\n');
}

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
  log(`Computing brute-force L2 ground truth (${queries.length} x ${train.length} x ${dim}D)...`);
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
    if ((q + 1) % 25 === 0) {
      const elapsed = (performance.now() - t0) / 1000;
      const eta = (elapsed / (q + 1)) * (queries.length - q - 1);
      process.stdout.write(`  ${q + 1}/${queries.length} (${elapsed.toFixed(0)}s elapsed, ~${eta.toFixed(0)}s remaining)\r`);
    }
  }
  log(`  Ground truth computed in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  return gt;
}

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

async function benchPancake(train, queries, groundTruth, dim, quantized, runtimeLabel) {
  if (runtimeLabel === 'native') {
    const label = quantized ? 'pancake-int8-native' : 'pancake-f32-native';
    const h = native.pancake_init(dim, train.length, quantized ? 1 : 0, 0, M, EF_CONSTRUCTION, EF_SEARCH);
    if (h === 0xFFFFFFFF) throw new Error(`Failed to init ${label}`);
    const flat = new Float32Array(train.length * dim);
    for (let i = 0; i < train.length; i++) flat.set(train[i], i * dim);
    const t0 = performance.now();
    const batchSize = 10000;
    for (let start = 0; start < train.length; start += batchSize) {
      const end = Math.min(start + batchSize, train.length);
      const batch = flat.subarray(start * dim, end * dim);
      native.pancake_bulk_insert(h, batch, end - start);
      if (end % 10000 === 0) log(`    ${end.toLocaleString()}/${train.length.toLocaleString()}`);
    }
    const buildMs = performance.now() - t0;
    const memBytes = native.pancake_memory(h);
    for (let i = 0; i < WARMUP_QUERIES && i < queries.length; i++) native.pancake_query(h, queries[i], K);
    const latencies = [];
    let totalRecall = 0;
    for (let rep = 0; rep < REPETITIONS; rep++) {
      for (let i = 0; i < queries.length; i++) {
        const st = performance.now();
        const result = native.pancake_query(h, queries[i], K);
        latencies.push(performance.now() - st);
        if (rep === 0) totalRecall += recall(Array.from(result.ids), groundTruth[i]);
      }
    }
    latencies.sort((a, b) => a - b);
    native.pancake_dispose(h);
    return {
      label,
      buildMs,
      recall: totalRecall / queries.length,
      qps: 1000 / mean(latencies),
      p50: percentile(latencies, 0.5),
      p99: percentile(latencies, 0.99),
      memMB: memBytes / 1024 / 1024,
      params: `M=${M} efC=${EF_CONSTRUCTION} efS=${EF_SEARCH}`
    };
  }

  const label = quantized ? 'pancake-int8-wasm' : 'pancake-f32-wasm';
  const index = await Pancake.create({
    dim,
    maxElements: train.length,
    quantized,
    metric: 'l2',
    M,
    efConstruction: EF_CONSTRUCTION,
    efSearch: EF_SEARCH,
  });
  const t0 = performance.now();
  const batchSize = 500;
  for (let start = 0; start < train.length; start += batchSize) {
    const end = Math.min(start + batchSize, train.length);
    index.addBatch(train.slice(start, end));
    if (end % 10000 < batchSize) log(`    ${end.toLocaleString()}/${train.length.toLocaleString()}`);
  }
  const buildMs = performance.now() - t0;
  index.setEfSearch(EF_SEARCH);
  for (let i = 0; i < WARMUP_QUERIES && i < queries.length; i++) index.search(queries[i], K);
  const latencies = [];
  let totalRecall = 0;
  for (let rep = 0; rep < REPETITIONS; rep++) {
    for (let i = 0; i < queries.length; i++) {
      const st = performance.now();
      const results = index.search(queries[i], K);
      latencies.push(performance.now() - st);
      if (rep === 0) totalRecall += recall(results.map(r => r.id), groundTruth[i]);
    }
  }
  latencies.sort((a, b) => a - b);
  const memMB = index.memory / 1024 / 1024;
  index.dispose();
  return {
    label,
    buildMs,
    recall: totalRecall / queries.length,
    qps: 1000 / mean(latencies),
    p50: percentile(latencies, 0.5),
    p99: percentile(latencies, 0.99),
    memMB,
    params: `M=${M} efC=${EF_CONSTRUCTION} efS=${EF_SEARCH}`
  };
}

function benchHnswlib(train, queries, groundTruth, dim) {
  const index = new HierarchicalNSW('l2', dim);
  index.initIndex(train.length, M, EF_CONSTRUCTION, 100);
  index.setEf(EF_SEARCH);
  const t0 = performance.now();
  for (let i = 0; i < train.length; i++) {
    index.addPoint(Array.from(train[i]), i);
    if ((i + 1) % 10000 === 0) log(`    ${(i + 1).toLocaleString()}/${train.length.toLocaleString()}`);
  }
  const buildMs = performance.now() - t0;
  for (let i = 0; i < WARMUP_QUERIES && i < queries.length; i++) {
    index.searchKnn(Array.from(queries[i]), K);
  }
  const latencies = [];
  let totalRecall = 0;
  for (let rep = 0; rep < REPETITIONS; rep++) {
    for (let i = 0; i < queries.length; i++) {
      const q = Array.from(queries[i]);
      const st = performance.now();
      const results = index.searchKnn(q, K);
      latencies.push(performance.now() - st);
      if (rep === 0) totalRecall += recall(results.neighbors, groundTruth[i]);
    }
  }
  latencies.sort((a, b) => a - b);
  return {
    label: 'hnswlib-f32-native',
    buildMs,
    recall: totalRecall / queries.length,
    qps: 1000 / mean(latencies),
    p50: percentile(latencies, 0.5),
    p99: percentile(latencies, 0.99),
    memMB: null,
    params: `M=${M} efC=${EF_CONSTRUCTION} efS=${EF_SEARCH}`
  };
}

function benchUsearch(train, queries, groundTruth, dim, dtype) {
  const quantization = dtype === 'i8' ? 'i8' : 'f32';
  const label = `usearch-${dtype}-native`;
  const index = new usearch.Index({
    metric: 'l2sq',
    connectivity: M,
    dimensions: dim,
    quantization,
    expansion_add: EF_CONSTRUCTION,
    expansion_search: EF_SEARCH,
  });
  const t0 = performance.now();
  for (let i = 0; i < train.length; i++) {
    index.add(BigInt(i), train[i]);
    if ((i + 1) % 10000 === 0) log(`    ${(i + 1).toLocaleString()}/${train.length.toLocaleString()}`);
  }
  const buildMs = performance.now() - t0;
  for (let i = 0; i < WARMUP_QUERIES && i < queries.length; i++) index.search(queries[i], K);
  const latencies = [];
  let totalRecall = 0;
  for (let rep = 0; rep < REPETITIONS; rep++) {
    for (let i = 0; i < queries.length; i++) {
      const st = performance.now();
      const results = index.search(queries[i], K);
      latencies.push(performance.now() - st);
      if (rep === 0) {
        const ids = Array.from(results.keys).map(Number);
        totalRecall += recall(ids, groundTruth[i]);
      }
    }
  }
  latencies.sort((a, b) => a - b);
  return {
    label,
    buildMs,
    recall: totalRecall / queries.length,
    qps: 1000 / mean(latencies),
    p50: percentile(latencies, 0.5),
    p99: percentile(latencies, 0.99),
    memMB: null,
    params: `M=${M} efC=${EF_CONSTRUCTION} efS=${EF_SEARCH}`
  };
}

function benchFaiss(train, queries, groundTruth, dim) {
  const index = faiss.Index.fromFactory(dim, `HNSW${M},Flat`, faiss.MetricType.METRIC_L2);
  const flat = new Float32Array(train.length * dim);
  for (let i = 0; i < train.length; i++) flat.set(train[i], i * dim);
  const t0 = performance.now();
  index.add(Array.from(flat));
  const buildMs = performance.now() - t0;
  for (let i = 0; i < WARMUP_QUERIES && i < queries.length; i++) index.search(Array.from(queries[i]), K);
  const latencies = [];
  let totalRecall = 0;
  for (let rep = 0; rep < REPETITIONS; rep++) {
    for (let i = 0; i < queries.length; i++) {
      const q = Array.from(queries[i]);
      const st = performance.now();
      const results = index.search(q, K);
      latencies.push(performance.now() - st);
      if (rep === 0) totalRecall += recall(results.labels, groundTruth[i]);
    }
  }
  latencies.sort((a, b) => a - b);
  return {
    label: 'faiss-hnsw',
    buildMs,
    recall: totalRecall / queries.length,
    qps: 1000 / mean(latencies),
    p50: percentile(latencies, 0.5),
    p99: percentile(latencies, 0.99),
    memMB: null,
    params: `M=${M} efC=40(default) efS=16(default)`
  };
}

function writeCsv(results) {
  const rows = [['label', 'build_s', 'recall', 'qps', 'p50_ms', 'p99_ms', 'mem_mb', 'params']];
  for (const r of results) {
    rows.push([
      r.label,
      (r.buildMs / 1000).toFixed(3),
      r.recall.toFixed(6),
      r.qps.toFixed(2),
      r.p50.toFixed(4),
      r.p99.toFixed(4),
      r.memMB == null ? '' : r.memMB.toFixed(2),
      r.params
    ]);
  }
  fs.writeFileSync(CSV_PATH, rows.map(r => r.join(',')).join('\n') + '\n');
}

async function main() {
  const basePath = path.join(DBPEDIA_DIR, 'dbpedia_base_100k.fvecs');
  const queryPath = path.join(DBPEDIA_DIR, 'dbpedia_query.fvecs');
  for (const f of [basePath, queryPath]) {
    if (!fs.existsSync(f)) {
      log(`Missing file: ${f}`);
      process.exit(1);
    }
  }

  log('='.repeat(70));
  log('Full DBpedia Benchmark (50K, L2, 1536D)');
  log('='.repeat(70));
  log();
  log('Loading dataset...');
  const { vectors: train, dim } = readFvecs(basePath, N_BASE);
  const { vectors: queries } = readFvecs(queryPath, N_QUERIES);
  log(`  Base:    ${train.length.toLocaleString()} vectors, ${dim}D`);
  log(`  Queries: ${queries.length.toLocaleString()}`);
  log(`  k=${K}, M=${M}, ef_construction=${EF_CONSTRUCTION}, ef_search=${EF_SEARCH}`);
  log('  Faiss note: HNSW efConstruction/efSearch are library defaults in faiss-node');
  log();

  let groundTruth = loadGroundTruth();
  if (groundTruth) {
    log(`Loaded cached ground truth (${groundTruth.length} queries)`);
  } else {
    groundTruth = computeGroundTruth(train, queries, dim);
    saveGroundTruth(groundTruth);
  }
  log();

  const results = [];

  log('--- Pancake Int8 WASM ---');
  results.push(await benchPancake(train, queries, groundTruth, dim, true, 'wasm'));
  log('--- Pancake Int8 Native ---');
  results.push(await benchPancake(train, queries, groundTruth, dim, true, 'native'));
  log('--- Pancake FP32 WASM ---');
  results.push(await benchPancake(train, queries, groundTruth, dim, false, 'wasm'));
  log('--- Pancake FP32 Native ---');
  results.push(await benchPancake(train, queries, groundTruth, dim, false, 'native'));
  log('--- hnswlib FP32 ---');
  results.push(benchHnswlib(train, queries, groundTruth, dim));
  log('--- USearch Int8 ---');
  results.push(benchUsearch(train, queries, groundTruth, dim, 'i8'));
  log('--- USearch FP32 ---');
  results.push(benchUsearch(train, queries, groundTruth, dim, 'f32'));
  log('--- Faiss HNSW ---');
  results.push(benchFaiss(train, queries, groundTruth, dim));

  log();
  log('='.repeat(90));
  log('Results');
  log('='.repeat(90));
  log(
    'Library'.padEnd(22)
    + 'Build(s)'.padStart(10)
    + 'Recall'.padStart(10)
    + 'QPS'.padStart(10)
    + 'p50(ms)'.padStart(10)
    + 'p99(ms)'.padStart(10)
    + 'Mem(MB)'.padStart(10)
    + '  Params'
  );
  log('-'.repeat(90));
  for (const r of results) {
    log(
      r.label.padEnd(22)
      + (r.buildMs / 1000).toFixed(2).padStart(10)
      + `${(r.recall * 100).toFixed(2)}%`.padStart(10)
      + r.qps.toFixed(0).padStart(10)
      + r.p50.toFixed(3).padStart(10)
      + r.p99.toFixed(3).padStart(10)
      + (r.memMB == null ? '-'.padStart(10) : r.memMB.toFixed(1).padStart(10))
      + `  ${r.params}`
    );
  }

  writeCsv(results);
  fs.writeFileSync(JSON_PATH, JSON.stringify({
    benchmark: 'dbpedia-50k-full-l2',
    timestamp: new Date().toISOString(),
    dataset: { vectors: train.length, queries: queries.length, dim, metric: 'l2', source: DBPEDIA_DIR },
    params: { K, M, EF_CONSTRUCTION, EF_SEARCH, REPETITIONS, WARMUP_QUERIES },
    results,
  }, null, 2));

  log();
  log('Outputs:');
  log(`  log:  ${LOG_PATH}`);
  log(`  json: ${JSON_PATH}`);
  log(`  csv:  ${CSV_PATH}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
