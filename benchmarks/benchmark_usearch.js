#!/usr/bin/env node
'use strict';

/**
 * Pancake vs USearch Benchmark (DBpedia 100K, L2, 1536D)
 *
 * Compares four configurations on the same recall-QPS sweep:
 *   1. Pancake Int8  (WASM, quantized)
 *   2. Pancake FP32  (WASM, full precision)
 *   3. USearch Int8   (native, quantized)
 *   4. USearch FP32   (native, full precision)
 *
 * Optional fifth config (hnswlib-node) included if installed.
 *
 * Expected files in dbpedia/:
 *   dbpedia_base_100k.fvecs   - 100K base vectors (float32, 1536D)
 *   dbpedia_query.fvecs       - query vectors
 *
 * Ground truth is computed via brute-force L2 on the 100K subset,
 * cached to disk on first run. Pass --regenerate-gt to force recompute.
 *
 * Usage:
 *   node benchmarks/benchmark_usearch.js
 *   node benchmarks/benchmark_usearch.js --regenerate-gt
 *   node benchmarks/benchmark_usearch.js --m 16 --ef-construction 100
 */

const fs = require('fs');
const path = require('path');
const Pancake = require('../pancake.js');
const { parseBenchmarkArgs, resolveSingleValue, resolveSweepValues } = require('./bench_args');

// --- Optional libraries ---
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
  // optional — will skip hnswlib config
}

const parsedArgs = parseBenchmarkArgs();
const DBPEDIA_DIR = parsedArgs.args.find(a => !a.startsWith('-')) || path.join(__dirname, '..', 'dbpedia');
const REGENERATE_GT = parsedArgs.args.includes('--regenerate-gt');

// --- Config ---
const N_BASE = 100_000;
const N_QUERIES = 1_000;
const K = 10;
const M = resolveSingleValue(parsedArgs.m, 16);
const EF_CONSTRUCTION = resolveSingleValue(parsedArgs.efConstruction, 50);
const EF_SEARCH_VALUES = resolveSweepValues(parsedArgs, [10, 20, 40, 60, 80, 100, 150, 200, 300, 500, 800]);
const REPETITIONS = 3;
const WARMUP_QUERIES = 200;

const CONFIGS = [
  { label: 'pancake-int8-wasm',   library: 'pancake',  dtype: 'i8'  },
  { label: 'pancake-f32-wasm',    library: 'pancake',  dtype: 'f32' },
  { label: 'usearch-i8-native',   library: 'usearch',  dtype: 'i8'  },
  { label: 'usearch-f32-native',  library: 'usearch',  dtype: 'f32' },
];
if (HierarchicalNSW) {
  CONFIGS.push({ label: 'hnswlib-f32-native', library: 'hnswlib', dtype: 'f32' });
}

// --- Logging / output ---
const RESULTS_DIR = path.join(__dirname, '..', 'benchmark_results');
if (!fs.existsSync(RESULTS_DIR)) fs.mkdirSync(RESULTS_DIR);
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const LOG_PATH  = path.join(RESULTS_DIR, `usearch_${timestamp}.log`);
const JSON_PATH = path.join(RESULTS_DIR, `usearch_${timestamp}.json`);
const CSV_PATH  = path.join(RESULTS_DIR, `usearch_${timestamp}.csv`);
const logStream = fs.createWriteStream(LOG_PATH);

function log(msg = '') {
  console.log(msg);
  logStream.write(msg + '\n');
}

// --- .fvecs / .ivecs readers ---
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

// --- Ground truth (brute-force L2 for the 100K subset, cached to disk) ---
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
  log(`Computing brute-force L2 ground truth (${nq} × ${train.length} × ${dim}D)...`);
  log(`  This will take several minutes at 1536D.`);
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

  const elapsed = (performance.now() - t0) / 1000;
  log(`  Ground truth computed in ${elapsed.toFixed(1)}s                    `);
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

// --- Pancake: build + query ---
async function buildPancake({ train, dim, dtype }) {
  const quantized = dtype === 'i8';
  log(`  [pancake-${dtype}] building index (M=${M}, ef_c=${EF_CONSTRUCTION}, metric=l2)...`);
  const index = await Pancake.create({
    dim,
    maxElements: train.length,
    quantized,
    metric: 'l2',
    M,
    efConstruction: EF_CONSTRUCTION,
    efSearch: EF_SEARCH_VALUES[0],
  });

  const t0 = performance.now();
  const batchSize = 500;
  for (let start = 0; start < train.length; start += batchSize) {
    const end = Math.min(start + batchSize, train.length);
    index.addBatch(train.slice(start, end));
    if (end % 10000 < batchSize) {
      log(`    ${end.toLocaleString()}/${train.length.toLocaleString()}`);
    }
  }
  const buildMs = performance.now() - t0;
  log(`  [pancake-${dtype}] build: ${(buildMs / 1000).toFixed(1)}s, memory: ${(index.memory / 1024 / 1024).toFixed(1)} MB`);
  return { index, buildMs, memBytes: index.memory };
}

function queryPancake(index, test, groundTruth, efSearch) {
  index.setEfSearch(efSearch);
  for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) {
    index.search(test[i], K);
  }
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

// --- USearch: build + query ---
// USearch's JS binding only respects expansion_search at construction time.
// Setting the property after construction is silently ignored by the native layer.
// Workaround: build once, save to disk, then view() with a fresh Index per ef_search.
function buildUsearch({ train, dim, dtype }) {
  const quantization = dtype === 'i8' ? 'i8' : 'f32';
  log(`  [usearch-${dtype}] building index (M=${M}, ef_c=${EF_CONSTRUCTION}, metric=l2sq, quantization=${quantization})...`);
  const index = new usearch.Index({
    metric: 'l2sq',
    connectivity: M,
    dimensions: dim,
    quantization,
    expansion_add: EF_CONSTRUCTION,
    expansion_search: EF_SEARCH_VALUES[0],
  });

  const t0 = performance.now();
  for (let i = 0; i < train.length; i++) {
    index.add(BigInt(i), train[i]);
    if ((i + 1) % 10000 === 0) {
      log(`    ${(i + 1).toLocaleString()}/${train.length.toLocaleString()}`);
    }
  }
  const buildMs = performance.now() - t0;

  // Save to disk so we can view() with different expansion_search values
  const savePath = path.join(RESULTS_DIR, `_usearch_${dtype}_temp.bin`);
  index.save(savePath);
  log(`  [usearch-${dtype}] build: ${(buildMs / 1000).toFixed(1)}s (saved to ${savePath})`);
  return { index, buildMs, memBytes: null, savePath, quantization };
}

function queryUsearch(built, test, groundTruth, efSearch, dim) {
  // Load a view of the saved index with the desired expansion_search
  const viewIndex = new usearch.Index({
    metric: 'l2sq',
    connectivity: M,
    dimensions: dim,
    quantization: built.quantization,
    expansion_search: efSearch,
  });
  viewIndex.view(built.savePath);

  for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) {
    viewIndex.search(test[i], K);
  }

  const latencies = new Array(test.length);
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const st = performance.now();
    const results = viewIndex.search(test[i], K);
    latencies[i] = performance.now() - st;
    // USearch returns { keys: BigUint64Array, distances: Float32Array }
    const ids = Array.from(results.keys).map(Number);
    totalRecall += recall(ids, groundTruth[i]);
  }
  return { latencies, meanRecall: totalRecall / test.length };
}

// --- hnswlib-node: build + query ---
function buildHnswlib({ train, dim }) {
  if (!HierarchicalNSW) return null;
  log(`  [hnswlib-f32] building index (M=${M}, ef_c=${EF_CONSTRUCTION}, metric=l2)...`);
  const index = new HierarchicalNSW('l2', dim);
  index.initIndex(train.length, M, EF_CONSTRUCTION, 100);

  const t0 = performance.now();
  for (let i = 0; i < train.length; i++) {
    index.addPoint(Array.from(train[i]), i);
    if ((i + 1) % 10000 === 0) log(`    ${(i + 1).toLocaleString()}/${train.length.toLocaleString()}`);
  }
  const buildMs = performance.now() - t0;
  log(`  [hnswlib-f32] build: ${(buildMs / 1000).toFixed(1)}s`);
  return { index, buildMs, memBytes: null };
}

function queryHnswlib(index, test, groundTruth, efSearch) {
  index.setEf(efSearch);
  for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) {
    index.searchKnn(Array.from(test[i]), K);
  }
  const latencies = new Array(test.length);
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const query = Array.from(test[i]);
    const st = performance.now();
    const results = index.searchKnn(query, K);
    latencies[i] = performance.now() - st;
    totalRecall += recall(results.neighbors, groundTruth[i]);
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
  if (config.library === 'pancake') {
    built = await buildPancake({ train, dim, dtype: config.dtype });
  } else if (config.library === 'usearch') {
    built = buildUsearch({ train, dim, dtype: config.dtype });
  } else {
    built = buildHnswlib({ train, dim });
    if (!built) return null;
  }

  const { index, buildMs, memBytes } = built;
  const points = [];

  for (const efSearch of EF_SEARCH_VALUES) {
    log(`\n  ef_search=${efSearch}`);
    const reps = [];
    for (let rep = 0; rep < REPETITIONS; rep++) {
      let result;
      if (config.library === 'pancake') {
        result = queryPancake(index, test, groundTruth, efSearch);
      } else if (config.library === 'usearch') {
        result = queryUsearch(built, test, groundTruth, efSearch, dim);
      } else {
        result = queryHnswlib(index, test, groundTruth, efSearch);
      }
      const { latencies, meanRecall } = result;
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

  if (typeof index.dispose === 'function') index.dispose();
  // Clean up temp file for usearch save/view pattern
  if (built.savePath && fs.existsSync(built.savePath)) {
    fs.unlinkSync(built.savePath);
  }

  return {
    label: config.label,
    library: config.library,
    dtype: config.dtype,
    buildMs,
    memMB: memBytes ? memBytes / 1024 / 1024 : null,
    params: { M, ef_construction: EF_CONSTRUCTION, K },
    points,
  };
}

// --- CSV output ---
function writeCsv(allResults, csvPath) {
  const rows = [['label', 'library', 'dtype', 'ef_search', 'recall', 'recall_std',
                 'qps', 'qps_std', 'p50_ms', 'p95_ms', 'p99_ms']];
  for (const r of allResults) {
    if (!r) continue;
    for (const p of r.points) {
      rows.push([
        r.label, r.library, r.dtype, p.ef_search,
        p.recall_mean.toFixed(5), p.recall_std.toFixed(5),
        p.qps_mean.toFixed(2), p.qps_std.toFixed(2),
        p.p50_mean.toFixed(4), p.p95_mean.toFixed(4), p.p99_mean.toFixed(4),
      ]);
    }
  }
  fs.writeFileSync(csvPath, rows.map(r => r.join(',')).join('\n') + '\n');
}

// --- Matched-recall comparison ---
function matchedRecallComparison(results) {
  log(`\n${'='.repeat(70)}`);
  log('Matched-recall comparison');
  log('='.repeat(70));

  const valid = results.filter(r => r != null);
  if (valid.length < 2) { log('  (need at least 2 configs to compare)'); return; }

  function qpsAtRecall(points, targetRecall) {
    const sorted = [...points].sort((a, b) => a.recall_mean - b.recall_mean);
    if (targetRecall < sorted[0].recall_mean || targetRecall > sorted[sorted.length - 1].recall_mean) return null;
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i].recall_mean >= targetRecall) {
        const r0 = sorted[i - 1].recall_mean, r1 = sorted[i].recall_mean;
        const q0 = sorted[i - 1].qps_mean, q1 = sorted[i].qps_mean;
        const t = (targetRecall - r0) / (r1 - r0);
        return Math.exp(Math.log(q0) + t * (Math.log(q1) - Math.log(q0)));
      }
    }
    return null;
  }

  // Build summary table
  log('\n  Build time comparison:');
  log('  ' + '-'.repeat(55));
  log(`  ${'Config'.padEnd(25)} ${'Build (s)'.padStart(12)} ${'Memory (MB)'.padStart(14)}`);
  log('  ' + '-'.repeat(55));
  for (const r of valid) {
    const mem = r.memMB != null ? r.memMB.toFixed(1) : 'N/A';
    log(`  ${r.label.padEnd(25)} ${(r.buildMs / 1000).toFixed(1).padStart(12)} ${mem.padStart(14)}`);
  }

  // Pairwise QPS comparison
  for (let i = 0; i < valid.length; i++) {
    for (let j = i + 1; j < valid.length; j++) {
      const A = valid[i], B = valid[j];
      log(`\n  ${A.label} vs ${B.label}:`);
      for (const pA of A.points) {
        const qB = qpsAtRecall(B.points, pA.recall_mean);
        if (qB === null) continue;
        const ratio = pA.qps_mean / qB;
        log(`    recall=${(pA.recall_mean * 100).toFixed(1)}%:  `
          + `${A.label}=${pA.qps_mean.toFixed(0)} qps, `
          + `${B.label}=${qB.toFixed(0)} qps, `
          + `ratio=${ratio.toFixed(2)}x`);
      }
    }
  }
}

// --- Main ---
async function main() {
  const basePath = path.join(DBPEDIA_DIR, 'dbpedia_base_100k.fvecs');
  const queryPath = path.join(DBPEDIA_DIR, 'dbpedia_query.fvecs');
  for (const f of [basePath, queryPath]) {
    if (!fs.existsSync(f)) {
      log(`Missing file: ${f}`);
      log(`\nTo get the DBpedia dataset, see benchmarks/README or the project wiki.`);
      process.exit(1);
    }
  }

  log('='.repeat(70));
  log('Pancake vs USearch Benchmark (DBpedia 100K, L2, 1536D)');
  log('='.repeat(70));

  const configs = CONFIGS.map(c => c.label).join(', ');
  log(`Configs: ${configs}`);

  // Load dataset
  log('\nLoading dataset...');
  const { vectors: train, dim } = readFvecs(basePath, N_BASE);
  const { vectors: test } = readFvecs(queryPath, N_QUERIES);

  log(`  Base:    ${train.length.toLocaleString()} vectors, ${dim}D`);
  log(`  Queries: ${test.length.toLocaleString()}`);
  log(`  Metric:  L2`);
  log(`  k=${K}, M=${M}, ef_construction=${EF_CONSTRUCTION}`);
  log(`  ef_search sweep: [${EF_SEARCH_VALUES.join(', ')}]`);
  log(`  Repetitions: ${REPETITIONS}, Warmup: ${WARMUP_QUERIES}`);

  // Ground truth
  let groundTruth;
  if (!REGENERATE_GT) groundTruth = loadGroundTruth();
  if (groundTruth) {
    log(`\nLoaded cached ground truth (${groundTruth.length} queries) from ${GT_CACHE_PATH}`);
  } else {
    log('');
    groundTruth = computeGroundTruth(train, test, dim);
    saveGroundTruth(groundTruth);
  }

  const dataset = { train, test, groundTruth, dim };

  // Run each config
  const allResults = [];
  for (const config of CONFIGS) {
    const result = await sweepOne(config, dataset);
    allResults.push(result);
  }

  // Save outputs
  fs.writeFileSync(JSON_PATH, JSON.stringify({
    benchmark: 'pancake-vs-usearch-dbpedia-100k-l2',
    timestamp: new Date().toISOString(),
    dataset: { vectors: train.length, queries: test.length, dim, metric: 'l2', source: DBPEDIA_DIR },
    params: { K, M, EF_CONSTRUCTION, EF_SEARCH_VALUES, REPETITIONS, WARMUP_QUERIES },
    results: allResults,
  }, null, 2) + '\n');
  writeCsv(allResults, CSV_PATH);

  matchedRecallComparison(allResults);

  log(`\n${'='.repeat(70)}`);
  log('Outputs:');
  log(`  log:  ${LOG_PATH}`);
  log(`  json: ${JSON_PATH}`);
  log(`  csv:  ${CSV_PATH}`);
  log('\nPlot with:');
  log(`  python3 benchmarks/plot_sweep.py ${CSV_PATH}`);
  logStream.end();
}

main().catch(e => { console.error(e); process.exit(1); });
