#!/usr/bin/env node
'use strict';

/**
 * Recall-QPS sweep on SIFT-1M: Pancake Int8 WASM vs hnswlib-node Float32 Native.
 *
 * Uses the standard SIFT-1M dataset (1M vectors, 128D, L2).
 *
 * Expected files in [sift-dir]:
 *   sift_base.fvecs        - 1M base vectors (float32, 128D)
 *   sift_query.fvecs       - 10K query vectors
 *   sift_groundtruth.ivecs - ground truth neighbor IDs
 *
 * Usage:
 *   node qps-recall_sweep_sift1m.js [sift-dir]
 */

const fs = require('fs');
const path = require('path');
const Pancake = require('../pancake.js');

const SIFT_DIR = process.argv[2] || '/mnt/c/pancake_gt/sift';

// --- Sweep configuration ---
const K = 10;
const M = 16;
const EF_CONSTRUCTION = 200;
const EF_SEARCH_VALUES = [10, 20, 40, 60, 80, 100, 150, 200, 300, 500, 800];
const REPETITIONS = 3;
const WARMUP_QUERIES = 200;

const CONFIGS = [
  { label: 'pancake-int8-wasm',   library: 'pancake' },
  { label: 'hnswlib-f32-native',  library: 'hnswlib' },
];

// --- Logging setup ---
const RESULTS_DIR = path.join(__dirname, '..', 'benchmark_results');
if (!fs.existsSync(RESULTS_DIR)) fs.mkdirSync(RESULTS_DIR);
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const LOG_PATH  = path.join(RESULTS_DIR, `sweep_sift1m_${timestamp}.log`);
const JSON_PATH = path.join(RESULTS_DIR, `sweep_sift1m_${timestamp}.json`);
const CSV_PATH  = path.join(RESULTS_DIR, `sweep_sift1m_${timestamp}.csv`);
const logStream = fs.createWriteStream(LOG_PATH);

function log(msg = '') {
  console.log(msg);
  logStream.write(msg + '\n');
}

// --- Data loading (fvecs/ivecs) ---
function readFvecs(filePath) {
  log(`  Loading ${filePath}...`);
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

// --- Metric helpers ---
function recall(predicted, truth) {
  const truthSet = new Set(truth);
  let hits = 0;
  for (const id of predicted) if (truthSet.has(id)) hits++;
  return hits / truth.length;
}

function percentile(sortedAsc, p) {
  const idx = Math.min(sortedAsc.length - 1,
                       Math.floor(sortedAsc.length * p));
  return sortedAsc[idx];
}

function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function stddev(arr) {
  const m = mean(arr);
  const v = arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length;
  return Math.sqrt(v);
}

// --- Pancake: build and query ---
async function buildPancake({ train, dim }) {
  log(`  [pancake-int8] building index (M=${M}, ef_c=${EF_CONSTRUCTION})...`);
  const index = await Pancake.create({
    dim,
    metric: 'l2',
    maxElements: train.length,
    quantized: true,
    M,
    efConstruction: EF_CONSTRUCTION,
    efSearch: EF_SEARCH_VALUES[0],
  });

  const t0 = performance.now();
  if (typeof index.addBatch === 'function') {
    const batchSize = 4096;
    for (let start = 0; start < train.length; start += batchSize) {
      const end = Math.min(start + batchSize, train.length);
      index.addBatch(train.slice(start, end));
      if (end % 100000 < batchSize) {
        const elapsed = (performance.now() - t0) / 1000;
        log(`    ${end}/${train.length} (${(end / elapsed).toFixed(0)} vec/s)`);
      }
    }
  } else {
    for (let i = 0; i < train.length; i++) {
      index.add(train[i]);
      if ((i + 1) % 100000 === 0) {
        const elapsed = (performance.now() - t0) / 1000;
        log(`    ${i + 1}/${train.length} (${((i + 1) / elapsed).toFixed(0)} vec/s)`);
      }
    }
  }
  const buildMs = performance.now() - t0;
  log(`  [pancake-int8] build: ${(buildMs / 1000).toFixed(1)}s (${(train.length / (buildMs / 1000)).toFixed(0)} vec/s), memory: ${(index.memory / 1024 / 1024).toFixed(0)} MB`);
  return { index, buildMs };
}

function queryPancake(index, test, groundTruth, efSearch) {
  if (typeof index._setEfSearch === 'function') {
    index._setEfSearch(efSearch);
  } else if (index._e && index._e._i8_set_ef && index._useInt8) {
    index._e._i8_set_ef(efSearch);
  } else if (index._e && index._e._float_set_ef && !index._useInt8) {
    index._e._float_set_ef(efSearch);
  } else {
    throw new Error('Cannot set ef_search on Pancake index');
  }

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

// --- hnswlib-node: build and query ---
let HierarchicalNSW;
try {
  HierarchicalNSW = require('hnswlib-node').HierarchicalNSW;
} catch (e) {
  log('WARNING: hnswlib-node not installed; skipping hnswlib configs');
}

function buildHnswlib({ train, dim }) {
  if (!HierarchicalNSW) return null;
  log(`  [hnswlib-f32] building index (M=${M}, ef_c=${EF_CONSTRUCTION})...`);
  const index = new HierarchicalNSW('l2', dim);
  index.initIndex(train.length, M, EF_CONSTRUCTION, 100);

  const t0 = performance.now();
  for (let i = 0; i < train.length; i++) {
    index.addPoint(Array.from(train[i]), i);
    if ((i + 1) % 100000 === 0) {
      const elapsed = (performance.now() - t0) / 1000;
      log(`    ${i + 1}/${train.length} (${((i + 1) / elapsed).toFixed(0)} vec/s)`);
    }
  }
  const buildMs = performance.now() - t0;
  log(`  [hnswlib-f32] build: ${(buildMs / 1000).toFixed(1)}s (${(train.length / (buildMs / 1000)).toFixed(0)} vec/s)`);
  return { index, buildMs };
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
  log(`\n${'='.repeat(60)}`);
  log(`Config: ${config.label}`);
  log('='.repeat(60));

  let built;
  if (config.library === 'pancake') {
    built = await buildPancake({ train, dim });
  } else {
    built = buildHnswlib({ train, dim });
    if (!built) return null;
  }

  const { index, buildMs } = built;
  const points = [];

  for (const efSearch of EF_SEARCH_VALUES) {
    log(`\n  ef_search=${efSearch}`);
    const reps = [];
    for (let rep = 0; rep < REPETITIONS; rep++) {
      const { latencies, meanRecall } = config.library === 'pancake'
        ? queryPancake(index, test, groundTruth, efSearch)
        : queryHnswlib(index, test, groundTruth, efSearch);
      const sorted = [...latencies].sort((a, b) => a - b);
      const avgLatency = mean(latencies);
      reps.push({
        recall: meanRecall,
        qps: 1000 / avgLatency,
        meanLatencyMs: avgLatency,
        p50: percentile(sorted, 0.5),
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
      p99_mean:    mean(reps.map(r => r.p99)),
      reps,
    };
    log(`    summary: recall=${(summary.recall_mean * 100).toFixed(2)}% `
      + `(+/-${(summary.recall_std * 100).toFixed(2)}) `
      + `qps=${summary.qps_mean.toFixed(0)} (+/-${summary.qps_std.toFixed(0)})`);
    points.push(summary);
  }

  if (typeof index.dispose === 'function') index.dispose();

  return {
    label: config.label,
    library: config.library,
    dtype: config.library === 'pancake' ? 'i8' : 'f32',
    buildMs,
    params: { M, ef_construction: EF_CONSTRUCTION, K },
    points,
  };
}

// --- Write CSV ---
function writeCsv(allResults, csvPath) {
  const rows = [['label', 'library', 'dtype', 'ef_search', 'recall', 'recall_std',
                 'qps', 'qps_std', 'p50_ms', 'p99_ms']];
  for (const r of allResults) {
    if (!r) continue;
    for (const p of r.points) {
      rows.push([
        r.label, r.library, r.dtype, p.ef_search,
        p.recall_mean.toFixed(5), p.recall_std.toFixed(5),
        p.qps_mean.toFixed(2), p.qps_std.toFixed(2),
        p.p50_mean.toFixed(4), p.p99_mean.toFixed(4),
      ]);
    }
  }
  fs.writeFileSync(csvPath, rows.map(r => r.join(',')).join('\n') + '\n');
}

// --- Matched-recall comparison ---
function matchedRecallComparison(results) {
  log(`\n${'='.repeat(60)}`);
  log('Matched-recall comparison');
  log('='.repeat(60));

  const byLabel = {};
  for (const r of results) if (r) byLabel[r.label] = r;

  function qpsAtRecall(points, targetRecall) {
    const sorted = [...points].sort((a, b) => a.recall_mean - b.recall_mean);
    if (targetRecall < sorted[0].recall_mean) return null;
    if (targetRecall > sorted[sorted.length - 1].recall_mean) return null;
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

  const labels = Object.keys(byLabel);
  for (let i = 0; i < labels.length; i++) {
    for (let j = 0; j < labels.length; j++) {
      if (i === j) continue;
      const A = byLabel[labels[i]], B = byLabel[labels[j]];
      log(`\n  ${A.label} vs ${B.label}:`);
      for (const pA of A.points) {
        const qB = qpsAtRecall(B.points, pA.recall_mean);
        if (qB === null) continue;
        const ratio = pA.qps_mean / qB;
        log(`    at recall=${(pA.recall_mean * 100).toFixed(1)}%: `
          + `${A.label} qps=${pA.qps_mean.toFixed(0)}, `
          + `${B.label} interp. qps=${qB.toFixed(0)}, `
          + `ratio=${ratio.toFixed(2)}x`);
      }
    }
  }
}

// --- Main ---
async function main() {
  const baseFile = path.join(SIFT_DIR, 'sift_base.fvecs');
  const queryFile = path.join(SIFT_DIR, 'sift_query.fvecs');
  const gtFile = path.join(SIFT_DIR, 'sift_groundtruth.ivecs');

  for (const f of [baseFile, queryFile, gtFile]) {
    if (!fs.existsSync(f)) {
      log(`Missing: ${f}`);
      log('Download SIFT-1M from http://corpus-texmex.irisa.fr/');
      process.exit(1);
    }
  }

  log('Loading SIFT-1M dataset...');
  const { vectors: train, dim } = readFvecs(baseFile);
  const { vectors: test } = readFvecs(queryFile);
  const groundTruth = readIvecs(gtFile);

  log(`\n${'='.repeat(60)}`);
  log('Recall-QPS sweep on SIFT-1M (Pancake vs hnswlib)');
  log(`${train.length} vectors, ${dim}D, ${test.length} queries`);
  log(`k=${K}, M=${M}, ef_construction=${EF_CONSTRUCTION}`);
  log(`ef_search sweep: [${EF_SEARCH_VALUES.join(', ')}]`);
  log(`Repetitions: ${REPETITIONS}, Warmup: ${WARMUP_QUERIES}`);
  log('='.repeat(60));

  const allResults = [];
  for (const config of CONFIGS) {
    const result = await sweepOne(config, { train, test, groundTruth, dim });
    allResults.push(result);
  }

  fs.writeFileSync(JSON_PATH, JSON.stringify({
    benchmark: 'sift1m-sweep-hnswlib',
    timestamp: new Date().toISOString(),
    dataset: { n_train: train.length, n_test: test.length, dim },
    params: { K, M, EF_CONSTRUCTION, EF_SEARCH_VALUES, REPETITIONS, WARMUP_QUERIES },
    results: allResults,
  }, null, 2) + '\n');
  writeCsv(allResults, CSV_PATH);

  matchedRecallComparison(allResults);

  log(`\n${'='.repeat(60)}`);
  log('Outputs:');
  log(`  log:  ${LOG_PATH}`);
  log(`  json: ${JSON_PATH}`);
  log(`  csv:  ${CSV_PATH}`);
  log(`\nPlot with: python3 plot_sweep.py ${CSV_PATH}`);

  logStream.end();
}

main().catch(e => { console.error(e); process.exit(1); });
