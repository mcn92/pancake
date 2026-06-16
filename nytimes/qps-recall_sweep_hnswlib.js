#!/usr/bin/env node
'use strict';

/**
 * Recall-QPS sweep on NYTimes-256: Pancake vs hnswlib-node.
 *
 * For each library, we:
 *   1. Build one index at a fixed (M, ef_construction)
 *   2. Query the same index at many different ef_search values
 *   3. Measure recall@10, QPS, p50, p99 at each point
 *   4. Repeat queries N times per point for variance
 *
 * Output: JSON file with full per-point measurements + CSV for plotting.
 *
 * Usage:
 *   node nytimes/qps-recall_sweep_hnswlib.js [path-to-hdf5]
 *
 * Requires: pancake.js (your wrapper), hnswlib-node (npm), h5py (for loading).
 */

const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Pancake = require('../pancake.js');

const HDF5_PATH = process.argv[2] || './nytimes/nytimes-256-angular.hdf5';

// --- Sweep configuration ---
const K = 10;
const M = 16;
const EF_CONSTRUCTION = 100;
const EF_SEARCH_VALUES = [20, 40, 60, 80, 100, 150, 200, 300, 500, 800];
const REPETITIONS = 3;
const WARMUP_QUERIES = 200;

const CONFIGS = [
  { label: 'pancake-i8-wasm',    library: 'pancake',  dtype: 'i8' },
  { label: 'hnswlib-f32-native',  library: 'hnswlib',  dtype: 'f32' },
];

// --- Logging setup ---
const RESULTS_DIR = path.join(__dirname, 'benchmark_results');
if (!fs.existsSync(RESULTS_DIR)) fs.mkdirSync(RESULTS_DIR);
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const LOG_PATH  = path.join(RESULTS_DIR, `sweep_hnswlib_${timestamp}.log`);
const JSON_PATH = path.join(RESULTS_DIR, `sweep_hnswlib_${timestamp}.json`);
const CSV_PATH  = path.join(RESULTS_DIR, `sweep_hnswlib_${timestamp}.csv`);
const logStream = fs.createWriteStream(LOG_PATH);

function log(msg = '') {
  console.log(msg);
  logStream.write(msg + '\n');
}

// --- Data loading ---
function loadDataset(hdf5Path) {
  log(`Loading dataset from ${hdf5Path}...`);
  const tmpBin = path.join(os.tmpdir(), `nytimes_sweep_${process.pid}.bin`);
  const tmpPy = path.join(os.tmpdir(), `nytimes_sweep_${process.pid}.py`);
  const script = `
import h5py, json, sys, struct
src, dst = sys.argv[1], sys.argv[2]
f = h5py.File(src, "r")
train = f["train"][:]
test = f["test"][:]
neighbors = f["neighbors"][:, :${K}]
info = {"n_train": int(train.shape[0]), "n_test": int(test.shape[0]),
        "dim": int(train.shape[1])}
with open(dst, "wb") as out:
    info_bytes = json.dumps(info).encode()
    out.write(struct.pack("<I", len(info_bytes)))
    out.write(info_bytes)
    out.write(train.astype("float32").tobytes())
    out.write(test.astype("float32").tobytes())
    out.write(neighbors.astype("int32").tobytes())
print(json.dumps(info))
`;
  fs.writeFileSync(tmpPy, script);
  const info = JSON.parse(execFileSync('python3', [tmpPy, hdf5Path, tmpBin], { encoding: 'utf8' }));
  log(`  Train: ${info.n_train} vectors, ${info.dim}D`);
  log(`  Test:  ${info.n_test} queries`);

  const buf = fs.readFileSync(tmpBin);
  let offset = 0;
  const infoLen = buf.readUInt32LE(offset); offset += 4;
  offset += infoLen;

  const dim = info.dim;
  const trainFloats = new Float32Array(buf.buffer, buf.byteOffset + offset,
                                        info.n_train * dim);
  offset += info.n_train * dim * 4;
  const testFloats = new Float32Array(buf.buffer, buf.byteOffset + offset,
                                       info.n_test * dim);
  offset += info.n_test * dim * 4;
  const neighborsInt = new Int32Array(buf.buffer, buf.byteOffset + offset,
                                       info.n_test * K);

  const train = new Array(info.n_train);
  for (let i = 0; i < info.n_train; i++)
    train[i] = trainFloats.subarray(i * dim, (i + 1) * dim);
  const test = new Array(info.n_test);
  for (let i = 0; i < info.n_test; i++)
    test[i] = testFloats.subarray(i * dim, (i + 1) * dim);
  const groundTruth = new Array(info.n_test);
  for (let i = 0; i < info.n_test; i++)
    groundTruth[i] = Array.from(neighborsInt.subarray(i * K, (i + 1) * K));

  try { fs.unlinkSync(tmpBin); } catch (_) {}
  try { fs.unlinkSync(tmpPy); } catch (_) {}

  return { train, test, groundTruth, dim, info };
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
async function buildPancake({ train, dim, dtype }) {
  const quantized = dtype === 'i8';
  log(`  [pancake-${dtype}] building index (M=${M}, ef_c=${EF_CONSTRUCTION})...`);
  const index = await Pancake.create({
    dim,
    maxElements: train.length,
    quantized,
    M,
    efConstruction: EF_CONSTRUCTION,
    efSearch: EF_SEARCH_VALUES[0],
  });

  const t0 = performance.now();
  if (typeof index.addBatch === 'function' && quantized) {
    const batchSize = 4096;
    for (let start = 0; start < train.length; start += batchSize) {
      const end = Math.min(start + batchSize, train.length);
      index.addBatch(train.slice(start, end));
    }
  } else {
    for (let i = 0; i < train.length; i++) index.add(train[i]);
  }
  const buildMs = performance.now() - t0;
  log(`  [pancake-${dtype}] build: ${(buildMs / 1000).toFixed(1)}s, memory: ${(index.memory / 1024 / 1024).toFixed(0)} MB`);
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
    throw new Error('Pancake index does not expose ef_search mutation; '
                  + 'add a setEfSearch() wrapper or export _i8_set_ef/_float_set_ef.');
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
  const index = new HierarchicalNSW('cosine', dim);
  index.initIndex(train.length, M, EF_CONSTRUCTION, 100);

  const t0 = performance.now();
  for (let i = 0; i < train.length; i++) {
    index.addPoint(Array.from(train[i]), i);
  }
  const buildMs = performance.now() - t0;
  log(`  [hnswlib-f32] build: ${(buildMs / 1000).toFixed(1)}s`);
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
    built = await buildPancake({ train, dim, dtype: config.dtype });
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
      + `(±${(summary.recall_std * 100).toFixed(2)}) `
      + `qps=${summary.qps_mean.toFixed(0)} (±${summary.qps_std.toFixed(0)})`);
    points.push(summary);
  }

  if (typeof index.dispose === 'function') index.dispose();

  return {
    label: config.label,
    library: config.library,
    dtype: config.dtype,
    buildMs,
    params: { M, ef_construction: EF_CONSTRUCTION, K },
    points,
  };
}

// --- Write CSV for plotting ---
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
  log('For each pair (A, B), interpolate B to match A\'s recall points,');
  log('and report the QPS ratio A/B at those points.');

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
  if (!fs.existsSync(HDF5_PATH)) {
    log(`Dataset not found: ${HDF5_PATH}`);
    log('Download: curl -L -o ./nytimes/nytimes-256-angular.hdf5 '
      + 'http://ann-benchmarks.com/nytimes-256-angular.hdf5');
    process.exit(1);
  }

  const dataset = loadDataset(HDF5_PATH);
  log(`\n${'='.repeat(60)}`);
  log('Recall-QPS sweep on NYTimes-256 (Pancake vs hnswlib)');
  log(`${dataset.info.n_train} vectors, ${dataset.dim}D, ${dataset.info.n_test} queries`);
  log(`k=${K}, M=${M}, ef_construction=${EF_CONSTRUCTION}`);
  log(`ef_search sweep: [${EF_SEARCH_VALUES.join(', ')}]`);
  log(`Repetitions: ${REPETITIONS}, Warmup: ${WARMUP_QUERIES}`);
  log('='.repeat(60));

  const allResults = [];
  for (const config of CONFIGS) {
    const result = await sweepOne(config, dataset);
    allResults.push(result);
  }

  fs.writeFileSync(JSON_PATH, JSON.stringify({
    benchmark: 'nytimes-256-sweep-hnswlib',
    timestamp: new Date().toISOString(),
    dataset: dataset.info,
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
  log('\nPlot with:');
  log(`  python3 plot_sweep.py ${CSV_PATH}`);

  logStream.end();
}

main().catch(e => { console.error(e); process.exit(1); });
