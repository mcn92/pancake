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
 * Fairness notes (read before publishing numbers):
 *   - Both libraries are called single-threaded. The underlying C++ hnswlib
 *     parallelizes insertion inside its batch API (Python's add_items uses
 *     OpenMP across threads). hnswlib-node only exposes addPoint, which
 *     inserts one vector per call with no thread parameter, so we get one
 *     core regardless of what we do from JS. Pancake's addBatch is also
 *     single-threaded, so this is a symmetric comparison -- but note that
 *     a Python user of hnswlib would see substantially faster builds than
 *     the numbers here suggest.
 *   - Vectors are loaded into one big Float32Array and accessed via subarray
 *     views (zero-copy). Pancake's addBatch accepts these views directly
 *     and writes into WASM heap without per-element boxing.
 *   - hnswlib-node's addPoint REQUIRES a plain Array (it does Array.isArray
 *     internally and rejects typed arrays). To avoid allocating 290k Arrays
 *     during build, we reuse one scratch Array(dim) and copy values into it
 *     before each call. The native binding consumes the array eagerly via
 *     N-API element accessors, so mutating between calls is safe. We still
 *     pay per-element float->double conversions in the v8 array, but that's
 *     unavoidable given the binding's contract. Pancake's wrapper accepts
 *     Float32Array directly, so it pays nothing analogous.
 *   - addBatch is used for Pancake regardless of dtype; the wrapper supports
 *     it for both quantized and float modes.
 *   - Warmup queries run before each timed sweep point, since v8 may
 *     re-tier hot functions when ef_search changes the work shape.
 *   - Latency percentiles use linear interpolation (numpy default).
 *
 * Usage:
 *   node benchmark_nytimes.js [path-to-hdf5]
 */

const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Pancake = require('../pancake.js');
const { parseBenchmarkArgs, resolveSingleValue, resolveSweepValues } = require('./bench_args');

const parsedArgs = parseBenchmarkArgs();

// --- Angular dataset registry --------------------------------------------
// All entries are ann-benchmarks HDF5 files (train/test/neighbors, angular
// metric). dim is read from the file at load time, so only the path, a display
// label, and the download URL differ between datasets. Select with
// --dataset <name>; a positional path still overrides the resolved default.
const RAW_ARGS = process.argv.slice(2);
function getStrArg(name, def) {
  const i = RAW_ARGS.indexOf('--' + name);
  return i >= 0 && i + 1 < RAW_ARGS.length ? RAW_ARGS[i + 1] : def;
}
const DATASETS = {
  nytimes: {
    label: 'NYTimes-256',
    file: path.join(__dirname, '..', 'nytimes', 'nytimes-256-angular.hdf5'),
    url: 'http://ann-benchmarks.com/nytimes-256-angular.hdf5',
  },
  glove: {
    label: 'GloVe-100',
    file: path.join(__dirname, '..', 'glove', 'glove-100-angular.hdf5'),
    url: 'http://ann-benchmarks.com/glove-100-angular.hdf5',
  },
};
const DATASET = getStrArg('dataset', 'nytimes').toLowerCase();
if (!DATASETS[DATASET]) {
  console.error(`Unknown --dataset ${DATASET}. Options: ${Object.keys(DATASETS).join(', ')}.`);
  process.exit(1);
}
const DS = DATASETS[DATASET];
// Positional arg (a bare path, not a flag value) still overrides the default
// file. parseBenchmarkArgs() doesn't know --dataset/--count, so it leaves both
// the flags and their values in .args; skip any value following such a flag.
const VALUE_FLAGS = new Set(['--dataset', '--count']);
const POSITIONAL = parsedArgs.args.find(
  (a, i, arr) => a && !a.startsWith('-') && !(i > 0 && VALUE_FLAGS.has(arr[i - 1]))
);
const HDF5_PATH = POSITIONAL || DS.file;

// --count N indexes only the first N train vectors. The shipped HDF5 ground
// truth is computed over the FULL train set, so a subset needs its own GT;
// loadDataset() brute-forces cosine GT for the subset (these are angular
// datasets). 0 / unset => use the whole file and the shipped GT.
const COUNT = (() => {
  const i = RAW_ARGS.indexOf('--count');
  if (i < 0 || i + 1 >= RAW_ARGS.length) return 0;
  const n = parseInt(RAW_ARGS[i + 1], 10);
  return Number.isInteger(n) && n > 0 ? n : 0;
})();

// --- Sweep configuration ---
const K = 10;
const M = resolveSingleValue(parsedArgs.m, 16);
const EF_CONSTRUCTION = resolveSingleValue(parsedArgs.efConstruction, 150);
const EF_SEARCH_VALUES = resolveSweepValues(parsedArgs, [20, 40, 60, 80, 100, 150, 200, 300, 500, 800]);
const REPETITIONS = 3;
const WARMUP_QUERIES = 200;
const PANCAKE_BATCH_SIZE = 16384;

let usearch;
try {
  usearch = require('usearch');
} catch (e) {
  // optional — will skip usearch configs
}

let HierarchicalNSW;
try {
  HierarchicalNSW = require('hnswlib-node').HierarchicalNSW;
} catch (e) {
  // optional — will skip hnswlib configs
}

let native;
try {
  native = require('../native');
} catch (e) {
  // optional — will skip pancake-native configs (addon not built)
}

const CONFIGS = [
  { label: 'pancake-i8-wasm',     library: 'pancake',        dtype: 'i8' },
  { label: 'pancake-f32-wasm',    library: 'pancake',        dtype: 'f32' },
  { label: 'pancake-i8-native',   library: 'pancake-native', dtype: 'i8' },
  { label: 'pancake-f32-native',  library: 'pancake-native', dtype: 'f32' },
  { label: 'usearch-i8-native',   library: 'usearch',        dtype: 'i8' },
  { label: 'usearch-f32-native',  library: 'usearch',        dtype: 'f32' },
  { label: 'hnswlib-f32-native',  library: 'hnswlib',        dtype: 'f32' },
].filter(c => {
  if (c.library === 'pancake-native' && !native) return false;
  if (c.library === 'usearch' && !usearch) return false;
  if (c.library === 'hnswlib' && !HierarchicalNSW) return false;
  return true;
});

// --- Logging setup ---
const RESULTS_DIR = path.join(__dirname, '..', 'benchmark_results');
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

// Brute-force cosine ground truth for a (possibly subset) train set. These are
// angular datasets, so neighbors are ranked by cosine distance (1 - dot/|q||x|),
// NOT L2. Used only when --count subsets the data, since the shipped GT is
// computed over the full train set and is invalid for a subset.
function computeCosineGroundTruth(train, test, dim, k) {
  const nq = test.length;
  log(`Computing brute-force cosine ground truth (${nq} x ${train.length} x ${dim}D)...`);
  const t0 = performance.now();

  // Pre-norm train rows once (queries normalized per-query below).
  const trainNorm = new Float32Array(train.length);
  for (let i = 0; i < train.length; i++) {
    let s = 0; const v = train[i];
    for (let d = 0; d < dim; d++) s += v[d] * v[d];
    trainNorm[i] = Math.sqrt(s) || 1;
  }

  const gt = new Array(nq);
  for (let q = 0; q < nq; q++) {
    const query = test[q];
    let qn = 0;
    for (let d = 0; d < dim; d++) qn += query[d] * query[d];
    qn = Math.sqrt(qn) || 1;

    const sims = new Float32Array(train.length);
    for (let i = 0; i < train.length; i++) {
      let dot = 0; const v = train[i];
      for (let d = 0; d < dim; d++) dot += query[d] * v[d];
      sims[i] = dot / (qn * trainNorm[i]);  // cosine similarity (higher = closer)
    }
    const idx = new Array(train.length);
    for (let i = 0; i < train.length; i++) idx[i] = i;
    idx.sort((a, b) => sims[b] - sims[a]);  // descending similarity
    gt[q] = new Set(idx.slice(0, k));

    if ((q + 1) % 25 === 0) {
      const el = (performance.now() - t0) / 1000;
      const eta = (el / (q + 1)) * (nq - q - 1);
      process.stdout.write(`  ${q + 1}/${nq} (${el.toFixed(0)}s, ~${eta.toFixed(0)}s left)\r`);
    }
  }
  log(`  Ground truth computed in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  return gt;
}

// --- Data loading ---
//
// Loads train + test + ground-truth into flat typed arrays, then exposes
// per-row subarray views. The flat buffers are owned by us (allocated fresh,
// not aliasing Buffer memory) so alignment is guaranteed for Float32Array.
function loadDataset(hdf5Path) {
  log(`Loading dataset from ${hdf5Path}...`);

  // Write the python loader to a temp file rather than inlining via -c, so
  // the dataset path can contain shell metacharacters without breaking us.
  const tmpBin = path.join(os.tmpdir(), `nytimes_sweep_${process.pid}.bin`);
  const tmpPy  = path.join(os.tmpdir(), `nytimes_sweep_${process.pid}.py`);
  const script = `
import h5py, json, struct, sys
src, dst, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
f = h5py.File(src, "r")
train = f["train"][:]
test  = f["test"][:]
neighbors = f["neighbors"][:, :k]
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
  const info = JSON.parse(execFileSync(
    'python3',
    [tmpPy, hdf5Path, tmpBin, String(K)],
    { encoding: 'utf8' }
  ));
  log(`  Train: ${info.n_train} vectors, ${info.dim}D`);
  log(`  Test:  ${info.n_test} queries`);

  const buf = fs.readFileSync(tmpBin);
  let offset = 0;
  const infoLen = buf.readUInt32LE(offset); offset += 4;
  offset += infoLen;

  const dim = info.dim;

  // Copy into freshly-allocated typed arrays. The source offset (4 + JSON
  // header length) is not guaranteed to be 4-byte aligned — and it isn't for
  // every dataset (the JSON length varies) — so a Float32Array view over
  // buf.buffer at that offset throws "start offset should be a multiple of 4".
  // Copy the raw bytes through Buffer.copy(), which has no alignment
  // requirement, into the typed arrays' own (aligned) backing buffers.
  const trainFlat = new Float32Array(info.n_train * dim);
  buf.copy(Buffer.from(trainFlat.buffer), 0, offset, offset + info.n_train * dim * 4);
  offset += info.n_train * dim * 4;

  const testFlat = new Float32Array(info.n_test * dim);
  buf.copy(Buffer.from(testFlat.buffer), 0, offset, offset + info.n_test * dim * 4);
  offset += info.n_test * dim * 4;

  const neighborsFlat = new Int32Array(info.n_test * K);
  buf.copy(Buffer.from(neighborsFlat.buffer), 0, offset, offset + info.n_test * K * 4);

  // Per-row views. subarray() is O(1) -- no copy.
  const train = new Array(info.n_train);
  for (let i = 0; i < info.n_train; i++) train[i] = trainFlat.subarray(i * dim, (i + 1) * dim);

  const test = new Array(info.n_test);
  for (let i = 0; i < info.n_test; i++) test[i] = testFlat.subarray(i * dim, (i + 1) * dim);

  // Ground truth: each query's K neighbor ids stored as a Set for O(1) recall lookup.
  const groundTruth = new Array(info.n_test);
  for (let i = 0; i < info.n_test; i++) {
    const s = new Set();
    for (let j = 0; j < K; j++) s.add(neighborsFlat[i * K + j]);
    groundTruth[i] = s;
  }

  // Cleanup temp files (best-effort).
  try { fs.unlinkSync(tmpBin); } catch (_) {}
  try { fs.unlinkSync(tmpPy);  } catch (_) {}

  // --count subset: index only the first N train vectors and recompute GT over
  // that subset (the shipped GT indexes the full set and would be invalid).
  if (COUNT > 0 && COUNT < info.n_train) {
    const subTrain = train.slice(0, COUNT);
    log(`  Subsetting to first ${COUNT.toLocaleString()} train vectors (--count); recomputing GT.`);
    const subGt = computeCosineGroundTruth(subTrain, test, dim, K);
    return {
      train: subTrain, test, groundTruth: subGt, dim,
      info: { ...info, n_train: subTrain.length },
    };
  }

  return { train, test, groundTruth, dim, info };
}

// --- Metric helpers ---

// numpy-default linear-interpolation percentile.
function percentile(sortedAsc, p) {
  if (sortedAsc.length === 0) return NaN;
  if (sortedAsc.length === 1) return sortedAsc[0];
  const rank = p * (sortedAsc.length - 1);
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  if (lo === hi) return sortedAsc[lo];
  return sortedAsc[lo] + (sortedAsc[hi] - sortedAsc[lo]) * (rank - lo);
}

function mean(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

function stddev(arr) {
  const m = mean(arr);
  let v = 0;
  for (let i = 0; i < arr.length; i++) { const d = arr[i] - m; v += d * d; }
  return Math.sqrt(v / arr.length);
}

// --- USearch: build and query ---
// USearch's JS binding only respects expansion_search at construction time.
// Workaround: build once, save to disk, then view() with a fresh Index per ef_search.
function buildUsearch({ train, dim, dtype }) {
  const quantization = dtype === 'i8' ? 'i8' : 'f32';
  log(`  [usearch-${dtype}] building index (M=${M}, ef_c=${EF_CONSTRUCTION}, metric=cos, quantization=${quantization})...`);
  const index = new usearch.Index({
    metric: 'cos',
    connectivity: M,
    dimensions: dim,
    quantization,
    expansion_add: EF_CONSTRUCTION,
    expansion_search: EF_SEARCH_VALUES[0],
  });

  const t0 = performance.now();
  for (let i = 0; i < train.length; i++) {
    index.add(BigInt(i), train[i]);
    if ((i + 1) % 50000 === 0) {
      log(`    ${(i + 1).toLocaleString()}/${train.length.toLocaleString()}`);
    }
  }
  const buildMs = performance.now() - t0;

  const savePath = path.join(RESULTS_DIR, `_usearch_nyt_${dtype}_temp.bin`);
  index.save(savePath);
  log(`  [usearch-${dtype}] build: ${(buildMs / 1000).toFixed(1)}s (saved to ${savePath})`);
  return { index, buildMs, memoryMB: null, savePath, quantization };
}

function queryUsearch(built, test, groundTruth, efSearch, dim) {
  const viewIndex = new usearch.Index({
    metric: 'cos',
    connectivity: M,
    dimensions: dim,
    quantization: built.quantization,
    expansion_search: efSearch,
  });
  viewIndex.view(built.savePath);

  for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) {
    viewIndex.search(test[i], K);
  }

  const latencies = new Float64Array(test.length);
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const st = performance.now();
    const results = viewIndex.search(test[i], K);
    latencies[i] = performance.now() - st;
    let hits = 0;
    const truth = groundTruth[i];
    const keys = results.keys;
    for (let j = 0; j < keys.length; j++) if (truth.has(Number(keys[j]))) hits++;
    totalRecall += hits / truth.size;
  }
  return { latencies, meanRecall: totalRecall / test.length };
}

// --- Pancake: build and query ---
async function buildPancake({ train, dim, dtype }) {
  const quantized = dtype === 'i8';
  log(`  [pancake-${dtype}] building index (M=${M}, ef_c=${EF_CONSTRUCTION})...`);
  // create() defaults to cosine, which matches these angular datasets; no
  // explicit metric needed.
  const index = await Pancake.create({
    dim,
    maxElements: train.length,
    quantized,
    M,
    efConstruction: EF_CONSTRUCTION,
    efSearch: EF_SEARCH_VALUES[0],
  });

  const t0 = performance.now();
  // addBatch supports both quantized and float modes; previous version of
  // this script gated it on `quantized`, which silently penalized float
  // Pancake. Use it unconditionally.
  for (let start = 0; start < train.length; start += PANCAKE_BATCH_SIZE) {
    const end = Math.min(start + PANCAKE_BATCH_SIZE, train.length);
    // train.slice() returns a plain Array of Float32Array views (the views
    // themselves are not copied). addBatch requires Array.isArray(input).
    index.addBatch(train.slice(start, end));
  }
  const buildMs = performance.now() - t0;
  const memMB = index.memory / 1024 / 1024;
  log(`  [pancake-${tag}] build: ${(buildMs / 1000).toFixed(1)}s, memory: ${memMB.toFixed(0)} MB`);
  return { index, buildMs, memoryMB: memMB };
}

function queryPancake(index, test, groundTruth, efSearch) {
  index.setEfSearch(efSearch);

  // Warmup: let v8 specialize for the new ef_search work shape.
  for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) {
    index.search(test[i], K);
  }

  const latencies = new Float64Array(test.length);
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const st = performance.now();
    const results = index.search(test[i], K);
    latencies[i] = performance.now() - st;
    let hits = 0;
    const truth = groundTruth[i];
    for (let j = 0; j < results.length; j++) if (truth.has(results[j].id)) hits++;
    totalRecall += hits / truth.size;
  }
  return { latencies, meanRecall: totalRecall / test.length };
}

// --- pancake native addon: build and query ---
// Same graph/quantization as the WASM engine but with AVX2 distance kernels;
// isolates WASM runtime overhead from graph quality. metric=1 (cosine) — these
// are angular datasets, so unlike the L2 pareto_frontier path this must NOT
// pass L2.
function buildPancakeNative({ train, dim, dtype }) {
  const quantized = dtype === 'i8' ? 1 : 0;
  log(`  [pancake-${dtype}-native] building index (M=${M}, ef_c=${EF_CONSTRUCTION})...`);
  const h = native.pancake_init(dim, train.length, quantized, 1 /* 1=cosine */, M, EF_CONSTRUCTION, EF_SEARCH_VALUES[0]);
  if (h === 0xFFFFFFFF) throw new Error('Failed to init native pancake index');

  const t0 = performance.now();
  const flat = new Float32Array(train.length * dim);
  for (let i = 0; i < train.length; i++) flat.set(train[i], i * dim);
  const batchSize = 10000;
  for (let start = 0; start < train.length; start += batchSize) {
    const end = Math.min(start + batchSize, train.length);
    native.pancake_bulk_insert(h, flat.subarray(start * dim, end * dim), end - start);
  }
  const buildMs = performance.now() - t0;
  const memMB = native.pancake_memory(h) / 1024 / 1024;
  log(`  [pancake-${dtype}-native] build: ${(buildMs / 1000).toFixed(1)}s, memory: ${memMB.toFixed(0)} MB`);
  return { handle: h, buildMs, memoryMB: memMB };
}

function queryPancakeNative(built, test, groundTruth, efSearch) {
  native.pancake_set_ef(built.handle, efSearch);
  for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) {
    native.pancake_query(built.handle, test[i], K);
  }

  const latencies = new Float64Array(test.length);
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const st = performance.now();
    const result = native.pancake_query(built.handle, test[i], K);
    latencies[i] = performance.now() - st;
    let hits = 0;
    const truth = groundTruth[i];
    const ids = result.ids;
    for (let j = 0; j < ids.length; j++) if (truth.has(Number(ids[j]))) hits++;
    totalRecall += hits / truth.size;
  }
  return { latencies, meanRecall: totalRecall / test.length };
}

// --- hnswlib-node: build and query ---
function buildHnswlib({ train, dim }) {
  if (!HierarchicalNSW) return null;
  log(`  [hnswlib-f32] building index (M=${M}, ef_c=${EF_CONSTRUCTION})...`);
  log(`  [hnswlib-f32] note: hnswlib-node only exposes addPoint (no batch API);`
    + ` build runs on one core. Python's hnswlib via add_items would be faster.`);
  const index = new HierarchicalNSW('cosine', dim);
  index.initIndex(train.length, M, EF_CONSTRUCTION, 100);

  const t0 = performance.now();
  // hnswlib-node's addon checks Array.isArray() on the input and rejects
  // typed arrays. We can't pass Float32Array directly, but we can avoid
  // the per-row allocation cost of Array.from() by reusing one plain Array
  // across all calls and copying values into it. The native binding reads
  // the array's elements via N-API element accessors immediately, so it's
  // safe to mutate the buffer between calls.
  const scratch = new Array(dim);
  for (let i = 0; i < train.length; i++) {
    const v = train[i];
    for (let j = 0; j < dim; j++) scratch[j] = v[j];
    index.addPoint(scratch, i);
  }
  const buildMs = performance.now() - t0;
  log(`  [hnswlib-f32] build: ${(buildMs / 1000).toFixed(1)}s`);
  return { index, buildMs, memoryMB: null };
}

function queryHnswlib(index, test, groundTruth, efSearch, scratch) {
  index.setEf(efSearch);

  const dim = test[0].length;

  // Warmup. Same scratch-buffer trick as build to avoid 10k * dim allocations
  // per repetition.
  for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) {
    const v = test[i];
    for (let j = 0; j < dim; j++) scratch[j] = v[j];
    index.searchKnn(scratch, K);
  }

  const latencies = new Float64Array(test.length);
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const v = test[i];
    // Copy into scratch BEFORE the timer starts, so we measure search, not
    // float64<->int conversion.
    for (let j = 0; j < dim; j++) scratch[j] = v[j];
    const st = performance.now();
    const results = index.searchKnn(scratch, K);
    latencies[i] = performance.now() - st;
    let hits = 0;
    const truth = groundTruth[i];
    const neigh = results.neighbors;
    for (let j = 0; j < neigh.length; j++) if (truth.has(neigh[j])) hits++;
    totalRecall += hits / truth.size;
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
  } else if (config.library === 'pancake-native') {
    built = buildPancakeNative({ train, dim, dtype: config.dtype });
  } else if (config.library === 'usearch') {
    built = buildUsearch({ train, dim, dtype: config.dtype });
  } else {
    built = buildHnswlib({ train, dim });
    if (!built) return null;
  }

  const { index, buildMs, memoryMB } = built;
  const points = [];

  for (const efSearch of EF_SEARCH_VALUES) {
    log(`\n  ef_search=${efSearch}`);
    const reps = [];
    // Reusable scratch buffer for hnswlib query path; allocated once per
    // sweep point, reused across all repetitions and queries.
    const hnswScratch = config.library === 'hnswlib' ? new Array(dim) : null;
    for (let rep = 0; rep < REPETITIONS; rep++) {
      let queryResult;
      if (config.library === 'pancake') {
        queryResult = queryPancake(index, test, groundTruth, efSearch);
      } else if (config.library === 'pancake-native') {
        queryResult = queryPancakeNative(built, test, groundTruth, efSearch);
      } else if (config.library === 'usearch') {
        queryResult = queryUsearch(built, test, groundTruth, efSearch, dim);
      } else {
        queryResult = queryHnswlib(index, test, groundTruth, efSearch, hnswScratch);
      }
      const { latencies, meanRecall } = queryResult;
      const sorted = Float64Array.from(latencies).sort();
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

  if (config.library === 'pancake-native') {
    native.pancake_dispose(built.handle);
  } else if (index && typeof index.dispose === 'function') {
    index.dispose();
  }
  if (built.savePath && fs.existsSync(built.savePath)) {
    fs.unlinkSync(built.savePath);
  }

  return {
    label: config.label,
    library: config.library,
    dtype: config.dtype,
    buildMs,
    memoryMB, // null for hnswlib (binding doesn't expose it)
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
//
// For each pair (A, B), at every recall level achieved by A, log-interpolate
// B's QPS at that recall and report the ratio. Skips A's points that fall
// outside B's recall range (and logs the skip so the output isn't silently
// sparse).
function matchedRecallComparison(results) {
  log(`\n${'='.repeat(60)}`);
  log('Matched-recall comparison');
  log('='.repeat(60));
  log("For each pair (A, B), interpolate B's QPS at A's recall points,");
  log('then report A_qps / B_qps. Ratios > 1 mean A is faster at that recall.');

  const byLabel = {};
  for (const r of results) if (r) byLabel[r.label] = r;

  function qpsAtRecall(points, targetRecall) {
    const sorted = [...points].sort((a, b) => a.recall_mean - b.recall_mean);
    if (targetRecall < sorted[0].recall_mean - 1e-9) return { qps: null, reason: 'below B range' };
    if (targetRecall > sorted[sorted.length - 1].recall_mean + 1e-9) return { qps: null, reason: 'above B range' };
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i].recall_mean >= targetRecall) {
        const r0 = sorted[i - 1].recall_mean, r1 = sorted[i].recall_mean;
        const q0 = sorted[i - 1].qps_mean,    q1 = sorted[i].qps_mean;
        if (r1 === r0) return { qps: q0, reason: null };
        const t = (targetRecall - r0) / (r1 - r0);
        // Log-linear interpolation in QPS, since QPS varies over orders of
        // magnitude across an HNSW recall sweep.
        return { qps: Math.exp(Math.log(q0) + t * (Math.log(q1) - Math.log(q0))), reason: null };
      }
    }
    return { qps: null, reason: 'no bracket' };
  }

  const labels = Object.keys(byLabel);
  for (let i = 0; i < labels.length; i++) {
    for (let j = 0; j < labels.length; j++) {
      if (i === j) continue;
      const A = byLabel[labels[i]], B = byLabel[labels[j]];
      log(`\n  ${A.label} vs ${B.label}:`);
      for (const pA of A.points) {
        const { qps: qB, reason } = qpsAtRecall(B.points, pA.recall_mean);
        if (qB === null) {
          log(`    at recall=${(pA.recall_mean * 100).toFixed(1)}%: skipped (${reason})`);
          continue;
        }
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
    log(`Download: curl -L -o ${HDF5_PATH} ${DS.url}`);
    process.exit(1);
  }

  const dataset = loadDataset(HDF5_PATH);
  log(`\n${'='.repeat(60)}`);
  log(`Recall-QPS sweep on ${DS.label} (Pancake vs USearch vs hnswlib)`);
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
    benchmark: `${DATASET}-angular-sweep`,
    timestamp: new Date().toISOString(),
    dataset: dataset.info,
    params: { K, M, EF_CONSTRUCTION, EF_SEARCH_VALUES, REPETITIONS, WARMUP_QUERIES, PANCAKE_BATCH_SIZE, ROTATION },
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
