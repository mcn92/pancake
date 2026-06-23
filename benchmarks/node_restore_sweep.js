#!/usr/bin/env node
'use strict';

/**
 * Node restore/load sweep
 *
 * Measures the Node deployment path from snapshot bytes on disk to first query:
 *   - file read time
 *   - Pancake.create() / WASM engine instantiation time
 *   - index.import() / deserialize time
 *   - first query latency
 *   - warm query latency
 *
 * Usage:
 *   node benchmarks/node_restore_sweep.js
 *   node benchmarks/node_restore_sweep.js --counts 1000,5000,10000 --dim 256
 *   node benchmarks/node_restore_sweep.js --snapshot ./index.pnck --dim 256 --metric cosine --quantized true
 *
 * For package v3 snapshots, dim/metric/quantized/count are inferred. For raw
 * engine snapshots, pass --dim, --metric, --quantized, and --max-elements.
 */

const fs = require('fs');
const os = require('os');
const path = require('path');
const { performance } = require('perf_hooks');
const Pancake = require('../pancake.js');
const { parseBenchmarkArgs, resolveSingleValue } = require('./bench_args');

const PANCAKE_MAGIC = 0x504E434B;
const FLOAT_HNSW_MAGIC_V0 = 0x464C4857;
const FLOAT_HNSW_MAGIC_V1 = 0x464C4831;
const INT8_HNSW_MAGIC_V0 = 0x49384857;
const INT8_HNSW_MAGIC_V1 = 0x49384831;

const parsedArgs = parseBenchmarkArgs();
const rawArgs = process.argv.slice(2);

function getArg(name, fallback = undefined) {
  const idx = rawArgs.indexOf(`--${name}`);
  return idx >= 0 && idx + 1 < rawArgs.length ? rawArgs[idx + 1] : fallback;
}

function getIntArg(name, fallback) {
  const value = getArg(name);
  if (value === undefined) return fallback;
  const parsed = parseInt(value, 10);
  if (!Number.isInteger(parsed)) throw new Error(`--${name} must be an integer`);
  return parsed;
}

function getBoolArg(name, fallback) {
  const value = getArg(name);
  if (value === undefined) return fallback;
  if (value === 'true' || value === '1') return true;
  if (value === 'false' || value === '0') return false;
  throw new Error(`--${name} must be true or false`);
}

function getCounts() {
  const value = getArg('counts');
  if (!value) return [1000, 5000, 10000];
  const counts = value.split(',').map((v) => parseInt(v, 10)).filter((v) => Number.isInteger(v) && v > 0);
  if (counts.length === 0) throw new Error('--counts must include at least one positive integer');
  return counts;
}

const SNAPSHOT_PATH = getArg('snapshot');
const DIM = getIntArg('dim', 256);
const METRIC = getArg('metric', 'cosine');
const QUANTIZED = getBoolArg('quantized', true);
const M = resolveSingleValue(parsedArgs.m, 16);
const EF_CONSTRUCTION = resolveSingleValue(parsedArgs.efConstruction, 50);
const EF_SEARCH = resolveSingleValue(parsedArgs.efSearch, 100);
const RESTORE_REPS = getIntArg('restore-reps', 5);
const QUERY_REPS = getIntArg('query-reps', 20);
const COUNTS = getCounts();
const KEEP_SNAPSHOTS = rawArgs.includes('--keep-snapshots');
const OUT_DIR = getArg('out-dir', path.join(os.tmpdir(), 'pancake-node-restore'));

function percentile(sorted, p) {
  if (sorted.length === 0) return NaN;
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))];
}

function fmtMs(value) {
  return Number.isFinite(value) ? value.toFixed(2) : '-';
}

function fmtKb(value) {
  return Number.isFinite(value) ? (value / 1024).toFixed(1) : '-';
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

function normalize(vec) {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  return vec;
}

function makeVector(seed, dim, normalized) {
  const rand = makeRng(seed);
  const vec = new Float32Array(dim);
  for (let i = 0; i < dim; i++) vec[i] = rand() * 2 - 1;
  return normalized ? normalize(vec) : vec;
}

function rawMetadata(bytes) {
  if (!bytes || bytes.byteLength < 12) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = view.getUint32(0, true);
  if (magic === FLOAT_HNSW_MAGIC_V1 || magic === INT8_HNSW_MAGIC_V1) {
    if (bytes.byteLength < 40) return null;
    return {
      dim: view.getUint32(4, true),
      count: view.getUint32(12, true),
      metric: view.getUint32(32, true) === 0 ? 'l2' : 'cosine',
      quantized: magic === INT8_HNSW_MAGIC_V1,
    };
  }
  if (magic === FLOAT_HNSW_MAGIC_V0 || magic === INT8_HNSW_MAGIC_V0) {
    if (bytes.byteLength < 36) return null;
    return {
      dim: view.getUint32(4, true),
      count: view.getUint32(8, true),
      metric: view.getUint32(28, true) === 0 ? 'l2' : 'cosine',
      quantized: magic === INT8_HNSW_MAGIC_V0,
    };
  }
  return null;
}

function inferSnapshotOptions(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (u8.byteLength < 4) return null;
  const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (view.getUint32(0, true) !== PANCAKE_MAGIC) return rawMetadata(u8);

  const version = view.getUint32(4, true);
  const dim = view.getUint32(8, true);
  const metric = view.getUint32(12, true) === 0 ? 'l2' : 'cosine';
  const quantized = view.getUint32(16, true) !== 0;
  let raw;
  if (version === 3) {
    const mappingCount = view.getUint32(24, true);
    const wasmSize = view.getUint32(28, true);
    const wasmOffset = 32 + mappingCount * 8;
    raw = u8.subarray(wasmOffset, wasmOffset + wasmSize);
  } else {
    raw = u8.subarray(20);
  }
  const metadata = rawMetadata(raw);
  return {
    dim,
    metric,
    quantized,
    count: metadata ? metadata.count : undefined,
  };
}

async function buildSyntheticSnapshot(count) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const normalized = METRIC === 'cosine';
  const opts = {
    dim: DIM,
    maxElements: count,
    metric: METRIC,
    quantized: QUANTIZED,
    M,
    efConstruction: EF_CONSTRUCTION,
    efSearch: EF_SEARCH,
  };

  const index = await Pancake.create(opts);
  try {
    const vectors = new Array(count);
    for (let i = 0; i < count; i++) vectors[i] = makeVector(i + 1, DIM, normalized);
    index.addBatch(vectors);

    const snapshot = index.export();
    const filePath = path.join(OUT_DIR, `node_restore_${count}_${DIM}d_${METRIC}_${QUANTIZED ? 'i8' : 'f32'}.pnck`);
    fs.writeFileSync(filePath, snapshot);
    return {
      filePath,
      opts,
      dim: DIM,
      metric: METRIC,
      quantized: QUANTIZED,
      query: makeVector(999999, DIM, normalized),
      snapshotBytes: snapshot.byteLength,
      builtHere: true,
    };
  } finally {
    index.dispose();
  }
}

function resolveSnapshotCase(filePath) {
  const bytes = fs.readFileSync(filePath);
  const inferred = inferSnapshotOptions(bytes) || {};
  const dim = inferred.dim || DIM;
  const metric = inferred.metric || METRIC;
  const quantized = inferred.quantized !== undefined ? inferred.quantized : QUANTIZED;
  const count = inferred.count;
  const maxElements = getIntArg('max-elements', count || 100000);

  return {
    filePath,
    opts: { dim, maxElements, metric, quantized, M, efConstruction: EF_CONSTRUCTION, efSearch: EF_SEARCH },
    dim,
    metric,
    quantized,
    query: makeVector(999999, dim, metric === 'cosine'),
    snapshotBytes: bytes.byteLength,
    builtHere: false,
  };
}

async function measureRestore(snapshotCase) {
  const readTimes = [];
  const createTimes = [];
  const importTimes = [];
  const firstQueryTimes = [];
  const totalToQueryTimes = [];
  const warmTimes = [];
  let memoryBytes = null;
  let count = null;

  for (let rep = 0; rep < RESTORE_REPS; rep++) {
    const totalStart = performance.now();

    const readStart = performance.now();
    const bytes = fs.readFileSync(snapshotCase.filePath);
    const readMs = performance.now() - readStart;

    const createStart = performance.now();
    const index = await Pancake.create(snapshotCase.opts);
    const createMs = performance.now() - createStart;

    const importStart = performance.now();
    index.import(bytes);
    const importMs = performance.now() - importStart;

    const firstQueryStart = performance.now();
    index.search(snapshotCase.query, 10);
    const firstQueryMs = performance.now() - firstQueryStart;

    const totalToQueryMs = performance.now() - totalStart;

    const localWarm = [];
    for (let i = 0; i < QUERY_REPS; i++) {
      const t0 = performance.now();
      index.search(snapshotCase.query, 10);
      localWarm.push(performance.now() - t0);
    }
    localWarm.sort((a, b) => a - b);

    readTimes.push(readMs);
    createTimes.push(createMs);
    importTimes.push(importMs);
    firstQueryTimes.push(firstQueryMs);
    totalToQueryTimes.push(totalToQueryMs);
    warmTimes.push(percentile(localWarm, 0.5));
    memoryBytes = index.memory;
    count = index.count;
    index.dispose();
  }

  readTimes.sort((a, b) => a - b);
  createTimes.sort((a, b) => a - b);
  importTimes.sort((a, b) => a - b);
  firstQueryTimes.sort((a, b) => a - b);
  totalToQueryTimes.sort((a, b) => a - b);
  warmTimes.sort((a, b) => a - b);

  return {
    filePath: snapshotCase.filePath,
    count,
    snapshotBytes: snapshotCase.snapshotBytes,
    memoryBytes,
    readP50Ms: percentile(readTimes, 0.5),
    createP50Ms: percentile(createTimes, 0.5),
    importP50Ms: percentile(importTimes, 0.5),
    firstQueryP50Ms: percentile(firstQueryTimes, 0.5),
    totalToQueryP50Ms: percentile(totalToQueryTimes, 0.5),
    warmQueryP50Ms: percentile(warmTimes, 0.5),
    totalToQueryP95Ms: percentile(totalToQueryTimes, 0.95),
  };
}

async function main() {
  if (METRIC !== 'cosine' && METRIC !== 'l2') {
    throw new Error('--metric must be cosine or l2');
  }
  if (RESTORE_REPS <= 0 || QUERY_REPS <= 0) {
    throw new Error('--restore-reps and --query-reps must be positive');
  }

  const cases = [];
  if (SNAPSHOT_PATH) {
    cases.push(resolveSnapshotCase(SNAPSHOT_PATH));
  } else {
    for (const count of COUNTS) cases.push(await buildSyntheticSnapshot(count));
  }

  console.log('Node restore/load sweep');
  console.log(`restore_reps=${RESTORE_REPS} query_reps=${QUERY_REPS}`);
  if (!SNAPSHOT_PATH) {
    console.log(`dim=${DIM} metric=${METRIC} quantized=${QUANTIZED} M=${M} efConstruction=${EF_CONSTRUCTION} efSearch=${EF_SEARCH}`);
  }
  console.log('');
  console.log('count\tdim\tmetric\tquantized\tsnapshot_kb\tmemory_mb\tread_p50_ms\tcreate_p50_ms\timport_p50_ms\tfirst_query_p50_ms\ttotal_to_query_p50_ms\ttotal_to_query_p95_ms\twarm_query_p50_ms\tpath');

  for (const snapshotCase of cases) {
    const row = await measureRestore(snapshotCase);
    console.log([
      row.count,
      snapshotCase.dim,
      snapshotCase.metric,
      snapshotCase.quantized,
      fmtKb(row.snapshotBytes),
      (row.memoryBytes / 1024 / 1024).toFixed(2),
      fmtMs(row.readP50Ms),
      fmtMs(row.createP50Ms),
      fmtMs(row.importP50Ms),
      fmtMs(row.firstQueryP50Ms),
      fmtMs(row.totalToQueryP50Ms),
      fmtMs(row.totalToQueryP95Ms),
      fmtMs(row.warmQueryP50Ms),
      row.filePath,
    ].join('\t'));

    if (snapshotCase.builtHere && !KEEP_SNAPSHOTS) {
      try { fs.unlinkSync(snapshotCase.filePath); } catch {}
    }
  }

  if (!SNAPSHOT_PATH && !KEEP_SNAPSHOTS) {
    try { fs.rmdirSync(OUT_DIR); } catch {}
  }
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
