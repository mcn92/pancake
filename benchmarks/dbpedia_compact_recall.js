#!/usr/bin/env node
'use strict';

/**
 * DBpedia uint8 compaction recall benchmark.
 *
 * Measures recall@K against brute-force L2 ground truth before and after:
 *   1. deleting a deterministic fraction of base vectors
 *   2. compacting the index
 *
 * This is meant to answer one specific question: does compaction materially
 * reduce uint8 search quality on the real 1536D DBpedia workload?
 *
 * Usage:
 *   node benchmarks/dbpedia_compact_recall.js
 *   node benchmarks/dbpedia_compact_recall.js /path/to/dbpedia --delete-fraction 0.2
 *   node benchmarks/dbpedia_compact_recall.js --queries 250 --m 12 --ef-construction 100 --ef-search 200
 */

const fs = require('fs');
const path = require('path');
const Pancake = require('../pancake.js');
const { parseBenchmarkArgs, resolveSingleValue } = require('./bench_args');

const parsedBenchArgs = parseBenchmarkArgs();
const rawArgs = parsedBenchArgs.args;

function takeFlagValue(name, defaultValue) {
  const idx = rawArgs.indexOf(name);
  if (idx === -1) return defaultValue;
  if (idx + 1 >= rawArgs.length) {
    throw new Error(`Missing value for ${name}`);
  }
  return rawArgs[idx + 1];
}

function hasFlag(name) {
  return rawArgs.includes(name);
}

function firstPositionalArg() {
  for (let i = 0; i < rawArgs.length; i++) {
    const arg = rawArgs[i];
    if (arg.startsWith('-')) {
      if (arg === '--delete-fraction' || arg === '--queries' || arg === '--base-count' || arg === '--seed') i++;
      continue;
    }
    return arg;
  }
  return null;
}

function parsePositiveInt(value, flagName) {
  const parsed = parseInt(String(value), 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`Invalid value for ${flagName}: ${value}`);
  }
  return parsed;
}

function parseUnitFraction(value, flagName) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed >= 1) {
    throw new Error(`Invalid value for ${flagName}: ${value}`);
  }
  return parsed;
}

const DBPEDIA_DIR = firstPositionalArg() || path.join(__dirname, '..', 'dbpedia');
const N_BASE = parsePositiveInt(takeFlagValue('--base-count', 50_000), '--base-count');
const N_QUERIES = parsePositiveInt(takeFlagValue('--queries', 1_000), '--queries');
const DELETE_FRACTION = parseUnitFraction(takeFlagValue('--delete-fraction', 0.2), '--delete-fraction');
const DELETE_SEED = parsePositiveInt(takeFlagValue('--seed', 42), '--seed');
const REGENERATE_GT = hasFlag('--regenerate-gt');

const K = 10;
const M = resolveSingleValue(parsedBenchArgs.m, 16);
const EF_CONSTRUCTION = resolveSingleValue(parsedBenchArgs.efConstruction, 50);
const EF_SEARCH = resolveSingleValue(parsedBenchArgs.efSearch, 100);

const RESULTS_DIR = path.join(__dirname, '..', 'benchmark_results');
const CACHE_DIR = path.join(RESULTS_DIR, 'cache');
if (!fs.existsSync(RESULTS_DIR)) fs.mkdirSync(RESULTS_DIR);
if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR);

const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const LOG_PATH = path.join(RESULTS_DIR, `dbpedia_compact_recall_${timestamp}.log`);
const JSON_PATH = path.join(RESULTS_DIR, `dbpedia_compact_recall_${timestamp}.json`);
const logStream = fs.createWriteStream(LOG_PATH);

const BASE_GT_CACHE_PATH = path.join(CACHE_DIR, `gt_dbpedia_l2_n${N_BASE}_q${N_QUERIES}_k${K}.bin`);
const POST_GT_CACHE_PATH = path.join(
  CACHE_DIR,
  `gt_dbpedia_l2_n${N_BASE}_q${N_QUERIES}_k${K}_delete${Math.round(DELETE_FRACTION * 1000)}_seed${DELETE_SEED}.bin`
);

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

function saveGroundTruth(filePath, gt) {
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
  fs.writeFileSync(filePath, buf);
  log(`  Cached ground truth to ${filePath}`);
}

function loadGroundTruth(filePath) {
  if (!fs.existsSync(filePath)) return null;
  const buf = fs.readFileSync(filePath);
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

function computeGroundTruth(train, queries, dim, liveIds = null) {
  const nq = queries.length;
  const ids = liveIds || Array.from({ length: train.length }, (_, i) => i);
  log(`Computing brute-force L2 ground truth (${nq} × ${ids.length} × ${dim}D)...`);
  const t0 = performance.now();
  const gt = new Array(nq);

  for (let q = 0; q < nq; q++) {
    const query = queries[q];
    const scored = new Array(ids.length);
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      const vec = train[id];
      let sum = 0;
      for (let d = 0; d < dim; d++) {
        const diff = query[d] - vec[d];
        sum += diff * diff;
      }
      scored[i] = { id, dist: sum };
    }
    scored.sort((a, b) => a.dist - b.dist);
    gt[q] = scored.slice(0, K).map(entry => entry.id);

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

function recall(predicted, truth) {
  const truthSet = new Set(truth);
  let hits = 0;
  for (const id of predicted) {
    if (truthSet.has(id)) hits++;
  }
  return hits / truth.length;
}

function percentile(sorted, p) {
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
}

function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function chooseDeletedIds(count, fraction, seed) {
  const target = Math.max(1, Math.floor(count * fraction));
  const ids = Array.from({ length: count }, (_, i) => i);
  const rng = mulberry32(seed);
  for (let i = ids.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = ids[i];
    ids[i] = ids[j];
    ids[j] = tmp;
  }
  const deleted = ids.slice(0, target);
  deleted.sort((a, b) => a - b);
  return deleted;
}

async function buildIndex(train, dim) {
  const index = await Pancake.create({
    dim,
    maxElements: train.length,
    quantized: true,
    metric: 'l2',
    M,
    efConstruction: EF_CONSTRUCTION,
    efSearch: EF_SEARCH,
  });

  log(`Building Pancake u8 index (M=${M}, ef_c=${EF_CONSTRUCTION}, ef_s=${EF_SEARCH}, metric=l2)...`);
  const t0 = performance.now();
  const batchSize = 500;
  for (let start = 0; start < train.length; start += batchSize) {
    const end = Math.min(start + batchSize, train.length);
    index.addBatch(train.slice(start, end));
    if (end % 10000 < batchSize) {
      log(`  ${end.toLocaleString()}/${train.length.toLocaleString()}`);
    }
  }
  const buildMs = performance.now() - t0;
  log(`  Build complete in ${(buildMs / 1000).toFixed(1)}s, memory=${(index.memory / 1024 / 1024).toFixed(1)} MB`);
  return { index, buildMs };
}

function measureRecall(index, queries, groundTruth) {
  index.setEfSearch(EF_SEARCH);
  const latencies = new Array(queries.length);
  let totalRecall = 0;
  for (let i = 0; i < queries.length; i++) {
    const t0 = performance.now();
    const results = index.search(queries[i], K);
    latencies[i] = performance.now() - t0;
    totalRecall += recall(results.map(r => r.id), groundTruth[i]);
  }
  const sorted = [...latencies].sort((a, b) => a - b);
  return {
    recall: totalRecall / queries.length,
    qps: 1000 / mean(latencies),
    p50: percentile(sorted, 0.5),
    p99: percentile(sorted, 0.99),
  };
}

async function main() {
  const basePath = path.join(DBPEDIA_DIR, 'dbpedia_base_100k.fvecs');
  const queryPath = path.join(DBPEDIA_DIR, 'dbpedia_query.fvecs');
  for (const filePath of [basePath, queryPath]) {
    if (!fs.existsSync(filePath)) {
      log(`Missing file: ${filePath}`);
      process.exit(1);
    }
  }

  log('='.repeat(72));
  log('DBpedia uint8 compaction recall benchmark');
  log('='.repeat(72));
  log(`Dataset: ${DBPEDIA_DIR}`);
  log(`Base vectors: ${N_BASE.toLocaleString()}  Queries: ${N_QUERIES.toLocaleString()}  k=${K}`);
  log(`Params: M=${M}  efConstruction=${EF_CONSTRUCTION}  efSearch=${EF_SEARCH}`);
  log(`Delete fraction: ${(DELETE_FRACTION * 100).toFixed(1)}%  seed=${DELETE_SEED}`);

  log('\nLoading dataset...');
  const { vectors: train, dim } = readFvecs(basePath, N_BASE);
  const { vectors: queries } = readFvecs(queryPath, N_QUERIES);

  let baseGroundTruth = null;
  if (!REGENERATE_GT) baseGroundTruth = loadGroundTruth(BASE_GT_CACHE_PATH);
  if (baseGroundTruth) {
    log(`Loaded cached baseline ground truth from ${BASE_GT_CACHE_PATH}`);
  } else {
    baseGroundTruth = computeGroundTruth(train, queries, dim);
    saveGroundTruth(BASE_GT_CACHE_PATH, baseGroundTruth);
  }

  const { index, buildMs } = await buildIndex(train, dim);

  log('\nMeasuring pre-compact recall...');
  const before = measureRecall(index, queries, baseGroundTruth);
  log(`  recall@${K}=${(before.recall * 100).toFixed(2)}%  qps=${before.qps.toFixed(0)}  p50=${before.p50.toFixed(3)}ms  p99=${before.p99.toFixed(3)}ms`);

  const deletedIds = chooseDeletedIds(train.length, DELETE_FRACTION, DELETE_SEED);
  const deletedSet = new Set(deletedIds);
  const liveIds = [];
  for (let i = 0; i < train.length; i++) {
    if (!deletedSet.has(i)) liveIds.push(i);
  }

  log(`\nDeleting ${deletedIds.length.toLocaleString()} vectors and compacting...`);
  for (const id of deletedIds) index.delete(id);
  const compactStart = performance.now();
  index.compact();
  const compactMs = performance.now() - compactStart;
  log(`  compact() finished in ${(compactMs / 1000).toFixed(2)}s; live count=${index.count}`);

  let postGroundTruth = null;
  if (!REGENERATE_GT) postGroundTruth = loadGroundTruth(POST_GT_CACHE_PATH);
  if (postGroundTruth) {
    log(`Loaded cached post-delete ground truth from ${POST_GT_CACHE_PATH}`);
  } else {
    postGroundTruth = computeGroundTruth(train, queries, dim, liveIds);
    saveGroundTruth(POST_GT_CACHE_PATH, postGroundTruth);
  }

  log('\nMeasuring post-compact recall...');
  const after = measureRecall(index, queries, postGroundTruth);
  log(`  recall@${K}=${(after.recall * 100).toFixed(2)}%  qps=${after.qps.toFixed(0)}  p50=${after.p50.toFixed(3)}ms  p99=${after.p99.toFixed(3)}ms`);

  const delta = after.recall - before.recall;
  log(`\nRecall delta after delete+compact: ${(delta * 100).toFixed(2)} points`);

  const results = {
    benchmark: 'dbpedia-compact-recall-l2-uint8',
    timestamp: new Date().toISOString(),
    dataset: {
      source: DBPEDIA_DIR,
      vectors: train.length,
      queries: queries.length,
      dim,
      metric: 'l2',
    },
    params: {
      K,
      M,
      efConstruction: EF_CONSTRUCTION,
      efSearch: EF_SEARCH,
      deleteFraction: DELETE_FRACTION,
      deleteSeed: DELETE_SEED,
    },
    buildMs,
    compactMs,
    before,
    after,
    recallDelta: delta,
    deletedCount: deletedIds.length,
    liveCount: liveIds.length,
    groundTruthCache: {
      baseline: BASE_GT_CACHE_PATH,
      postDelete: POST_GT_CACHE_PATH,
    },
  };

  fs.writeFileSync(JSON_PATH, JSON.stringify(results, null, 2) + '\n');
  log(`\nOutputs:`);
  log(`  log:  ${LOG_PATH}`);
  log(`  json: ${JSON_PATH}`);
  index.dispose();
  logStream.end();
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
