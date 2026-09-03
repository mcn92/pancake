#!/usr/bin/env node
'use strict';

/**
 * Distinct-vector churn benchmark
 *
 * Keeps a fixed live population while replacing it completely for several
 * generations. Five replacement rounds leave five populations soft-deleted
 * and one live: deletedRatio = 5/6 = 83.3%. Recall and result counts are
 * measured after every turnover and again after compact().
 *
 * Usage:
 *   node benchmarks/churn_scale.js
 *   node benchmarks/churn_scale.js --population 1000 --queries 10 --rounds 2
 */

const fs = require('fs');
const path = require('path');
const Pikelet = require('../pikelet.js');

const args = process.argv.slice(2);
function intArg(name, fallback) {
  const index = args.indexOf(`--${name}`);
  if (index < 0) return fallback;
  const value = Number.parseInt(args[index + 1], 10);
  if (!Number.isInteger(value) || value <= 0) throw new Error(`--${name} must be a positive integer`);
  return value;
}
function stringArg(name, fallback) {
  const index = args.indexOf(`--${name}`);
  return index < 0 ? fallback : args[index + 1];
}

const POPULATION = intArg('population', 100_000);
const DIMS = intArg('dim', 64);
const CLUSTERS = intArg('clusters', 32);
const ROUNDS = intArg('rounds', 5);
const QUERY_COUNT = intArg('queries', 25);
const K = intArg('k', 10);
const M = intArg('m', 8);
const EF_CONSTRUCTION = intArg('ef-construction', 100);
const EF_SEARCH = intArg('ef-search', 200);
const BATCH_SIZE = intArg('batch-size', 1000);
const OUTPUT = stringArg('output', path.join(
  __dirname,
  '..',
  'benchmark_results',
  `churn_scale_${new Date().toISOString().replace(/[:.]/g, '-')}.json`
));

function mulberry32(seed) {
  return function random() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(random) {
  const u = Math.max(Number.MIN_VALUE, random());
  const v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function normalize(flat, offset) {
  let normSq = 0;
  for (let d = 0; d < DIMS; d++) normSq += flat[offset + d] * flat[offset + d];
  const inverse = 1 / Math.sqrt(normSq);
  for (let d = 0; d < DIMS; d++) flat[offset + d] *= inverse;
}

function makeCenters() {
  const random = mulberry32(0x50414e43);
  const centers = new Float32Array(CLUSTERS * DIMS);
  for (let cluster = 0; cluster < CLUSTERS; cluster++) {
    const offset = cluster * DIMS;
    for (let d = 0; d < DIMS; d++) centers[offset + d] = gaussian(random);
    normalize(centers, offset);
  }
  return centers;
}

const centers = makeCenters();

function makePopulation(round, count = POPULATION, spread = 0.08) {
  const random = mulberry32((0x9e3779b9 ^ Math.imul(round + 1, 0x85ebca6b)) >>> 0);
  const vectors = new Float32Array(count * DIMS);
  for (let row = 0; row < count; row++) {
    const cluster = row % CLUSTERS;
    const centerOffset = cluster * DIMS;
    const rowOffset = row * DIMS;
    for (let d = 0; d < DIMS; d++) {
      vectors[rowOffset + d] = centers[centerOffset + d] + gaussian(random) * spread;
    }
    normalize(vectors, rowOffset);
  }
  return vectors;
}

function rows(flat, start, end) {
  const result = new Array(end - start);
  for (let row = start; row < end; row++) {
    result[row - start] = flat.subarray(row * DIMS, (row + 1) * DIMS);
  }
  return result;
}

function exactTopK(vectors, ids, query) {
  const best = [];
  for (let row = 0; row < ids.length; row++) {
    const offset = row * DIMS;
    let dot = 0;
    for (let d = 0; d < DIMS; d++) dot += vectors[offset + d] * query[d];
    const candidate = { id: ids[row], distance: 1 - dot };
    if (best.length < K) {
      best.push(candidate);
      best.sort((a, b) => a.distance - b.distance);
    } else if (candidate.distance < best[K - 1].distance) {
      best[K - 1] = candidate;
      best.sort((a, b) => a.distance - b.distance);
    }
  }
  return new Set(best.map((entry) => entry.id));
}

function measure(index, vectors, ids, queries) {
  let recall = 0;
  let resultCount = 0;
  const latencies = [];
  for (const query of queries) {
    const truth = exactTopK(vectors, ids, query);
    const start = performance.now();
    const results = index.search(query, K, { efSearch: EF_SEARCH });
    latencies.push(performance.now() - start);
    resultCount += results.length;
    let hits = 0;
    for (const result of results) if (truth.has(result.id)) hits++;
    recall += hits / K;
  }
  latencies.sort((a, b) => a - b);
  return {
    recall: recall / queries.length,
    averageResultCount: resultCount / queries.length,
    p50Ms: latencies[Math.floor((latencies.length - 1) * 0.50)],
    p99Ms: latencies[Math.floor((latencies.length - 1) * 0.99)],
    count: index.count,
    liveCount: index.liveCount,
    deletedCount: index.deletedCount,
    deletedRatio: index.deletedRatio,
    logicalIndexBytes: index.memoryUsage.logicalIndexBytes,
    wasmHeapBytes: index.memoryUsage.wasmHeapBytes,
  };
}

async function insertPopulation(index, vectors) {
  const ids = [];
  for (let start = 0; start < POPULATION; start += BATCH_SIZE) {
    const end = Math.min(POPULATION, start + BATCH_SIZE);
    ids.push(...index.addBatch(rows(vectors, start, end)));
  }
  return ids;
}

async function replacePopulation(index, oldIds, vectors) {
  const nextIds = [];
  for (let start = 0; start < POPULATION; start += BATCH_SIZE) {
    const end = Math.min(POPULATION, start + BATCH_SIZE);
    for (let row = start; row < end; row++) index.delete(oldIds[row]);
    nextIds.push(...index.addBatch(rows(vectors, start, end)));
  }
  return nextIds;
}

async function main() {
  if (K > POPULATION) throw new Error('--k cannot exceed --population');
  if (CLUSTERS > POPULATION) throw new Error('--clusters cannot exceed --population');

  const capacity = POPULATION * (ROUNDS + 1);
  const index = await Pikelet.create({
    dim: DIMS,
    maxElements: capacity,
    metric: 'cosine',
    quantized: true,
    M,
    efConstruction: EF_CONSTRUCTION,
    efSearch: EF_SEARCH,
  });

  const queriesFlat = makePopulation(0x51554552, QUERY_COUNT, 0.04);
  const queries = rows(queriesFlat, 0, QUERY_COUNT);
  const measurements = [];

  try {
    let liveVectors = makePopulation(0);
    let liveIds = await insertPopulation(index, liveVectors);
    measurements.push({ stage: 'baseline', round: 0, ...measure(index, liveVectors, liveIds, queries) });

    for (let round = 1; round <= ROUNDS; round++) {
      liveVectors = makePopulation(round);
      liveIds = await replacePopulation(index, liveIds, liveVectors);
      const result = { stage: 'turnover', round, ...measure(index, liveVectors, liveIds, queries) };
      measurements.push(result);
      console.log(`round=${round} deleted=${(result.deletedRatio * 100).toFixed(1)}% recall=${(result.recall * 100).toFixed(2)}% results=${result.averageResultCount.toFixed(1)}`);
    }

    const compactStart = performance.now();
    index.compact();
    const compactMs = performance.now() - compactStart;
    measurements.push({
      stage: 'compacted',
      round: ROUNDS,
      compactMs,
      ...measure(index, liveVectors, liveIds, queries),
    });

    // A clean build of the final live population is the recovery ceiling for
    // this exact dataset. It keeps compaction quality separate from changes in
    // difficulty between generations.
    const fresh = await Pikelet.create({
      dim: DIMS,
      maxElements: POPULATION,
      metric: 'cosine',
      quantized: true,
      M,
      efConstruction: EF_CONSTRUCTION,
      efSearch: EF_SEARCH,
    });
    try {
      const freshStart = performance.now();
      const freshIds = await insertPopulation(fresh, liveVectors);
      const freshBuildMs = performance.now() - freshStart;
      measurements.push({
        stage: 'fresh-reference',
        round: ROUNDS,
        freshBuildMs,
        ...measure(fresh, liveVectors, freshIds, queries),
      });
    } finally {
      fresh.dispose();
    }

    const output = {
      timestamp: new Date().toISOString(),
      params: { POPULATION, DIMS, CLUSTERS, ROUNDS, QUERY_COUNT, K, M, EF_CONSTRUCTION, EF_SEARCH },
      measurements,
    };
    fs.mkdirSync(path.dirname(OUTPUT), { recursive: true });
    fs.writeFileSync(OUTPUT, `${JSON.stringify(output, null, 2)}\n`);
    console.log(`output=${OUTPUT}`);
  } finally {
    index.dispose();
  }
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
