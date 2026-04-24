#!/usr/bin/env node
'use strict';

/**
 * NYTimes-256 Benchmark: Pancake Int8 (WASM) vs USearch Float32 (Native)
 *
 * Uses the NYTimes dataset in fvecs/ivecs format (290K vectors, 256D, angular).
 *
 * Expected files in nytimes/ directory:
 *   nytimes_base.fvecs        - base vectors (float32)
 *   nytimes_query.fvecs       - query vectors (float32)
 *   nytimes_groundtruth.ivecs - ground truth neighbor IDs (int32)
 *
 * Usage:
 *   node benchmark_nytimes.js [nytimes-dir]
 */

const fs = require('fs');
const path = require('path');
const Pancake = require('./pancake.js');

const NYTIMES_DIR = process.argv[2] || path.join(__dirname, 'nytimes');

const RESULTS_DIR = path.join(__dirname, 'benchmark_results');
if (!fs.existsSync(RESULTS_DIR)) fs.mkdirSync(RESULTS_DIR);
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const LOG_PATH = path.join(RESULTS_DIR, `nytimes_${timestamp}.log`);
const JSON_PATH = path.join(RESULTS_DIR, `nytimes_${timestamp}.json`);
const logStream = fs.createWriteStream(LOG_PATH);

function log(msg = '') {
  console.log(msg);
  logStream.write(msg + '\n');
}
const K = 10;
const M = 12;
const EF_CONSTRUCTION = 100;
const EF_SEARCH = 100;

/**
 * Read an fvecs file: each record is [dim (int32), dim x float32].
 * Returns { vectors: Float32Array[], dim: number }
 */
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

/**
 * Read an ivecs file: each record is [dim (int32), dim x int32].
 * Returns arrays of integer arrays (neighbor IDs), truncated to K.
 */
function readIvecs(filePath) {
  log(`  Loading ${filePath}...`);
  const buf = fs.readFileSync(filePath);
  const rows = [];
  let offset = 0;
  while (offset < buf.length) {
    const dim = buf.readInt32LE(offset); offset += 4;
    const row = new Array(Math.min(dim, K));
    for (let d = 0; d < dim; d++) {
      const val = buf.readInt32LE(offset); offset += 4;
      if (d < K) row[d] = val;
    }
    rows.push(row);
  }
  return rows;
}

function loadDataset(dir) {
  log(`Loading dataset from ${dir}...`);
  const basePath = path.join(dir, 'nytimes_base.fvecs');
  const queryPath = path.join(dir, 'nytimes_query.fvecs');
  const gtPath = path.join(dir, 'nytimes_groundtruth.ivecs');

  const { vectors: train, dim } = readFvecs(basePath);
  const { vectors: test } = readFvecs(queryPath);
  const groundTruth = readIvecs(gtPath);

  const info = { n_train: train.length, n_test: test.length, dim };
  log(`  Train: ${info.n_train} vectors, ${dim}D`);
  log(`  Test:  ${info.n_test} queries`);

  return { train, test, groundTruth, dim, info };
}

function recall(predicted, truth) {
  const truthSet = new Set(truth);
  let hits = 0;
  for (const id of predicted) if (truthSet.has(id)) hits++;
  return hits / truth.length;
}

function percentile(sorted, p) {
  return sorted[Math.floor(sorted.length * p)];
}

async function benchPancake(train, test, groundTruth, dim) {
  log('\n  Pancake Int8 (WASM)');
  log('  Building index...');

  const index = await Pancake.create({
    dim, maxElements: train.length, quantized: true,
    M, efConstruction: EF_CONSTRUCTION, efSearch: EF_SEARCH
  });

  const t0 = performance.now();
  const batchSize = 1000;
  for (let i = 0; i < train.length; i += batchSize) {
    const batch = train.slice(i, Math.min(i + batchSize, train.length));
    index.addBatch(batch);
    if (Math.min(i + batchSize, train.length) % 50000 < batchSize) {
      log(`    ${Math.min(i + batchSize, train.length)}/${train.length}`);
    }
  }
  const buildMs = performance.now() - t0;
  log(`    Build: ${(buildMs / 1000).toFixed(1)}s`);

  // Warmup
  for (let i = 0; i < 10; i++) index.search(test[i], K);

  log('  Searching...');
  const latencies = [];
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const st = performance.now();
    const results = index.search(test[i], K);
    latencies.push(performance.now() - st);
    totalRecall += recall(results.map(r => r.id), groundTruth[i]);
  }

  latencies.sort((a, b) => a - b);
  const memBytes = index.memory;
  index.dispose();

  return {
    name: 'Pancake Int8 (WASM)',
    buildSec: buildMs / 1000,
    memMB: memBytes / 1024 / 1024,
    recall: totalRecall / test.length,
    qps: 1000 / (latencies.reduce((a, b) => a + b) / latencies.length),
    p50: percentile(latencies, 0.5),
    p99: percentile(latencies, 0.99)
  };
}

async function benchUSearch(train, test, groundTruth, dim) {
  let usearch;
  try {
    usearch = require('usearch');
  } catch (e) {
    log('\n  USearch not installed, skipping');
    return null;
  }

  log('\n  USearch Float32 (Native)');
  log('  Building index...');

  const index = new usearch.Index({
    metric: 'cos',
    connectivity: M,
    dimensions: dim,
    dtype: 'f32',
    expansion_add: EF_CONSTRUCTION,
    expansion_search: EF_SEARCH
  });

  const t0 = performance.now();
  for (let i = 0; i < train.length; i++) {
    index.add(BigInt(i), train[i]);
    if ((i + 1) % 50000 === 0) log(`    ${i + 1}/${train.length}`);
  }
  const buildMs = performance.now() - t0;
  log(`    Build: ${(buildMs / 1000).toFixed(1)}s`);

  // Warmup
  for (let i = 0; i < 10; i++) index.search(test[i], K);

  log('  Searching...');
  const latencies = [];
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const st = performance.now();
    const results = index.search(test[i], K);
    latencies.push(performance.now() - st);
    const ids = Array.from(results.keys).map(Number);
    totalRecall += recall(ids, groundTruth[i]);
  }

  latencies.sort((a, b) => a - b);

  return {
    name: 'USearch Float32 (Native)',
    buildSec: buildMs / 1000,
    memMB: null,
    recall: totalRecall / test.length,
    qps: 1000 / (latencies.reduce((a, b) => a + b) / latencies.length),
    p50: percentile(latencies, 0.5),
    p99: percentile(latencies, 0.99)
  };
}

function printResult(r) {
  if (!r) return;
  log(`\n  ${r.name}:`);
  log(`    Build time:  ${r.buildSec.toFixed(1)}s`);
  if (r.memMB) log(`    Index size:  ${r.memMB.toFixed(0)} MB`);
  log(`    Recall@${K}:  ${(r.recall * 100).toFixed(1)}%`);
  log(`    QPS:         ${r.qps.toFixed(0)}`);
  log(`    p50 latency: ${r.p50.toFixed(3)} ms`);
  log(`    p99 latency: ${r.p99.toFixed(3)} ms`);
}

async function main() {
  const requiredFiles = ['nytimes_base.fvecs', 'nytimes_query.fvecs', 'nytimes_groundtruth.ivecs'];
  for (const f of requiredFiles) {
    if (!fs.existsSync(path.join(NYTIMES_DIR, f))) {
      log(`Missing file: ${path.join(NYTIMES_DIR, f)}`);
      process.exit(1);
    }
  }

  const { train, test, groundTruth, dim, info } = loadDataset(NYTIMES_DIR);

  log(`\n${'='.repeat(60)}`);
  log(`NYTimes-256 Benchmark`);
  log(`${info.n_train} vectors, ${dim}D, ${info.n_test} queries, k=${K}`);
  log(`M=${M}, efConstruction=${EF_CONSTRUCTION}, efSearch=${EF_SEARCH}`);
  log('='.repeat(60));

  const pancake = await benchPancake(train, test, groundTruth, dim);
  const usearchResult = await benchUSearch(train, test, groundTruth, dim);

  log(`\n${'='.repeat(60)}`);
  log('Results');
  log('='.repeat(60));

  printResult(pancake);
  printResult(usearchResult);

  if (pancake && usearchResult) {
    log(`\n  Head-to-head:`);
    log(`    QPS:     Pancake ${(pancake.qps / usearchResult.qps).toFixed(2)}x vs USearch`);
    log(`    Recall:  ${(pancake.recall * 100).toFixed(1)}% vs ${(usearchResult.recall * 100).toFixed(1)}%`);
    log(`    Latency: ${pancake.p50.toFixed(3)} vs ${usearchResult.p50.toFixed(3)} ms (p50)`);
  }

  const results = {
    benchmark: 'nytimes-256',
    timestamp: new Date().toISOString(),
    dataset: { vectors: info.n_train, queries: info.n_test, dim, source: NYTIMES_DIR },
    params: { K, M, efConstruction: EF_CONSTRUCTION, efSearch: EF_SEARCH },
    pancake: pancake || null,
    usearch: usearchResult || null
  };
  if (pancake && usearchResult) {
    results.comparison = {
      qpsRatio: +(pancake.qps / usearchResult.qps).toFixed(2),
      recallDelta: +((pancake.recall - usearchResult.recall) * 100).toFixed(1),
      p50Ratio: +(pancake.p50 / usearchResult.p50).toFixed(2)
    };
  }
  fs.writeFileSync(JSON_PATH, JSON.stringify(results, null, 2) + '\n');
  log(`\nResults saved to:\n  ${LOG_PATH}\n  ${JSON_PATH}`);
  logStream.end();
}

main().catch(console.error);
