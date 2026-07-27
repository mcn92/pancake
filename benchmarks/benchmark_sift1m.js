#!/usr/bin/env node
'use strict';

/**
 * SIFT-1M Benchmark: Pancake u8 (WASM)
 *
 * Uses the standard SIFT-1M dataset (1M vectors, 128D, L2).
 *
 * Expected files:
 *   sift_base.fvecs        - 1M base vectors (float32, 128D)
 *   sift_query.fvecs       - 10K query vectors
 *   sift_groundtruth.ivecs - ground truth neighbor IDs
 *
 * Usage:
 *   node benchmarks/benchmark_sift1m.js [sift-dir]
 */

const fs = require('fs');
const path = require('path');
const Pancake = require('../pancake.js');
const { parseBenchmarkArgs, resolveSingleValue } = require('./bench_args');

const parsedArgs = parseBenchmarkArgs();
const SIFT_DIR = parsedArgs.args[0] || path.join(__dirname, '..', 'sift');

const RESULTS_DIR = path.join(__dirname, '..', 'benchmark_results');
if (!fs.existsSync(RESULTS_DIR)) fs.mkdirSync(RESULTS_DIR);
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const LOG_PATH = path.join(RESULTS_DIR, `sift1m_${timestamp}.log`);
const JSON_PATH = path.join(RESULTS_DIR, `sift1m_${timestamp}.json`);
const logStream = fs.createWriteStream(LOG_PATH);

function log(msg = '') {
  console.log(msg);
  logStream.write(msg + '\n');
}

const K = 10;
const M = resolveSingleValue(parsedArgs.m, 16);
const EF_CONSTRUCTION = resolveSingleValue(parsedArgs.efConstruction, 200);
const EF_SEARCH = resolveSingleValue(parsedArgs.efSearch, 100);

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
  const { vectors: train, dim } = readFvecs(path.join(dir, 'sift_base.fvecs'));
  const { vectors: test } = readFvecs(path.join(dir, 'sift_query.fvecs'));
  const groundTruth = readIvecs(path.join(dir, 'sift_groundtruth.ivecs'));

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
  log('\n  Pancake u8 (WASM)');
  log('  Building index...');

  const index = await Pancake.create({
    dim, maxElements: train.length, quantized: true,
    metric: 'l2',
    M, efConstruction: EF_CONSTRUCTION, efSearch: EF_SEARCH
  });

  const t0 = performance.now();
  const batchSize = 1000;
  for (let i = 0; i < train.length; i += batchSize) {
    const batch = train.slice(i, Math.min(i + batchSize, train.length));
    index.addBatch(batch);
    if (Math.min(i + batchSize, train.length) % 100000 < batchSize) {
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
    name: 'Pancake u8 (WASM)',
    buildSec: buildMs / 1000,
    memMB: memBytes / 1024 / 1024,
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
  const requiredFiles = ['sift_base.fvecs', 'sift_query.fvecs', 'sift_groundtruth.ivecs'];
  for (const f of requiredFiles) {
    if (!fs.existsSync(path.join(SIFT_DIR, f))) {
      log(`Missing file: ${path.join(SIFT_DIR, f)}`);
      process.exit(1);
    }
  }

  const { train, test, groundTruth, dim, info } = loadDataset(SIFT_DIR);

  log(`\n${'='.repeat(60)}`);
  log(`SIFT-1M Benchmark`);
  log(`${info.n_train} vectors, ${dim}D, ${info.n_test} queries, k=${K}`);
  log(`M=${M}, efConstruction=${EF_CONSTRUCTION}, efSearch=${EF_SEARCH}`);
  log('='.repeat(60));

  const pancake = await benchPancake(train, test, groundTruth, dim);

  log(`\n${'='.repeat(60)}`);
  log('Results');
  log('='.repeat(60));

  printResult(pancake);

  const results = {
    benchmark: 'sift-1m',
    timestamp: new Date().toISOString(),
    dataset: { vectors: info.n_train, queries: info.n_test, dim, source: SIFT_DIR },
    params: { K, M, efConstruction: EF_CONSTRUCTION, efSearch: EF_SEARCH },
    pancake: pancake || null
  };
  fs.writeFileSync(JSON_PATH, JSON.stringify(results, null, 2) + '\n');
  log(`\nResults saved to:\n  ${LOG_PATH}\n  ${JSON_PATH}`);
  logStream.end();
}

main().catch(console.error);
