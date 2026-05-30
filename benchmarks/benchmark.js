#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const loadEngine = require('./dist/engine.js');
const { parseBenchmarkArgs, resolveSingleValue } = require('./bench_args');

const parsedArgs = parseBenchmarkArgs();

const RESULTS_DIR = path.join(__dirname, 'benchmark_results');
if (!fs.existsSync(RESULTS_DIR)) fs.mkdirSync(RESULTS_DIR);
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const LOG_PATH = path.join(RESULTS_DIR, `synthetic_${timestamp}.log`);
const JSON_PATH = path.join(RESULTS_DIR, `synthetic_${timestamp}.json`);
const logStream = fs.createWriteStream(LOG_PATH);
const PROFILE_BUILD = parsedArgs.args.includes('--profile-build');

const DIM = 384;
const N = 10000;
const N_QUERIES = 500;
const K = 10;

// Compile-time 384D path uses hardcoded HNSW params in the WASM engine:
//   M=12, efConstruction=150, efSearch=250
const HNSW_M = resolveSingleValue(parsedArgs.m, 16);
const HNSW_EF_CONSTRUCTION = resolveSingleValue(parsedArgs.efConstruction, 100);
const HNSW_EF_SEARCH = resolveSingleValue(parsedArgs.efSearch, 100);

function log(msg = '') {
  console.log(msg);
  logStream.write(msg + '\n');
}

function getEnvironmentInfo() {
  const wasmPath = path.join(__dirname, 'dist', 'engine.wasm');
  const wasmBinary = fs.readFileSync(wasmPath);
  return {
    timestamp: new Date().toISOString(),
    cwd: process.cwd(),
    node: process.version,
    platform: process.platform,
    arch: process.arch,
    pid: process.pid,
    wasmPath,
    wasmBytes: wasmBinary.byteLength,
    wasmSha256: crypto.createHash('sha256').update(wasmBinary).digest('hex')
  };
}

function normalize(v) {
  let norm = 0;
  for (let d = 0; d < v.length; d++) norm += v[d] * v[d];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let d = 0; d < v.length; d++) v[d] /= norm;
  return v;
}

function generateVectors(count, dim) {
  const nClusters = 50;
  const centroids = new Array(nClusters);
  for (let c = 0; c < nClusters; c++) {
    const v = new Float32Array(dim);
    for (let d = 0; d < dim; d++) v[d] = Math.random() * 2 - 1;
    centroids[c] = normalize(v);
  }

  const vectors = new Array(count);
  for (let i = 0; i < count; i++) {
    const c = centroids[i % nClusters];
    const v = new Float32Array(dim);
    for (let d = 0; d < dim; d++) v[d] = c[d] + (Math.random() - 0.5) * 0.3;
    vectors[i] = normalize(v);
  }
  return vectors;
}

function bruteForceKNN(vectors, query, k) {
  const dists = vectors.map((v, i) => {
    let dot = 0;
    for (let d = 0; d < v.length; d++) dot += v[d] * query[d];
    return { id: i, distance: 1 - dot };
  });
  dists.sort((a, b) => a.distance - b.distance);
  return dists.slice(0, k).map(r => r.id);
}

function recall(predicted, truth) {
  const truthSet = new Set(truth);
  let hits = 0;
  for (const id of predicted) if (truthSet.has(id)) hits++;
  return hits / truth.length;
}

async function loadWasmEngine() {
  const wasmBinary = fs.readFileSync(path.join(__dirname, 'dist', 'engine.wasm'));
  return loadEngine({
    wasmBinary: wasmBinary.buffer.slice(
      wasmBinary.byteOffset,
      wasmBinary.byteOffset + wasmBinary.byteLength
    )
  });
}

// Benchmark using the compile-time QuantizedHNSW<384> path (pi/pa/pq/pc/pm)
async function benchPancake384(engine, vectors, queries, k, groundTruth) {
  const vecPtr = engine._emsc_malloc(DIM * 4);
  const idPtr = engine._emsc_malloc(k * 8);
  const distPtr = engine._emsc_malloc(k * 4);
  if (!vecPtr || !idPtr || !distPtr) throw new Error('WASM malloc failed');

  try {
    // Init compile-time 384D index
    engine._pi(vectors.length);

    // Build
    const t0 = performance.now();
    for (const v of vectors) {
      engine.HEAPF32.set(v, vecPtr >> 2);
      engine._pa(vecPtr);
    }
    const buildMs = performance.now() - t0;

    // Search
    const latencies = [];
    let totalRecall = 0;
    const dv = new DataView(engine.HEAPU8.buffer);
    for (let i = 0; i < queries.length; i++) {
      engine.HEAPF32.set(queries[i], vecPtr >> 2);
      const st = performance.now();
      const found = engine._pq(vecPtr, k, idPtr, distPtr);
      latencies.push(performance.now() - st);

      const ids = [];
      for (let j = 0; j < found; j++) {
        const lo = dv.getUint32(idPtr + j * 8, true);
        const hi = dv.getUint32(idPtr + j * 8 + 4, true);
        ids.push(hi * 0x100000000 + lo);
      }
      totalRecall += recall(ids, groundTruth[i]);
    }

    latencies.sort((a, b) => a - b);
    const memBytes = engine._pm();

    return {
      name: 'Pancake Int8 (WASM, QuantizedHNSW<384>)',
      buildMs,
      memBytes,
      recall: totalRecall / queries.length,
      p50: latencies[Math.floor(latencies.length * 0.5)],
      p99: latencies[Math.floor(latencies.length * 0.99)],
      qps: 1000 / (latencies.reduce((a, b) => a + b) / latencies.length)
    };
  } finally {
    engine._emsc_free(vecPtr);
    engine._emsc_free(idPtr);
    engine._emsc_free(distPtr);
  }
}

async function profilePancakeBuild(engine, vectors) {
  const flatStart = performance.now();
  const flat = new Float32Array(vectors.length * DIM);
  for (let i = 0; i < vectors.length; i++) flat.set(vectors[i], i * DIM);
  const flattenMs = performance.now() - flatStart;

  const allocStart = performance.now();
  const dataPtr = engine._emsc_malloc(flat.length * 4);
  const allocMs = performance.now() - allocStart;
  if (!dataPtr) throw new Error('WASM malloc failed for profile-build');

  try {
    const initIndexStart = performance.now();
    engine._pi(vectors.length);
    const indexInitMs = performance.now() - initIndexStart;

    const copyStart = performance.now();
    engine.HEAPF32.set(flat, dataPtr >> 2);
    const heapCopyMs = performance.now() - copyStart;

    const insertStart = performance.now();
    const hasBulk = typeof engine._bulk_insert === 'function';
    let inserted = 0;
    if (hasBulk) {
      inserted = engine._bulk_insert(dataPtr, vectors.length);
    } else {
      for (let i = 0; i < vectors.length; i++) {
        const id = engine._pa(dataPtr + i * DIM * 4);
        if (id !== 0xFFFFFFFF && id >= 0) inserted++;
      }
    }
    const wasmInsertMs = performance.now() - insertStart;

    return {
      flattenMs,
      allocMs,
      indexInitMs,
      heapCopyMs,
      wasmInsertMs,
      totalBuildMs: flattenMs + allocMs + indexInitMs + heapCopyMs + wasmInsertMs,
      insertMode: hasBulk ? 'bulk' : 'single',
      inserted,
      expected: vectors.length,
      memBytes: engine._pm()
    };
  } finally {
    engine._emsc_free(dataPtr);
  }
}

async function benchUSearch(vectors, queries, k, groundTruth) {
  let usearch;
  try {
    usearch = require('usearch');
  } catch (e) {
    log('  usearch not available, skipping');
    return null;
  }

  const index = new usearch.Index({
    metric: 'cos',
    connectivity: HNSW_M,
    dimensions: DIM,
    dtype: 'f32',
    expansion_add: HNSW_EF_CONSTRUCTION,
    expansion_search: HNSW_EF_SEARCH
  });

  // Build
  const t0 = performance.now();
  for (let i = 0; i < vectors.length; i++) {
    index.add(BigInt(i), vectors[i]);
  }
  const buildMs = performance.now() - t0;

  // Search
  const latencies = [];
  let totalRecall = 0;
  for (let i = 0; i < queries.length; i++) {
    const st = performance.now();
    const results = index.search(queries[i], k);
    latencies.push(performance.now() - st);
    const ids = Array.from(results.keys).map(Number);
    totalRecall += recall(ids, groundTruth[i]);
  }

  latencies.sort((a, b) => a - b);

  return {
    name: 'USearch Float32 (Native)',
    buildMs,
    memBytes: null,
    recall: totalRecall / queries.length,
    p50: latencies[Math.floor(latencies.length * 0.5)],
    p99: latencies[Math.floor(latencies.length * 0.99)],
    qps: 1000 / (latencies.reduce((a, b) => a + b) / latencies.length)
  };
}

function printResult(r) {
  if (!r) return;
  log(`  ${r.name}`);
  log(`    Build:    ${r.buildMs.toFixed(0)} ms`);
  if (r.memBytes) log(`    Memory:   ${(r.memBytes / 1024 / 1024).toFixed(1)} MB`);
  log(`    Recall@${K}: ${(r.recall * 100).toFixed(1)}%`);
  log(`    QPS:      ${r.qps.toFixed(0)}`);
  log(`    p50:      ${r.p50.toFixed(3)} ms`);
  log(`    p99:      ${r.p99.toFixed(3)} ms`);
  log();
}

async function main() {
  const environment = getEnvironmentInfo();

  log(`\nBenchmark: ${N} vectors, ${DIM}D (compile-time path), ${N_QUERIES} queries, k=${K}`);
  log(`HNSW params: M=${HNSW_M}, efConstruction=${HNSW_EF_CONSTRUCTION}, efSearch=${HNSW_EF_SEARCH}\n`);
  log('Environment:');
  log(`  timestamp:   ${environment.timestamp}`);
  log(`  cwd:         ${environment.cwd}`);
  log(`  node:        ${environment.node}`);
  log(`  platform:    ${environment.platform}/${environment.arch}`);
  log(`  wasm bytes:  ${environment.wasmBytes}`);
  log(`  wasm sha256: ${environment.wasmSha256}`);
  log('');

  log('Generating vectors...');
  const vectors = generateVectors(N, DIM);
  const queries = generateVectors(N_QUERIES, DIM);

  log('Computing ground truth (brute force)...');
  const groundTruth = queries.map(q => bruteForceKNN(vectors, q, K));

  log('');

  const engine = await loadWasmEngine();

  let buildProfile = null;
  if (PROFILE_BUILD) {
    log('Build profile:');
    buildProfile = await profilePancakeBuild(engine, vectors);
    log(`  flatten:       ${buildProfile.flattenMs.toFixed(3)} ms`);
    log(`  wasm malloc:   ${buildProfile.allocMs.toFixed(3)} ms`);
    log(`  index init:    ${buildProfile.indexInitMs.toFixed(3)} ms`);
    log(`  heap copy:     ${buildProfile.heapCopyMs.toFixed(3)} ms`);
    log(`  wasm insert:   ${buildProfile.wasmInsertMs.toFixed(3)} ms (${buildProfile.insertMode})`);
    log(`  total build:   ${buildProfile.totalBuildMs.toFixed(3)} ms`);
    log(`  inserted:      ${buildProfile.inserted}/${buildProfile.expected}`);
    log(`  index memory:  ${(buildProfile.memBytes / 1024 / 1024).toFixed(1)} MB`);
    log('');
  }

  const pancakeResult = await benchPancake384(engine, vectors, queries, K, groundTruth);
  printResult(pancakeResult);

  if (typeof engine._shutdown_all === 'function') engine._shutdown_all();

  const usearchResult = await benchUSearch(vectors, queries, K, groundTruth);
  printResult(usearchResult);

  if (pancakeResult && usearchResult) {
    log('  Comparison:');
    if (usearchResult.memBytes && pancakeResult.memBytes) {
      log(`    Memory:  ${((1 - pancakeResult.memBytes / usearchResult.memBytes) * 100).toFixed(0)}% reduction`);
    }
    log(`    QPS:     ${(pancakeResult.qps / usearchResult.qps).toFixed(2)}x`);
    log(`    Recall:  ${(pancakeResult.recall * 100).toFixed(1)}% vs ${(usearchResult.recall * 100).toFixed(1)}%`);
  }

  const results = {
    benchmark: 'synthetic',
    timestamp: environment.timestamp,
    environment,
    params: { N, DIM, N_QUERIES, K, M: HNSW_M, efConstruction: HNSW_EF_CONSTRUCTION, efSearch: HNSW_EF_SEARCH },
    backend: 'QuantizedHNSW<384> (compile-time)',
    profileBuild: PROFILE_BUILD,
    buildProfile: buildProfile,
    pancake: pancakeResult || null,
    usearch: usearchResult || null
  };
  if (pancakeResult && usearchResult) {
    results.comparison = {
      memoryReduction: usearchResult.memBytes && pancakeResult.memBytes
        ? `${((1 - pancakeResult.memBytes / usearchResult.memBytes) * 100).toFixed(1)}%` : null,
      qpsRatio: +(pancakeResult.qps / usearchResult.qps).toFixed(2),
      recallDelta: +((pancakeResult.recall - usearchResult.recall) * 100).toFixed(1)
    };
  }
  fs.writeFileSync(JSON_PATH, JSON.stringify(results, null, 2) + '\n');
  log(`\nResults saved to:\n  ${LOG_PATH}\n  ${JSON_PATH}`);
  logStream.end();
}

main().catch(console.error);
