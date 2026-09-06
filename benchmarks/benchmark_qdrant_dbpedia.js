#!/usr/bin/env node
'use strict';

/**
 * Qdrant DBpedia benchmark (REST service comparison).
 *
 * This is intentionally separate from the in-process Pareto harness: Qdrant is
 * a server and this path includes REST/JSON overhead. Use it as a deployment
 * shape comparison, not as an apples-to-apples engine microbenchmark.
 *
 * Example:
 *   docker run --rm -p 6333:6333 qdrant/qdrant
 *   node benchmarks/benchmark_qdrant_dbpedia.js --count 50000 --m 16 --ef-construction 50 --ef-search 100 --recreate
 *
 * Environment:
 *   QDRANT_URL      default http://127.0.0.1:6333
 *   QDRANT_API_KEY  optional API key
 */

const fs = require('fs');
const os = require('os');
const path = require('path');
const { parseBenchmarkArgs, resolveSingleValue } = require('./bench_args');

if (typeof fetch !== 'function') {
  console.error('This benchmark requires a Node runtime with global fetch support.');
  process.exit(1);
}

const parsedArgs = parseBenchmarkArgs();
const rawArgs = process.argv.slice(2);

function getIntArg(name, defaultVal) {
  const idx = rawArgs.indexOf('--' + name);
  return idx >= 0 && idx + 1 < rawArgs.length ? parseInt(rawArgs[idx + 1], 10) : defaultVal;
}

function getStrArg(name, defaultVal) {
  const idx = rawArgs.indexOf('--' + name);
  return idx >= 0 && idx + 1 < rawArgs.length ? rawArgs[idx + 1] : defaultVal;
}

function getBoolArg(name) {
  return rawArgs.includes('--' + name);
}

const DBPEDIA_DIR = path.resolve(getStrArg('data-dir', path.join(__dirname, '..', 'dbpedia')));
const QDRANT_URL = (getStrArg('url', process.env.QDRANT_URL || 'http://127.0.0.1:6333')).replace(/\/+$/, '');
const COLLECTION = getStrArg('collection', 'pikelet_dbpedia_l2');
const N_BASE = getIntArg('count', 50_000);
const N_QUERIES = getIntArg('queries', 1_000);
const K = getIntArg('k', 10);
const M = resolveSingleValue(parsedArgs.m, 16);
const EF_CONSTRUCTION = resolveSingleValue(parsedArgs.efConstruction, 50);
const EF_SEARCH = resolveSingleValue(parsedArgs.efSearch, 100);
const BATCH_SIZE = getIntArg('batch-size', 64);
const REPETITIONS = getIntArg('repetitions', 3);
const WARMUP_QUERIES = getIntArg('warmup', 200);
const RECREATE = getBoolArg('recreate');
const SKIP_UPSERT = getBoolArg('skip-upsert');

const RESULTS_DIR = path.join(__dirname, '..', 'benchmark_results');
const CACHE_DIR = path.join(RESULTS_DIR, 'cache');
fs.mkdirSync(RESULTS_DIR, { recursive: true });
fs.mkdirSync(CACHE_DIR, { recursive: true });

const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const LOG_PATH = path.join(RESULTS_DIR, `qdrant_dbpedia_${timestamp}.log`);
const JSON_PATH = path.join(RESULTS_DIR, `qdrant_dbpedia_${timestamp}.json`);
const CSV_PATH = path.join(RESULTS_DIR, `qdrant_dbpedia_${timestamp}.csv`);
const GT_CACHE_PATHS = [
  path.join(CACHE_DIR, `gt_dbpedia_l2_n${N_BASE}_q${N_QUERIES}_k${K}.bin`),
  path.join(CACHE_DIR, `gt_dbpedia_l2_n${N_BASE}_k${K}.bin`),
];
const logStream = fs.createWriteStream(LOG_PATH);

function log(msg = '') {
  console.log(msg);
  logStream.write(msg + '\n');
}

function formatMB(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function systemInfo() {
  const cpus = os.cpus() || [];
  return {
    platform: process.platform,
    arch: process.arch,
    node: process.version,
    cpu: cpus[0]?.model || 'unknown',
    logicalCpus: cpus.length,
    totalMemoryBytes: os.totalmem(),
  };
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

function saveGroundTruth(gt) {
  const target = GT_CACHE_PATHS[0];
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
  fs.writeFileSync(target, buf);
  log(`  Cached ground truth to ${target}`);
}

function loadGroundTruth() {
  for (const filePath of GT_CACHE_PATHS) {
    if (!fs.existsSync(filePath)) continue;
    const buf = fs.readFileSync(filePath);
    const nq = buf.readUInt32LE(0);
    const k = buf.readUInt32LE(4);
    if (k !== K || nq < N_QUERIES) continue;
    const gt = new Array(N_QUERIES);
    let offset = 8;
    for (let i = 0; i < nq; i++) {
      const row = new Array(k);
      for (let j = 0; j < k; j++) {
        row[j] = buf.readInt32LE(offset);
        offset += 4;
      }
      if (i < N_QUERIES) gt[i] = row;
    }
    log(`Loaded cached ground truth from ${filePath}`);
    return gt;
  }
  return null;
}

function computeGroundTruth(train, queries, dim) {
  log(`Computing brute-force L2 ground truth (${queries.length} x ${train.length} x ${dim}D)...`);
  const t0 = performance.now();
  const gt = new Array(queries.length);
  for (let q = 0; q < queries.length; q++) {
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
    const indices = Array.from({ length: train.length }, (_, i) => i);
    indices.sort((a, b) => dists[a] - dists[b]);
    gt[q] = indices.slice(0, K);
    if ((q + 1) % 25 === 0) {
      const elapsed = (performance.now() - t0) / 1000;
      const eta = (elapsed / (q + 1)) * (queries.length - q - 1);
      process.stdout.write(`  ${q + 1}/${queries.length} (${elapsed.toFixed(0)}s elapsed, ~${eta.toFixed(0)}s remaining)\r`);
    }
  }
  log(`  Ground truth computed in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  return gt;
}

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

async function qdrant(method, route, body) {
  const headers = { 'content-type': 'application/json' };
  if (process.env.QDRANT_API_KEY) headers['api-key'] = process.env.QDRANT_API_KEY;
  const res = await fetch(`${QDRANT_URL}${route}`, {
    method,
    headers,
    body: body == null ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  let data = null;
  if (text) {
    try { data = JSON.parse(text); }
    catch { data = text; }
  }
  if (!res.ok) {
    throw new Error(`${method} ${route} failed with ${res.status}: ${typeof data === 'string' ? data : JSON.stringify(data)}`);
  }
  return data;
}

async function ensureCollection(dim) {
  if (RECREATE) {
    try {
      await qdrant('DELETE', `/collections/${encodeURIComponent(COLLECTION)}`);
      log(`Deleted existing collection ${COLLECTION}`);
    } catch (err) {
      if (!String(err.message).includes('404')) throw err;
    }
  }

  try {
    await qdrant('GET', `/collections/${encodeURIComponent(COLLECTION)}`);
    log(`Using existing collection ${COLLECTION}`);
    return;
  } catch (err) {
    if (!String(err.message).includes('404')) throw err;
  }

  await qdrant('PUT', `/collections/${encodeURIComponent(COLLECTION)}`, {
    vectors: {
      size: dim,
      distance: 'Euclid',
    },
    hnsw_config: {
      m: M,
      ef_construct: EF_CONSTRUCTION,
    },
  });
  log(`Created collection ${COLLECTION}`);
}

async function upsertVectors(train) {
  if (SKIP_UPSERT) {
    log('Skipping upsert (--skip-upsert)');
    return 0;
  }

  const t0 = performance.now();
  for (let start = 0; start < train.length; start += BATCH_SIZE) {
    const end = Math.min(start + BATCH_SIZE, train.length);
    const points = [];
    for (let i = start; i < end; i++) {
      points.push({ id: i, vector: Array.from(train[i]) });
    }
    await qdrant('PUT', `/collections/${encodeURIComponent(COLLECTION)}/points?wait=true`, { points });
    if (end % 1000 < BATCH_SIZE || end === train.length) {
      log(`    ${end.toLocaleString()}/${train.length.toLocaleString()}`);
    }
  }
  return performance.now() - t0;
}

async function searchOne(query) {
  const data = await qdrant('POST', `/collections/${encodeURIComponent(COLLECTION)}/points/query`, {
    query: Array.from(query),
    limit: K,
    with_payload: false,
    with_vector: false,
    params: {
      hnsw_ef: EF_SEARCH,
      exact: false,
    },
  });
  return data.result.points.map((point) => Number(point.id));
}

async function benchSearch(queries, groundTruth) {
  for (let i = 0; i < WARMUP_QUERIES && i < queries.length; i++) {
    await searchOne(queries[i]);
  }

  const latencies = [];
  let totalRecall = 0;
  for (let rep = 0; rep < REPETITIONS; rep++) {
    for (let i = 0; i < queries.length; i++) {
      const st = performance.now();
      const ids = await searchOne(queries[i]);
      latencies.push(performance.now() - st);
      if (rep === 0) totalRecall += recall(ids, groundTruth[i]);
    }
  }
  latencies.sort((a, b) => a - b);
  return {
    label: 'qdrant-rest-fp32',
    buildMs: null,
    recall: totalRecall / queries.length,
    qps: 1000 / mean(latencies),
    p50: percentile(latencies, 0.5),
    p99: percentile(latencies, 0.99),
    meanMs: mean(latencies),
    stddevMs: stddev(latencies),
    memMB: null,
    params: `M=${M} efC=${EF_CONSTRUCTION} efS=${EF_SEARCH} REST batch=${BATCH_SIZE}`,
  };
}

function writeCsv(result) {
  const rows = [
    ['label', 'build_s', 'recall', 'qps', 'mean_ms', 'p50_ms', 'p99_ms', 'stddev_ms', 'mem_mb', 'params'],
    [
      result.label,
      result.buildMs == null ? '' : (result.buildMs / 1000).toFixed(3),
      result.recall.toFixed(6),
      result.qps.toFixed(2),
      result.meanMs.toFixed(4),
      result.p50.toFixed(4),
      result.p99.toFixed(4),
      result.stddevMs.toFixed(4),
      result.memMB == null ? '' : result.memMB.toFixed(2),
      result.params,
    ],
  ];
  fs.writeFileSync(CSV_PATH, rows.map((r) => r.join(',')).join('\n') + '\n');
}

async function main() {
  const basePath = path.join(DBPEDIA_DIR, N_BASE <= 5000 ? 'dbpedia_base_5k.fvecs' : 'dbpedia_base_100k.fvecs');
  const queryPath = path.join(DBPEDIA_DIR, 'dbpedia_query.fvecs');
  for (const filePath of [basePath, queryPath]) {
    if (!fs.existsSync(filePath)) throw new Error(`Missing file: ${filePath}`);
  }

  log('='.repeat(72));
  log('Qdrant DBpedia REST Benchmark');
  log('='.repeat(72));
  log(`Qdrant URL: ${QDRANT_URL}`);
  log(`Collection: ${COLLECTION}`);
  log(`count=${N_BASE.toLocaleString()} queries=${N_QUERIES.toLocaleString()} k=${K} M=${M} efConstruction=${EF_CONSTRUCTION} efSearch=${EF_SEARCH}`);
  log('Note: latency includes REST/JSON/client-server overhead.');
  log();

  const sysInfo = systemInfo();
  log(`System: ${sysInfo.platform}/${sysInfo.arch}, Node ${sysInfo.node}`);
  log(`CPU: ${sysInfo.cpu} (${sysInfo.logicalCpus} logical), RAM ${formatMB(sysInfo.totalMemoryBytes)}`);
  log();

  const { vectors: train, dim } = readFvecs(basePath, N_BASE);
  const { vectors: queries } = readFvecs(queryPath, N_QUERIES);
  log(`  Base: ${train.length.toLocaleString()} vectors, ${dim}D`);
  log(`  Queries: ${queries.length.toLocaleString()}`);
  log();

  let groundTruth = loadGroundTruth();
  if (!groundTruth) {
    groundTruth = computeGroundTruth(train, queries, dim);
    saveGroundTruth(groundTruth);
  }
  log();

  await qdrant('GET', '/');
  await ensureCollection(dim);
  const buildMs = await upsertVectors(train);
  log();

  const result = await benchSearch(queries, groundTruth);
  result.buildMs = buildMs;

  log('='.repeat(96));
  log('Results');
  log('='.repeat(96));
  log(
    'Library'.padEnd(22)
    + 'Build(s)'.padStart(10)
    + 'Recall'.padStart(10)
    + 'QPS'.padStart(10)
    + 'Mean(ms)'.padStart(10)
    + 'p50(ms)'.padStart(10)
    + 'p99(ms)'.padStart(10)
    + '  Params'
  );
  log('-'.repeat(96));
  log(
    result.label.padEnd(22)
    + (result.buildMs / 1000).toFixed(2).padStart(10)
    + `${(result.recall * 100).toFixed(2)}%`.padStart(10)
    + result.qps.toFixed(0).padStart(10)
    + result.meanMs.toFixed(3).padStart(10)
    + result.p50.toFixed(3).padStart(10)
    + result.p99.toFixed(3).padStart(10)
    + `  ${result.params}`
  );

  writeCsv(result);
  fs.writeFileSync(JSON_PATH, JSON.stringify({
    benchmark: 'qdrant-dbpedia-rest-l2',
    timestamp: new Date().toISOString(),
    qdrant: {
      url: QDRANT_URL,
      collection: COLLECTION,
      apiKeyProvided: Boolean(process.env.QDRANT_API_KEY),
    },
    dataset: {
      vectors: train.length,
      queries: queries.length,
      dim,
      metric: 'l2',
      source: DBPEDIA_DIR,
      baseFile: basePath,
      queryFile: queryPath,
    },
    params: { K, M, EF_CONSTRUCTION, EF_SEARCH, REPETITIONS, WARMUP_QUERIES, BATCH_SIZE },
    system: sysInfo,
    note: 'Qdrant is benchmarked over REST; build and query timings include JSON serialization and local/client-server overhead.',
    results: [result],
  }, null, 2));

  log();
  log('Outputs:');
  log(`  log:  ${LOG_PATH}`);
  log(`  json: ${JSON_PATH}`);
  log(`  csv:  ${CSV_PATH}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
