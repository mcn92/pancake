#!/usr/bin/env node
'use strict';

/**
 * Synthetic 1536D Benchmark: Pancake vs hnswlib-node
 *
 * 100K clustered unit-norm vectors in 1536D (OpenAI Ada dimension).
 * Sweeps ef_search to produce recall-QPS curves comparable to the
 * NYTimes-256 and SIFT-1M benchmarks.
 *
 * Data: 50 clusters, spread=0.25, gaussian noise + re-normalize.
 * Queries drawn from the same distribution. Deterministic (seed=42).
 *
 * Ground truth is brute-force cosine distance, cached to disk on
 * first run. Pass --regenerate-gt to force recomputation.
 *
 * Usage:
 *   node benchmarks/benchmark_synthetic_1536.js
 *   node benchmarks/benchmark_synthetic_1536.js --regenerate-gt
 */

const fs = require('fs');
const path = require('path');
const Pancake = require('../pancake.js');
const { parseBenchmarkArgs, resolveSingleValue, resolveSweepValues } = require('./bench_args');

const parsedArgs = parseBenchmarkArgs();

const RESULTS_DIR = path.join(__dirname, '..', 'benchmark_results');
const CACHE_DIR = path.join(__dirname, '..', 'benchmark_results', 'cache');
if (!fs.existsSync(RESULTS_DIR)) fs.mkdirSync(RESULTS_DIR);
if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR);
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const LOG_PATH = path.join(RESULTS_DIR, `synthetic_1536_${timestamp}.log`);
const JSON_PATH = path.join(RESULTS_DIR, `synthetic_1536_${timestamp}.json`);
const CSV_PATH = path.join(RESULTS_DIR, `synthetic_1536_${timestamp}.csv`);
const logStream = fs.createWriteStream(LOG_PATH);

const REGENERATE_GT = parsedArgs.args.includes('--regenerate-gt');

function log(msg = '') {
  console.log(msg);
  logStream.write(msg + '\n');
}

// --- Config ---
const DIM = 1536;
const N_VECTORS = 50_000;
const N_QUERIES = 1000;
const N_CLUSTERS = 50;
const K = 10;
const M = resolveSingleValue(parsedArgs.m, 16);
const EF_CONSTRUCTION = resolveSingleValue(parsedArgs.efConstruction, 200);
const EF_SEARCH_VALUES = resolveSweepValues(parsedArgs, [10, 20, 40, 60, 80, 100, 150, 200, 300, 500, 800]);
const REPETITIONS = 3;
const WARMUP_QUERIES = 200;
const CLUSTER_SPREAD = 0.02;
const SEED = 42;

// --- Deterministic RNG (mulberry32) ---
function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianPair(rng) {
  const u1 = Math.max(rng(), 1e-10);
  const u2 = rng();
  const r = Math.sqrt(-2 * Math.log(u1));
  return [r * Math.cos(2 * Math.PI * u2), r * Math.sin(2 * Math.PI * u2)];
}

function randomUnitVector(dim, rng) {
  const v = new Float32Array(dim);
  let norm = 0;
  for (let i = 0; i < dim; i += 2) {
    const [a, b] = gaussianPair(rng);
    v[i] = a;
    if (i + 1 < dim) v[i + 1] = b;
    norm += a * a;
    if (i + 1 < dim) norm += b * b;
  }
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < dim; i++) v[i] /= norm;
  return v;
}

function generateCentroids(nClusters, dim, rng) {
  const centroids = [];
  for (let c = 0; c < nClusters; c++) centroids.push(randomUnitVector(dim, rng));
  return centroids;
}

function generateClusteredVectors(count, dim, centroids, spread, rng) {
  const nClusters = centroids.length;
  const vectors = new Array(count);
  for (let i = 0; i < count; i++) {
    const c = Math.floor(rng() * nClusters);
    const centroid = centroids[c];
    const v = new Float32Array(dim);
    let norm = 0;
    for (let d = 0; d < dim; d += 2) {
      const [a, b] = gaussianPair(rng);
      v[d] = centroid[d] + a * spread;
      if (d + 1 < dim) v[d + 1] = centroid[d + 1] + b * spread;
      norm += v[d] * v[d];
      if (d + 1 < dim) norm += v[d + 1] * v[d + 1];
    }
    norm = Math.sqrt(norm) || 1;
    for (let d = 0; d < dim; d++) v[d] /= norm;
    vectors[i] = v;
  }
  return vectors;
}

// --- Ground truth with disk cache ---
const GT_CACHE_PATH = path.join(CACHE_DIR,
  `gt_syn1536_n${N_VECTORS}_q${N_QUERIES}_c${N_CLUSTERS}_s${CLUSTER_SPREAD}_seed${SEED}.bin`);

function saveGroundTruth(gt) {
  // Format: [N_QUERIES(u32)][K(u32)] then N_QUERIES × K int32 IDs
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
  fs.writeFileSync(GT_CACHE_PATH, buf);
  log(`  Cached ground truth to ${GT_CACHE_PATH}`);
}

function loadGroundTruth() {
  if (!fs.existsSync(GT_CACHE_PATH)) return null;
  const buf = fs.readFileSync(GT_CACHE_PATH);
  const nq = buf.readUInt32LE(0);
  const k = buf.readUInt32LE(4);
  if (nq !== N_QUERIES || k !== K) return null;
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

async function computeGroundTruth(vectors, queries, k) {
  log(`Computing brute-force ground truth (${queries.length} × ${vectors.length} × ${DIM}D)...`);
  log(`  Pure JS — this will take several minutes at 1536D.`);
  const t0 = performance.now();
  const gt = new Array(queries.length);

  for (let q = 0; q < queries.length; q++) {
    const query = queries[q];
    // Compute dot product with every database vector
    const dots = new Float32Array(vectors.length);
    for (let i = 0; i < vectors.length; i++) {
      let dot = 0;
      for (let d = 0; d < DIM; d++) dot += query[d] * vectors[i][d];
      dots[i] = dot;
    }
    // Find top-k by highest dot product (= nearest for unit-norm cosine)
    const indices = new Array(vectors.length);
    for (let i = 0; i < vectors.length; i++) indices[i] = i;
    indices.sort((a, b) => dots[b] - dots[a]);
    gt[q] = indices.slice(0, k);

    if ((q + 1) % 25 === 0) {
      const elapsed = (performance.now() - t0) / 1000;
      const eta = (elapsed / (q + 1)) * (queries.length - q - 1);
      process.stdout.write(`  ${q + 1}/${queries.length} (${elapsed.toFixed(0)}s elapsed, ~${eta.toFixed(0)}s remaining)\r`);
    }
  }

  const elapsed = (performance.now() - t0) / 1000;
  log(`  Ground truth computed in ${elapsed.toFixed(1)}s                    `);
  return gt;
}

// --- Helpers ---
function recall(predicted, truth) {
  const truthSet = new Set(truth);
  let hits = 0;
  for (const id of predicted) if (truthSet.has(id)) hits++;
  return hits / truth.length;
}

function percentile(sorted, p) {
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
}

function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }

function stddev(arr) {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
}

// --- hnswlib-node ---
let HierarchicalNSW;
try {
  HierarchicalNSW = require('hnswlib-node').HierarchicalNSW;
} catch (e) {
  log('WARNING: hnswlib-node not installed; skipping hnswlib comparison');
}

// --- Main ---
async function main() {
  log('='.repeat(70));
  log('Synthetic 1536D Benchmark');
  log(`${N_VECTORS.toLocaleString()} vectors, ${DIM}D, ${N_CLUSTERS} clusters, spread=${CLUSTER_SPREAD}`);
  log(`${N_QUERIES} queries, k=${K}`);
  log(`M=${M}, ef_construction=${EF_CONSTRUCTION}`);
  log(`ef_search sweep: [${EF_SEARCH_VALUES.join(', ')}]`);
  log(`Repetitions: ${REPETITIONS}, Warmup: ${WARMUP_QUERIES}`);
  log(`Note: synthetic clustered data; recall will be higher than real-world embeddings`);
  log('='.repeat(70));

  // Generate data (deterministic, shared centroids)
  log('\nGenerating dataset...');
  const rng = mulberry32(SEED);
  const centroids = generateCentroids(N_CLUSTERS, DIM, rng);
  const vectors = generateClusteredVectors(N_VECTORS, DIM, centroids, CLUSTER_SPREAD, rng);
  const queries = generateClusteredVectors(N_QUERIES, DIM, centroids, CLUSTER_SPREAD, rng);
  log(`  ${N_VECTORS.toLocaleString()} vectors + ${N_QUERIES} queries generated (${N_CLUSTERS} shared centroids)`);

  // Ground truth (cached)
  let groundTruth;
  if (!REGENERATE_GT) groundTruth = loadGroundTruth();
  if (groundTruth) {
    log(`\nLoaded cached ground truth from ${GT_CACHE_PATH}`);
  } else {
    log('');
    groundTruth = await computeGroundTruth(vectors, queries, K);
    saveGroundTruth(groundTruth);
  }

  const memBefore = process.memoryUsage();

  // =============================================
  // Pancake
  // =============================================
  log(`\n${'='.repeat(70)}`);
  log('Pancake u8 (WASM)');
  log('='.repeat(70));

  log('Building index...');
  const pkIndex = await Pancake.create({
    dim: DIM, maxElements: N_VECTORS, quantized: false,
    M, efConstruction: EF_CONSTRUCTION, efSearch: EF_SEARCH_VALUES[0],
  });

  const pkBuildT0 = performance.now();
  const batchSize = 500;
  for (let i = 0; i < vectors.length; i += batchSize) {
    pkIndex.addBatch(vectors.slice(i, Math.min(i + batchSize, vectors.length)));
    if (Math.min(i + batchSize, vectors.length) % 10000 < batchSize) {
      log(`  ${Math.min(i + batchSize, vectors.length).toLocaleString()}/${N_VECTORS.toLocaleString()}`);
    }
  }
  const pkBuildSec = (performance.now() - pkBuildT0) / 1000;
  const pkIndexMem = pkIndex.memory;
  const memAfterPk = process.memoryUsage();

  log(`  Build: ${pkBuildSec.toFixed(1)}s`);
  log(`  Index memory: ${(pkIndexMem / 1024 / 1024).toFixed(1)} MB`);
  log(`  Process RSS Δ (index build only): ${((memAfterPk.rss - memBefore.rss) / 1024 / 1024).toFixed(1)} MB`);

  const pkPoints = [];
  for (const efS of EF_SEARCH_VALUES) {
    pkIndex.setEfSearch(efS);

    for (let i = 0; i < WARMUP_QUERIES && i < queries.length; i++) {
      pkIndex.search(queries[i], K);
    }

    const reps = [];
    for (let rep = 0; rep < REPETITIONS; rep++) {
      const lats = [];
      let rec = 0;
      for (let i = 0; i < queries.length; i++) {
        const t0 = performance.now();
        const results = pkIndex.search(queries[i], K);
        lats.push(performance.now() - t0);
        rec += recall(results.map(r => r.id), groundTruth[i]);
      }
      reps.push({ recall: rec / queries.length, lats });
    }

    const avgRecall = mean(reps.map(r => r.recall));
    const recallStd = stddev(reps.map(r => r.recall));
    const allLats = reps.flatMap(r => r.lats).sort((a, b) => a - b);
    const qps = 1000 / mean(allLats);
    const qpsStd = stddev(reps.map(r => 1000 / mean(r.lats)));

    const point = {
      ef_search: efS,
      recall_mean: avgRecall, recall_std: recallStd,
      qps_mean: qps, qps_std: qpsStd,
      p50: percentile(allLats, 0.5),
      p95: percentile(allLats, 0.95),
      p99: percentile(allLats, 0.99),
      p999: percentile(allLats, 0.999),
    };
    pkPoints.push(point);
    log(`  ef=${String(efS).padStart(4)}  recall=${(avgRecall * 100).toFixed(1).padStart(5)}%  qps=${String(qps.toFixed(0)).padStart(5)}  p50=${point.p50.toFixed(3)}ms  p95=${point.p95.toFixed(3)}ms  p999=${point.p999.toFixed(3)}ms`);
  }
  pkIndex.dispose();

  // =============================================
  // hnswlib-node
  // =============================================
  let hlPoints = null;
  let hlBuildSec = null;
  if (HierarchicalNSW) {
    log(`\n${'='.repeat(70)}`);
    log('hnswlib Float32 (Native)');
    log('='.repeat(70));

    log('Building index...');
    const hlIndex = new HierarchicalNSW('cosine', DIM);
    hlIndex.initIndex(N_VECTORS, M, EF_CONSTRUCTION, 100);

    const hlBuildT0 = performance.now();
    for (let i = 0; i < vectors.length; i++) {
      hlIndex.addPoint(Array.from(vectors[i]), i);
      if ((i + 1) % 10000 === 0) log(`  ${(i + 1).toLocaleString()}/${N_VECTORS.toLocaleString()}`);
    }
    hlBuildSec = (performance.now() - hlBuildT0) / 1000;
    log(`  Build: ${hlBuildSec.toFixed(1)}s`);

    hlPoints = [];
    for (const efS of EF_SEARCH_VALUES) {
      hlIndex.setEf(efS);

      for (let i = 0; i < WARMUP_QUERIES && i < queries.length; i++) {
        hlIndex.searchKnn(Array.from(queries[i]), K);
      }

      const reps = [];
      for (let rep = 0; rep < REPETITIONS; rep++) {
        const lats = [];
        let rec = 0;
        for (let i = 0; i < queries.length; i++) {
          const query = Array.from(queries[i]);
          const t0 = performance.now();
          const results = hlIndex.searchKnn(query, K);
          lats.push(performance.now() - t0);
          rec += recall(results.neighbors, groundTruth[i]);
        }
        reps.push({ recall: rec / queries.length, lats });
      }

      const avgRecall = mean(reps.map(r => r.recall));
      const recallStd = stddev(reps.map(r => r.recall));
      const allLats = reps.flatMap(r => r.lats).sort((a, b) => a - b);
      const qps = 1000 / mean(allLats);
      const qpsStd = stddev(reps.map(r => 1000 / mean(r.lats)));

      const point = {
        ef_search: efS,
        recall_mean: avgRecall, recall_std: recallStd,
        qps_mean: qps, qps_std: qpsStd,
        p50: percentile(allLats, 0.5),
        p95: percentile(allLats, 0.95),
        p99: percentile(allLats, 0.99),
        p999: percentile(allLats, 0.999),
      };
      hlPoints.push(point);
      log(`  ef=${String(efS).padStart(4)}  recall=${(avgRecall * 100).toFixed(1).padStart(5)}%  qps=${String(qps.toFixed(0)).padStart(5)}  p50=${point.p50.toFixed(3)}ms  p95=${point.p95.toFixed(3)}ms  p999=${point.p999.toFixed(3)}ms`);
    }
  }

  // =============================================
  // Output
  // =============================================

  // CSV (same format as other sweeps)
  const csvRows = ['label,library,dtype,ef_search,recall,recall_std,qps,qps_std,p50_ms,p95_ms,p99_ms,p999_ms'];
  for (const p of pkPoints) {
    csvRows.push([
      'pancake-u8-wasm', 'pancake', 'u8', p.ef_search,
      p.recall_mean.toFixed(5), p.recall_std.toFixed(5),
      p.qps_mean.toFixed(2), p.qps_std.toFixed(2),
      p.p50.toFixed(4), p.p95.toFixed(4), p.p99.toFixed(4), p.p999.toFixed(4),
    ].join(','));
  }
  if (hlPoints) {
    for (const p of hlPoints) {
      csvRows.push([
        'hnswlib-f32-native', 'hnswlib', 'f32', p.ef_search,
        p.recall_mean.toFixed(5), p.recall_std.toFixed(5),
        p.qps_mean.toFixed(2), p.qps_std.toFixed(2),
        p.p50.toFixed(4), p.p95.toFixed(4), p.p99.toFixed(4), p.p999.toFixed(4),
      ].join(','));
    }
  }
  fs.writeFileSync(CSV_PATH, csvRows.join('\n') + '\n');

  // JSON
  const bytesPerVec = pkIndexMem / N_VECTORS;
  fs.writeFileSync(JSON_PATH, JSON.stringify({
    benchmark: 'synthetic-1536',
    timestamp: new Date().toISOString(),
    note: 'Synthetic clustered data; recall will be higher than real-world embeddings',
    config: {
      dim: DIM, vectors: N_VECTORS, queries: N_QUERIES,
      clusters: N_CLUSTERS, spread: CLUSTER_SPREAD, seed: SEED,
      K, M, ef_construction: EF_CONSTRUCTION,
      ef_search_values: EF_SEARCH_VALUES,
      repetitions: REPETITIONS, warmup: WARMUP_QUERIES,
    },
    pancake: {
      build_s: +pkBuildSec.toFixed(1),
      index_memory_mb: +(pkIndexMem / 1024 / 1024).toFixed(1),
      bytes_per_vector: +bytesPerVec.toFixed(0),
      points: pkPoints,
    },
    hnswlib: hlPoints ? {
      build_s: +hlBuildSec.toFixed(1),
      points: hlPoints,
    } : null,
  }, null, 2) + '\n');

  log(`\nOutputs:\n  ${LOG_PATH}\n  ${JSON_PATH}\n  ${CSV_PATH}`);
  log(`\nPlot with:\n  python3 benchmarks/plot_sweep.py ${CSV_PATH}`);
  logStream.end();
}

main().catch(e => { console.error(e); process.exit(1); });
