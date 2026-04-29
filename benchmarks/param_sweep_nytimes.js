#!/usr/bin/env node
'use strict';

/**
 * Full parameter sweep on NYTimes-256: Pancake vs hnswlib-node.
 *
 * Sweeps M × ef_construction × ef_search. Each (M, ef_c) pair builds
 * the index once; ef_search is tuned at runtime without rebuilding.
 *
 * Usage:
 *   node benchmarks/param_sweep_nytimes.js [path-to-nytimes-dir]
 */

const fs = require('fs');
const path = require('path');
const Pancake = require('../pancake.js');

const NYTIMES_DIR = process.argv[2] || path.join(__dirname, '..', 'nytimes');
const RESULTS_DIR = path.join(__dirname, '..', 'benchmark_results');
if (!fs.existsSync(RESULTS_DIR)) fs.mkdirSync(RESULTS_DIR);
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const LOG_PATH = path.join(RESULTS_DIR, `param_sweep_${timestamp}.log`);
const JSON_PATH = path.join(RESULTS_DIR, `param_sweep_${timestamp}.json`);
const CSV_PATH = path.join(RESULTS_DIR, `param_sweep_${timestamp}.csv`);
const logStream = fs.createWriteStream(LOG_PATH);

function log(msg = '') {
  console.log(msg);
  logStream.write(msg + '\n');
}

// --- Sweep grid ---
const K = 10;
const M_VALUES = [8, 12, 16, 32];
const EF_C_VALUES = [50, 100, 150, 200];
const EF_SEARCH_VALUES = [50, 100, 150, 200];
const REPETITIONS = 3;
const WARMUP_QUERIES = 200;

// --- Data loading (fvecs/ivecs) ---
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

// --- Main ---
async function main() {
  const requiredFiles = ['nytimes_base.fvecs', 'nytimes_query.fvecs', 'nytimes_groundtruth.ivecs'];
  for (const f of requiredFiles) {
    if (!fs.existsSync(path.join(NYTIMES_DIR, f))) {
      log(`Missing: ${path.join(NYTIMES_DIR, f)}`);
      process.exit(1);
    }
  }

  const { vectors: train, dim } = readFvecs(path.join(NYTIMES_DIR, 'nytimes_base.fvecs'));
  const { vectors: test } = readFvecs(path.join(NYTIMES_DIR, 'nytimes_query.fvecs'));
  const groundTruth = readIvecs(path.join(NYTIMES_DIR, 'nytimes_groundtruth.ivecs'));

  log(`\n${'='.repeat(70)}`);
  log(`NYTimes-256 Pancake Parameter Sweep`);
  log(`${train.length} vectors, ${dim}D, ${test.length} queries, k=${K}`);
  log(`M: [${M_VALUES}]  ef_c: [${EF_C_VALUES}]  ef_search: [${EF_SEARCH_VALUES}]`);
  log(`${M_VALUES.length * EF_C_VALUES.length} index builds, ${M_VALUES.length * EF_C_VALUES.length * EF_SEARCH_VALUES.length} measurement points`);
  log(`Repetitions: ${REPETITIONS}, Warmup: ${WARMUP_QUERIES}`);
  log('='.repeat(70));

  const allResults = [];

  for (const M of M_VALUES) {
    for (const efC of EF_C_VALUES) {
      log(`\n${'─'.repeat(70)}`);
      log(`Building: M=${M}, ef_c=${efC}`);
      log('─'.repeat(70));

      const index = await Pancake.create({
        dim, maxElements: train.length, quantized: true,
        M, efConstruction: efC, efSearch: EF_SEARCH_VALUES[0],
      });
      const t0 = performance.now();
      const batchSize = 1000;
      for (let i = 0; i < train.length; i += batchSize) {
        index.addBatch(train.slice(i, Math.min(i + batchSize, train.length)));
      }
      const buildMs = performance.now() - t0;
      log(`  Built in ${(buildMs / 1000).toFixed(1)}s`);

      for (const efS of EF_SEARCH_VALUES) {
        index._setEfSearch(efS);

        // Warmup
        for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) {
          index.search(test[i], K);
        }

        const repLats = [];
        let totalRecall = 0;
        for (let rep = 0; rep < REPETITIONS; rep++) {
          const lats = [];
          let rec = 0;
          for (let i = 0; i < test.length; i++) {
            const st = performance.now();
            const results = index.search(test[i], K);
            lats.push(performance.now() - st);
            rec += recall(results.map(r => r.id), groundTruth[i]);
          }
          repLats.push(lats);
          totalRecall += rec / test.length;
        }

        const avgRecall = totalRecall / REPETITIONS;
        const allLats = repLats.flat().sort((a, b) => a - b);
        const qps = 1000 / mean(allLats);
        const p50 = percentile(allLats, 0.5);
        const p99 = percentile(allLats, 0.99);

        log(`  ef_s=${String(efS).padStart(3)}  recall=${(avgRecall * 100).toFixed(1).padStart(5)}%  qps=${String(qps.toFixed(0)).padStart(5)}  p50=${p50.toFixed(3)}ms  p99=${p99.toFixed(3)}ms`);

        allResults.push({
          M, ef_c: efC, ef_search: efS,
          build_s: buildMs / 1000,
          recall: avgRecall,
          qps, p50, p99,
        });
      }

      index.dispose();
    }
  }

  // --- Write CSV ---
  const csvRows = ['M,ef_c,ef_search,build_s,recall,qps,p50,p99'];
  for (const r of allResults) {
    csvRows.push([
      r.M, r.ef_c, r.ef_search,
      r.build_s.toFixed(1),
      (r.recall * 100).toFixed(2),
      r.qps.toFixed(1),
      r.p50.toFixed(4),
      r.p99.toFixed(4),
    ].join(','));
  }
  fs.writeFileSync(CSV_PATH, csvRows.join('\n') + '\n');

  // --- Write JSON ---
  fs.writeFileSync(JSON_PATH, JSON.stringify({
    benchmark: 'nytimes-256-param-sweep',
    timestamp: new Date().toISOString(),
    dataset: { vectors: train.length, queries: test.length, dim },
    grid: { M: M_VALUES, ef_c: EF_C_VALUES, ef_search: EF_SEARCH_VALUES, K, REPETITIONS },
    results: allResults,
  }, null, 2) + '\n');

  // --- Summary table ---
  log(`\n${'='.repeat(70)}`);
  log('Recall ceiling (ef_search=200) per (M, ef_c)');
  log('='.repeat(70));
  log(`  ${'M'.padStart(4)}  ${'ef_c'.padStart(5)}  ${'recall'.padStart(8)}  ${'qps'.padStart(6)}  ${'build'.padStart(7)}`);
  for (const M of M_VALUES) {
    for (const efC of EF_C_VALUES) {
      const maxEf = Math.max(...EF_SEARCH_VALUES);
      const r = allResults.find(x => x.M === M && x.ef_c === efC && x.ef_search === maxEf);
      if (r) {
        log(`  ${String(M).padStart(4)}  ${String(efC).padStart(5)}  ${(r.recall * 100).toFixed(1).padStart(7)}%  ${r.qps.toFixed(0).padStart(6)}  ${r.build_s.toFixed(0).padStart(6)}s`);
      }
    }
  }

  log(`\nOutputs:\n  ${LOG_PATH}\n  ${JSON_PATH}\n  ${CSV_PATH}`);
  logStream.end();
}

main().catch(e => { console.error(e); process.exit(1); });
