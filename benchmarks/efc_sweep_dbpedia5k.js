#!/usr/bin/env node
'use strict';

/**
 * efConstruction sensitivity sweep on DBpedia-5K (1536D, L2).
 * Measures build time and recall@10 at fixed efSearch=100 across
 * efConstruction values to find the minimum ef_c that holds recall.
 */

const fs = require('fs');
const path = require('path');
const Pikelet = require('../pikelet.js');
const { parseBenchmarkArgs, resolveSingleValue } = require('./bench_args');

const parsedArgs = parseBenchmarkArgs();

const DBPEDIA_DIR = path.join(__dirname, '..', 'dbpedia');
const BASE_PATH = path.join(DBPEDIA_DIR, 'dbpedia_base_5k.fvecs');
const QUERY_PATH = path.join(DBPEDIA_DIR, 'dbpedia_query.fvecs');

const K = 10;
const M = resolveSingleValue(parsedArgs.m, 16);
const EF_SEARCH = resolveSingleValue(parsedArgs.efSearch, 100);
const N_QUERIES = 200;
const EFC_VALUES = parsedArgs.efConstruction !== undefined ? [parsedArgs.efConstruction] : [16, 32, 50, 64, 75];

function readFvecs(filePath, maxVectors) {
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

function bruteForceL2(train, queries, k) {
  const dim = train[0].length;
  const gt = [];
  for (const query of queries) {
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
    gt.push(indices.slice(0, k));
  }
  return gt;
}

function recall(predicted, truth) {
  const s = new Set(truth);
  let hits = 0;
  for (const id of predicted) if (s.has(id)) hits++;
  return hits / truth.length;
}

async function main() {
  if (!fs.existsSync(BASE_PATH)) {
    console.log(`Missing: ${BASE_PATH}`);
    process.exit(1);
  }

  const { vectors: train, dim } = readFvecs(BASE_PATH);
  const { vectors: queries } = readFvecs(QUERY_PATH, N_QUERIES);
  console.log(`Dataset: ${train.length} vectors, ${dim}D, ${queries.length} queries`);
  console.log(`K=${K}, M=${M}, efSearch=${EF_SEARCH}`);
  console.log(`efConstruction sweep: [${EFC_VALUES.join(', ')}]\n`);

  console.log('Computing ground truth (brute-force L2)...');
  const gt = bruteForceL2(train, queries, K);
  console.log('Done.\n');

  console.log('efc  | backend  | build_s | recall@10 | qps');
  console.log('-----|----------|---------|-----------|------');

  for (const efc of EFC_VALUES) {
    for (const quantized of [true, false]) {
      const label = quantized ? 'uint8' : 'fp32';
      const index = await Pikelet.create({
        dim, maxElements: train.length, quantized,
        metric: 'l2', M, efConstruction: efc, efSearch: EF_SEARCH,
      });

      const t0 = performance.now();
      const batchSize = 500;
      for (let i = 0; i < train.length; i += batchSize) {
        index.addBatch(train.slice(i, Math.min(i + batchSize, train.length)));
      }
      const buildSec = (performance.now() - t0) / 1000;

      let totalRecall = 0;
      const searchT0 = performance.now();
      for (let i = 0; i < queries.length; i++) {
        const results = index.search(queries[i], K);
        totalRecall += recall(results.map(r => r.id), gt[i]);
      }
      const searchMs = performance.now() - searchT0;
      const qps = (queries.length / searchMs) * 1000;
      const avgRecall = totalRecall / queries.length;

      console.log(
        `${String(efc).padStart(4)} | ${label.padEnd(8)} | ${buildSec.toFixed(1).padStart(7)} | ${(avgRecall * 100).toFixed(2).padStart(8)}% | ${qps.toFixed(0).padStart(5)}`
      );
      index.dispose();
    }
  }
}

main().catch(e => { console.error(e); process.exit(1); });
