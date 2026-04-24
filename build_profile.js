#!/usr/bin/env node
'use strict';

/**
 * Build-profiling benchmark.
 *
 * Goal: isolate WHY Pancake's insert path is slow, in a run that completes
 * fast enough (<30s) to iterate on.
 *
 * This benchmark measures:
 *   1. Per-insert cost as N grows (is it O(log N) like it should be, or worse?)
 *   2. Scaling with ef_construction (how much of the cost is candidate search?)
 *   3. Scaling with dimension (how much is distance computation?)
 *   4. Float32 vs Int8 on same data (how much is quantization overhead?)
 *   5. Pancake vs USearch at each setting (where exactly is the gap?)
 *
 * Usage: node bench_build_profile.js
 */

const Pancake = require('./pancake.js');

let usearch;
try { usearch = require('usearch'); }
catch (e) { console.warn('USearch not installed; running Pancake-only'); }

// -------- data generation (normalized, reproducible) --------
function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function genVectors(n, dim, seed = 42) {
  const rng = mulberry32(seed);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const v = new Float32Array(dim);
    let norm = 0;
    // box-muller for gaussian (more realistic than uniform for angular data)
    for (let j = 0; j < dim; j += 2) {
      const u1 = Math.max(rng(), 1e-10);
      const u2 = rng();
      const r = Math.sqrt(-2 * Math.log(u1));
      v[j] = r * Math.cos(2 * Math.PI * u2);
      if (j + 1 < dim) v[j + 1] = r * Math.sin(2 * Math.PI * u2);
      norm += v[j] * v[j];
      if (j + 1 < dim) norm += v[j + 1] * v[j + 1];
    }
    norm = Math.sqrt(norm);
    for (let j = 0; j < dim; j++) v[j] /= norm;
    out[i] = v;
  }
  return out;
}

// -------- timing helpers --------
function now() { return performance.now(); }

function fmt(ms) {
  if (ms < 1) return `${(ms * 1000).toFixed(1)}µs`;
  if (ms < 1000) return `${ms.toFixed(2)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

// -------- build measurement with per-batch timing --------
async function buildPancake({ vectors, dim, M, efConstruction, quantized }) {
  const index = await Pancake.create({
    dim,
    maxElements: vectors.length,
    quantized,
    M,
    efConstruction,
    efSearch: 50, // irrelevant for build
  });

  // Measure cost in batches to see how it scales with N
  const batchSize = Math.max(1, Math.floor(vectors.length / 10));
  const batchTimes = [];
  const t0 = now();

  for (let start = 0; start < vectors.length; start += batchSize) {
    const end = Math.min(start + batchSize, vectors.length);
    const bt0 = now();
    for (let i = start; i < end; i++) index.add(vectors[i]);
    batchTimes.push({ nAtEnd: end, ms: now() - bt0, count: end - start });
  }

  const totalMs = now() - t0;
  const mem = index.memory;
  index.dispose();
  return { totalMs, batchTimes, mem };
}

function buildUSearch({ vectors, dim, M, efConstruction, quantized }) {
  if (!usearch) return null;
  const index = new usearch.Index({
    metric: 'cos',
    connectivity: M,
    dimensions: dim,
    dtype: quantized ? 'i8' : 'f32',
    expansion_add: efConstruction,
    expansion_search: 50,
  });

  const batchSize = Math.max(1, Math.floor(vectors.length / 10));
  const batchTimes = [];
  const t0 = now();

  for (let start = 0; start < vectors.length; start += batchSize) {
    const end = Math.min(start + batchSize, vectors.length);
    const bt0 = now();
    for (let i = start; i < end; i++) index.add(BigInt(i), vectors[i]);
    batchTimes.push({ nAtEnd: end, ms: now() - bt0, count: end - start });
  }

  const totalMs = now() - t0;
  return { totalMs, batchTimes, mem: null };
}

// -------- experiments --------

/**
 * Experiment 1: Does per-insert cost scale as O(log N) like HNSW should?
 * If Pancake's last batch is dramatically slower than its first, something
 * is growing superlinearly (visited set, candidate list, etc).
 */
async function exp1_scalingWithN() {
  console.log('\n=== Experiment 1: per-insert cost vs N ===');
  console.log('HNSW insert should be O(log N * ef_c * M * dim).');
  console.log('We insert 20k vectors in 10 batches and watch per-insert time.\n');

  const vectors = genVectors(20000, 128);
  const cfg = { dim: 128, M: 16, efConstruction: 100, quantized: true };

  const pk = await buildPancake({ vectors, ...cfg });
  const us = buildUSearch({ vectors, ...cfg });

  console.log('  batch_end     pancake_us/insert   usearch_us/insert   ratio');
  for (let i = 0; i < pk.batchTimes.length; i++) {
    const pb = pk.batchTimes[i];
    const pkUs = (pb.ms * 1000) / pb.count;
    if (us) {
      const ub = us.batchTimes[i];
      const usUs = (ub.ms * 1000) / ub.count;
      console.log(`  ${String(pb.nAtEnd).padStart(8)}   ${pkUs.toFixed(1).padStart(16)}   ${usUs.toFixed(1).padStart(16)}   ${(pkUs / usUs).toFixed(2)}x`);
    } else {
      console.log(`  ${String(pb.nAtEnd).padStart(8)}   ${pkUs.toFixed(1).padStart(16)}`);
    }
  }

  // Diagnostic: compare first batch to last batch.
  const first = pk.batchTimes[0];
  const last = pk.batchTimes[pk.batchTimes.length - 1];
  const firstUs = (first.ms * 1000) / first.count;
  const lastUs = (last.ms * 1000) / last.count;
  const growth = lastUs / firstUs;
  const expectedGrowth = Math.log2(last.nAtEnd) / Math.log2(first.nAtEnd);
  console.log(`\n  Pancake growth: ${growth.toFixed(2)}x (expected ~${expectedGrowth.toFixed(2)}x for O(log N))`);
  if (growth > expectedGrowth * 1.5) {
    console.log('  WARNING: growing faster than log(N). Check visited-set allocation.');
  }
}

/**
 * Experiment 2: How much does ef_construction cost?
 * Distance-compute cost should scale roughly linearly with ef_construction.
 * If Pancake grows faster than USearch here, the candidate-search loop is
 * inefficient.
 */
async function exp2_scalingWithEfC() {
  console.log('\n=== Experiment 2: per-insert cost vs ef_construction ===');
  console.log('Fixed N=5000, dim=128. Varying ef_construction.\n');

  const vectors = genVectors(5000, 128);
  const efs = [20, 50, 100, 200, 400];

  console.log('  ef_c    pancake(s)   usearch(s)   ratio');
  for (const ef of efs) {
    const cfg = { dim: 128, M: 16, efConstruction: ef, quantized: true };
    const pk = await buildPancake({ vectors, ...cfg });
    const us = buildUSearch({ vectors, ...cfg });
    if (us) {
      console.log(`  ${String(ef).padStart(4)}    ${(pk.totalMs / 1000).toFixed(3).padStart(9)}    ${(us.totalMs / 1000).toFixed(3).padStart(9)}    ${(pk.totalMs / us.totalMs).toFixed(2)}x`);
    } else {
      console.log(`  ${String(ef).padStart(4)}    ${(pk.totalMs / 1000).toFixed(3).padStart(9)}`);
    }
  }
  console.log('\n  If ratio GROWS with ef_c, candidate-search loop is the bottleneck.');
  console.log('  If ratio is FLAT, overhead is per-insert fixed cost (allocation, etc).');
}

/**
 * Experiment 3: How much does dimension cost?
 * Distance computation cost scales linearly with dim.
 * SIMD should give ~4-8x speedup. If Pancake's per-dim cost is much higher
 * than USearch, SIMD isn't engaging or distance isn't vectorized.
 */
async function exp3_scalingWithDim() {
  console.log('\n=== Experiment 3: per-insert cost vs dimension ===');
  console.log('Fixed N=3000, ef_c=100. Varying dim.\n');

  const dims = [64, 128, 256, 512];

  console.log('  dim    pancake(s)   usearch(s)   ratio   pancake_us/insert');
  for (const dim of dims) {
    const vectors = genVectors(3000, dim);
    const cfg = { dim, M: 16, efConstruction: 100, quantized: true };
    const pk = await buildPancake({ vectors, ...cfg });
    const us = buildUSearch({ vectors, ...cfg });
    const pkPer = (pk.totalMs * 1000) / vectors.length;
    if (us) {
      console.log(`  ${String(dim).padStart(3)}    ${(pk.totalMs / 1000).toFixed(3).padStart(9)}    ${(us.totalMs / 1000).toFixed(3).padStart(9)}    ${(pk.totalMs / us.totalMs).toFixed(2)}x     ${pkPer.toFixed(1)}`);
    } else {
      console.log(`  ${String(dim).padStart(3)}    ${(pk.totalMs / 1000).toFixed(3).padStart(9)}                           ${pkPer.toFixed(1)}`);
    }
  }
  console.log('\n  If cost scales >linearly with dim, distance func is poorly vectorized.');
  console.log('  If ratio grows with dim, Pancake SIMD is worse than USearch SIMD.');
}

/**
 * Experiment 4: Int8 vs Float32 on the same data.
 * If you have both paths, this isolates the cost of quantization.
 */
async function exp4_quantizationCost() {
  console.log('\n=== Experiment 4: Int8 vs Float32 quantization overhead ===');
  console.log('Same data, same params. Only quantization changes.\n');

  const vectors = genVectors(5000, 128);
  const base = { dim: 128, M: 16, efConstruction: 100 };

  const pkI8 = await buildPancake({ vectors, ...base, quantized: true });
  const pkF32 = await buildPancake({ vectors, ...base, quantized: false });

  console.log(`  Pancake int8:    ${(pkI8.totalMs / 1000).toFixed(3)}s`);
  console.log(`  Pancake float32: ${(pkF32.totalMs / 1000).toFixed(3)}s`);
  console.log(`  Quantization overhead: ${((pkI8.totalMs / pkF32.totalMs - 1) * 100).toFixed(0)}%`);
  console.log('\n  If int8 is significantly slower than float32, quantization is');
  console.log('  happening per-insert instead of being amortized.');
}

/**
 * Experiment 5: Native baseline sanity check.
 * Pure JS insert-loop overhead vs WASM call overhead. If this is high,
 * the JS/WASM boundary is part of the problem.
 */
async function exp5_boundaryCost() {
  console.log('\n=== Experiment 5: JS→WASM call overhead ===');

  const vectors = genVectors(10000, 128);
  const cfg = { dim: 128, M: 16, efConstruction: 100, quantized: true };

  // Measure time for just the `index.add` call overhead by comparing
  // N=100 ef_c=1 (minimal graph work) against the main benchmark.
  const minimalVectors = genVectors(5000, 128, 99);
  const minimal = await buildPancake({
    vectors: minimalVectors, dim: 128, M: 4, efConstruction: 4, quantized: true,
  });
  const perInsertMinimal = (minimal.totalMs * 1000) / minimalVectors.length;
  console.log(`  Near-zero-work insert (M=4, ef=4, N=5000): ${perInsertMinimal.toFixed(1)}µs/insert`);
  console.log(`  This is approx. the JS→WASM call floor. If it's >5µs, consider bulk_insert.`);
}

async function main() {
  const started = now();
  console.log('Build-profiling benchmark — designed to isolate insert cost');
  console.log('='.repeat(60));

  await exp1_scalingWithN();
  await exp2_scalingWithEfC();
  await exp3_scalingWithDim();
  await exp4_quantizationCost();
  await exp5_boundaryCost();

  console.log(`\nTotal wall time: ${fmt(now() - started)}`);
  console.log('\nInterpretation cheat-sheet:');
  console.log('  Exp1 last/first ratio > log growth  → visited-set or candidate growth bug');
  console.log('  Exp2 ratio grows with ef_c          → candidate-search loop slow');
  console.log('  Exp3 ratio grows with dim           → distance SIMD weak');
  console.log('  Exp4 int8 slower than float32       → quantization not amortized');
  console.log('  Exp5 >5µs/insert at minimal work    → JS/WASM boundary overhead');
}

main().catch(e => { console.error(e); process.exit(1); });