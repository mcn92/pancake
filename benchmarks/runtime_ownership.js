#!/usr/bin/env node
'use strict';

/**
 * Runtime ownership benchmark
 *
 * Quantifies Model C: one cached compiled module plus one isolated WASM
 * instance per public index.
 *
 * Usage:
 *   node benchmarks/runtime_ownership.js
 *   node benchmarks/runtime_ownership.js --indexes 8 --warm-reps 20 --dim 384
 */

const { performance } = require('perf_hooks');
const Pancake = require('../pancake.js');

const args = process.argv.slice(2);
function intArg(name, fallback) {
  const index = args.indexOf(`--${name}`);
  if (index < 0) return fallback;
  const value = Number.parseInt(args[index + 1], 10);
  if (!Number.isInteger(value) || value <= 0) throw new Error(`--${name} must be a positive integer`);
  return value;
}

const DIM = intArg('dim', 384);
const INDEXES = intArg('indexes', 8);
const WARM_REPS = intArg('warm-reps', 20);
const CAPACITY = intArg('capacity', 32);

const options = {
  dim: DIM,
  maxElements: CAPACITY,
  metric: 'l2',
  quantized: true,
};

function percentile(values, p) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))];
}

function formatMs(value) {
  return `${value.toFixed(3)} ms`;
}

function formatMb(value) {
  return `${(value / 1024 / 1024).toFixed(2)} MB`;
}

async function timedCreate() {
  const start = performance.now();
  const index = await Pancake.create(options);
  return { index, ms: performance.now() - start };
}

async function main() {
  const cold = await timedCreate();
  cold.index.dispose();

  const warmTimes = [];
  for (let i = 0; i < WARM_REPS; i++) {
    const warm = await timedCreate();
    warmTimes.push(warm.ms);
    warm.index.dispose();
  }

  const rssBefore = process.memoryUsage().rss;
  const concurrentStart = performance.now();
  const indexes = await Promise.all(Array.from({ length: INDEXES }, () => Pancake.create(options)));
  const concurrentMs = performance.now() - concurrentStart;
  const rssDelta = Math.max(0, process.memoryUsage().rss - rssBefore);

  try {
    const logicalBytes = indexes.reduce((sum, index) => sum + index.memoryUsage.logicalIndexBytes, 0);
    const wasmHeapBytes = indexes.reduce((sum, index) => sum + index.memoryUsage.wasmHeapBytes, 0);
    const distinctHeaps = new Set(indexes.map((index) => index._e.HEAPU8.buffer)).size;

    console.log('Runtime ownership benchmark');
    console.log(`dim=${DIM} capacity=${CAPACITY} indexes=${INDEXES} warm_reps=${WARM_REPS}`);
    console.log(`cold create:             ${formatMs(cold.ms)}`);
    console.log(`warm create p50:         ${formatMs(percentile(warmTimes, 0.50))}`);
    console.log(`warm create p95:         ${formatMs(percentile(warmTimes, 0.95))}`);
    console.log(`${INDEXES} concurrent creates:   ${formatMs(concurrentMs)}`);
    console.log(`distinct WASM heaps:     ${distinctHeaps}/${INDEXES}`);
    console.log(`logical index bytes:     ${formatMb(logicalBytes)}`);
    console.log(`WASM heap bytes:         ${formatMb(wasmHeapBytes)}`);
    console.log(`process RSS delta:       ${formatMb(rssDelta)}`);

    if (distinctHeaps !== INDEXES) throw new Error('Indexes unexpectedly share WASM heaps');
  } finally {
    for (const index of indexes) index.dispose();
  }
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
