#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const DEFAULTS = Object.freeze({
  out: path.join(process.cwd(), 'dist', 'vectors.bin'),
  count: 5000,
  dims: 384,
  clusters: 32,
  spread: 0.08,
  seed: 0x504E434B,
});

function parsePositiveInteger(value, flag) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`${flag} must be a positive integer`);
  }
  return parsed;
}

function parseArgs(argv) {
  const options = { ...DEFAULTS };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    if (i + 1 >= argv.length) throw new Error(`Missing value for ${flag}`);
    const value = argv[++i];
    switch (flag) {
    case '--out':
      options.out = path.resolve(value);
      break;
    case '--count':
      options.count = parsePositiveInteger(value, flag);
      break;
    case '--dims':
      options.dims = parsePositiveInteger(value, flag);
      break;
    case '--clusters':
      options.clusters = parsePositiveInteger(value, flag);
      break;
    case '--spread':
      options.spread = Number(value);
      if (!Number.isFinite(options.spread) || options.spread <= 0) {
        throw new Error('--spread must be a positive finite number');
      }
      break;
    case '--seed':
      options.seed = parsePositiveInteger(value, flag) >>> 0;
      break;
    default:
      throw new Error(`Unknown option: ${flag}`);
    }
  }
  if (options.clusters > options.count) {
    throw new Error('--clusters cannot exceed --count');
  }
  return options;
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return function random() {
    state = (state + 0x6D2B79F5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 0x100000000;
  };
}

function gaussian(random) {
  let u = 0;
  let v = 0;
  while (u === 0) u = random();
  while (v === 0) v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function normalize(values, offset, dims) {
  let normSq = 0;
  for (let d = 0; d < dims; d++) {
    const value = values[offset + d];
    normSq += value * value;
  }
  const scale = 1 / Math.sqrt(normSq);
  for (let d = 0; d < dims; d++) values[offset + d] *= scale;
}

export function generateClusteredVectors(options) {
  const { count, dims, clusters, spread, seed } = options;
  const random = mulberry32(seed);
  const centers = new Float32Array(clusters * dims);
  for (let cluster = 0; cluster < clusters; cluster++) {
    const offset = cluster * dims;
    for (let d = 0; d < dims; d++) centers[offset + d] = gaussian(random);
    normalize(centers, offset, dims);
  }

  const vectors = new Float32Array(count * dims);
  for (let row = 0; row < count; row++) {
    const cluster = row % clusters;
    const centerOffset = cluster * dims;
    const rowOffset = row * dims;
    for (let d = 0; d < dims; d++) {
      vectors[rowOffset + d] = centers[centerOffset + d] + spread * gaussian(random);
    }
    normalize(vectors, rowOffset, dims);
  }
  return vectors;
}

export function writeDemoVectors(options) {
  const vectors = generateClusteredVectors(options);
  fs.mkdirSync(path.dirname(options.out), { recursive: true });
  const bytes = Buffer.from(vectors.buffer, vectors.byteOffset, vectors.byteLength);
  fs.writeFileSync(options.out, bytes);

  const manifest = {
    format: 'raw-float32-le',
    generator: 'clustered-unit-vectors-v1',
    count: options.count,
    dims: options.dims,
    clusters: options.clusters,
    spread: options.spread,
    seed: options.seed >>> 0,
    bytes: bytes.byteLength,
  };
  const manifestPath = `${options.out}.manifest.json`;
  fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  return { manifest, manifestPath };
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    const options = parseArgs(process.argv.slice(2));
    const { manifest, manifestPath } = writeDemoVectors(options);
    console.log(`Wrote ${manifest.count.toLocaleString()} clustered ${manifest.dims}D vectors`);
    console.log(`  data:     ${options.out} (${manifest.bytes.toLocaleString()} bytes)`);
    console.log(`  manifest: ${manifestPath}`);
  } catch (error) {
    console.error(error && error.message ? error.message : String(error));
    process.exit(1);
  }
}
