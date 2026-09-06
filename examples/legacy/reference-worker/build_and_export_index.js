#!/usr/bin/env node
'use strict';

/**
 * Build a Pikelet snapshot from vectors.bin for upload to R2.
 *
 * Usage:
 *   node build_and_export_index.js \
 *     --vectors ./vectors.bin \
 *     --dims 384 \
 *     --count 5000 \
 *     --out /tmp/pikelet-index.pnck
 */

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

function parseArgs(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i++) {
    const key = argv[i];
    if (!key.startsWith('--')) continue;
    out[key.slice(2)] = argv[i + 1];
    i++;
  }
  return out;
}

function positiveInt(value, name) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

async function loadPikelet() {
  const url = pathToFileURL(path.join(__dirname, '..', '..', 'pikelet.node.mjs')).href;
  const mod = await import(url);
  return mod.default;
}

function readVectors(filePath, dims, count) {
  const expectedBytes = count * dims * 4;
  const stat = fs.statSync(filePath);
  if (stat.size < expectedBytes) {
    throw new Error(`vectors.bin is ${stat.size} bytes, expected at least ${expectedBytes}`);
  }

  const buffer = fs.readFileSync(filePath).subarray(0, expectedBytes);
  const raw = new Float32Array(buffer.buffer, buffer.byteOffset, count * dims);
  const vectors = new Array(count);
  for (let i = 0; i < count; i++) {
    vectors[i] = raw.subarray(i * dims, (i + 1) * dims);
  }
  return vectors;
}

async function main() {
  const args = parseArgs(process.argv);
  const engineDir = args.engineDir || path.resolve(__dirname, '..', '..', 'dist');
  const vectorsPath = args.vectors || path.join(engineDir, 'vectors.bin');
  const dims = positiveInt(args.dims || '384', '--dims');
  const count = positiveInt(args.count || '5000', '--count');
  const outPath = args.out || path.join(os.tmpdir(), 'pikelet-index.pnck');
  const M = positiveInt(args.M || '8', '--M');
  const efConstruction = positiveInt(args.efC || args.efConstruction || '150', '--efC');
  const efSearch = positiveInt(args.efS || args.efSearch || '100', '--efS');
  const maxElements = positiveInt(args.maxElements || String(Math.max(count, 1_000_000)), '--maxElements');

  console.log(`[config] vectors=${vectorsPath}`);
  console.log(`[config] dims=${dims}, count=${count}, maxElements=${maxElements}`);
  console.log(`[config] M=${M}, efConstruction=${efConstruction}, efSearch=${efSearch}`);
  console.log(`[config] out=${outPath}`);

  const vectors = readVectors(vectorsPath, dims, count);
  const Pikelet = await loadPikelet();
  const index = await Pikelet.create({
    dim: dims,
    maxElements,
    metric: 'cosine',
    quantized: true,
    M,
    efConstruction,
    efSearch
  });

  try {
    const t0 = Date.now();
    const ids = index.addBatch(vectors);
    const buildSec = (Date.now() - t0) / 1000;
    console.log(`[insert] ${ids.length} vectors in ${buildSec.toFixed(1)}s (${(ids.length / buildSec).toFixed(0)} vec/s)`);

    const snapshot = index.export();
    fs.writeFileSync(outPath, snapshot);
    console.log(`[export] wrote ${snapshot.byteLength} bytes (${(snapshot.byteLength / 1024 / 1024).toFixed(2)} MB) to ${outPath}`);
  } finally {
    index.dispose();
  }

  console.log('');
  console.log('done. next:');
  console.log(`  npx wrangler r2 object put pikelet-indexes/pikelet-index.pnck --file=${outPath} --remote`);
}

main().catch(error => {
  console.error('FATAL:', error && error.message ? error.message : error);
  process.exit(1);
});
