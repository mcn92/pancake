#!/usr/bin/env node
/**
 * Build a pancake index from vectors.bin and export to a file ready for R2.
 *
 * Usage:
 *   node build-and-export-index.js \
 *     --vectors /mnt/c/pancake1.0.0/dist/vectors.bin \
 *     --dims 384 \
 *     --count 5000 \
 *     --out /tmp/pancake-index.bin
 *
 * Defaults match the technical_demo_worker.js dataset (5000 x 384D from vectors.bin).
 *
 * After this finishes, upload to R2 with:
 *   npx wrangler r2 object put pancake-indexes/pancake-index.bin --file=/tmp/pancake-index.bin --remote
 *
 * The cloudflare worker will then auto-restore from R2 on its next cold start.
 */

const fs = require('node:fs');
const path = require('node:path');

// ------------------ args ------------------
function parseArgs(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i++) {
    const k = argv[i];
    if (k.startsWith('--')) {
      const v = argv[i + 1];
      out[k.slice(2)] = v;
      i++;
    }
  }
  return out;
}
const args = parseArgs(process.argv);

const ENGINE_DIR  = args.engineDir || '/mnt/c/pancake1.0.0/dist';
const VECTORS_PATH = args.vectors  || path.join(ENGINE_DIR, 'vectors.bin');
const DIMS         = parseInt(args.dims  || '384', 10);
const COUNT        = parseInt(args.count || '5000', 10);
const OUT_PATH     = args.out      || '/tmp/pancake-index.bin';

// init params (mirror worker.js restore-path defaults: int8 backend, M=8, efC=150, efS=100)
const M             = parseInt(args.M  || '8',   10);
const EF_CONSTR     = parseInt(args.efC || '150', 10);
const EF_SEARCH     = parseInt(args.efS || '100', 10);
const MAX_ELEMENTS  = parseInt(args.maxElements || String(Math.max(COUNT, 1_000_000)), 10);

console.log(`[config] vectors=${VECTORS_PATH}`);
console.log(`[config] dims=${DIMS}, count=${COUNT}, maxElements=${MAX_ELEMENTS}`);
console.log(`[config] M=${M}, efC=${EF_CONSTR}, efS=${EF_SEARCH}`);
console.log(`[config] out=${OUT_PATH}`);

// ------------------ load vectors.bin ------------------
//
// Worker.js / technical_demo_worker.js treat vectors.bin as a flat float32 array.
// COUNT vectors of DIMS floats each, little-endian.
const expectedBytes = COUNT * DIMS * 4;
const stat = fs.statSync(VECTORS_PATH);
if (stat.size < expectedBytes) {
  console.error(`vectors.bin is ${stat.size} bytes, expected at least ${expectedBytes}`);
  process.exit(1);
}
const vecBuf = fs.readFileSync(VECTORS_PATH).slice(0, expectedBytes);
const vectors = new Float32Array(vecBuf.buffer, vecBuf.byteOffset, COUNT * DIMS);
console.log(`[load] read ${COUNT} vectors x ${DIMS}D from ${VECTORS_PATH}`);

// ------------------ load wasm engine ------------------
async function main() {
  const engineJsPath = path.join(ENGINE_DIR, 'engine.js');
  const wasmPath     = path.join(ENGINE_DIR, 'engine.wasm');

  // engine.js exports a default factory function via `var P = (() => { ... })()`.
  // Easiest way to get it in node is to require() and inspect, but it's an ESM-style
  // IIFE. We can require it as commonjs since emscripten emits a CJS wrapper when
  // it detects node. Try require first.
  let P;
  try {
    P = require(engineJsPath);
    if (typeof P !== 'function' && P && typeof P.default === 'function') P = P.default;
  } catch (e) {
    // Fallback: read and eval (engine.js leaks `P` into the surrounding scope via `var P=...`)
    const code = fs.readFileSync(engineJsPath, 'utf8');
    const wrapped = `(function(){${code}\nreturn P;})()`;
    P = eval(wrapped);
  }
  if (typeof P !== 'function') {
    throw new Error(`Could not obtain engine factory from ${engineJsPath}; got ${typeof P}`);
  }

  const wasmBinary = fs.readFileSync(wasmPath);

  console.log(`[wasm] instantiating engine...`);
  const engine = await P({ wasmBinary });

  // sanity check exports
  const required = [
    '_pancake_init', '_pancake_add', '_pancake_count', '_pancake_export',
    '_emsc_malloc', '_emsc_free'
  ];
  for (const fn of required) {
    if (typeof engine[fn] !== 'function') {
      throw new Error(`wasm export missing: ${fn}`);
    }
  }

  // ------------------ init index ------------------
  // signature observed in worker.js: _pancake_init(dims, maxElements, ?, ?, M, efC, efS)
  // Args 3 and 4 are backend flags; (1, 1) selects int8 in the worker's restore path.
  const handle = engine._pancake_init(DIMS, MAX_ELEMENTS, 1, 1, M, EF_CONSTR, EF_SEARCH);
  if (handle === 0xFFFFFFFF) throw new Error('_pancake_init returned -1');
  console.log(`[init] handle=${handle}`);

  // ------------------ insert vectors ------------------
  // Allocate a single dim-sized buffer and reuse it for each insert.
  const vecBytes = DIMS * 4;
  const vecPtr = engine._emsc_malloc(vecBytes);
  if (!vecPtr) throw new Error('alloc failed');

  const t0 = Date.now();
  let lastReport = t0;
  for (let i = 0; i < COUNT; i++) {
    // copy vector i into wasm memory
    engine.HEAPF32.set(
      vectors.subarray(i * DIMS, (i + 1) * DIMS),
      vecPtr / 4
    );
    const id = engine._pancake_add(handle, vecPtr);
    if (id === 0xFFFFFFFF || id < 0) {
      throw new Error(`_pancake_add failed at i=${i}`);
    }
    if (i > 0 && i % 1000 === 0) {
      const now = Date.now();
      const rate = ((i + 1) / ((now - t0) / 1000)).toFixed(0);
      console.log(`[insert] ${i}/${COUNT} (${rate} vec/s)`);
      lastReport = now;
    }
  }
  engine._emsc_free(vecPtr);
  const buildMs = Date.now() - t0;
  const finalCount = engine._pancake_count(handle);
  console.log(`[insert] done: ${finalCount} vectors in ${(buildMs / 1000).toFixed(1)}s ` +
              `(${(finalCount / (buildMs / 1000)).toFixed(0)} vec/s)`);

  // ------------------ export ------------------
  // pattern from worker.js exportBinary():
  //   sizePtr = malloc(8); dataPtr = _pancake_export(handle, sizePtr);
  //   read 4-byte LE size at sizePtr; bytes at dataPtr.
  const sizePtr = engine._emsc_malloc(8);
  if (!sizePtr) throw new Error('alloc sizePtr failed');
  const dataPtr = engine._pancake_export(handle, sizePtr);
  if (!dataPtr) throw new Error('_pancake_export returned null');

  const size =
    engine.HEAPU8[sizePtr]            |
    (engine.HEAPU8[sizePtr + 1] << 8) |
    (engine.HEAPU8[sizePtr + 2] << 16) |
    (engine.HEAPU8[sizePtr + 3] << 24);

  if (size <= 0) {
    throw new Error(`export produced size=${size}`);
  }

  const bytes = Buffer.from(engine.HEAPU8.subarray(dataPtr, dataPtr + size));
  fs.writeFileSync(OUT_PATH, bytes);
  console.log(`[export] wrote ${size} bytes (${(size / 1024 / 1024).toFixed(2)} MB) to ${OUT_PATH}`);

  engine._emsc_free(sizePtr);
  // dataPtr is owned by the engine; don't free it here.

  // ------------------ summary ------------------
  console.log('');
  console.log('done. next:');
  console.log(`  npx wrangler r2 object put pancake-indexes/pancake-index.bin --file=${OUT_PATH} --remote`);
}

main().catch(e => {
  console.error('FATAL:', e);
  process.exit(1);
});