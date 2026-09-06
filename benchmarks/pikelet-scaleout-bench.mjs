// pikelet-scaleout-bench.mjs
// Measures aggregate QPS scaling across Node worker_threads, each worker owning an
// isolated WASM instance importing the same snapshot ("scale out with copies").
//
// Usage (from the pikelet repo root):
//   node pikelet-scaleout-bench.mjs [--vectors path.bin] [--dim 384] [--synthN 30000] [--synthD 768]
//                                   [--workers 1,2,4,8] [--duration 8] [--ef 100] [--k 10] [--quantized 1]
//
// With --vectors: loads float32 vectors from a raw .bin (row-major, dim from --dim).
// Without: generates --synthN x --synthD clustered synthetic vectors (recall is not
// measured here; throughput scaling is representation-faithful either way).

import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).map((a, i, arr) =>
  a.startsWith('--') ? [a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : '1'] : null
).filter(Boolean));

const DIM = parseInt(args.dim ?? '384');
const SYNTH_N = parseInt(args.synthN ?? '30000');
const SYNTH_D = parseInt(args.synthD ?? '768');
const WORKERS = (args.workers ?? '1,2,4,8').split(',').map(Number);
const DURATION_S = parseFloat(args.duration ?? '8');
const EF = parseInt(args.ef ?? '100');
const K = parseInt(args.k ?? '10');
const QUANTIZED = (args.quantized ?? '1') !== '0';
const N_QUERIES = 512; // pool per worker, cycled

const pikeletPath = path.resolve('./pikelet.node.mjs');

function makeSynthetic(n, d) {
  const centers = 128, C = new Float32Array(centers * d);
  let seed = 42; const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff * 2 - 1;
  for (let i = 0; i < C.length; i++) C[i] = rnd();
  const out = new Float32Array(n * d);
  for (let i = 0; i < n; i++) {
    const c = (i * 2654435761 >>> 0) % centers;
    let s = 0;
    for (let j = 0; j < d; j++) { const v = C[c * d + j] + rnd() * 0.25; out[i * d + j] = v; s += v * v; }
    s = Math.sqrt(s) || 1;
    for (let j = 0; j < d; j++) out[i * d + j] /= s;
  }
  return out;
}

if (isMainThread) {
  const Pikelet = (await import(pikeletPath)).default;
  let data, dim, n;
  if (args.vectors) {
    const buf = fs.readFileSync(args.vectors);
    data = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    dim = DIM; n = Math.floor(data.length / dim);
    console.log(`data: ${n} x ${dim}D from ${args.vectors}`);
  } else {
    dim = SYNTH_D; n = SYNTH_N;
    console.log(`data: generating ${n} x ${dim}D clustered synthetic...`);
    data = makeSynthetic(n, dim);
  }

  console.log(`building index (quantized=${QUANTIZED}, M=12, efc=100)...`);
  const t0 = performance.now();
  const index = await Pikelet.create({ dim, maxElements: n, metric: 'cosine', quantized: QUANTIZED, M: 12, efConstruction: 100, efSearch: EF });
  const BATCH = 5000;
  for (let i = 0; i < n; i += BATCH) {
    const chunk = [];
    for (let j = i; j < Math.min(i + BATCH, n); j++) chunk.push(data.subarray(j * dim, (j + 1) * dim));
    index.addBatch(chunk);
  }
  console.log(`build: ${((performance.now() - t0) / 1000).toFixed(1)}s, engine memory ${(index.memory / 1e6).toFixed(1)} MB`);
  index.compact();
  const snapshot = index.export();
  console.log(`snapshot: ${(snapshot.byteLength / 1e6).toFixed(1)} MB`);
  index.dispose();

  // Query pool: perturbed base vectors (shared read-only buffer to workers)
  const queries = new Float32Array(N_QUERIES * dim);
  for (let q = 0; q < N_QUERIES; q++) {
    const src = (q * 7919) % n;
    for (let j = 0; j < dim; j++) queries[q * dim + j] = data[src * dim + j] + (((q * 31 + j) % 17) - 8) * 1e-3;
  }

  const baselineRss = process.memoryUsage().rss;
  const results = [];
  for (const W of WORKERS) {
    const workers = [];
    const ready = [];
    for (let w = 0; w < W; w++) {
      const worker = new Worker(fileURLToPath(import.meta.url), {
        workerData: {
          snapshot, queries, dim, ef: EF, k: K, quantized: QUANTIZED,
          durationMs: DURATION_S * 1000, workerId: w, pikeletPath, maxElements: n,
        },
      });
      workers.push(worker);
      ready.push(new Promise((res, rej) => {
        worker.once('message', m => m.type === 'ready' ? res(m) : rej(new Error(m.error)));
        worker.once('error', rej);
      }));
    }
    const readyInfo = await Promise.all(ready);          // all imported before anyone starts
    const importMs = readyInfo.map(r => r.importMs);
    const rssAfterSpawn = process.memoryUsage().rss;

    const done = workers.map(w => new Promise((res, rej) => {
      w.on('message', m => m.type === 'done' && res(m));
      w.on('error', rej);
    }));
    const start = performance.now();
    workers.forEach(w => w.postMessage('go'));
    const stats = await Promise.all(done);
    const wall = (performance.now() - start) / 1000;
    await Promise.all(workers.map(w => w.terminate()));

    const totalQ = stats.reduce((a, s) => a + s.count, 0);
    const qps = totalQ / wall;
    const p50s = stats.map(s => s.p50), p99s = stats.map(s => s.p99);
    results.push({ W, qps, p50: Math.max(...p50s), p99: Math.max(...p99s), importMs: Math.max(...importMs), rss: rssAfterSpawn - baselineRss });
    console.log(`workers=${W}: aggregate ${Math.round(qps)} QPS | worst p50 ${Math.max(...p50s).toFixed(2)}ms p99 ${Math.max(...p99s).toFixed(2)}ms | import(max) ${Math.max(...importMs).toFixed(0)}ms | ΔRSS ${(rssAfterSpawn - baselineRss) / 1e6 | 0}MB`);
  }

  const base = results.find(r => r.W === Math.min(...WORKERS));
  console.log('\nscaling summary (vs ' + base.W + '-worker baseline):');
  for (const r of results) {
    const ideal = r.W / base.W;
    console.log(`  ${String(r.W).padStart(2)} workers: ${(r.qps / base.qps).toFixed(2)}x  (ideal ${ideal.toFixed(0)}x, efficiency ${(100 * r.qps / base.qps / ideal).toFixed(0)}%)  p99 ${(r.p99 / base.p99).toFixed(2)}x baseline`);
  }
} else {
  // ---- worker ----
  const { snapshot, queries, dim, ef, k, quantized, durationMs, pikeletPath, maxElements } = workerData;
  try {
    const Pikelet = (await import(pikeletPath)).default;
    const t0 = performance.now();
    const index = await Pikelet.create({ dim, maxElements, metric: 'cosine', quantized, M: 12, efConstruction: 100, efSearch: ef });
    index.import(snapshot);
    const importMs = performance.now() - t0;

    const nQ = queries.length / dim;
    const qv = [];
    for (let i = 0; i < nQ; i++) qv.push(queries.subarray(i * dim, (i + 1) * dim));
    for (let i = 0; i < 100; i++) index.search(qv[i % nQ], k);   // warmup

    parentPort.postMessage({ type: 'ready', importMs });
    await new Promise(res => parentPort.once('message', res));   // barrier: start together

    const lat = [];
    let count = 0;
    const end = performance.now() + durationMs;
    while (performance.now() < end) {
      const t = performance.now();
      index.search(qv[count % nQ], k);
      lat.push(performance.now() - t);
      count++;
    }
    lat.sort((a, b) => a - b);
    parentPort.postMessage({
      type: 'done', count,
      p50: lat[lat.length >> 1] ?? 0,
      p99: lat[Math.floor(lat.length * 0.99)] ?? 0,
    });
  } catch (e) {
    parentPort.postMessage({ type: 'ready', error: String(e && e.stack || e) });
  }
}
