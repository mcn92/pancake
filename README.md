# Pancake

HNSW vector search in about 45 KB of gzipped WebAssembly (122 KB uncompressed). Runs in Node.js and browser-bundled web apps with no native dependencies.

Most ANN libraries ship as platform-specific native binaries, which means they do not work in browser tabs or JavaScript runtimes without native extensions. Pancake is a single portable WASM module built for JavaScript environments where native addons are not an option.

Pancake ships two backends: an int8 quantized backend that cuts memory ~4x with an explicit recall-ceiling tradeoff, and a full float32 backend for higher precision distances. Both use WASM SIMD acceleration. On the current 1536D DBpedia benchmark, the float32 backend is competitive with native hnswlib across the full recall range, while the int8 backend is strongest at lower recall and much lower memory. On the current 256D NYTimes benchmark, Pancake reaches slightly higher recall in the tested sweep but trails native throughput at matched recall.

Pancake is an ANN library -- it doesn't ship an embedding model. Use any embedder (sentence-transformers, OpenAI, Cohere, etc.) and feed the resulting vectors to Pancake.

## Install

```bash
npm install pancake-wasm
```

## Runtime entry points

```js
// Node.js CJS
const Pancake = require('pancake-wasm');

// Node.js ESM
import Pancake from 'pancake-wasm';

// Explicit Node entry
import Pancake from 'pancake-wasm/node';

// Browser bundlers and other runtimes that support packaged wasm asset resolution
import Pancake from 'pancake-wasm/web';
```

The `pancake-wasm/web` entry expects a bundler or runtime that can resolve the packaged
`./dist/engine.wasm` asset. This is tested with a bundled Vite + Chromium flow. For raw
no-bundler demos, see `dist/technical-demo.html`.

## Quick start

```js
import Pancake from 'pancake-wasm';

const index = await Pancake.create({
  dim: 384,
  maxElements: 100000,
  metric: 'cosine',
  quantized: true,
});

const id = index.add(new Float32Array(384)); // populate with your embedding
index.addBatch([
  new Float32Array(384), // populate with your embeddings
  new Float32Array(384),
]);

const results = index.search(new Float32Array(384), 10);
// [{ id: 0, distance: 0.12 }, ...]

index.delete(id);
index.compact();

// Persist and restore
// If you have deleted anything, compact() before export().
const snapshot = index.export();
const restored = await Pancake.create({
  dim: 384,
  maxElements: 100000,
  metric: 'cosine',
  quantized: true,
});
restored.import(snapshot);
```

If `ghostCount > 0`, `export()` throws. Call `compact()` first to produce a clean snapshot.
`import()` also requires the target index config to match the export's `dim`, `metric`, and `quantized` mode.

## API

### `await Pancake.create(opts)`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `dim` | number | required | Vector dimension |
| `maxElements` | number | `100000` | Pre-allocated capacity |
| `metric` | string | `'cosine'` | `'cosine'` or `'l2'` |
| `quantized` | boolean | `true` | Use int8 storage. Reduces memory ~4x with minimal recall loss. Set to `false` for exact float32 distances. |
| `M` | number | `16` | HNSW connectivity |
| `efConstruction` | number | `200` | Build-time beam width |
| `efSearch` | number | `100` | Query-time beam width |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add(vector)` | `number` | Insert one vector, returns its ID |
| `addBatch(vectors)` | `number[]` | Insert multiple vectors |
| `search(query, k)` | `{id, distance}[]` | k-nearest-neighbor search |
| `delete(id)` | -- | Soft-delete by ID |
| `compact()` | -- | Rebuild graph without soft-deleted entries |
| `export()` | `Uint8Array` | Serialize index state. Requires `ghostCount === 0`; call `compact()` first after deletions. |
| `import(data)` | -- | Restore a previous export |
| `dispose()` | -- | Free WASM buffers |

### Properties

| Property | Description |
|----------|-------------|
| `count` | Vectors stored (includes soft-deleted until `compact()`) |
| `ghostCount` | Soft-deleted vectors awaiting compaction |
| `ghostRatio` | `ghostCount / count` |
| `memory` | Estimated index memory in bytes (vectors + graph structure) |
| `dim` | Vector dimension |

### Backend dispatch

Pancake picks one of two HNSW backends based on the `quantized` option. This is transparent -- the API is the same regardless of backend:

| Condition | Backend | Notes |
|-----------|---------|-------|
| `quantized: true` | Int8 HNSW | Asymmetric distance: queries stay in float32, database vectors stored as int8. Preserves query-side precision at ~4x memory savings. |
| `quantized: false` | Float32 HNSW | Full precision, any dimension |

## Performance

All benchmarks are single-threaded on an AMD Ryzen 9 4900HS laptop. Pancake Int8 and Float32 are WASM; hnswlib is native C++ with SIMD.

### DBpedia-50K (50k x 1536D, L2)

Three-way comparison at OpenAI embedding dimension. M=16, efConstruction=100.

| ef_search | Pancake Int8 recall | Pancake Int8 QPS | Pancake FP32 recall | Pancake FP32 QPS | hnswlib recall | hnswlib QPS |
|----------:|--------------------:|-----------------:|--------------------:|-----------------:|---------------:|------------:|
| 10 | 86.0% | 3,752 | 87.4% | 2,994 | 81.5% | 2,452 |
| 40 | 95.4% | 1,728 | 97.2% | 1,298 | 95.7% | 1,321 |
| 100 | 97.1% | 874 | 99.2% | 659 | 98.7% | 763 |
| 200 | 97.4% | 496 | 99.7% | 380 | 99.5% | 473 |
| 500 | 97.6% | 239 | 99.96% | 178 | 99.87% | 228 |
| 800 | 97.6% | 159 | 99.98% | 121 | 99.97% | 156 |

On this 1536D L2 workload, Pancake FP32 is the strongest result: it tracks hnswlib's recall curve closely and is slightly ahead on much of the matched-recall frontier. Pancake Int8 is still attractive when memory matters or when target recall is in the low-to-mid `90%` range, but it plateaus at about `97.65%` recall while both float32 systems continue toward full recall.

Build times from this run: Pancake Int8 `129.3s`, Pancake FP32 `248.3s`, hnswlib `80.8s`. Measured index memory: Int8 `86.3 MB`, Pancake FP32 `299.3 MB`.

### NYTimes-256 (290k x 256D, cosine)

Three-way comparison. M=16, efConstruction=100.

| ef_search | Pancake Int8 recall | Pancake Int8 QPS | Pancake FP32 recall | Pancake FP32 QPS | hnswlib recall | hnswlib QPS |
|----------:|--------------------:|-----------------:|--------------------:|-----------------:|---------------:|------------:|
| 20 | 69.9% | 3,617 | 70.4% | 3,444 | 66.9% | 4,825 |
| 100 | 85.0% | 1,233 | 85.4% | 1,042 | 83.8% | 1,741 |
| 200 | 88.5% | 752 | 88.9% | 593 | 87.5% | 1,017 |
| 500 | 92.2% | 321 | 92.5% | 255 | 91.4% | 448 |
| 800 | 93.8% | 203 | 94.2% | 157 | 93.3% | 285 |

On this 256D cosine workload, Pancake reaches slightly higher recall than hnswlib at the same `ef_search`, but native hnswlib is still faster at matched recall across the overlapping range. Pancake Int8 is roughly `0.84x-0.90x` of native throughput at comparable recall points; Pancake FP32 is slower still. The recall gap suggests competitive graph quality, while the throughput gap is dominated by WASM-vs-native execution cost on lower-dimensional vectors.

### Deletion tolerance

DBpedia 5K x 1536D, cosine, int8, M=12, 100 held-out queries, brute-force ground truth recomputed against the live set at each step.

| Ghost % | Live Vectors | Recall@10 | p50 Latency |
|---:|---:|---:|---:|
| 0% | 4,900 | 97.0% | 1.16ms |
| 30% | 3,430 | 96.9% | 0.92ms |
| 50% | 2,450 | 96.9% | 0.73ms |
| 70% | 1,470 | 96.0% | 0.53ms |
| 90% | 490 | 84.1% | 0.27ms |

Recall holds within about 1 point of baseline through 70% ghosts. Search latency drops as ghosts accumulate (fewer live nodes to visit). The cliff beyond 90% is graph disconnection: the live subgraph is no longer well-connected enough to preserve recall. Compaction can be deferred until the main thread is idle.

### Reproducing

Benchmark scripts are in `benchmarks/`. A shared runner and parameter parser are
included, so you can list and run them with:

```bash
npm run bench -- --list
npm run bench -- benchmark_dbpedia_50k --m 16 --ef-construction 50 --ef-search 100
```

Results depend on dimension, corpus size, HNSW parameters, and hardware.

## Snapshot-first Worker deployment

Pancake can also run inside Cloudflare Workers, but the right mental model is
**snapshot search at the edge**, not **a durable mutable vector database inside
one long-lived isolate**.

The reference Worker in [`examples/worker/`](examples/worker/) keeps a Pancake
index warm in process when possible and restores snapshots from R2 on cold
start. That makes it a good fit for:

- read-heavy semantic search
- modest-sized indexes that fit comfortably inside Worker memory limits
- periodic snapshot rebuilds or explicit import/export admin flows
- edge serving where cold restore is acceptable

It is a weaker fit for:

- high-write online mutation as the primary production path
- strict cross-request write durability inside plain Worker memory
- “one always-live authoritative index instance” semantics

The Worker example still exposes add/delete/compact routes because they are
useful for demos, admin tooling, and local validation, but the production story
is snapshot-backed search rather than a fully stateful ANN service.

See [`examples/worker/README.md`](examples/worker/README.md) for:

- endpoint documentation
- local development and deployment steps
- environment variables and Wrangler configuration
- memory, cold-start, and persistence tradeoffs

## When to use Pancake

Pancake makes sense when native ANN libraries aren't an option:

- **Browser-based search** -- client-side retrieval without a server round-trip
- **Portable applications** -- one artifact, same behavior across every JavaScript runtime
- **Node.js without native addons** -- in-process ANN without native binary packaging
- **Small to medium indexes** -- in-process search without external infrastructure

If you have a server where native dependencies are fine, Faiss, hnswlib, or USearch will generally be faster -- especially at low dimensions where native SIMD width matters most. On the current 1536D DBpedia run, Pancake's float32 backend is competitive with native hnswlib and the int8 backend offers a strong memory/speed tradeoff up to about `97.6%` recall. Pancake exists for the places native libraries can't go -- and on some high-dimensional workloads, it is still competitive even where they can.

## How it works

### Quantization

Stored vectors use row-wise affine quantization. Each vector gets its own scale and offset derived from its own min/max:

```
scale  = (max - min) / 255
offset = min
q[i]   = uint8(clamp((x[i] - offset) / scale, 0, 255))
```

This preserves per-vector dynamic range better than global quantization, at the cost of 8 bytes of overhead per vector for the scale and offset.

The quantized backend uses asymmetric distance: queries stay in float32 while database vectors are dequantized from int8 on the fly. This preserves full query-side precision -- quantization error only affects the stored side.

### HNSW

Standard HNSW graph search with one modification: the neighbor selection heuristic uses **backfill** (the "keep pruned connections" option from the HNSW paper). After the diversity heuristic rejects candidates, the closest rejected candidates are added back until M slots are filled. This guarantees minimum connectivity and prevents weakly-connected nodes in clustered data, where the diversity heuristic can otherwise reject most same-cluster candidates.

`M` controls graph connectivity, `efConstruction` controls build quality, `efSearch` controls query quality. Higher values improve recall at the cost of build or query speed.

### Handle-based C ABI

The WASM module exposes a handle-based C API: `pancake_init` returns an opaque handle, and all operations take a handle as the first argument. This enables multiple independent indexes per WASM instance. The JavaScript wrapper (`pancake-core.js`) manages handles, ID translation, and a binary envelope format for export/import.

## Examples

- `examples/cli/build_index.js` -- build an HNSW index from precomputed embeddings
- `examples/demo/technical_demo_cli.js` -- interactive local proof/demo REPL
- `examples/demo/technical_demo_worker.js` -- interactive Worker-targeted proof/demo REPL
- `dist/technical-demo.html` -- browser proof/demo page with latency and stress views
- `examples/browser-vite/` -- minimal bundled browser consumer fixture used by `npm run test:browser`
- `examples/demo-portable-search/` -- packaged browser semantic-search demo using exported snapshots
- `examples/demo-ecommerce/` -- browser demo built on top of the packaged web entry
- `examples/worker/` -- reference Cloudflare Worker deployment built on top of Pancake

## Architecture

See [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) for a detailed design document covering the C++ engine, WASM compilation, JavaScript wrapper, serialization formats, and the reference Worker deployment.

## Compatibility

- **Node.js** >= 16 (uses `WebAssembly`, `performance.now`)
- **Browsers**: any browser with [WebAssembly SIMD](https://caniuse.com/wasm-simd) (Chrome 91+, Firefox 89+, Safari 16.4+) plus a bundler/runtime that supports importing the packaged `engine.wasm`
- **TypeScript**: type definitions included (`pancake.d.ts`)

## Tests

```bash
node run_tests.js
```

This covers the core API, Node CJS and ESM entrypoints, browser-style
`instantiateWasm` loading, held-out brute-force recall oracles, filtered-search
oracles, golden search-output checks, and determinism checks.

For a real bundled browser-consumer check of `import Pancake from 'pancake-wasm/web'`, run:

```bash
npm run test:browser
```

This starts a minimal Vite app in `examples/browser-vite/` and verifies the published web entry
in Chromium via Playwright.

For SIMD parity coverage, run:

```bash
npm run test:simd
```

For the Worker reference deployment, run:

```bash
node test/test_worker_features.js
```

Current core suite status on this tree: **755 passed, 0 failed**.

## Building from source

The npm package ships prebuilt WASM artifacts. Rebuild only if you're modifying the C++ engine:

```bash
./build.sh
```

Requires an Emscripten toolchain with WASM SIMD support. Produces `dist/engine.js` (~18 KB) and `dist/engine.wasm` (~113 KB).

## Tradeoffs

- **Single-threaded by design.** Pancake is meant for runtimes where background threads are unavailable or unreliable. That is a deployment advantage, not just a limitation.
- **Quantization is a real trade.** The int8 path wins on memory and can win on throughput at lower recall, but on the current 1536D DBpedia benchmark it tops out around `97.6%` recall. Use float32 when you need `>99%`.
- **Compaction is rebuild-based.** Deletes are soft deletes. Compaction rewrites the graph rather than patching edges in place. That keeps behavior predictable and avoids relying on background maintenance threads.
- **Index instances are not a shared-memory concurrency primitive.** Treat a Pancake index like ordinary mutable in-process state: safe within one JavaScript thread/event loop, but not something to share concurrently across Node worker threads or isolates without your own coordination.
- **Workers are best used as snapshot-serving search frontends.** In a Cloudflare Worker, in-memory state is a warm cache, not durable authority. Persist snapshots explicitly and treat isolate reuse as opportunistic.
- **This is an index, not an embedding stack.** Pancake does vector search only. Bring your own embedding pipeline.

## License

Apache 2.0
