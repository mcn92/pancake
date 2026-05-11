# Pancake

HNSW vector search in 43 KB of gzipped WebAssembly (113 KB uncompressed). Runs in Node.js, browsers, and Cloudflare Workers with no native dependencies.

Most ANN libraries ship as platform-specific native binaries, which means they don't work in edge runtimes, browser tabs, or any JavaScript environment without native extensions. Pancake is a single portable WASM module that runs wherever WebAssembly runs.

Pancake ships two backends: an int8 quantized backend that cuts memory ~4x with minimal recall loss, and a full float32 backend for higher precision distances. Both use WASM SIMD acceleration. At high dimensions (1536D), the int8 backend is faster than native hnswlib at low-to-mid recall because quantized vectors fit in CPU cache where float32 vectors don't. The float32 backend matches or exceeds native throughput across the full recall range. At 256D, Pancake Int8 runs at 93-100% of native throughput at matched recall.

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

// Browser, Cloudflare Workers, other V8 isolates
import Pancake from 'pancake-wasm/web';
```

## Quick start

```js
import Pancake from 'pancake-wasm';

const index = await Pancake.create({
  dim: 384,
  maxElements: 100000,
  metric: 'cosine',
  quantized: true,
});

const id = index.add(new Float32Array(384));
index.addBatch([new Float32Array(384), new Float32Array(384)]);

const results = index.search(new Float32Array(384), 10);
// [{ id: 0, distance: 0.12 }, ...]

index.delete(id);
index.compact();

// Persist and restore
const snapshot = index.export();
const restored = await Pancake.create({ dim: 384, maxElements: 100000 });
restored.import(snapshot);
```

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
| `export()` | `Uint8Array` | Serialize index state |
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

Three-way comparison at OpenAI embedding dimension. M=16, efConstruction=50.

| ef_search | Pancake Int8 recall | Pancake Int8 QPS | Pancake FP32 recall | Pancake FP32 QPS | hnswlib recall | hnswlib QPS |
|----------:|--------------------:|-----------------:|--------------------:|-----------------:|---------------:|------------:|
| 10 | 84.1% | 4,877 | 86.5% | 3,584 | 80.0% | 3,049 |
| 40 | 94.5% | 2,171 | 96.5% | 1,562 | 94.9% | 1,707 |
| 100 | 96.7% | 1,061 | 98.8% | 790 | 98.2% | 993 |
| 200 | 97.3% | 623 | 99.4% | 466 | 99.2% | 608 |
| 500 | 97.6% | 295 | 99.8% | 221 | 99.8% | 300 |

At 1536D, Pancake Int8 is the fastest option at low-to-mid recall because quantized vectors fit in CPU cache where float32 vectors don't. The tradeoff is a recall ceiling around 97.6% where quantization noise dominates. Pancake Float32 matches hnswlib's recall curve almost exactly, running within 95-100% of native throughput.

Build times: Pancake Int8 87s, Pancake FP32 160s, hnswlib 42s. Index memory: Int8 80 MB, FP32 and hnswlib 299 MB.

### NYTimes-256 (290k x 256D, cosine)

Pancake Int8 vs native hnswlib. M=16, efConstruction=150.

| ef_search | Pancake Int8 recall | Pancake Int8 QPS | hnswlib recall | hnswlib QPS |
|----------:|--------------------:|-----------------:|---------------:|------------:|
| 20 | 73.4% | 4,883 | 69.7% | 5,827 |
| 100 | 87.1% | 1,569 | 85.7% | 2,007 |
| 200 | 89.8% | 917 | 88.8% | 1,157 |
| 500 | 93.2% | 387 | 92.5% | 485 |
| 800 | 94.6% | 244 | 94.2% | 324 |

Pancake reaches 2-3 percentage points higher recall at the same efSearch. At matched recall, Pancake runs at 93-100% of native throughput -- the gap reflects the cost of WASM vs native SIMD, not an algorithmic difference. Pancake Int8 reaches 94.6% recall where hnswlib tops out at 94.2%.

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

Benchmark scripts are in `benchmarks/`. Results depend on dimension, corpus size, HNSW parameters, and hardware.

## Cloudflare Workers

Pancake was designed with edge workers in mind. The Worker *is* the vector search service -- no external ANN backend, no database round-trip, no cold-start connection pool. The index runs in-process and serves queries at the edge.

The included Worker (`examples/worker/`) is a complete vector search HTTP API with optional bearer-token auth, per-IP rate limiting, and automatic persistence to Cloudflare R2.

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /init` | Create an index (`{ dims, maxElements, M?, efConstruction?, efSearch?, vectors? }`) |
| `POST /add` | Insert a vector (`{ vector: float[] }`) |
| `POST /add_batch` | Insert multiple vectors (`{ vectors: float[][] }`) |
| `POST /delete` | Soft-delete by ID (`{ id: number }`) |
| `POST /compact` | Rebuild graph without deleted entries |
| `POST /search` | k-NN search (`{ query: float[], k?, ef? }`) |
| `GET /export` | Serialize index to binary blob |
| `POST /import` | Restore from binary (`?dims=N` required) |
| `GET /stats` | Index count, memory, ghost ratio |
| `GET /health` | Health check (public, no auth required) |

### Running locally

```bash
cd examples/worker
npx wrangler dev --port 8787
```

### Deploying

```bash
cd examples/worker
wrangler r2 bucket create pancake-indexes
wrangler deploy
```

See `examples/worker/wrangler.toml` for the full configuration. Key env vars: `API_KEY` (bearer auth), `ALLOWED_ORIGIN` (CORS), `RATE_LIMIT_RPM` (per-IP rate limit).

### Deployment notes

**Memory.** Workers have a 128 MB memory ceiling per isolate. Rough formula for quantized index memory:

```
~(dim + 8 + 7 * M) * num_vectors bytes
```

At M=16 this is `(dim + 120)` bytes per vector. Examples: 30k x 256D = 10 MB, 200k x 384D = 100 MB. The fp32 backend uses `(4*dim + 8 + 7*M)` bytes per vector -- roughly 4x more.

**CPU time.** Workers paid plan allows 50ms CPU per request (free tier: 10ms). Search comfortably fits within both tiers. Heavy operations (`/import`, `/compact`, `/add_batch` with large batches) can exceed free-tier limits on larger indexes.

**Cold starts and R2 restore.** Worker isolates are not persistent. On cold start, the Worker fetches the index from R2 and deserializes it lazily on the first request. For a large index, the first request after idle will be slow (1-2 seconds).

**Persistence.** The Worker debounces R2 writes with a 2-second timer and uses `ctx.waitUntil()` to complete writes after the response is sent. If the isolate is terminated before the timer fires, the last ~2 seconds of mutations may be lost. For workflows where every write must be durable, call `/export` explicitly after critical mutations.

**Rate limiting** is in-memory per-isolate. Each isolate tracks its own sliding-window counter, so the effective rate across multiple isolates is approximately `limit * number_of_isolates`.

## When to use Pancake

Pancake makes sense when native ANN libraries aren't an option:

- **Edge workers** -- the Worker *is* the search service; no separate vector database
- **Browser-based search** -- client-side retrieval without a server round-trip
- **Portable applications** -- one artifact, same behavior across every JavaScript runtime
- **Small to medium indexes** -- in-process search without external infrastructure

If you have a server where native dependencies are fine, Faiss, hnswlib, or USearch will generally be faster -- especially at low dimensions where native SIMD width matters most. At high dimensions (1536D), Pancake's float32 backend matches native hnswlib throughput, and the int8 backend beats it at low-to-mid recall. Pancake exists for the places native libraries can't go -- and at high dimensions, it holds its own even where they can.

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

- `examples/worker/` -- reference Cloudflare Worker with persistence, auth, and rate limiting
- `examples/cli/build_index.js` -- build an HNSW index from precomputed embeddings (see [QUICKSTART.md](QUICKSTART.md))
- `examples/demo/technical_demo_cli.js` -- interactive REPL exercising the full API against 5,000 precomputed 384D embeddings
- `examples/demo/technical_demo_worker.js` -- the same proof suite against a running Worker
- `dist/technical-demo.html` -- in-browser version of the proof suite with live latency chart and adversarial stress tests

## Architecture

See [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) for a detailed design document covering the C++ engine, WASM compilation, JavaScript wrapper, serialization formats, and Worker deployment.

## Compatibility

- **Node.js** >= 16 (uses `WebAssembly`, `performance.now`)
- **Browsers**: any browser with [WebAssembly SIMD](https://caniuse.com/wasm-simd) (Chrome 91+, Firefox 89+, Safari 16.4+)
- **Cloudflare Workers**: tested with Wrangler >= 4.x
- **TypeScript**: type definitions included (`pancake.d.ts`)

## Tests

```bash
node run_tests.js
```

213 assertions covering insertion, search, deletion, compaction, export/import, batch operations, quantized and float32 modes, error paths, edge cases, and metric correctness.

## Building from source

The npm package ships prebuilt WASM artifacts. Rebuild only if you're modifying the C++ engine:

```bash
./build.sh
```

Requires an Emscripten toolchain with WASM SIMD support. Produces `dist/engine.js` (~18 KB) and `dist/engine.wasm` (~113 KB).

## Tradeoffs

- **Single-threaded by design.** Pancake is meant for runtimes where background threads are unavailable or unreliable. That is a deployment advantage, not just a limitation.
- **Quantization is a real trade.** The int8 path wins on memory and often on throughput, but tops out around 97.6% recall on 1536D benchmarks. Use float32 when you need >99%.
- **Compaction is rebuild-based.** Deletes are soft deletes. Compaction rewrites the graph rather than patching edges in place. That keeps behavior predictable and avoids relying on background maintenance threads.
- **This is an index, not an embedding stack.** Pancake does vector search only. Bring your own embedding pipeline.

## License

Apache 2.0
