# Pancake

HNSW vector search in about 45 KB of gzipped WebAssembly (122 KB uncompressed). Runs in Node.js, browser-bundled web apps, and Cloudflare Workers with no native dependencies in the default package path.

Most ANN libraries ship as platform-specific native binaries, which means they do not work in browser tabs or JavaScript runtimes without native extensions. Pancake's primary package is a single portable WASM module built for JavaScript environments where native addons are not an option.

Pancake ships two backends in the WASM engine: an int8 quantized backend that cuts memory by about `3.5x` on the current 1536D runs, with roughly a 1-2 point recall-ceiling tradeoff versus float32, and a full float32 backend for higher precision distances. Both use WASM SIMD acceleration. The repo also includes an experimental native Node addon with AVX2 distance kernels for direct native-vs-WASM comparisons and local benchmarks.

Pancake is an ANN library -- it doesn't ship an embedding model. Use any embedder (sentence-transformers, OpenAI, Cohere, etc.) and feed the resulting vectors to Pancake.

## Install

```bash
npm install pancake-wasm
```

The native addon under `native/` is not part of the npm package. It is an
opt-in benchmarking tool used in this repo to separate runtime overhead from
graph quality.

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

The core package export is the raw engine snapshot. The Cloudflare Worker example wraps that
payload in a small `WRK1` envelope so it can persist Worker-specific metadata such as
`maxElements`, init params, and external/internal ID mappings.

## API

### `await Pancake.create(opts)`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `dim` | number | required | Vector dimension |
| `maxElements` | number | `100000` | Pre-allocated capacity |
| `metric` | string | `'cosine'` | `'cosine'` or `'l2'` |
| `quantized` | boolean | `true` | Use int8 storage. Cuts memory by about `3.5x` on the current 1536D runs, with roughly a 1-2 point recall-ceiling tradeoff versus float32. Set to `false` for full float32 storage. |
| `M` | number | `16` | HNSW connectivity |
| `efConstruction` | number | `200` | Build-time beam width |
| `efSearch` | number | `100` | Query-time beam width |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add(vector)` | `number` | Insert one vector, returns its ID |
| `addBatch(vectors)` | `number[]` | Insert multiple vectors |
| `search(query, k)` | `{id, distance}[]` | k-nearest-neighbor search |
| `searchFiltered(query, k, allowedIds)` | `{id, distance}[]` | k-NN restricted to an ID set |
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
| `quantized: true` | Int8 HNSW | Asymmetric distance: queries stay in float32, database vectors stored as int8. Preserves query-side precision while cutting memory by about `3.5x` on the current 1536D runs. |
| `quantized: false` | Float32 HNSW | Full precision, any dimension |

## Filtered search

`searchFiltered(query, k, allowedIds)` finds the k nearest neighbors restricted to a caller-supplied `Set<number>` of IDs. Pancake is an index, not a database, so it doesn't store metadata. The caller maintains their own ID-to-metadata mapping and builds the allowed set before searching.

```js
// Maintain metadata alongside the index
const metadata = new Map();

const id1 = index.add(embedding1);
metadata.set(id1, { tenant: 'acme', category: 'shoes' });

const id2 = index.add(embedding2);
metadata.set(id2, { tenant: 'acme', category: 'hats' });

const id3 = index.add(embedding3);
metadata.set(id3, { tenant: 'other', category: 'shoes' });

// Filter by tenant, then search within that set
const acmeIds = new Set();
for (const [id, meta] of metadata) {
  if (meta.tenant === 'acme') acmeIds.add(id);
}

const results = index.searchFiltered(query, 10, acmeIds);
```

Filtering happens during HNSW layer-0 traversal, not as a post-filter on top of `search()`. Non-matching nodes still participate in graph navigation (they stay in the candidate queue) but are excluded from the result set. The search widens ef dynamically within a single traversal when too few filtered results have been found.

This works well for moderate selectivity (roughly 1% of the index or more). At very low selectivity (< 1%), the graph may not contain enough navigable paths to the sparse target set, and recall drops. For extremely selective filters, brute-force over the allowed set is more reliable than in-graph filtering.

## Performance

All benchmarks are single-threaded on an AMD Ryzen 9 4900HS laptop. Pancake's
default package path is WASM. The native addon rows below come from an
experimental internal benchmarking tool in this repo, not from the installed
npm package. External baselines in the current harness are hnswlib, USearch,
and Faiss.

### DBpedia-50K full comparison (50k x 1536D, efSearch=100)

This is the current unified 50K comparison run across Pancake WASM/native,
hnswlib, USearch, and Faiss. Parameters are `M=16`, `efConstruction=50`,
`efSearch=100` where the library exposes them.

| Library | Mode | Build | Recall@10 | QPS | P50 | P99 | Memory |
|:--------|:-----|------:|----------:|----:|----:|----:|-------:|
| Pancake | int8 WASM | 75.61s | 96.70% | 1107 | 0.914ms | 1.422ms | 86.3 MB |
| Pancake | int8 native | 40.87s | 96.69% | 1305 | 0.772ms | 1.282ms | 86.3 MB |
| Pancake | f32 WASM | 162.89s | 98.83% | 771 | 1.307ms | 2.102ms | 299.3 MB |
| Pancake | f32 native | 97.43s | 98.83% | 1002 | 1.013ms | 1.543ms | 299.3 MB |
| hnswlib | f32 native | 43.01s | 98.20% | 928 | 1.091ms | 1.575ms | — |
| USearch | i8 native | 15.40s | 88.91% | 1633 | 0.617ms | 0.948ms | — |
| USearch | f32 native | 45.16s | 98.19% | 769 | 1.329ms | 1.859ms | — |
| Faiss HNSW | native | 35.88s | 85.99% | 2251 | 0.428ms | 0.705ms | — |

Takeaways:

- At the mid-recall operating point, `pancake-int8-native` reaches
  `1305 QPS` at `96.69%` recall.
- At the high-recall operating point, `pancake-f32-native` reaches
  `1002 QPS` at `98.83%` recall.
- At matched-or-better recall, `pancake-f32-wasm` lands within about `15-20%`
  of native hnswlib throughput, while `pancake-f32-native` edges ahead.
- Native Pancake now beats WASM Pancake clearly in both int8 and float32
  modes, which is useful for separating engine quality from runtime overhead.
- `usearch-i8` and `faiss-hnsw` win raw QPS, but only at materially lower
  recall than the comparable Pancake operating points.

Important caveat: `faiss-node` does not expose the same `efConstruction` and
`efSearch` controls as the other libraries in this harness, so the Faiss row
is informative but not perfectly parameter-matched.

### DBpedia-5K sanity run (50K harness, count=5000, efSearch=40)

The same full-comparison harness is also useful as a smaller smoke benchmark:

| Library | Mode | Build | Recall@10 | QPS | P50 | P99 | Memory |
|:--------|:-----|------:|----------:|----:|----:|----:|-------:|
| Pancake | int8 WASM | 5.54s | 96.67% | 3219 | 0.308ms | 0.484ms | 8.6 MB |
| Pancake | int8 native | 2.65s | 96.67% | 4088 | 0.235ms | 0.456ms | 8.6 MB |
| Pancake | f32 WASM | 13.16s | 98.83% | 2031 | 0.488ms | 0.761ms | 29.9 MB |
| Pancake | f32 native | 7.53s | 98.83% | 2615 | 0.378ms | 0.600ms | 29.9 MB |
| hnswlib | f32 native | 3.16s | 97.73% | 1977 | 0.499ms | 0.729ms | — |
| USearch | i8 native | 0.72s | 88.97% | 6100 | 0.155ms | 0.289ms | — |
| USearch | f32 native | 3.03s | 97.68% | 1996 | 0.496ms | 0.703ms | — |
| Faiss HNSW | native | 2.08s | 91.27% | 2447 | 0.382ms | 0.905ms | — |

This run is not the main benchmark, but it is a useful quick check
that the harness, native addon, and external baselines are all behaving.

### NYTimes-256 (290k x 256D, cosine)

On the lower-dimensional NYTimes cosine sweep, Pancake still reaches slightly
higher recall than hnswlib at the same `ef_search`, but native hnswlib remains
faster at matched recall across the overlapping range. This is the more typical
“portability costs throughput” picture at lower dimensions, where native SIMD
width matters more than it does on 1536D vectors.

### Deletion tolerance

DBpedia 5K x 1536D, cosine, int8, M=12, 100 held-out queries, brute-force ground truth recomputed against the live set at each step.

This sweep predates the current `M=16` comparison harness. Keep it as a
ghost-tolerance shape check rather than a directly comparable row against the
main 50K tables.

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
npm run bench -- benchmark_dbpedia_50k_full --count 50000 --m 16 --ef-construction 50 --ef-search 100
npm run bench -- benchmark_dbpedia_100k --m 16 --ef-construction 50 --ef-search 100
```

Results depend on dimension, corpus size, HNSW parameters, and hardware.

Recent benchmark additions include:

- `benchmark_dbpedia_50k_full` — unified 50K comparison across Pancake WASM/native, hnswlib, USearch, and Faiss
- `benchmark_native` — direct Pancake native-vs-WASM comparison
- `benchmark_usearch` and `benchmark_faiss` — external baseline runners
- `worker_restore_sweep` — cold/warm restore measurements through the Worker API

## Snapshot-first Worker deployment

Pancake can also run inside Cloudflare Workers, but the right mental model is
**snapshot search at the edge**, not **a durable mutable vector database inside
one long-lived isolate**.

The reference Worker in [`examples/worker/`](examples/worker/) keeps a Pancake
index warm in process when possible and restores snapshots from R2 on cold
start. It works well for:

- read-heavy semantic search
- modest-sized indexes that fit comfortably inside Worker memory limits
- periodic snapshot rebuilds or explicit import/export admin flows
- edge serving where cold restore is acceptable

As a rough guide, the current 1536D DBpedia runs put `50K` vectors at about
`86 MB` in int8 mode and `299 MB` in float32 mode. In practice that means
Worker deployments are most comfortable in the tens-of-thousands range unless
you are being very deliberate about memory headroom.

It is a weaker fit for:

- high-write online mutation as the primary production path
- strict cross-request write durability inside plain Worker memory
- “one always-live authoritative index instance” semantics

The Worker example still exposes add/delete/compact routes because they are
useful for demos, admin tooling, and local validation, but the main use case
is snapshot-backed search rather than a fully stateful ANN service.

See [`examples/worker/README.md`](examples/worker/README.md) for:

- endpoint documentation
- local development and deployment steps
- environment variables and Wrangler configuration
- memory, cold-start, and persistence tradeoffs

## When to use Pancake

Pancake makes sense when you want one engine that spans browser, Worker, and Node without much fuss:

- **Browser-based search** -- client-side retrieval without a server round-trip
- **Portable applications** -- one artifact, same behavior across every JavaScript runtime
- **Node.js without native addons** -- in-process ANN without native binary packaging
- **Small to medium indexes** -- in-process search without external infrastructure

If you only care about native server throughput and do not need portability, Faiss, hnswlib, or USearch are good baselines to compare against. On the current 1536D DBpedia run, `pancake-f32-wasm` lands within about `15-20%` of native hnswlib throughput at matched-or-better recall, and `pancake-f32-native` edges ahead. The int8 backend offers a lower-memory operating point with about a 1-2 point recall-ceiling tradeoff versus float32 in the current 50K sweep. The native addon in this repo is there to help separate runtime overhead from graph quality.

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

The WASM module exposes a handle-based C API: `pancake_init` returns an opaque handle, and all operations take a handle as the first argument. This enables multiple independent indexes per WASM instance. The JavaScript wrapper (`pancake-core.js`) manages handles and ID translation. The Worker deployment wraps the raw engine export in the `WRK1` metadata envelope.

## Examples

- `examples/cli/build_index.js` -- build an HNSW index from precomputed embeddings
- `examples/demo/technical_demo_cli.js` -- interactive local REPL
- `examples/demo/technical_demo_worker.js` -- interactive Worker-targeted REPL
- `dist/technical-demo.html` -- browser demo page with latency and stress views
- `examples/browser-vite/` -- minimal bundled browser consumer fixture used by `npm run test:browser`
- `examples/worker/` -- reference Cloudflare Worker deployment built on top of Pancake
- `examples/worker-semantic-search/` -- snapshot-first semantic docs search demo for Cloudflare Workers

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

Requires an Emscripten toolchain with WASM SIMD support. The default build is plain WASM SIMD for broader compatibility. To opt into relaxed SIMD on supporting runtimes:

```bash
WASM_RELAXED_SIMD=1 ./build.sh
```

Relaxed SIMD is an opt-in fast path for supporting runtimes. The default build
stays on plain WASM SIMD so the checked-in artifact remains broadly compatible,
but local builds can enable relaxed SIMD when you want the faster vectorized
path.

The script auto-detects whether the current Node runtime still needs
`--experimental-wasm-relaxed-simd` for its post-build test step.

## Tradeoffs

- **Single-threaded by design.** Pancake is meant for runtimes where background threads are unavailable or unreliable. That is a deployment advantage, not just a limitation.
- **Quantization is a real trade.** On the current 1536D DBpedia benchmark, the int8 path uses about `3.5x` less memory than float32 and gives up roughly 1-2 points of recall ceiling (`96.7%` vs `98.8%` in the 50K sweep at `efSearch=100`). Use float32 when you need the higher ceiling.
- **Compaction is rebuild-based.** Deletes are soft deletes. Compaction rewrites the graph rather than patching edges in place. That keeps behavior predictable and avoids relying on background maintenance threads.
- **Index instances are not a shared-memory concurrency primitive.** Treat a Pancake index like ordinary mutable in-process state: safe within one JavaScript thread/event loop, but not something to share concurrently across Node worker threads or isolates without your own coordination.
- **Workers are best used as snapshot-serving search frontends.** In a Cloudflare Worker, in-memory state is a warm cache, not durable authority. Persist snapshots explicitly and treat isolate reuse as opportunistic.
- **This is an index, not an embedding stack.** Pancake does vector search only. Bring your own embedding pipeline.

## License

Apache 2.0
