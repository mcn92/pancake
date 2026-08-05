# Pancake

HNSW vector search in about 50 KB of gzipped WebAssembly (141 KB uncompressed). Runs in Node.js, browser-bundled web apps, and Cloudflare Workers with no native dependencies in the default package path.

Most ANN libraries ship as platform-specific native binaries, which means they do not work in browser tabs or JavaScript runtimes without native extensions. Pancake's primary package is a single portable WASM module built for JavaScript environments where native addons are not an option.

Pancake ships two backends in the WASM engine: a uint8 quantized backend that cuts memory by about `3.5x` on the current 1536D DBpedia-50K run, with roughly a 2.3 point recall-ceiling tradeoff versus float32, and a full float32 backend for higher precision distances. Both use WASM SIMD acceleration. The repo also includes an experimental native Node addon with AVX2 distance kernels for direct native-vs-WASM comparisons and local benchmarks.

Pancake is an ANN library -- it doesn't ship an embedding model. Use any embedder (sentence-transformers, OpenAI, Cohere, etc.) and feed the resulting vectors to Pancake.

## Install

Install from npm:

```bash
npm install pancake-wasm
```

The API documented in this README is the Pancake 0.2 contract, which ships in
`pancake-wasm@0.2.0` and later.

Or work from a repository checkout. The checkout ships prebuilt WASM in
`dist/`, so the library works with no build step and no install — you can
`require('./pancake.js')` and search immediately:

```bash
git clone https://github.com/mcn92/pancake.git
cd pancake
node -e "const P=require('./pancake.js'); P.create({dim:8,metric:'l2'}).then(i=>{i.add(new Float32Array(8).fill(1)); console.log(i.search(new Float32Array(8).fill(1),1)); i.dispose();})"
```

To run the demos and tests, install the dev dependencies first:

```bash
npm install          # dev deps for demos, benchmarks, and tests
npm test             # optional: verify the shipped build (core + conformance)
```

You only need to rebuild the WASM engine if you are modifying the C++ under
`src/`; the prebuilt artifacts already match the source. Building requires an
Emscripten toolchain (see [Building from source](#building-from-source)):

```bash
./build.sh           # rebuilds dist/engine.{js,wasm} — only if changing the engine
```

The root `pancake.js` / `pancake.node.mjs` / `pancake.web.mjs` entry points are
the same files that ship in the npm package, so code written against the repo
checkout works unchanged against the installed package.

See [`DEMOS.md`](DEMOS.md) for copy-paste commands to run each demo.

The native addon under `native/` is not part of the npm package. It is an
opt-in benchmarking tool used in this repo to separate runtime overhead from
graph quality.

## Runtime entry points

From the published package, import the entry points by name:

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

From a repository checkout, import the same entry points by relative path
instead:

```js
// Node.js CJS
const Pancake = require('./pancake.js');

// Node.js ESM
import Pancake from './pancake.node.mjs';

// Browser bundlers
import Pancake from './pancake.web.mjs';
```

The `pancake-wasm/web` entry (`./pancake.web.mjs` in a checkout) expects a bundler or runtime that can resolve the packaged
`./dist/engine.wasm` asset. This is tested with a bundled Vite + Chromium flow. For a raw
no-bundler demo, see the repo copy of
[`dist/technical-demo.html`](https://github.com/mcn92/pancake/blob/main/dist/technical-demo.html).

## Quick start

The examples below import `pancake-wasm` by name. From a repository checkout,
replace that import with a relative path — `import Pancake from './pancake.node.mjs'`
(ESM) or `const Pancake = require('./pancake.js')` (CJS). The API is identical
either way.

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
const restored = await Pancake.restore(snapshot, { maxElements: 100000 });
```

If you already have vectors in memory, `fromVectors()` is the easiest ingest path:

```js
const rows = [
  { id: 'doc-1', vector: embedding1 },
  { id: 'doc-2', vector: embedding2 },
];

const { index, ids, idMap } = await Pancake.fromVectors(rows, {
  metric: 'cosine',
  quantized: true,
});

const hits = index.search(queryEmbedding, 5);
console.log(idMap.get(hits[0].id)); // -> 'doc-1'
```

On the Node.js entrypoints, there are file helpers for JSON/JSONL vectors and
Pancake snapshots:

```js
const { index, idMap } = await Pancake.loadJsonFile('vectors.jsonl', {
  metric: 'cosine',
  vectorKey: 'embedding', // default: 'vector'
  idKey: 'docId',         // default: 'id'
  maxFileBytes: 64 * 1024 * 1024,
});

const restored = await Pancake.loadSnapshotFile('index.pnck', {
  dim: 384,
  maxElements: 100000,
  metric: 'cosine',
  quantized: true,
  maxFileBytes: 512 * 1024 * 1024,
});
```

If `ghostCount > 0`, `export()` throws. Call `compact()` first to produce a clean snapshot.
`import()` also requires the target index config to match the export's `dim`, `metric`, and `quantized` mode.
`import()` is atomic: the engine deserializes into a fresh internal index and only swaps it in on success, so a failed import — whether it's envelope validation or a backend-level rejection — leaves the destination index and its ID mappings unchanged.

`compact()` is a stop-the-world maintenance pass over the surviving graph:
below 50% deletions it remaps live vectors in place and repairs their edges; at
50% or more it rebuilds the topology from the live vectors because a mostly
hollow graph is no longer a reliable skeleton. Both paths preserve external
IDs. Its cost scales with the number of live vectors that remain after
deletions, not just the number of ghosts removed, so treat it as maintenance
work rather than something to run in a latency-sensitive path.

In the committed 100K clustered churn run, five complete population turnovers
left 83.3% deleted nodes and reduced recall@10 from 96.0% to 7.2% (4.8 results
returned on average). Compaction rebuilt the 100K live-vector graph in 17.6s,
restoring 99.2% recall and a full top-10; a clean build of the same final
population reached 97.2%. The rebuild reused the old graph's allocations, so
the WASM heap did not grow during compaction. Run
`node benchmarks/churn_scale.js` to reproduce it.

The JavaScript package `export()` returns a `PNCK` envelope, not the bare raw engine payload.
That wrapper preserves package-level metadata such as stable external IDs across
`compact()`/`import()` round-trips. The underlying raw engine snapshot is stored inside that
envelope. The Cloudflare Worker stores this standard envelope directly and keeps
capacity and runtime query policy in R2 custom metadata.

## API

### `await Pancake.create(opts)`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `dim` | number | required | Vector dimension |
| `maxElements` | number | `100000` | Pre-allocated capacity |
| `metric` | string | `'cosine'` | `'cosine'` or `'l2'` |
| `quantized` | boolean | `true` | Use uint8 storage. Cuts memory by about `3.5x` on the current 1536D DBpedia-50K run, with roughly a 2.3 point recall-ceiling tradeoff versus float32. Set to `false` for full float32 storage. |
| `M` | number | `12` | HNSW connectivity |
| `efConstruction` | number | `75` | Build-time beam width (`1`–`4096`) |
| `efSearch` | number | `100` | Default query-time beam width (`1`–`4096`) |
| `seed` | number | `108` | Deterministic HNSW level-assignment seed |

`maxElements` pre-allocates graph structure eagerly: the default `100000`
costs about `25 MB` of index memory (and roughly `75 MB` of WASM heap at
384D) before the first vector is added. In memory-constrained runtimes
(Workers, browser tabs), size it to your expected corpus —
`fromVectors()` does this automatically by defaulting it to `rows.length`.

### `await Pancake.fromVectors(rows, opts)`

Creates an index and bulk-loads either:

- `Float32Array[]` / `number[][]`
- `{ id?, vector }[]`

Returns:

- `index` — the populated `PancakeIndex`
- `ids` — Pancake's stable numeric IDs in insertion order
- `idMap` — `Map<number, sourceId>` for rows that supplied caller IDs

`dim` defaults to the first vector's length. `maxElements` defaults to `rows.length`.

### Snapshot restore

`await Pancake.restore(snapshot, overrides?)` infers `dim`, `metric`,
`quantized`, `M`, and `efConstruction` from a Pancake envelope. Capacity
defaults to the restored count, so pass `{ maxElements }` when the restored
index must accept more vectors. `efSearch` is runtime policy and defaults to
`100`; it can also be overridden during restore.

`Pancake.inspectSnapshot(snapshot)` validates the headers without creating a
WASM instance and returns the format, version, resolved construction fields,
count, and next external ID. Legacy raw engine snapshots remain supported, but
`restore()` requires their construction fields to be supplied explicitly.

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add(vector)` | `number` | Insert one vector, returns its ID |
| `addBatch(vectors)` | `number[]` | Insert multiple vectors |
| `search(query, k, { efSearch? })` | `{id, distance}[]` | k-nearest-neighbor search, optionally overriding beam width for this call |
| `searchFiltered(query, k, allowedIds, { efSearch? })` | `{id, distance}[]` | k-NN restricted to an ID set, with the same per-call override |
| `setEfSearch(value)` | -- | Change the default beam width for future searches |
| `delete(id)` | `boolean` | Soft-delete a live ID; false means unknown or already deleted |
| `has(id)` | `boolean` | Whether an ID is live |
| `isDeleted(id)` | `boolean` | Whether an ID is awaiting compaction |
| `compact()` | -- | Rebuild graph without soft-deleted entries |
| `export()` | `Uint8Array` | Serialize index state. Requires `ghostCount === 0`; call `compact()` first after deletions. |
| `import(data)` | -- | Restore a previous export |
| `dispose()` | -- | Free WASM buffers |
| `Pancake.withIndex(opts, fn)` | callback result | Create, use, and always dispose an index |

### Distance values

`search()` and `searchFiltered()` report a `distance` per hit:

- `metric: 'l2'` — Euclidean distance (the square root is applied; this is
  not squared L2).
- `metric: 'cosine'` — cosine distance, `1 - cosine_similarity`, ranging
  over `[0, 2]`: `0` is identical direction, `1` is orthogonal, `2` is
  opposite. Vectors are normalized internally at insert and query time, so
  inputs do not need to be pre-normalized.

These semantics are part of the API contract and stable across snapshot
envelope versions. (On the uint8 backend, quantization adds small error to
stored-side values, as described under Quantization.)

### Node-only file helpers

| Helper | Returns | Description |
|--------|---------|-------------|
| `loadJsonFile(path, opts)` | `{ index, ids, idMap }` | Build an index from a JSON array or JSONL file |
| `loadSnapshotFile(path, overrides?)` | `PancakeIndex` | Restore a previous `export()` from disk with inferred config |

File-helper notes:

- `loadJsonFile()` only auto-detects `.json`, `.jsonl`, and `.ndjson`. For other extensions, pass `opts.format`.
- `loadJsonFile()` defaults `maxFileBytes` to `64 MiB`.
- `loadSnapshotFile()` defaults `maxFileBytes` to `512 MiB`.
- `loadSnapshotFile()` checks the file header before import and rejects files that do not look like a Pancake envelope or raw engine snapshot.

### Properties

| Property | Description |
|----------|-------------|
| `count` | Vectors stored (includes soft-deleted until `compact()`) |
| `liveCount` | Live, searchable vectors |
| `deletedCount` / `deletedRatio` | Soft-deleted vectors awaiting compaction and their fraction |
| `capacity` / `remainingCapacity` | Fixed insertion capacity and unused slots; soft deletion does not free a slot |
| `config` | Fully resolved construction config and current default `efSearch` |
| `memoryUsage` | `{ logicalIndexBytes, wasmHeapBytes, snapshotBufferBytes }` |
| `ghostCount` / `ghostRatio` | Backward-compatible aliases for the deleted state |
| `memory` | Alias for `memoryUsage.logicalIndexBytes` |
| `dim` | Vector dimension |

### Backend dispatch

Pancake picks one of two HNSW backends based on the `quantized` option. This is transparent -- the API is the same regardless of backend:

| Condition | Backend | Notes |
|-----------|---------|-------|
| `quantized: true` | Uint8 HNSW | Asymmetric distance: queries stay in float32, database vectors stored as uint8. Preserves query-side precision while cutting memory by about `3.5x` on the current 1536D DBpedia-50K run. |
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

The primary benchmark is DBpedia-50K (`50,000 x 1536D`, L2), the operating
point Pancake is built for: in-process ANN at moderate scale across portable
JavaScript runtimes. Pancake's default package path is WASM; the native addon
is an experimental in-repo tool, included here to separate runtime overhead
from graph quality, and is not part of the npm package.

The Pareto harness uses the canonical operating point by default: `k=10`,
`M=12`, `efConstruction=75`, 3 repetitions, and an `efSearch` sweep from 10 to
800. It writes CSV/JSON/log/PNG artifacts under
[`benchmark_results/`](benchmark_results/).

Current Pareto CSVs split memory reporting into stable index size and runtime
overhead. `memory_mb` is the comparable index-size proxy: Pancake logical index
bytes, USearch serialized index bytes, and blank for hnswlib because hnswlib-node
does not expose a serialized index-size API. `wasm_heap_mb` reports the full
WASM heap when available, and `rss_delta_mb` reports approximate process RSS
growth during build.

### DBpedia-50K: WASM vs native (50k x 1536D, L2)

This DBpedia-50K release run used `M=16` and `efConstruction=50`; the
cross-dataset table below uses the parameters recorded in each result artifact.

| Mode | Build | Memory | ef=100 Recall@10 | ef=100 QPS | ef=800 Recall@10 | ef=800 QPS |
|:-----|------:|-------:|-----------------:|-----------:|-----------------:|-----------:|
| u8 WASM | 73.4s | 86.3 MB | 96.70% | 1,134 | 97.58% | 214 |
| f32 WASM | 155.1s | 299.4 MB | 98.83% | 829 | 99.90% | 156 |
| u8 native | 37.4s | 86.3 MB | 96.68% | 1,396 | 97.56% | 274 |
| f32 native | 93.3s | 299.4 MB | 98.83% | 1,025 | 99.90% | 203 |

These numbers are the Pancake rows from
[`pareto_dbpedia_2026-07-15T08-18-46-756Z`](benchmark_results/release/pareto_dbpedia_2026-07-15T08-18-46-756Z.csv).
The same run includes USearch and hnswlib baselines.

Takeaways:

- **uint8 is the memory/throughput operating point:** on DBpedia-50K, u8 WASM
  uses `86.3 MB` vs `299.4 MB` for float32, about `3.5x` less memory.
- **float32 is the high-recall operating point:** DBpedia float32 reaches
  `99.90%` recall at the high-`efSearch` end of the frontier; uint8 reaches
  `97.58%` on the same sweep.
- **Native vs WASM isolates runtime overhead:** the native addon runs the same
  engine and lands at essentially identical recall. It is faster on DBpedia,
  but the gap narrows or reverses depending on dataset dimensionality and
  memory behavior.

### QPS-recall frontier

Sweeping `efSearch` (10-800) traces Pancake's full QPS-recall frontier. The
harness can cover DBpedia, NYTimes, SIFT, GloVe, MNIST, or custom fvecs/HDF5
inputs, and can include Pancake WASM/native, USearch native, USearch WASM, and
hnswlib when the matching dependencies and artifacts are present.

Current release sweeps cover DBpedia-50K (1536D, L2), NYTimes-256 (256D,
cosine), SIFT-1M (128D, L2), and GloVe-100 (100D, cosine):

| Dataset | Backend | ef=100 Recall@10 | ef=100 QPS | ef=800 Recall@10 | ef=800 QPS | Memory |
|:--------|:--------|-----------------:|-----------:|-----------------:|-----------:|-------:|
| DBpedia-50K | u8 WASM | 96.70% | 1,134 | 97.58% | 214 | 86.3 MB |
| DBpedia-50K | f32 WASM | 98.83% | 829 | 99.90% | 156 | 299.4 MB |
| NYTimes-256 | u8 WASM | 79.75% | 2,050 | 91.86% | 320 | 129.1 MB |
| NYTimes-256 | f32 WASM | 79.91% | 1,514 | 92.43% | 240 | 311.5 MB |
| SIFT-1M | u8 WASM | 96.58% | 2,212 | 99.20% | 372 | 323.0 MB |
| SIFT-1M | f32 WASM | 96.97% | 1,821 | 99.80% | 315 | 585.9 MB |
| GloVe-100 | u8 WASM | 75.38% | 1,894 | 90.41% | 320 | 350.7 MB |
| GloVe-100 | f32 WASM | 76.24% | 1,512 | 90.93% | 258 | 567.0 MB |

The cross-library comparison is in the benchmark result artifacts rather than
asserted here; exact numbers depend on library versions, build flags, CPU, and
memory pressure. Run `pareto_frontier` to reproduce or extend the sweeps on
your own data (see [Reproducing](#reproducing)).

### Deletion tolerance

DBpedia 5K x 1536D, L2, uint8, `M=16`, `efConstruction=50`, `efSearch=100`, 100
held-out queries. Vectors are progressively soft-deleted; brute-force ground
truth is recomputed against the live set at each step.

| Ghost % | Live Vectors | Recall@10 | p50 Latency |
|---:|---:|---:|---:|
| 0% | 4,900 | 96.8% | 0.71ms |
| 30% | 3,430 | 97.5% | 0.56ms |
| 50% | 2,450 | 96.5% | 0.43ms |
| 70% | 1,470 | 95.5% | 0.32ms |
| 90% | 490 | 81.4% | 0.17ms |

Recall holds within about 1.5 points of baseline through 70% ghosts. Search latency drops as ghosts accumulate (fewer live nodes to visit). The cliff beyond 90% is graph disconnection: the live subgraph is no longer well-connected enough to preserve recall. Compaction can be deferred until the main thread is idle.

That progressive-delete result is not a substitute for insert/delete churn:
replacing the live population changes graph topology much more aggressively.
For sustained churn, use `deletedRatio >= 0.5` as the default maintenance
trigger and validate a different threshold against your own data and latency
budget.

### Reproducing

Benchmark scripts are in `benchmarks/`. A shared runner discovers them by name,
so you can list and run them with:

```bash
npm run bench -- --list

# The Pancake-only WASM vs native comparison:
npm run bench -- benchmark_native

# Full comparison including hnswlib, USearch, and Faiss on the same data:
npm run bench -- benchmark_dbpedia_50k_full --count 50000 --m 12 --ef-construction 75 --ef-search 100

# QPS-recall frontier (efSearch sweep) behind the plot above:
npm run bench -- pareto_frontier

# Include a locally built USearch WASM bundle in the frontier:
npm run bench -- pareto_frontier -- --usearch-wasm /path/to/usearch-wasm
```

`pareto_frontier` does not vendor USearch WASM binaries. By default, it looks
for local, untracked dtype-specific artifacts under `external/usearch-wasm/`:
1 GB builds for int8/f16 and a 2 GB build for fp32. Use `--usearch-wasm` to
point all USearch WASM configs at a custom artifact.

The local USearch WASM builds need a writable Emscripten cache and should use
the installed emsdk checkout instead of fetching Emscripten during CMake
configure:

```bash
cmake -E env EM_CACHE=/tmp/pancake-emscripten-cache cmake \
  -S external/usearch-wasm/USearch \
  -B external/usearch-wasm/USearch/build_wasm_pancake_1gb_growth \
  -DCMAKE_TOOLCHAIN_FILE=/home/mcn9284/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake \
  -DFETCHCONTENT_SOURCE_DIR_EMSCRIPTEN=/home/mcn9284/emsdk/upstream/emscripten \
  -DUSEARCH_BUILD_WASM=1 -DUSEARCH_BUILD_TEST_CPP=0 -DUSEARCH_BUILD_BENCH_CPP=0 \
  -DUSEARCH_BUILD_TEST_C=0 -DUSEARCH_BUILD_LIB_C=0 -DUSEARCH_BUILD_SQLITE=0 \
  -DUSEARCH_BUILD_JNI=0 -DUSEARCH_BUILD_WOLFRAM=0 -DUSEARCH_USE_OPENMP=0 \
  -DCMAKE_HAVE_LIBC_PTHREAD=1 -DCMAKE_HAVE_THREADS_LIBRARY=1 \
  -DCMAKE_USE_PTHREADS_INIT=1 -DCMAKE_THREAD_LIBS_INIT= \
  -DCMAKE_BUILD_TYPE=Release \
  '-DCMAKE_EXE_LINKER_FLAGS=-sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=16777216 -sMAXIMUM_MEMORY=1073741824'
cmake -E env EM_CACHE=/tmp/pancake-emscripten-cache cmake \
  --build external/usearch-wasm/USearch/build_wasm_pancake_1gb_growth \
  --target usearch_wasm --config Release --parallel 4
```

Use the same command with `build_wasm_pancake_2gb_growth` and
`-sMAXIMUM_MEMORY=2147483648` for the fp32 artifact.

The first run computes brute-force ground truth and caches it under
`benchmark_results/cache/`; later runs reuse it. Results depend on dimension,
corpus size, HNSW parameters, and hardware.

Other useful scripts:

- `benchmark_dbpedia_50k_full` — unified comparison vs hnswlib, USearch, and Faiss
- `pareto_frontier` — efSearch sweep producing the QPS-recall frontier CSVs
- `benchmark_native` — direct Pancake native-vs-WASM comparison
- `worker_restore_sweep` — cold/warm restore measurements through the Worker API
- `runtime_ownership` — cold vs warm `create()`, concurrent creation, and
  isolated per-index heap footprint

## Snapshot-first Worker deployment

Pancake can also run inside Cloudflare Workers, but the right mental model is
**snapshot search at the edge**, not **a durable mutable vector database inside
one long-lived isolate**.

The reference Worker in [`examples/reference-worker/`](examples/reference-worker/) keeps a Pancake
index warm in process when possible and restores snapshots from R2 on cold
start. It works well for:

- read-heavy semantic search
- modest-sized indexes that fit comfortably inside Worker memory limits
- periodic snapshot rebuilds or explicit import/export admin flows
- edge serving where cold restore is acceptable

As a rough guide, the current 1536D DBpedia runs put `50K` vectors at about
`86 MB` in uint8 mode and `299 MB` in float32 mode. In practice that means
Worker deployments are most comfortable in the tens-of-thousands range unless
you are being very deliberate about memory headroom.

It is a weaker fit for:

- high-write online mutation as the primary production path
- strict cross-request write durability inside plain Worker memory
- “one always-live authoritative index instance” semantics

The Worker example still exposes add/delete/compact routes because they are
useful for demos, admin tooling, and local validation, but the main use case
is snapshot-backed search rather than a fully stateful ANN service.

For a deployable public webpage with no storage or inference bindings, see the
[`worker-semantic-search`](examples/03-edge-docs-search/) demo. It bundles a
1.08 MB domain-distilled int8 query encoder, a quantized Pancake snapshot, and
the result corpus directly into one Worker. Query embedding and Pancake search
both run locally with zero outbound requests; the offline teacher never ships.

See [`examples/reference-worker/README.md`](examples/reference-worker/README.md) for:

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

If you only care about native server throughput and do not need portability, Faiss, hnswlib, or USearch are good baselines to compare against — run `benchmark_dbpedia_50k_full` to see how they land against Pancake on your own hardware. On the DBpedia-50K run above, the uint8 backend offers a lower-memory operating point with about a 2.3 point recall-ceiling tradeoff versus float32 (`97.58%` vs `99.90%` on DBpedia-50K). The native addon in this repo is there to help separate runtime overhead from graph quality.

## How it works

### Quantization

Stored vectors use row-wise affine quantization. Each vector gets its own scale and offset derived from its own min/max:

```
scale  = (max - min) / 255
offset = min
q[i]   = uint8(clamp((x[i] - offset) / scale, 0, 255))
```

This preserves per-vector dynamic range better than global quantization, at the cost of 8 bytes of overhead per vector for the scale and offset.

The quantized backend uses asymmetric distance: queries stay in float32 while database vectors are dequantized from uint8 on the fly. This preserves full query-side precision -- quantization error only affects the stored side.

### HNSW

Standard HNSW graph search with one modification: the neighbor selection heuristic uses **backfill** (the "keep pruned connections" option from the HNSW paper). After the diversity heuristic rejects candidates, the closest rejected candidates are added back until M slots are filled. This guarantees minimum connectivity and prevents weakly-connected nodes in clustered data, where the diversity heuristic can otherwise reject most same-cluster candidates.

`M` controls graph connectivity, `efConstruction` controls build quality, `efSearch` controls query quality. Higher values improve recall at the cost of build or query speed.

### Handle-based C ABI

The WASM module exposes a handle-based C API: `pancake_init` returns an opaque
handle, and all operations take a handle as the first argument. The supported
JavaScript loaders cache compiled modules but instantiate an isolated WASM heap
for every index. The wrapper (`pancake-core.js`) manages the handle and stable
external-ID translation inside that isolated instance.

## Examples

- [`examples/legacy/cli/build_index.js`](https://github.com/mcn92/pancake/blob/main/examples/legacy/cli/build_index.js) -- build an HNSW index from precomputed embeddings
- [`examples/legacy/technical-demo/technical_demo_cli.js`](https://github.com/mcn92/pancake/blob/main/examples/legacy/technical-demo/technical_demo_cli.js) -- interactive local REPL
- [`examples/legacy/technical-demo/technical_demo_worker.js`](https://github.com/mcn92/pancake/blob/main/examples/legacy/technical-demo/technical_demo_worker.js) -- interactive Worker-targeted REPL
- [`dist/technical-demo.html`](https://github.com/mcn92/pancake/blob/main/dist/technical-demo.html) -- browser demo page with latency and stress views
- [`examples/01-hello-pack/`](https://github.com/mcn92/pancake/tree/main/examples/01-hello-pack) -- minimal bundled browser consumer fixture used by `npm run test:browser`
- [`examples/reference-worker/`](https://github.com/mcn92/pancake/tree/main/examples/reference-worker) -- reference Cloudflare Worker deployment built on top of Pancake
- [`examples/03-edge-docs-search/`](https://github.com/mcn92/pancake/tree/main/examples/03-edge-docs-search) -- zero-runtime-dependency distilled docs search webpage bundled into one Worker

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

This starts a minimal Vite app in `examples/01-hello-pack/` and verifies the published web entry
in Chromium via Playwright.

For SIMD parity coverage, run:

```bash
npm run test:simd
```

For the Worker reference deployment, run:

```bash
node test/test_worker_features.js
```

Current core suite status on this tree: **1191 passed, 0 failed**.

## Building from source

The npm package ships prebuilt WASM artifacts. Rebuild only if you're modifying the C++ engine:

```bash
./build.sh          # SIMD build: dist/engine.{js,wasm}
npm run build:all   # SIMD + scalar fallback: dist/engine.scalar.{js,wasm} too
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
- **Quantization is a real trade.** On the current 1536D DBpedia-50K benchmark, the uint8 path uses about `3.5x` less memory than float32 and gives up roughly 2.3 points of recall ceiling (`97.58%` vs `99.90%` in the 50K sweep at `efSearch=800`). Use float32 when you need the higher ceiling.
- **Deterministic means per target/build.** Given the same inputs, the same build target will produce stable graph structure and query results, but bitwise-identical distances are not guaranteed across WASM SIMD, scalar WASM, AVX2, and SSE2 backends because their reduction orders differ. Treat tiny cross-backend floating-point deltas as expected, not as correctness bugs.
- **Each public index owns one isolated WASM instance.** Compiled module work is
  cached per runtime entrypoint, while heaps and handle tables are not shared.
  The engine retains a 64-slot handle table for future shared-runtime support,
  but a public index normally occupies only slot zero of its own instance.
- **Compaction is a stop-the-world maintenance pass.** Deletes are soft deletes. Below 50% deleted nodes, `compact()` remaps survivors in place and repairs connectivity; at 50% or more it rebuilds the graph from live vectors to recover from severe topology loss. Neither path runs in the background.
- **`export()` retains a WASM-side buffer.** The engine keeps the most recent snapshot copy inside WASM memory until the next `export()` on the same index or `dispose()`. Budget for that when exporting large indexes in memory-constrained runtimes.
- **Index instances are not a cross-thread concurrency primitive.** Treat a
  Pancake index like ordinary mutable in-process state: safe within one
  JavaScript thread/event loop, but not transferable across Node worker threads
  or isolates.
- **Workers are best used as snapshot-serving search frontends.** In a Cloudflare Worker, in-memory state is a warm cache, not durable authority. Persist snapshots explicitly and treat isolate reuse as opportunistic.
- **Inputs are validated, not coerced.** Vectors and queries accept a `Float32Array` or a plain numeric array, but plain-array elements must be actual numbers — a non-numeric element (string, boolean, nested array, `null`) is rejected with an error rather than silently coerced (e.g. an empty CSV field becoming `0`). Non-finite values (`NaN`/`Infinity`) and dimension mismatches are likewise rejected at the boundary.
- **This is an index, not an embedding stack.** Pancake does vector search only. Bring your own embedding pipeline.

## License

Apache 2.0
