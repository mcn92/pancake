# Pancake

HNSW vector search in about 49 KB of gzipped WebAssembly (137 KB uncompressed). Runs in Node.js, browser-bundled web apps, and Cloudflare Workers with no native dependencies in the default package path.

Most ANN libraries ship as platform-specific native binaries, which means they do not work in browser tabs or JavaScript runtimes without native extensions. Pancake's primary package is a single portable WASM module built for JavaScript environments where native addons are not an option.

Pancake ships two backends in the WASM engine: an int8 quantized backend that cuts memory by about `3.5x` on the current 1536D runs, with roughly a 1-2 point recall-ceiling tradeoff versus float32, and a full float32 backend for higher precision distances. Both use WASM SIMD acceleration. The repo also includes an experimental native Node addon with AVX2 distance kernels for direct native-vs-WASM comparisons and local benchmarks.

Pancake is an ANN library -- it doesn't ship an embedding model. Use any embedder (sentence-transformers, OpenAI, Cohere, etc.) and feed the resulting vectors to Pancake.

## Install

Install from npm:

```bash
npm install pancake-wasm
```

Or work from a repository checkout:

```bash
git clone https://github.com/mcn92/pancake.git
cd pancake
./build.sh          # produces dist/engine.js and dist/engine.wasm
node run_tests.js   # optional: verify the build
```

The root `pancake.js` / `pancake.node.mjs` / `pancake.web.mjs` entry points are
the same files that ship in the npm package, so code written against the repo
checkout works unchanged against the installed package.

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
const restored = await Pancake.create({
  dim: 384,
  maxElements: 100000,
  metric: 'cosine',
  quantized: true,
});
restored.import(snapshot);
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
Envelope/config validation failures reject the snapshot before mutating the destination index, but backend-level import failures may still leave the destination unusable. If recovery matters, export a backup first.

`compact()` is a stop-the-world rebuild of the surviving graph. Its cost scales
with the number of live vectors that remain after deletions, not just the
number of ghosts removed, so treat it as maintenance work rather than something
to run in a latency-sensitive path. On the current DBpedia 50K int8 benchmark
(`1536D`, `M=16`, `efConstruction=50`, `efSearch=100`), compacting after 20%,
50%, and 80% deletions took `23.7s`, `14.3s`, and `4.4s` respectively, while
recall stayed within `0.14` points of baseline.

The JavaScript package `export()` returns a `PNCK` envelope, not the bare raw engine payload.
That wrapper preserves package-level metadata such as stable external IDs across
`compact()`/`import()` round-trips. The underlying raw engine snapshot is stored inside that
envelope. The Cloudflare Worker example wraps the package export again in a small `WRK1`
envelope so it can persist Worker-specific metadata such as `maxElements`, init params, and
deployment state alongside the packaged snapshot.

## API

### `await Pancake.create(opts)`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `dim` | number | required | Vector dimension |
| `maxElements` | number | `100000` | Pre-allocated capacity |
| `metric` | string | `'cosine'` | `'cosine'` or `'l2'` |
| `quantized` | boolean | `true` | Use int8 storage. Cuts memory by about `3.5x` on the current 1536D runs, with roughly a 1-2 point recall-ceiling tradeoff versus float32. Set to `false` for full float32 storage. |
| `M` | number | `16` | HNSW connectivity |
| `efConstruction` | number | `50` | Build-time beam width |
| `efSearch` | number | `100` | Query-time beam width |

### `await Pancake.fromVectors(rows, opts)`

Creates an index and bulk-loads either:

- `Float32Array[]` / `number[][]`
- `{ id?, vector }[]`

Returns:

- `index` — the populated `PancakeIndex`
- `ids` — Pancake's stable numeric IDs in insertion order
- `idMap` — `Map<number, sourceId>` for rows that supplied caller IDs

`dim` defaults to the first vector's length. `maxElements` defaults to `rows.length`.

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

### Node-only file helpers

| Helper | Returns | Description |
|--------|---------|-------------|
| `loadJsonFile(path, opts)` | `{ index, ids, idMap }` | Build an index from a JSON array or JSONL file |
| `loadSnapshotFile(path, opts)` | `PancakeIndex` | Restore a previous `export()` from disk |

File-helper notes:

- `loadJsonFile()` only auto-detects `.json`, `.jsonl`, and `.ndjson`. For other extensions, pass `opts.format`.
- `loadJsonFile()` defaults `maxFileBytes` to `64 MiB`.
- `loadSnapshotFile()` defaults `maxFileBytes` to `512 MiB`.
- `loadSnapshotFile()` checks the file header before import and rejects files that do not look like a Pancake envelope or raw engine snapshot.

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

The primary benchmark is DBpedia-50K (`50,000 × 1536D`, L2), the operating
point Pancake is built for: in-process ANN at moderate scale across portable
JavaScript runtimes. Pancake's default package path is WASM; the native addon
is an experimental in-repo tool, included here to separate runtime overhead
from graph quality, and is not part of the npm package.

### DBpedia-50K: WASM vs native (50k x 1536D, L2)

Single-threaded on an AMD Ryzen 9 4900HS laptop. Parameters `M=16`,
`efConstruction=50`, `efSearch=100`; `k=10`, 1,000 held-out queries, 3
repetitions, brute-force L2 ground truth.

| Mode | Build | Recall@10 | QPS | p50 | p99 | Memory |
|:-----|------:|----------:|----:|----:|----:|-------:|
| int8 WASM | 73.0s | 96.70% | 1128 | 0.909ms | 1.385ms | 86.3 MB |
| f32 WASM | 158.6s | 98.83% | 823 | 1.232ms | 1.896ms | 299.4 MB |
| int8 native | 39.6s | 96.69% | 1394 | 0.726ms | 1.170ms | 86.3 MB |
| f32 native | 96.6s | 98.83% | 1010 | 1.003ms | 1.531ms | 299.3 MB |

These are the `efSearch=100` rows from the committed
[`pareto_frontier` DBpedia run](benchmark_results/release/); see that
directory for the full `efSearch` sweep and the cross-library frontier.

Takeaways:

- **int8 is the memory/throughput operating point:** `96.70%` recall at
  `1128 QPS` in WASM, using `3.5x` less memory than float32 (`86 MB` vs
  `299 MB`).
- **float32 is the high-recall operating point:** `98.83%` recall, at the cost
  of memory and a slower build.
- **Native vs WASM isolates runtime overhead:** the native addon runs the same
  engine and reaches the same recall at roughly `~25%` higher QPS (int8) and
  `~20%` higher (f32) — the gap is WASM runtime cost, not graph quality. (On
  larger, more memory-bound corpora the WASM gap narrows; the committed
  SIFT-1M sweep shows WASM and native within a few percent at the operating
  point.)

### QPS-recall frontier

Sweeping `efSearch` (10–800) traces Pancake's full QPS-recall frontier on the
DBpedia-50K data (`M=16`, `efConstruction=50`, single-threaded). The int8
backend climbs the curve to a ceiling near `97.6%` recall — the int8
quantization sets that limit — while the float32 backend continues on to
`~99%`. Use int8 for the memory/throughput operating point and float32 when
you need the higher recall ceiling.

The same `pareto_frontier` sweep also runs native hnswlib, USearch, and Faiss
on the identical data and writes every config's frontier to one set of CSVs, so
the cross-library comparison is in the committed results rather than asserted
here — numbers depend on each library's version, build flags, and hardware.
The committed sweeps in
[`benchmark_results/release/`](benchmark_results/release/) cover DBpedia-50K
(1536D, L2) plus SIFT-1M (128D, L2), GloVe-100 (100D, angular), and
NYTimes-256 (256D, angular); run `pareto_frontier` to reproduce or extend them
on your own data (see [Reproducing](#reproducing)).

### Deletion tolerance

DBpedia 5K x 1536D, L2, int8, `M=16`, `efConstruction=50`, `efSearch=100`, 100
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

### Reproducing

Benchmark scripts are in `benchmarks/`. A shared runner discovers them by name,
so you can list and run them with:

```bash
npm run bench -- --list

# The Pancake-only WASM vs native comparison:
npm run bench -- benchmark_native

# Full comparison including hnswlib, USearch, and Faiss on the same data:
npm run bench -- benchmark_dbpedia_50k_full --count 50000 --m 16 --ef-construction 50 --ef-search 100

# QPS-recall frontier (efSearch sweep) behind the plot above:
npm run bench -- pareto_frontier
```

The first run computes brute-force ground truth and caches it under
`benchmark_results/cache/`; later runs reuse it. Results depend on dimension,
corpus size, HNSW parameters, and hardware.

Other useful scripts:

- `benchmark_dbpedia_50k_full` — unified comparison vs hnswlib, USearch, and Faiss
- `pareto_frontier` — efSearch sweep producing the QPS-recall frontier CSVs
- `benchmark_native` — direct Pancake native-vs-WASM comparison
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

If you only care about native server throughput and do not need portability, Faiss, hnswlib, or USearch are good baselines to compare against — run `benchmark_dbpedia_50k_full` to see how they land against Pancake on your own hardware. On the DBpedia-50K run above, the int8 backend offers a lower-memory operating point with about a 1-2 point recall-ceiling tradeoff versus float32 (`96.70%` vs `98.83%`). The native addon in this repo is there to help separate runtime overhead from graph quality.

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

- [`examples/cli/build_index.js`](https://github.com/mcn92/pancake/blob/main/examples/cli/build_index.js) -- build an HNSW index from precomputed embeddings
- [`examples/demo/technical_demo_cli.js`](https://github.com/mcn92/pancake/blob/main/examples/demo/technical_demo_cli.js) -- interactive local REPL
- [`examples/demo/technical_demo_worker.js`](https://github.com/mcn92/pancake/blob/main/examples/demo/technical_demo_worker.js) -- interactive Worker-targeted REPL
- [`dist/technical-demo.html`](https://github.com/mcn92/pancake/blob/main/dist/technical-demo.html) -- browser demo page with latency and stress views
- [`examples/browser-vite/`](https://github.com/mcn92/pancake/tree/main/examples/browser-vite) -- minimal bundled browser consumer fixture used by `npm run test:browser`
- [`examples/worker/`](https://github.com/mcn92/pancake/tree/main/examples/worker) -- reference Cloudflare Worker deployment built on top of Pancake
- [`examples/worker-semantic-search/`](https://github.com/mcn92/pancake/tree/main/examples/worker-semantic-search) -- snapshot-first semantic docs search demo for Cloudflare Workers

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

Current core suite status on this tree: **781 passed, 0 failed**.

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
- **Deterministic means per target/build.** Given the same inputs, the same build target will produce stable graph structure and query results, but bitwise-identical distances are not guaranteed across WASM SIMD, scalar WASM, AVX2, and SSE2 backends because their reduction orders differ. Treat tiny cross-backend floating-point deltas as expected, not as correctness bugs.
- **One WASM instance can hold at most 64 live indexes.** The handle table in the shipped WASM engine is fixed at `MAX_HANDLES = 64`. If you need more concurrent indexes than that, dispose unused ones or spin up another module instance.
- **Compaction is rebuild-based.** Deletes are soft deletes. Compaction rewrites the graph rather than patching edges in place. That keeps behavior predictable and avoids relying on background maintenance threads.
- **Index instances are not a shared-memory concurrency primitive.** Treat a Pancake index like ordinary mutable in-process state: safe within one JavaScript thread/event loop, but not something to share concurrently across Node worker threads or isolates without your own coordination.
- **Workers are best used as snapshot-serving search frontends.** In a Cloudflare Worker, in-memory state is a warm cache, not durable authority. Persist snapshots explicitly and treat isolate reuse as opportunistic.
- **Inputs are validated, not coerced.** Vectors and queries accept a `Float32Array` or a plain numeric array, but plain-array elements must be actual numbers — a non-numeric element (string, boolean, nested array, `null`) is rejected with an error rather than silently coerced (e.g. an empty CSV field becoming `0`). Non-finite values (`NaN`/`Infinity`) and dimension mismatches are likewise rejected at the boundary.
- **This is an index, not an embedding stack.** Pancake does vector search only. Bring your own embedding pipeline.

## License

Apache 2.0
