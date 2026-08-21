# Pancake

Pancake is a portable search engine toolkit for JavaScript runtimes.

At the bottom is a small WebAssembly HNSW vector engine. On top of that are
immutable search artifact formats that package an index, corpus records,
query interpretation, calibration, and evaluation data into files that can be
served from ordinary static storage.

The current project focus is the **complete search artifact**: one `.pancake`
file that can answer natural-language queries with hydrated, confidence-scored
results using only the file, byte-range reads, and a reader.

```text
query text
  -> artifact-carried query interpreter
  -> vector search over compressed sketch/index data
  -> lazy range reads for records
  -> calibrated hydrated results
```

The original `pancake-wasm` package remains the vector engine and artifact
runtime. It runs in Node.js, browser-bundled apps, and Cloudflare Workers with
no native dependency in the default path.

## What This Repo Contains

Pancake has three layers:

1. **WASM vector engine**
   - In-memory HNSW index.
   - Float32 and row-wise affine uint8 backends.
   - JavaScript API: `Pancake.create()`, `add()`, `search()`, `export()`,
     `restore()`.
   - Source: `src/`, package entrypoints at `pancake.js`,
     `pancake.node.mjs`, `pancake.web.mjs`.

2. **Search artifacts**
   - Immutable, integrity-checked files built from indexes and corpus data.
   - Range-readable from local files, R2, S3, CDNs, or any source that can
     serve byte ranges.
   - Profiles:
     - `.pnck`: engine snapshot envelope.
     - `.pancake-range`: range-readable index/corpus profile.
     - `.pancake-sketch`: resident sketch plus lazy rerank rows.
     - `.pancake`: complete one-file search artifact.
   - Specs: [spec/SEARCH_ARTIFACT_CONTRACT.md](spec/SEARCH_ARTIFACT_CONTRACT.md),
     [spec/SKETCH_PROFILE.md](spec/SKETCH_PROFILE.md), and
     [spec/COMPLETE_PROFILE.md](spec/COMPLETE_PROFILE.md).

3. **Example products and compilers**
   - `examples/05-one-file-search/`: the current flagship path, including the
     complete `.pancake` container and reader.
   - `examples/04-static-wiki-pack/`: wiki-scale corpus preparation and sketch
     artifact pipeline.
   - `examples/03-edge-docs-search/`: Worker search app with bundled snapshot,
     distilled query encoder, and calibrated abstention.
   - `create-pancake-search/`: scaffolded docs-search app generator.

## Why Pancake Exists

Most vector search libraries assume a server process, native binaries, or a
database service. Pancake is aimed at a different deployment shape:

- static files instead of always-on index servers;
- byte-range reads instead of whole-index downloads;
- browser, Worker, and Node readers;
- explicit artifact identity and digest verification;
- query interpretation and evaluation data carried with the artifact;
- compact corpus-side vector representations that keep memory and transfer
  costs practical.

The result is a search package that can be hosted like an asset, not operated
like a database.

## Current Flagship: One-File Search

The complete profile (`.pancake`) packages the five search components into one
content-addressed file:

```text
index       embedded .pancake-sketch artifact
corpus      JSON records behind an offsets table
encoder     query interpreter declaration or inline model data
calibration abstention / match-quality model
evaluation  golden queries and expected behavior
```

The reader verifies the manifest and eager segments, opens the sketch index
against a range source, and hydrates only the records needed for the final
results.

The newest complete-profile path supports three query-interpretation kinds:

- **kind 1: `student-inline-v1`**
  - a small inline student encoder in JavaScript.
- **kind 2: `external-transformers-v1`**
  - the artifact declares and verifies an external host-supplied encoder.
- **kind 3: `inline-transformer-v1`**
  - the artifact carries WordPiece vocab and quantized MiniLM teacher weights
    as verified data; the reader ships the execution kernels.

Kind 3 is the first fully self-contained open-domain artifact in this repo:
the file carries the corpus, index, tokenizer, encoder weights, calibration,
and evaluation data. The reader supplies code; the artifact supplies data.

```bash
cd examples/05-one-file-search
node compile-wiki.mjs --inline-encoder
node compile.mjs --inspect pancake-wiki-inline.pancake
node test-inline.mjs
```

Recent local acceptance for the inline wiki artifact:

```text
456,153 records
~537 MiB complete artifact (562,737,657 bytes)
manifest identity: b25ff90d074fe889f02f6249ca5d4ce95099f2e1b04b9c1f71bd23f6d61b3828
recall@10: 82.4% over the 200-query pre-registered eval set, against brute-force ground truth
natural-language query served with no host encoder option, ~228 ms/query end to end locally
```

Release asset (v2 — declaration states the kernel's real `maxTokens: 128`
and carries verification vectors; index, corpus, and evaluation bytes are
identical to v1, which stays published for older checkouts):
`https://github.com/mcn92/pancake/releases/download/artifact-wiki-inline-v2/pancake-wiki-inline.pancake`

`node test-inline.mjs` downloads that file automatically when it is missing
locally, then verifies the manifest identity before running the acceptance
checks.

See [examples/05-one-file-search/README.md](examples/05-one-file-search/README.md)
for the full one-file walkthrough.

## Core Design Choice

The vector engine's quantized backend stores corpus vectors as row-wise affine
uint8:

```text
value ~= offset[row] + scale[row] * byte
```

That single constraint propagates through the system:

- HNSW search keeps the query as float32 and scores it against compressed
  corpus rows.
- Distance kernels fuse dequantization with dot/L2 accumulation.
- SIMD paths load bytes, widen lanes, apply scale/offset, and accumulate.
- Graph maintenance uses symmetric node-to-node distances derived from the
  compressed rows and row statistics.
- Snapshots serialize compact scale/offset/qdata arrays.
- Sketch artifacts scan compressed resident rows and fetch only rerank
  candidates.
- Range-backed readers keep the large corpus side compressed while queries
  stay precise and transient.

This is why Pancake's artifact work and engine work are connected: the same
corpus-side representation supports in-memory search, serialized snapshots,
resident sketches, range reads, and reranking.

## Install

From npm:

```bash
npm install pancake-wasm
```

From a checkout:

```bash
git clone https://github.com/mcn92/pancake.git
cd pancake
npm install
```

The checkout includes prebuilt WASM in `dist/`, so you do not need to rebuild
the engine unless you edit `src/`.

## Runtime Entry Points

Published package:

```js
// Node.js CJS
const Pancake = require('pancake-wasm');

// Node.js ESM
import Pancake from 'pancake-wasm';

// Explicit Node/browser entries
import Pancake from 'pancake-wasm/node';
import Pancake from 'pancake-wasm/web';

// Search Artifact layer (range + sketch readers/builders)
import { PancakeRangeArtifact, buildRangeArtifact } from 'pancake-wasm/artifact';

// Complete one-file profile: reader (any runtime) and builder (Node)
import { openPancakeFile } from 'pancake-wasm/complete';
import { assemblePancakeFile } from 'pancake-wasm/complete/builder';
```

Repository checkout:

```js
const Pancake = require('./pancake.js');
import Pancake from './pancake.node.mjs';
import Pancake from './pancake.web.mjs';
```

The browser entry expects the runtime or bundler to resolve
`dist/engine.wasm`.

## Engine Quick Start

Use the core engine when you already have vectors and want an in-memory ANN
index.

```js
import Pancake from 'pancake-wasm';

const index = await Pancake.create({
  dim: 384,
  maxElements: 100000,
  metric: 'cosine',
  quantized: true,
});

const docVector = new Float32Array(384).fill(0.1);   // your embedding here
const queryVector = new Float32Array(384).fill(0.1); // your query embedding here

const id = index.add(docVector);

const results = index.search(queryVector, 10);
// [{ id: 0, distance: 0.12 }, ...]

index.delete(id);
index.compact();

const snapshot = index.export();
const restored = await Pancake.restore(snapshot, { maxElements: 100000 });

index.dispose();
restored.dispose();
```

If you have records with vectors:

```js
const rows = [
  { id: 'doc-1', vector: embedding1 },
  { id: 'doc-2', vector: embedding2 },
];

const { index, idMap } = await Pancake.fromVectors(rows, {
  metric: 'cosine',
  quantized: true,
});

const hits = index.search(queryEmbedding, 5);
console.log(idMap.get(hits[0].id));
```

On Node entrypoints, file helpers are available:

```js
const { index, idMap } = await Pancake.loadJsonFile('vectors.jsonl', {
  metric: 'cosine',
  vectorKey: 'embedding',
  idKey: 'docId',
  maxFileBytes: 64 * 1024 * 1024,
});

const restored = await Pancake.loadSnapshotFile('index.pnck', {
  dim: 384,
  maxElements: 100000,
  metric: 'cosine',
  quantized: true,
});
```

## Complete Artifact Quick Start

Use the complete reader when you want a file that is itself the search
application data plane.

```js
import { openPancakeFile } from 'pancake-wasm/complete'; // or './complete/index.mjs' in a checkout

const search = await openPancakeFile('examples/05-one-file-search/pancake-docs.pancake');

const out = await search.query('how do workers restore snapshots', { k: 5 });

console.log(out.matchQuality, out.confidence);
console.log(out.results[0].title);

await search.close();
```

For kind-2 artifacts, pass `options.encodeQuery`. For kind-3 artifacts, no
host encoder is required:

```js
const search = await openPancakeFile('pancake-wiki-inline.pancake');
const out = await search.query('how do volcanoes form', { k: 5 });
```

The reader also accepts a range source:

```js
const search = await openPancakeFile({
  size,
  async read(offset, length) {
    // return bytes for [offset, offset + length)
  },
});
```

## Demos

Start with [DEMOS.md](DEMOS.md) for the maintained demo path.

Useful commands:

```bash
npm run demo                  # synthetic engine validation
npm run demo:artifact         # range artifact search
npm run demo:sketch           # sketch artifact search/rerank path
npm run test:browser          # browser smoke test
node examples/05-one-file-search/test-file.mjs
node examples/05-one-file-search/test-wiki.mjs
node examples/05-one-file-search/test-inline.mjs
```

The one-file example has the most current project direction:

```bash
cd examples/05-one-file-search
node compile.mjs
node test-file.mjs
node serve.mjs
```

## API Snapshot

Core index:

```js
const index = await Pancake.create(options);
index.add(vector);
index.addBatch(vectors);
index.search(query, k, options?);
index.searchFiltered(query, k, bitset, options?);
index.delete(id);
index.compact();
index.export();
index.dispose();
```

Common options:

```js
{
  dim: 384,
  maxElements: 100000,
  metric: 'cosine',       // 'cosine' or 'l2'
  quantized: true,
  M: 12,
  efConstruction: 75,
  efSearch: 100,
  seed: 108
}
```

Snapshot helpers:

```js
await Pancake.restore(bytes, options);
await Pancake.loadSnapshotFile(path, options);
await Pancake.loadJsonFile(path, options);
await Pancake.fromVectors(rows, options);
```

Artifact APIs live in `pancake-artifact.js` and the example readers. The
complete profile is still a draft profile, so its reference implementation is
kept under `examples/05-one-file-search/` while the format settles.

## Building From Source

You only need this if you change the C++ engine under `src/`.

Requirements:

- Node.js
- Emscripten SDK

Build:

```bash
npm run build
```

Build scalar and SIMD variants:

```bash
npm run build:all
```

The build emits `dist/engine.js`, `dist/engine.wasm`, and scalar variants.

## Tests

Core:

```bash
npm test
npm run test:fuzz
npm run test:simd
npm run test:browser
```

Complete profile:

```bash
node examples/05-one-file-search/test-file.mjs
node examples/05-one-file-search/test-wiki.mjs
node examples/05-one-file-search/test-inline.mjs
```

The inline wiki acceptance test downloads the large release artifact on demand
when `examples/05-one-file-search/pancake-wiki-inline.pancake` is missing.

## Performance Notes

Pancake's performance story is workload-dependent, but the main knobs are:

- `quantized: true` for much smaller corpus-side vector storage.
- `efSearch` for recall/latency tradeoff.
- sketch artifacts for depth-1 candidate selection before lazy rerank reads.
- corpus layout and clustering for fewer remote read rounds.
- kind-3 inline encoders when deployment should not depend on an external ML
  runtime.

The older engine-level benchmarks are still useful for HNSW behavior. The
newer artifact-level benchmarks are more representative of the current
project goal: search from static files and range-readable storage.

See `benchmarks/`, `benchmark_results/`, and the example READMEs for current
numbers.

## Tradeoffs

Pancake is a good fit when:

- you want search to run in JavaScript runtimes without native addons;
- you want static-file or edge deployment;
- you need reproducible, inspectable search artifacts;
- your corpus can be built offline;
- byte-range reads are cheaper than operating a dedicated search service.

Pancake is not the right tool when:

- you need high-rate online mutation;
- you need a multi-tenant database server;
- you require exact nearest-neighbor search at large scale;
- your deployment can already use a mature native ANN service with no
  portability constraints.

## Status

The npm engine is usable today. The artifact profiles are actively evolving.
The complete `.pancake` profile is still marked draft, but it is now the
clearest expression of the project direction: a search engine packaged as one
verified, range-readable file.

## License

Apache-2.0. See [LICENSE](LICENSE).
