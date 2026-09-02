# Pancake

Pancake is a search engine you host like a file.

At the bottom there's a small WebAssembly HNSW vector engine that runs
anywhere JavaScript does — Node, browsers, Cloudflare Workers — with no
native dependencies. On top of it sit immutable search artifact formats:
files that package an index, corpus records, query interpretation,
calibration, and evaluation data, and that can be served from any storage
that answers byte-range requests.

Where this has been heading is the **complete search artifact**: one
`.pancake` file that answers natural-language queries with hydrated,
confidence-scored results, using nothing but the file, range reads, and a
reader.

```text
query text
  -> artifact-carried query interpreter
  -> vector search over compressed sketch/index data
     + BM25 over the embedded lexical index, rank-fused
  -> lazy range reads for records
  -> calibrated hydrated results
```

The `pancake-wasm` package carries both the vector engine and the artifact
runtime.

## Knowledge Packs

The same file doubles as a **knowledge pack** for LLMs. A `.pancake` is a
portable, immutable corpus that contains its own query encoder, semantic
and lexical indexes, calibrated abstention, provenance, and integrity
metadata. Publish it as a static file; an MCP client attaches and queries
it directly over HTTP range requests — no index to rebuild, no vector
database to run, no embedding service to call:

```bash
npx create-pancake-search mcp install --client claude-code \
  --pack https://github.com/mcn92/pancake/releases/download/artifact-wiki-inline-v4/pancake-wiki-inline.pancake#1b180adf4c6cebb2dcd5615256df6a25dac5fda8738dbbc11d60af86046f97f3
```

Then ask questions. That line pins 456,153 passages of Simple English
Wikipedia by content hash: the mount transfers a ~52 MiB resident slice
of the 649 MiB file, each question costs ~127 range reads, results carry
their provenance (pack identity, title, section, source), and a pack that
cannot answer says so instead of guessing. Details in the
create-pancake-search README ("Attaching packs to an LLM") and `packs/`.

## What's in This Repo

Three layers, bottom to top:

1. **WASM vector engine**
   - In-memory HNSW index with float32 and row-wise affine uint8 backends.
   - JavaScript API: `Pancake.create()`, `add()`, `search()`, `export()`,
     `restore()`.
   - Source in `src/`, package entrypoints at `pancake.js`,
     `pancake.node.mjs`, `pancake.web.mjs`.

2. **Search artifacts**
   - Immutable, integrity-checked files built from indexes and corpus data,
     range-readable from local files, R2, S3, CDNs — anything that serves
     byte ranges.
   - One product format, one index layer beneath it:
     - `.pancake`: the complete one-file search artifact — the product.
     - `.pancake-sketch`: its index segment (resident sketch plus lazy
       rerank rows), also usable standalone; format 2 (the builder's
       default) interleaves per-row digests with the rows, so every row a
       query fetches is verified on that read.
   - Two adjacent formats that are not search artifacts in the contract
     sense:
     - `.pnck`: the engine's snapshot envelope (`export()`/`restore()`
       serialization), consumed by the compilers and by layer-1 users.
     - `.pancake-range`: **deprecated** range-readable index profile. The
       sketch geometry replaced it (5x faster over real networks at a
       third of the size); readers stay supported for existing artifacts,
       but no new ones should be built.
   - Specs: [spec/SEARCH_ARTIFACT_CONTRACT.md](spec/SEARCH_ARTIFACT_CONTRACT.md),
     [spec/SKETCH_PROFILE.md](spec/SKETCH_PROFILE.md), and
     [spec/COMPLETE_PROFILE.md](spec/COMPLETE_PROFILE.md).

3. **Products and compilers**
   - `create-pancake-search/`: the fast path. One command compiles a docs
     folder or a live site into a complete `.pancake` file:
     ```bash
     npx create-pancake-search compile --source ./docs --out search.pancake
     ```
     or scaffolds a deployable search app:
     ```bash
     npm create pancake-search -- --name my-docs-search --source ./docs --no-deploy --yes
     ```
   - `examples/05-one-file-search/`: the flagship path — the complete
     `.pancake` container, its compilers, and acceptance tests.
   - `examples/04-static-wiki-pack/`: wiki-scale corpus preparation and the
     sketch artifact pipeline.
   - `examples/03-edge-docs-search/`: a Worker search app with a bundled
     snapshot, distilled query encoder, and calibrated abstention.

## Why Pancake Exists

Most vector search libraries assume a server process, native binaries, or a
database service. Pancake is built for a different deployment shape:

- static files instead of always-on index servers;
- byte-range reads instead of whole-index downloads;
- readers for browsers, Workers, and Node;
- explicit artifact identity and digest verification;
- query interpretation and evaluation data carried inside the artifact;
- compact corpus-side vector representations that keep memory and transfer
  costs sane.

The result is search you can host like an asset instead of operating like a
database.

## The Flagship: One-File Search

The complete profile (`.pancake`) packs the five components of a search
application into one content-addressed file:

```text
index       embedded .pancake-sketch artifact
corpus      JSON records behind an offsets table
encoder     query interpreter declaration or inline model data
calibration abstention / match-quality model
evaluation  golden queries and expected behavior
lexical     BM25 inverted index for hybrid retrieval (optional; readers
            that predate it serve vector-only from the same file)
```

The reader verifies the manifest and eager segments, opens the sketch index
against a range source, and hydrates only the records the final results
need.

Query interpretation comes in three kinds. Each picks two of
**self-contained**, **small**, and **teacher-quality**, and gives up the
third — that trade-off, not history, is why all three exist:

- **kind 3: `inline-transformer-v1`** — self-contained + teacher quality.
  The artifact carries the WordPiece vocab and quantized MiniLM teacher
  weights as verified data; the reader ships the execution kernels. Costs
  ~25 MB of encoder no matter how small the corpus. This is the default:
  `create-pancake-search compile` builds kind 3.
- **kind 1: `student-inline-v1`** — self-contained + small. A
  corpus-distilled student encoder (~1 MiB), pure-JS execution. The only
  way a small docs corpus compiles to a file measured in single-digit MB;
  quality is bounded by the distillation.
- **kind 2: `external-transformers-v1`** — small + teacher quality, but
  not self-contained: the artifact pins a host-supplied encoder (model id,
  pooling, normalization, max tokens) and verifies it against embedded
  test vectors before serving. The escape hatch for encoders too big to
  inline and hosts that already run one.

Kind 3 is the flagship: the file carries the corpus, index, tokenizer,
encoder weights, calibration, and evaluation data — the reader supplies
code, the artifact supplies everything else. Ask it about volcanoes and no
service, model host, or network dependency is involved beyond range reads
of the file itself.

```bash
cd examples/05-one-file-search
node compile-wiki.mjs --inline-encoder
node compile.mjs --inspect pancake-wiki-inline.pancake
node test-inline.mjs
```

Where the inline wiki artifact currently stands:

```text
456,153 records
~649 MiB complete artifact (680,029,254 bytes)
manifest identity: 1b180adf4c6cebb2dcd5615256df6a25dac5fda8738dbbc11d60af86046f97f3
recall@10: 95.2% at the recommended rerank (97.0% at C=600), against exact
  fp32 brute-force ground truth on the 200-query pre-registered eval —
  98.5% on hand-written natural-language questions, 94.8% on single-title
  lookups (augmented retrieval: hybrid candidates, distance order)
serves over HTTP range reads: opens on ~52 MiB of the file, ~127 range
  requests per query (cluster-ordered row layout + manifest fetch hints)
natural-language query with no host encoder option, ~110 ms/query end to
  end locally (the resident scan auto-stages the engine's SIMD kernel at
  this scale; ~540 ms on the pure-JS scan it falls back to)
```

Release asset (v4 — recovers the pack's canonical build: k-means
cluster-ordered rows and the 192-dim 2:1-pooled sketch, where v1–v3
embedded a 96-dim 4:1 sketch in unpermuted order and measured 82.4%; adds
the hybrid BM25 lexical segment and a format-2 sketch with per-row read
verification; earlier versions stay published for older checkouts):
`https://github.com/mcn92/pancake/releases/download/artifact-wiki-inline-v4/pancake-wiki-inline.pancake`

`node test-inline.mjs` downloads that file automatically when it is missing
locally, verifies the manifest identity, and then runs the acceptance
checks.

See [examples/05-one-file-search/README.md](examples/05-one-file-search/README.md)
for the full one-file walkthrough.

## The Design Choice Everything Hangs On

The engine's quantized backend stores corpus vectors as row-wise affine
uint8:

```text
value ~= offset[row] + scale[row] * byte
```

That one constraint propagates through the whole system:

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

This is why the engine work and the artifact work belong in one project:
the same corpus-side representation serves in-memory search, serialized
snapshots, resident sketches, range reads, and reranking.

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

The checkout includes prebuilt WASM in `dist/`, so you don't need to
rebuild the engine unless you edit `src/`.

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

// Search Artifact layer (sketch readers/builders; the deprecated
// .pancake-range reader stays exported for existing artifacts)
import { PancakeSketchArtifact, buildSketchArtifact } from 'pancake-wasm/artifact';

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
application data plane. To build one from your own docs, see the compile
command above (`npx create-pancake-search compile`).

```js
import { openPancakeFile } from 'pancake-wasm/complete'; // or './complete/index.mjs' in a checkout

const search = await openPancakeFile('examples/05-one-file-search/pancake-docs.pancake');

const out = await search.query('how do workers restore snapshots', { k: 5 });

console.log(out.matchQuality, out.confidence);
console.log(out.results[0].title);

await search.close();
```

For kind-3 artifacts, no host encoder is needed:

```js
const search = await openPancakeFile('pancake-wiki-inline.pancake');
const out = await search.query('how do volcanoes form', { k: 5 });
```

For kind-2 artifacts, pass `options.encodeQuery`; the reader runs it
against the declaration's verification vectors before serving and refuses
the open if they disagree (`info().encoderVerified` reports the outcome).

The reader also accepts a range source, which is how a `.pancake` on R2 or
a CDN gets queried without downloading it:

```js
const search = await openPancakeFile({
  size,
  async read(offset, length) {
    // return bytes for [offset, offset + length)
  },
});
```

On integrity: every artifact-derived read is bounds-checked and budgeted
(`maxReadBytes`, `maxRecordBytes`). Format-2 files (`pancake-complete-v2`,
the builder's default) verify each hydrated record against a per-record
digest on its own range read (`info().corpusIntegrity`), and an embedded
format-2 sketch verifies each lazily fetched index row the same way
(`info().indexRowIntegrity`) — so a tampered transport fails the query
instead of skewing its results. Older files with a format-1 sketch commit
the lazy index rows to the identity but don't verify them per read; for
those, pass `verifyIndexVectors: true` (or call `verifyVectors()`) to
authenticate them in one full pass before trusting results from an
untrusted transport (spec/COMPLETE_PROFILE.md section 6).

### Attaching packs to an LLM

A complete artifact doubles as a portable knowledge pack: everything an
LLM needs to query a body of knowledge — corpus, indexes, encoder,
calibrated abstention, integrity commitments — in one immutable,
identity-pinned file. `create-pancake-search mcp` serves packs over the
Model Context Protocol so Claude Code, Claude Desktop, or any agent
framework can attach them as a retrieval tool with one config line — no
vector database, no embedding service, no retrieval backend. Results
carry full provenance (pack, manifest identity, title, heading path,
source), and a pack that cannot answer says so instead of returning
plausible noise. See the create-pancake-search README's
"Attaching packs to an LLM (MCP)" section.

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

The one-file example is the closest thing to the project's current center
of gravity:

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
index.searchFiltered(query, k, allowedIds, options?); // allowedIds: Set of ids
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

Range and sketch artifact APIs ship at `pancake-wasm/artifact`; the
complete-profile reader and builder ship at `pancake-wasm/complete` and
`pancake-wasm/complete/builder`. The wiki-scale compilers and acceptance
tests live in `examples/05-one-file-search/`.

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

Core (`npm test` also runs the artifact and complete-profile conformance
suites):

```bash
npm test
npm run test:fuzz
npm run test:simd
npm run test:browser
npm run test:scaffold
```

Complete profile, end to end:

```bash
node examples/05-one-file-search/test-file.mjs
node examples/05-one-file-search/test-wiki.mjs
node examples/05-one-file-search/test-inline.mjs
```

The inline wiki acceptance test downloads the large release artifact on
demand when `examples/05-one-file-search/pancake-wiki-inline.pancake` is
missing.

## Performance Notes

Performance is workload-dependent, but these are the knobs that matter:

- `quantized: true` for much smaller corpus-side vector storage.
- `efSearch` for the recall/latency tradeoff.
- sketch artifacts for depth-1 candidate selection before lazy rerank
  reads.
- corpus layout and clustering for fewer remote read rounds.
- kind-3 inline encoders when deployment shouldn't depend on an external
  ML runtime.

The older engine-level benchmarks are still useful for HNSW behavior. The
newer artifact-level benchmarks are more representative of what the project
is actually for: search from static files and range-readable storage.

The native baselines the engine benchmarks compare against (faiss-node,
hnswlib-node, usearch) are not dependencies of `pancake-wasm`; install
them on demand with `cd benchmarks && npm install` before running those
scripts. See `benchmarks/`, `benchmark_results/`, and the example READMEs
for current numbers.

## Tradeoffs

Pancake is a good fit when:

- you want search in JavaScript runtimes without native addons;
- you want static-file or edge deployment;
- you need reproducible, inspectable search artifacts;
- your corpus can be built offline;
- byte-range reads are cheaper than operating a dedicated search service.

It's the wrong tool when:

- you need high-rate online mutation;
- you need a multi-tenant database server;
- you require exact nearest-neighbor search at large scale;
- your deployment can already use a mature native ANN service with no
  portability constraints.

## Status

The engine and the artifact readers/builders are published and usable
today (`pancake-wasm`, with `create-pancake-search` as the app scaffold).
The complete `.pancake` profile's spec is still marked draft — Draft 2,
under review, not frozen — but its reader and builder ship in the package,
and it is the clearest expression of where the project is going: a search
engine packaged as one verified, range-readable file.

## License

Apache-2.0. See [LICENSE](LICENSE).
