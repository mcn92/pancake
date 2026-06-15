# Pancake Quick Start

End-to-end repository workflow for building an HNSW index from your own embeddings, searching it locally, and running the Cloudflare Worker example.

## What This Guide Covers

This guide is for the repository checkout. It uses the example CLI scripts in `examples/cli/` and the Worker/demo files in `examples/worker/` and `examples/demo/`.

> **Note:** Pancake is not yet published to npm — there is no `npm install
> pancake-wasm` today. Everything in this guide runs from a repository checkout,
> which is currently the supported way to use Pancake. npm publishing is coming
> soon. See [README.md](README.md#install) for details.

## Prerequisites

Build the WASM engine first:

```bash
./build.sh
```

That produces:

- `dist/engine.js`
- `dist/engine.wasm`

The CLI and Worker examples load those files directly from this repository.

## Local Document Workflow

Pancake is an ANN library — it doesn't ship an embedding model. Bring your own embeddings, then use the CLI scripts to build and query an index.

### 1. Generate Embeddings

Use whatever embedder fits your use case (sentence-transformers, OpenAI API, Cohere, etc.). Write the resulting vectors to a binary file with this header format:

- 4 bytes: dimension (uint32 LE)
- 4 bytes: count (uint32 LE)
- 8 bytes: reserved (must be zero — the loader skips these bytes; they exist for future use)
- count × dim × 4 bytes: float32 vectors, row-major

Minimal Node example using [`@xenova/transformers`](https://github.com/xenova/transformers.js) with `all-MiniLM-L6-v2` (384D):

```js
import { pipeline } from '@xenova/transformers';
import fs from 'node:fs';

const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

// Your documents
const texts = [
  "Your first document here",
  "Your second document here",
  "Your third document here",
];

const embeddings = [];
for (const text of texts) {
  const out = await embedder(text, { pooling: 'mean', normalize: true });
  embeddings.push(new Float32Array(out.data));
}

// Write in Pancake's format
const dim = embeddings[0].length;
const count = embeddings.length;
const buf = Buffer.alloc(16 + count * dim * 4); // alloc zero-fills (covers reserved bytes)
buf.writeUInt32LE(dim, 0);
buf.writeUInt32LE(count, 4);
for (let i = 0; i < count; i++) {
  for (let j = 0; j < dim; j++) {
    buf.writeFloatLE(embeddings[i][j], 16 + (i * dim + j) * 4);
  }
}
fs.writeFileSync('my_embeddings.bin', buf);

// Also save metadata alongside so you can map search results back to documents
fs.writeFileSync(
  'my_embeddings_metadata.json',
  JSON.stringify(texts.map((t, id) => ({ id, text: t })), null, 2)
);
```

### 2. Build An HNSW Index

```bash
node examples/cli/build_index.js \
  --embeddings my_embeddings.bin \
  --output my_index.bin \
  --m 16 \
  --ef-construction 200
```

The default metric is cosine. The build script infers the dimension and vector count from the binary header.

Useful parameters:

- `--m 16`: graph connectivity (typical range: 8–32; higher improves recall, costs more memory)
- `--ef-construction 200`: build beam width (typical range: 100–400; higher improves graph quality, slows build)
- `--no-quantize`: build a float32 index instead of the default int8 index

This produces a serialized index binary that can be imported via the JS API or the Worker.

### 3. Search The Index

The repo exposes the same JS API that will ship in the npm package via `pancake.js` at the root:

```js
const Pancake = require('./pancake.js');
const fs = require('fs');

// maxElements sets index capacity — can exceed current count to leave room for additions
const index = await Pancake.create({ dim: 384, maxElements: 10000 });
index.import(fs.readFileSync('my_index.bin'));

// Query with a pre-computed embedding vector
const results = index.search(queryVector, 5);
console.log(results); // [{ id: 0, distance: 0.12 }, ...]
```

Use the `id` field to look up the original document in `my_embeddings_metadata.json`.

Or import the index into the Worker (see [Worker Example](#worker-example) below) and query via HTTP. Note: the HTTP API uses `dims` (plural) while the JS API uses `dim` (singular) — a historical inconsistency kept for backward compatibility:

```bash
curl -X POST http://localhost:8787/import?dims=384 --data-binary @my_index.bin
curl -X POST http://localhost:8787/search \
  -H 'Content-Type: application/json' \
  -d '{"query": [0.1, 0.2, ...], "k": 5}'
```

## Worker Example

The Worker example is the main edge deployment story for Pancake: the Worker loads the WASM engine and runs ANN search in-process, so the Worker itself becomes the vector search service.

Typical flow:

1. Build or export an index locally.
2. Start the Worker example.
3. Load the index into the Worker — use `/init` to create a new empty index and add vectors via `/add`, or use `/import` to load a pre-built index binary from step 2.
4. Query `/search` directly without a separate ANN backend.

### Run The Worker Locally

```bash
cd examples/worker
npx wrangler dev --port 8787
```

From the repo root in another terminal:

```bash
node examples/demo/test_worker.js http://localhost:8787
node examples/demo/technical_demo_worker.js
```

The demos exercise different paths:

- `test_worker.js`: synthetic 1536D Worker/API integration test (no external data required)
- `technical_demo_worker.js`: interactive REPL against the Worker with 5,000 precomputed 384D embeddings from `dist/vectors.bin`; includes a full proof suite and adversarial stress tests

The Worker example includes:

- in-process ANN search inside the Worker
- `/init`, `/add`, `/add_batch`, `/delete`, `/compact`, `/search`, `/export`, `/import`, `/stats`, and `/health`
- optional R2-backed persistence
- optional bearer-token auth and per-IP rate limiting

See `examples/worker/worker.js`.

## Sizing And Tuning

| Parameter | Typical range | Effect |
|-----------|--------------|--------|
| `M` | 8–32 | Graph connectivity. Higher improves recall, increases memory and build time. Default: 16. |
| `ef-construction` | 100–400 | Build beam width. Higher improves graph quality, slows build. Default: 200. |
| `ef` (search) | 50–500 | Query beam width. Higher improves recall, slows search. Default: 100. |
| `quantized` | true/false | Int8 quantization reduces memory ~4× with typically <2% recall loss on standard benchmarks (SIFT-1M, NYTimes-256). Default: true. |
| `metric` | `'cosine'` / `'l2'` | Distance metric. Cosine for normalized embeddings, L2 for unnormalized. Default: `'cosine'`. |

For memory-constrained deployments such as Cloudflare Workers (128 MB limit), the int8 backend is the natural default. See [README.md](README.md#deployment-notes) for per-vector memory estimates.

## Benchmarks

Benchmark scripts are in `benchmarks/`. See the [Performance section of README.md](README.md#performance) for datasets and results. Benchmark outcomes depend heavily on dimension, corpus size, HNSW parameters, and quantization mode.

## Troubleshooting

### Out of memory during index build

- split large corpora into smaller jobs
- stay on the default int8 path
- reduce dimension or corpus size

### Low recall

- raise `--ef-construction` when building
- raise `--ef` when searching
- raise `--m` and rebuild
- verify your embeddings are L2-normalized if using cosine distance (the `metric` option defaults to `'cosine'`, which assumes unit-norm vectors for best results)

### Worker does not start from the repo root

Run Wrangler from `examples/worker` or pass the Worker config explicitly:

```bash
npx wrangler dev examples/worker/worker.js --config examples/worker/wrangler.toml --port 8787
```

## Next Steps

- [README.md](README.md) for the published JS/WASM API surface
- [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) for the full system design document
- `examples/cli/build_index.js` for building indexes from precomputed embeddings
- `examples/demo/` for validation and proof scripts
- `examples/worker/` for the Cloudflare Worker deployment example
- `benchmarks/` for reproducible benchmark scripts
