# Pancake Quick Start

End-to-end repository workflow for building an HNSW index from your own embeddings, searching it locally, and validating the Cloudflare Worker example.

## What This Guide Covers

This guide is for the repository checkout, not just the published npm package. It uses the example CLI scripts in `examples/cli/` and the Worker/demo files in `examples/worker/` and `examples/demo/`.

If you only want the published JS API, start with [README.md](README.md).

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
- 8 bytes: reserved (zeros)
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
const buf = Buffer.alloc(16 + count * dim * 4);
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

Useful parameters:

- `--m 16`: balanced default for graph connectivity
- `--ef-construction 200`: balanced default for build quality
- `--no-quantize`: build a float32 index instead of the default int8 index

This produces a serialized index binary that can be imported by the CLI search flow or by lower-level engine consumers.

### 3. Search The Index

Single query (requires the same embedder you used in step 1 to convert the query text into a vector):

```bash
node examples/cli/search_index.js \
  --index my_index.bin \
  --metadata my_embeddings_metadata.json \
  --query "neural networks and deep learning" \
  --k 5 \
  --ef 100
```

Interactive mode:

```bash
node examples/cli/search_index.js \
  --index my_index.bin \
  --metadata my_embeddings_metadata.json \
  --interactive
```

Useful parameters:

- `--k`: number of results
- `--ef`: query-time beam width
- `--float`: load a float32 index instead of the default int8 path

## Worker Example

The Worker example is the main edge deployment story for Pancake: the Worker loads the WASM engine and runs ANN search in-process, so the Worker itself becomes the vector search service.

Typical flow:

1. Build or export an index locally.
2. Start the Worker example.
3. Initialize/import the index into the Worker.
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

- `test_worker.js`: synthetic `1536D` Worker/API integration test (no external data required)
- `technical_demo_worker.js`: interactive REPL against the Worker with 5,000 precomputed 384D embeddings from `dist/vectors.bin`; includes a full proof suite and adversarial stress tests

The Worker example includes:

- in-process ANN search inside the Worker
- `/init`, `/add`, `/add_batch`, `/delete`, `/compact`, `/search`, `/export`, `/import`, `/stats`, and `/health`
- optional R2-backed persistence
- optional bearer-token auth and per-IP rate limiting

See `examples/worker/worker.js`.

## Sizing And Tuning

General levers:

- higher `M` improves recall and increases graph memory/build cost
- higher `ef-construction` improves build quality and slows index construction
- higher `ef` improves recall and slows search
- int8 quantization reduces memory substantially versus float32

For memory-constrained deployments such as Cloudflare Workers, the int8 backend is the natural default.

## Benchmarks

See the [Performance section of README.md](README.md#performance) for benchmark scripts, datasets, and results. Benchmark outcomes depend heavily on:

- dimension
- corpus size
- `M`
- `ef-construction`
- `ef`
- quantized vs float mode

## Troubleshooting

### Out of memory during index build

- split large corpora into smaller jobs
- stay on the default int8 path
- reduce dimension or corpus size

### Low recall

- raise `--ef-construction` when building
- raise `--ef` when searching
- raise `--m` and rebuild
- verify your embeddings are L2-normalized if using cosine distance

### Worker does not start from the repo root

Run Wrangler from `examples/worker` or pass the Worker config explicitly:

```bash
npx wrangler dev examples/worker/worker.js --config examples/worker/wrangler.toml --port 8787
```

## Next Steps

- [README.md](README.md) for the published JS/WASM API surface
- `examples/cli/` for repository-local indexing workflows
- `examples/demo/` for validation and proof scripts
- `examples/worker/` for the Cloudflare Worker deployment example