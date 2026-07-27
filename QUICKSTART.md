# Pancake Quick Start

Practical getting-started paths for building a Pancake index from your own embeddings, searching it locally, and running the Cloudflare Worker example.

## What This Guide Covers

This guide focuses on the current supported flows:

- Node.js indexing from vectors already in memory
- Node.js indexing from JSON / JSONL / NDJSON files
- Snapshot export / restore in Node.js
- Running the reference Cloudflare Worker example from a repository checkout

Pancake is an ANN library. It does not embed raw documents for you, and it is not a hosted search service. Bring your own embedding model or embedding pipeline, then feed the resulting vectors into Pancake.

## Install

### Published package

```bash
npm install pancake-wasm
```

This guide documents the Pancake 0.2 contract; it requires `pancake-wasm@0.2.0`
or later.

### Repository checkout

```bash
git clone https://github.com/mcn92/pancake.git
cd pancake
npm run build:all
```

`npm run build:all` produces all of the WASM engine artifacts used by the local entrypoints and Worker example:

- `dist/engine.js`
- `dist/engine.wasm`
- `dist/engine.scalar.js`
- `dist/engine.scalar.wasm`

(`./build.sh` builds only the SIMD pair, `dist/engine.{js,wasm}`; use `build:all` when you also need the scalar fallback.)

## Pick An Ingest Path

Use the path that matches what you already have:

- Vectors already in memory: `Pancake.fromVectors(...)`
- Vectors saved as JSON / JSONL: `Pancake.loadJsonFile(...)` on the Node entrypoints
- Existing Pancake snapshot on disk: `Pancake.loadSnapshotFile(...)` on the Node entrypoints

If you are working from a repo checkout, replace `import Pancake from 'pancake-wasm'` with `import Pancake from './pancake.node.mjs'` (or `require('./pancake.js')` from CommonJS code that awaits inside an async function).

## Local Node.js Workflow

### 1. Build An Index From In-Memory Vectors

```js
import Pancake from 'pancake-wasm';

const rows = [
  { id: 'doc-1', vector: [1, 0, 0, 0] },
  { id: 'doc-2', vector: [0, 1, 0, 0] },
  { id: 'doc-3', vector: [0, 0, 1, 0] },
];

const { index, ids, idMap } = await Pancake.fromVectors(rows, {
  metric: 'cosine',
  quantized: true,
});

const results = index.search(new Float32Array([1, 0, 0, 0]), 2);
console.log(results);
console.log(idMap.get(results[0].id)); // -> 'doc-1'
```

Use this path when your embedder already returns arrays or `Float32Array`s in the current process.

### 2. Build An Index From JSON Or JSONL

On the Node.js entrypoints, Pancake can load vectors directly from disk:

```js
import Pancake from 'pancake-wasm';

const { index, ids, idMap } = await Pancake.loadJsonFile('vectors.jsonl', {
  metric: 'cosine',
  quantized: true,
  vectorKey: 'embedding', // default: 'vector'
  idKey: 'docId',         // default: 'id'
  maxFileBytes: 64 * 1024 * 1024,
});

const results = index.search(new Float32Array([1, 0, 0, 0]), 5);
console.log(results);
console.log(idMap.get(results[0].id));
```

Supported file types:

- `.json`
- `.jsonl`
- `.ndjson`

Accepted row shapes:

- `[1, 2, 3, ...]`
- `{ "id": "doc-1", "vector": [1, 2, 3, ...] }`
- `{ "docId": "doc-1", "embedding": [1, 2, 3, ...] }` with `idKey` / `vectorKey`

If your embedding pipeline already writes JSONL, this is the simplest file-based path.

### 3. Export And Restore A Snapshot

If you want to reuse a built index later, export a snapshot:

```js
import fs from 'node:fs';
import Pancake from 'pancake-wasm';

const { index } = await Pancake.fromVectors([
  [1, 0, 0, 0],
  [0, 1, 0, 0],
], {
  metric: 'cosine',
  quantized: true,
});

// If you have deleted anything, compact() before export().
const snapshot = index.export();
fs.writeFileSync('index.pnck', snapshot);

const restored = await Pancake.loadSnapshotFile('index.pnck', {
  dim: 4,
  maxElements: 2,
  metric: 'cosine',
  quantized: true,
});

console.log(restored.search(new Float32Array([1, 0, 0, 0]), 1));
```

Snapshot notes:

- `export()` throws if `ghostCount > 0`; call `compact()` first after deletions
- `loadSnapshotFile()` is Node-only
- `loadSnapshotFile()` restores Pancake snapshots from disk, not arbitrary ANN binary formats

## Embedding Your Own Documents

Pancake does not care which embedder you use, as long as you end up with vectors.

Typical workflow:

1. Read your source documents.
2. Generate embeddings with your model or API of choice.
3. Store them either:
   - directly in memory and call `fromVectors()`, or
   - in JSON / JSONL and call `loadJsonFile()`
4. Query the built index with new query embeddings produced by the same model family.

If you already have parquet, numpy, or another upstream format, convert it into JSONL or feed the vectors into `fromVectors()` directly from your application code.

## Worker Example

The reference Worker example is repo-based. It loads the checked-in WASM engine artifacts and exposes Pancake over HTTP. Treat it as a deployment pattern you run in your own Cloudflare account, not as a centrally hosted Pancake service.

When you run or deploy this example, it runs in your own Cloudflare environment:

- `npx wrangler dev` starts a local dev instance on your machine
- `wrangler deploy` publishes the Worker into the Cloudflare account authenticated in your local Wrangler setup
- any R2 bucket, auth settings, and rate limits belong to your own deployment, not to this repository

### Run The Worker Locally

```bash
cd examples/worker
npx wrangler dev --port 8787 --var ALLOW_INSECURE_ADMIN:1
```

`ALLOW_INSECURE_ADMIN=1` is a local-only opt-in: without it (or an `API_KEY`), admin routes such as `/init`, `/add`, and `/import` return 403, so the curl examples below would be rejected.

### Deploy The Worker To Your Own Cloudflare Account

```bash
cd examples/worker
wrangler r2 bucket create pancake-indexes
wrangler deploy
```

### Initialize A Small Index Over HTTP

In another terminal:

```bash
curl -X POST http://localhost:8787/init \
  -H 'Content-Type: application/json' \
  -d '{
    "dims": 4,
    "maxElements": 16,
    "vectors": [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0]
    ]
  }'
```

Then query it:

```bash
curl -X POST http://localhost:8787/search \
  -H 'Content-Type: application/json' \
  -d '{"query": [1, 0, 0, 0], "k": 2}'
```

Useful local checks:

```bash
curl http://localhost:8787/health
curl http://localhost:8787/stats
```

### Worker Import / Export Contract

The Worker has its own binary envelope for `/export` and `/import`.

- `GET /export` returns a Worker snapshot blob
- `POST /import` expects a previous blob from that Worker `/export` path

Do not assume that a Node.js `index.export()` snapshot is interchangeable with the Worker `/import` format. The Worker wraps the engine snapshot with additional metadata for its own restore path.

### Worker Demo Scripts

From the repo root, with the Worker already running:

```bash
node examples/demo/test_worker.js http://localhost:8787
node examples/demo/technical_demo_worker.js
```

The demos exercise different paths:

- `test_worker.js`: synthetic 1536D Worker/API integration test
- `technical_demo_worker.js`: interactive REPL against the Worker using the `dist/vectors.bin` asset (gitignored — generate it first with `npm run demo:data`)

## Sizing And Tuning

| Parameter | Typical range | Effect |
|-----------|---------------|--------|
| `M` | 8–32 | Graph connectivity. Higher improves recall, increases memory and build time. |
| `efConstruction` | 50–400 | Build beam width. Higher improves graph quality, slows build. |
| `efSearch` / `ef` | 50–500 | Query beam width. Higher improves recall, slows search. |
| `quantized` | `true` / `false` | uint8 quantization cuts memory significantly with some recall tradeoff. |
| `metric` | `'cosine'` / `'l2'` | Use cosine for normalized embeddings and L2 for unnormalized vectors. |

For memory-constrained Worker deployments, the quantized path is usually the right default.

## Troubleshooting

### `loadJsonFile()` rejects my file

Check:

- the extension is `.json`, `.jsonl`, or `.ndjson`
- the rows contain vectors under the expected field name
- `vectorKey` / `idKey` match your actual JSON schema

### I already have vectors in another format

Pancake does not currently load formats like `.fvecs`, `.npy`, or `.parquet` directly.

Use one of these instead:

- convert the vectors to JSONL and call `loadJsonFile()`
- load them in your own code and pass them to `fromVectors()`

### Worker `/import` does not accept my local snapshot

The Worker `/import` route expects the Worker export format, not a raw local package snapshot. Use the Worker's own `/export` output when testing `/import`.

## Next Steps

- [README.md](README.md) for the full package API surface
- [examples/worker/README.md](examples/worker/README.md) for the reference Worker deployment model
- [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) for the deeper system design document
