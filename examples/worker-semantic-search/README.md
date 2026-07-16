# Zero-dependency distilled docs search

This is the public, webpage-shaped Pancake demo. A Cloudflare Worker serves the
UI, restores a bundled quantized Pancake snapshot, embeds each query locally,
and searches the in-memory index. The deployed Worker has no service bindings
and makes no outbound requests.

There is no embedding API and no runtime ML framework. The complete query path is:

```text
text
  -> hashed word and character n-grams
  -> 1.08 MB int8 distilled student
  -> normalized 384D query vector
  -> Pancake WASM
  -> matching documentation chunks
```

The large teacher model is an offline build dependency. It is never bundled
with the Worker or contacted at query time.

## What the demo proves

- Pancake runs inside a Cloudflare Worker without a native addon.
- A snapshot can be restored directly from a Worker data module on cold start.
- Query embedding and vector retrieval run without outbound API requests or storage bindings.
- The embedding model is application data, not a Pancake dependency.
- Embedding, Pancake search, and cold-restore latency are reported separately.
- `efSearch` and source filters are applied per query through the public API.

This is a domain-specific encoder, not a general-purpose replacement for a
large embedding model. It is deliberately small enough to make the entire demo
self-contained.

## Distillation design

Documents are embedded offline with `sentence-transformers/all-MiniLM-L6-v2`,
pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
The student learns to place queries in that same 384D space. Its loss combines:

- cosine imitation of the teacher query vector; and
- retrieval-distribution imitation against every documentation chunk.

The runtime student is:

```text
8,192 hashed n-gram buckets x 128 hidden dimensions
  -> mean pool
  -> tanh
  -> 128 x 384 projection
  -> L2 normalization
```

Both parameter matrices use symmetric per-row int8 quantization. The exported
`PSTU` artifact is approximately 1.08 MB.

On the current 195-chunk corpus, the selected model was evaluated on 218 query
forms that were not used for training or model selection:

- mean cosine to teacher query vectors: `0.891`
- teacher top-result agreement: `84.9%`
- overlap with the teacher's top five: `79.2%`
- JavaScript/Python exported-model parity: `218/218` top results

Validation queries are separate from this final test set and select the early-
stopping checkpoint. The full per-query results are written to
`student-evaluation.json` during training.

## Runtime assets

The build produces four runtime assets that Wrangler bundles into the Worker:

- `docs-index.bin` — standard Pancake snapshot
- `docs-corpus.json` — result metadata and previews
- `docs-manifest.json` — dimensions, construction config, hashes, and evaluation summary
- `docs-student.bin` — quantized query encoder

`docs-student-evaluation.json` is also produced for inspection but is not
imported by the Worker.

The current artifacts are committed under `assets/`, so deployment does not
require Python, PyTorch, or retraining. The build instructions below regenerate
them when the documentation or encoder changes.

## Build the demo

Build Pancake at the repository root first:

```bash
npm run build:all
```

Extract the documentation corpus:

```bash
node examples/worker-semantic-search/build_demo.mjs \
  --out /tmp/pancake-docs-demo \
  --corpus-only
```

Create an isolated training environment. These packages are offline build tools,
not runtime dependencies:

```bash
python3 -m venv /tmp/pancake-student-venv
/tmp/pancake-student-venv/bin/pip install \
  -r examples/worker-semantic-search/requirements-train.txt
```

Train and export the student. The first run downloads the teacher checkpoint to
the selected Hugging Face cache:

```bash
HF_HOME=/tmp/pancake-hf \
/tmp/pancake-student-venv/bin/python \
  examples/worker-semantic-search/train_student.py \
  --corpus /tmp/pancake-docs-demo/docs-corpus.json \
  --out /tmp/pancake-student
```

Verify that plain JavaScript reproduces the exported Python model:

```bash
node examples/worker-semantic-search/verify_student.mjs \
  --student-dir /tmp/pancake-student
```

Build the Pancake snapshot and bundled asset directory:

```bash
node examples/worker-semantic-search/build_demo.mjs \
  --out examples/worker-semantic-search/assets \
  --student-dir /tmp/pancake-student
```

The build prints preview results for the sample queries before deployment.

Build and exercise the actual workerd bundle with an in-process Worker and its
bundled binary data modules:

```bash
cd examples/worker-semantic-search
npx wrangler deploy --dry-run \
  --outdir ../../.tmp-test-work/student-worker-distilled
cd ../..
node examples/worker-semantic-search/test_worker.mjs
```

## Run and deploy

```bash
cd examples/worker-semantic-search
npx wrangler dev --port 8787
```

Open `http://localhost:8787/`, then deploy when ready:

```bash
npx wrangler deploy
```

The committed configuration sets `READ_ONLY=1`. Public visitors can search and
inspect `/health`; cache reset and readiness details remain authenticated admin
operations.

## Endpoints

- `GET /` — interactive webpage
- `GET /health` — public cache, encoder, and restore status
- `GET /readiness` — authenticated bundled-asset metadata
- `GET /search?q=...&k=5&ef=120` — public search
- `POST /search` — `{ query, k?, ef?, source? }`
- `POST /reset_cache` — authenticated and disabled in read-only deployments

Search responses report `embedding_ms`, `search_ms`, `restore_ms`, cache state,
encoder metadata, and the matching chunks.

## Security and deployment boundary

The bundled snapshot is immutable deployment data; the in-memory index is a
disposable query-optimized copy. Concurrent cold requests share one restore
promise. Snapshot and student-model byte limits are checked before restoration,
and the student artifact is checked against the manifest SHA-256 before use.

The demo intentionally does not provide live index mutation. Applications that
need authoritative writes should build and publish new snapshots or place their
write coordination in an appropriate durable system.
