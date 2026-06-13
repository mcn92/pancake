# Pancake Flagship Demo: Semantic Docs Search on Workers

This is the cleanest demonstration of `pancake-wasm` plus Cloudflare Workers:

- build a search index offline from markdown docs
- export a Pancake snapshot
- store the snapshot and corpus metadata in R2
- let the Worker restore on cold start
- serve semantic-ish text queries from hot in-memory state

The Worker is intentionally a **snapshot-serving search frontend**, not the
source of truth. That is the point of the demo.

The safest deployment shape for this demo is still read-only search:

- publish the snapshot assets to R2
- serve `/search` and `/health` publicly
- keep `/readiness` and `/reset_cache` as admin-only routes
- set `READ_ONLY=1` if you want to disable even admin cache resets

## Why this demo

It demonstrates the parts of Workers that fit Pancake well:

- read-heavy semantic search
- low-latency hot-path retrieval
- explicit restore from durable object storage
- honest cold-start semantics

It avoids the confusing parts:

- live mutable authoritative edge state
- strict read-after-write expectations across isolates
- treating Worker memory like a database

## How it works

1. `build_demo.mjs` chunks repo markdown files into searchable sections.
2. A deterministic local text embedder hashes text into 256D vectors.
3. The script builds a Pancake index and writes:
   - `docs-index.bin`
   - `docs-corpus.json`
   - `docs-manifest.json`
4. Upload those files to R2.
5. The Worker fetches them on first query, restores the index, and serves `/search`.

The local embedder is intentionally simple. It exists so the demo is runnable
without API keys. In a real deployment, replace it with your normal embedding
pipeline and keep the same snapshot-serving Worker shape.

## Build demo assets

```bash
cd /mnt/c/pancake1.0.0
node examples/worker-semantic-search/build_demo.mjs --out /tmp/pancake-docs-demo
```

That writes:

- `/tmp/pancake-docs-demo/docs-index.bin`
- `/tmp/pancake-docs-demo/docs-corpus.json`
- `/tmp/pancake-docs-demo/docs-manifest.json`

The script also prints a few preview queries and top matches so you can sanity-check
the corpus before deployment.

## Upload to R2

```bash
cd examples/worker-semantic-search
wrangler r2 bucket create pancake-docs-demo
wrangler r2 object put pancake-docs-demo/docs-index.bin --file=/tmp/pancake-docs-demo/docs-index.bin --remote
wrangler r2 object put pancake-docs-demo/docs-corpus.json --file=/tmp/pancake-docs-demo/docs-corpus.json --remote
wrangler r2 object put pancake-docs-demo/docs-manifest.json --file=/tmp/pancake-docs-demo/docs-manifest.json --remote
```

## Run locally

```bash
cd examples/worker-semantic-search
npx wrangler dev --remote --port 8787
```

Then open:

```txt
http://localhost:8787/
```

Use `--remote` so the local Worker can access the real R2 bucket binding.

## Deploy

```bash
cd examples/worker-semantic-search
wrangler deploy
```

## Endpoints

- `GET /` minimal interactive UI
- `GET /health` cache/restore status
- `GET /readiness` authenticated snapshot visibility and warm-load state
- `GET /search?q=...&k=5`
- `POST /search` with `{ query, k? }`
- `POST /reset_cache` authenticated admin cache reset

Search responses include:

- whether the request triggered an R2 restore
- restore latency
- search latency
- the top matching doc chunks

`/search` stays public because the point of the demo is the snapshot-serving
read path. `/readiness` and `/reset_cache` are the admin-facing routes. If
`API_KEY` is set, those routes require `Authorization: Bearer ...`. If
`READ_ONLY=1` is set, `/reset_cache` returns `403`.

## Good demo queries

- `How do Cloudflare Workers restore snapshots from R2?`
- `Why do I need compact before export after deletes?`
- `How does filtered search work in Pancake?`
- `What are the memory tradeoffs for quantized indexes?`

## Mental model

The important thing to demonstrate is not “look, the Worker owns the index.”

It is:

- R2 holds a durable snapshot
- the Worker restores a query-optimized copy from that snapshot
- hot memory serves search fast
- eviction is acceptable because the copy can be restored again
