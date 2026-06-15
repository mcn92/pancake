# Semantic Docs Search on Workers

> **Note:** `pancake-wasm` is not yet published to npm. This demo runs from the
> repository checkout (build the engine with `./build.sh` at the repo root
> first). npm publishing is coming soon. See the [root README](../../README.md#install).

A `pancake-wasm` + Cloudflare Workers example that:

- builds a search index offline from markdown docs
- exports a Pancake snapshot
- stores the snapshot and corpus metadata in R2
- lets the Worker restore on cold start
- serves text queries from in-memory state

The Worker is a snapshot-serving search frontend, not the source of truth.

The recommended deployment shape is read-only search:

- publish the snapshot assets to R2
- serve `/search` and `/health` publicly
- keep `/readiness` and `/reset_cache` as admin-only routes
- set `READ_ONLY=1` if you want to disable even admin cache resets

## Scope

This demo covers the Worker usage patterns that fit Pancake:

- read-heavy semantic search
- low-latency hot-path retrieval
- explicit restore from durable object storage
- cold-start restore from R2

It does not cover:

- live mutable authoritative edge state
- read-after-write across isolates
- using Worker memory as the system of record

## How it works

1. `build_demo.mjs` chunks repo markdown files into searchable sections.
2. A deterministic local text embedder hashes text into 256D vectors.
3. The script builds a Pancake index and writes:
   - `docs-index.bin`
   - `docs-corpus.json`
   - `docs-manifest.json`
4. Upload those files to R2.
5. The Worker fetches them on first query, restores the index, and serves `/search`.

The local embedder is a hash-based stand-in so the demo runs without API keys.
In a real deployment, replace it with your embedding pipeline and keep the same
snapshot-serving Worker shape.

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

`/search` is public; it is the snapshot-serving read path. `/readiness` and
`/reset_cache` are the admin-facing routes. If
`API_KEY` is set, those routes require `Authorization: Bearer ...`. If
`READ_ONLY=1` is set, `/reset_cache` returns `403`.

## Good demo queries

- `How do Cloudflare Workers restore snapshots from R2?`
- `Why do I need compact before export after deletes?`
- `How does filtered search work in Pancake?`
- `What are the memory tradeoffs for quantized indexes?`

## Architecture summary

- R2 holds the durable snapshot
- the Worker restores a copy from that snapshot
- in-memory state serves search
- on eviction, the copy is restored again from R2
