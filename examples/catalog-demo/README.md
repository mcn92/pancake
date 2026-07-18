# Catalog Demo

This demo shows the intended Cloudflare Worker shape for Pancake:

- Pancake Worker does semantic retrieval
- A separate catalog backend hydrates live product fields
- Search ranking comes from a snapshot
- Price and inventory come from the source of truth

It is intentionally small and deterministic. The vectors are hand-authored 4D
demo vectors, not model-generated embeddings.

## Files

- `products.json` — authoritative catalog data with live price/inventory
- `search_corpus.jsonl` — indexed search records and demo vectors
- `build_snapshot.mjs` — builds the Worker import snapshot (`catalog_index.pnck`)
- `mock_catalog_server.mjs` — serves live catalog hydration over HTTP
- `worker_import_snapshot.mjs` — imports the Worker snapshot blob into the reference Worker
- `demo_client.mjs` — queries the Worker, hydrates catalog data, and prints ranked results

## What This Demonstrates

The Worker is the retrieval layer, not the catalog database.

Example lifecycle:

1. Build a snapshot from `search_corpus.jsonl`
2. Import it into the reference Worker
3. Query the Worker for product IDs
4. Hydrate those IDs from the mock catalog API
5. Update inventory or price in the catalog API without rebuilding the snapshot
6. Re-run the same query and observe that live product fields changed immediately

## Run It

From the repo root:

```bash
./build.sh
node examples/catalog-demo/build_snapshot.mjs
node examples/catalog-demo/mock_catalog_server.mjs
```

In another terminal, start the reference Worker with unauthenticated admin
routes enabled for local demo use:

```bash
cd examples/worker
npx wrangler dev --port 8787 --log-level error --var ALLOW_INSECURE_ADMIN:1
```

Back at the repo root, import the generated Worker snapshot:

```bash
node examples/catalog-demo/worker_import_snapshot.mjs
```

Run a query and hydrate the product cards:

```bash
node examples/catalog-demo/demo_client.mjs "lightweight waterproof hiking jacket"
```

## Show Live Catalog Hydration

Mark the top jacket out of stock:

```bash
curl -X POST http://127.0.0.1:9090/admin/update \
  -H 'Content-Type: application/json' \
  -d '{"id":"jacket-rain-shell","inventory":0}'
```

Re-run the same search:

```bash
node examples/catalog-demo/demo_client.mjs "lightweight waterproof hiking jacket"
```

The Worker returns the same semantic match order, but the hydrated catalog
response now shows the item as out of stock.

## Rebuild The Snapshot

If you change the indexed search text or vectors in `search_corpus.jsonl`,
rebuild and re-import:

```bash
node examples/catalog-demo/build_snapshot.mjs
node examples/catalog-demo/worker_import_snapshot.mjs
```

That changes retrieval behavior, which is the right boundary for a snapshot-
backed search-serving layer.
