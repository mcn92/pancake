# Pikelet Quickstart

Run every command from the repo root unless noted otherwise.

```bash
git clone https://github.com/mcn92/pancake.git
cd pikelet
npm install
```

No engine build needed — the checkout ships prebuilt WASM in `dist/`.

---

## 1. Prove the engine

Builds a synthetic index and runs 8 validation checks (determinism, deletion,
compaction, export/import, self-recall, recall, stability):

```bash
npm run demo
```

Expected: `Validation summary: 8/8 checks passed.`

---

## 2. Search a packaged artifact

Opens the committed docs artifact from disk and searches it with lazy range
reads:

```bash
npm run demo:artifact
```

Expected: per-query IO stats and sample top-3 IDs.

Then the head-to-head that motivates the sketch profile — the same corpus and
queries served by graph traversal vs a resident sketch scan, cold per query:

```bash
npm run demo:sketch
```

Expected: the sketch artifact is derived from the range artifact in-process
(no re-embedding), both profiles fetch a similar number of bytes, and the
sequential fetch depth drops from ~11 rounds to exactly 1 — the property that
makes remote storage (R2/S3/CDN) practical, where each round is a network
round trip.

---

## 3. Hello pack (browser)

The smallest browser test — creates, snapshots, restores, and searches an
index in Chromium via Vite + Playwright:

```bash
npx playwright install chromium   # one-time
npm run test:browser
```

Expected: `chromium: passed`.

---

## 4. Catalog hydration (search + live data)

Shows Pikelet as the retrieval layer while a separate catalog owns live
product data. Needs three terminals:

**Terminal A** — mock catalog API:
```bash
node examples/02-catalog-hydration/mock_catalog_server.mjs
```

**Terminal B** — reference Worker (admin routes open for demo):
```bash
cd examples/reference-worker
npx wrangler dev --port 8787 --log-level error \
  --var ALLOW_INSECURE_ADMIN:1 --var READ_ONLY:0 --var API_KEY:""
```

**Terminal C** — build, import, query:
```bash
node examples/02-catalog-hydration/build_snapshot.mjs
node examples/02-catalog-hydration/worker_import_snapshot.mjs
node examples/02-catalog-hydration/demo_client.mjs "lightweight waterproof hiking jacket"
```

Then mark the top jacket out of stock and re-query:
```bash
curl -X POST http://127.0.0.1:9090/admin/update \
  -H 'Content-Type: application/json' \
  -d '{"id":"jacket-rain-shell","inventory":0}'
node examples/02-catalog-hydration/demo_client.mjs "lightweight waterproof hiking jacket"
```

Expected: same search ranking, but the rain shell now shows `inventory=0`.

Stop servers with Ctrl-C in terminals A and B when done.

> **Troubleshooting:** if you see `401 Unauthorized` or read-only rejection,
> check for a `.dev.vars` file in `examples/reference-worker/` — the `--var`
> overrides above take precedence, but a stale file can interfere.

---

## 5. Edge docs-search Worker

A self-contained Worker with a bundled snapshot, int8 distilled encoder,
and calibrated abstention. No outbound requests at query time.

```bash
cd examples/03-edge-docs-search
npx wrangler deploy --dry-run --outdir ../../.tmp-test-work/student-worker-distilled
cd ../..
node examples/03-edge-docs-search/test_worker.mjs
```

Expected: `Distilled Worker demo checks passed.`

---

## 6. Wiki knowledge pack (sample corpus)

The full pipeline from documents to a searchable pack. Requires Python with
`torch` and `transformers`.

```bash
cd examples/04-static-wiki-pack
ONNXRUNTIME_NODE_INSTALL_CUDA=skip npm install
```

Then run each step:

```bash
# Chunk and embed the sample documents
python embed_corpus.py --input sample-corpus.jsonl --out data-sample

# Generate eval queries and cluster layout
python eval_queries.py --data data-sample --sampled 5 --no-hand
python layout_sim.py --data data-sample --k 5

# Reorder corpus by cluster
python permute_corpus.py --src data-sample --out data-sample-perm

# Build the pack and query it
node build_pack.mjs data-sample-perm
node query_pack.mjs data-sample-perm "how is abstention calibrated"
```

Expected: encoder parity check passes, top result is the abstention
calibration chunk.

See `examples/04-static-wiki-pack/README.md` for the full Wikipedia pipeline
and `DEPLOY.md` for Pages/R2 deployment.

---

## 7. Scaffold a search app (`pikelet`)

Generates a complete Cloudflare Worker project from a docs folder:

```bash
npx pikelet create \
  --name my-docs-search --source ./docs \
  --runtime artifact --no-deploy --yes
cd my-docs-search
npm install
npm run dev
```

Add `--mode student` to bundle a distilled encoder (needs Python with
torch/transformers) so `wrangler dev` runs fully local. Without it, the
default `--mode workers-ai` uses Cloudflare Workers AI; pass
`--var LOCAL_STUB_AI:1` to `npm run dev` to exercise the endpoint mechanics
without a Cloudflare account.

---

## Test suites

```bash
npm test                 # core + model loader + sketch conformance
npm run test:browser     # Chromium/Firefox/WebKit browser smoke
npm run test:worker-web  # workerd entrypoint smoke
npm run test:simd        # SIMD vs scalar output parity
npm run test:fuzz        # malformed-snapshot import fuzzing
```

## Notes

- **npm audit warnings** are all in dev-only transitive dependencies
  (Wrangler, Vite). The published `pikelet-wasm` package has zero runtime
  dependencies.
- **Leftover wrangler processes:** if ports are stuck after tests, clear
  them with `pkill -f "wrangler dev"` and `pkill -f workerd`.
