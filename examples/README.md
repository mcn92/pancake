# Pancake Examples

Three examples carry the core story. Start with whichever matches your
deployment shape; the rest of the directory is listed below them.

## Start here

### [`search-artifact-demo/`](search-artifact-demo/) — search as a static file

The Search Artifact path: a `.pancake-range` index opened through HTTP range
requests, searched entirely client-side. The `static/` subdirectory is a
deployable browser demo — semantic search over the Pancake docs with a
bundled 1.08 MiB query encoder, no backend, no outbound API calls. The Node
demo (`npm run demo:artifact`) opens a SIFT1M artifact the same way from
disk. This is the newest and most representative example of where Pancake is
going.

### [`worker-semantic-search/`](worker-semantic-search/) — a complete search product in one Worker

A zero-dependency docs search service: quantized snapshot, distilled int8
query encoder, and calibrated abstention (strong / weak / none match
quality) bundled into a single Cloudflare Worker with no bindings and no
outbound requests at query time. Includes the training pipeline, parity
verifier, golden fixtures, and integration tests.

### [`catalog-demo/`](catalog-demo/) — the intended production shape

Pancake as the retrieval layer next to a real source of truth: search
ranking comes from an immutable snapshot, while price and inventory hydrate
live from a catalog API. Demonstrates why the Worker is not the database —
update inventory in the catalog and the next search reflects it without
touching the index.

## Reference deployments and fixtures

- [`worker/`](worker/) — the reference Cloudflare Worker: snapshot-first
  serving, R2 persistence, auth, rate limiting, read-only mode. Covered by
  `test/test_worker_features.js`.
- [`worker-semantic-search-pages/`](worker-semantic-search-pages/) — static
  Cloudflare Pages front-end shell for the semantic search Worker.
- [`browser-vite/`](browser-vite/) — minimal bundled browser consumer of
  `pancake-wasm/web`; the fixture behind `npm run test:browser`.
- [`cli/`](cli/) — `build_index.js`: build a `.pnck` snapshot from a binary
  embeddings file.
- [`demo/`](demo/) — interactive REPLs with built-in validation suites:
  `technical_demo_cli.js` (local engine; try `validate all`) and
  `technical_demo_worker.js` (same checks through a live Worker), plus the
  `test_worker.js` endpoint client.

There is also a raw, no-bundler browser page at
[`../dist/technical-demo.html`](../dist/technical-demo.html).
