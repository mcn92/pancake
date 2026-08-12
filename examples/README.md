# Pancake Examples

The examples tell the same story as `create-pancake-search`, but with each
layer exposed: build search offline, ship it as application data, run retrieval
near the user, and hydrate results from the system that actually owns the
records.

If you have 15 seconds, the hook is:

```bash
npm run demo:artifact
```

That opens a committed docs search artifact, searches it lazily, and prints the
range reads. The point is that search can be a file your app ships. If you want
that wrapped as a generated Worker project instead, start with:

```bash
npm create pancake-search -- --name my-docs-search --source ./docs --runtime artifact --mode student --no-deploy --yes
```

## Product Path

### [`01-hello-pack/`](01-hello-pack/) - consume a Pancake pack in a browser

The smallest bundled browser fixture for `pancake-wasm/web`. This is the
starting point for "can my app load and search a packaged index?"

### [`02-catalog-hydration/`](02-catalog-hydration/) - search is not the database

A catalog service owns price and inventory. Pancake owns ranking over an
immutable snapshot. Search returns IDs, then the app hydrates live product
state from the catalog API, so operational data can change without rebuilding
the search pack.

### [`03-edge-docs-search/`](03-edge-docs-search/) - a complete edge search product

A Cloudflare Worker bundles a quantized Pancake snapshot, a 1.08 MiB int8 query
encoder, docs metadata, and calibrated abstention. It has no storage binding
and no outbound query-time calls. The optional static Pages shell lives in
[`03-edge-docs-search/pages-ui/`](03-edge-docs-search/pages-ui/).

### [`04-static-wiki-pack/`](04-static-wiki-pack/) - scale the same idea to static bytes

The Wikipedia pack path: build embeddings and sketches offline, write a
range-readable pack, serve it from static/R2 storage, and let the browser fetch
only the segments it needs. This is the clearest version of Pancake as a
shippable search artifact rather than a running vector database.

## Scaffolded Product Path

`create-pancake-search` is the examples' productized front door. It takes a
folder or URL, produces the Pancake assets, writes the Worker/UI shell, and
keeps the rebuild command in the generated project. Use it when the demo needs
to become a deployable docs search app; use this directory when you want to
explain or test the pieces.

## Reference And Legacy

- [`reference-worker/`](reference-worker/) - full Cloudflare Worker reference:
  snapshot restore from R2, auth, rate limiting, read-only mode, and admin
  routes. Covered by `test/test_worker_features.js`.
- [`legacy/range-artifact-demo/`](legacy/range-artifact-demo/) - earlier
  `.pancake-range` artifact demo and SIFT-shaped smoke harness.
- [`legacy/technical-demo/`](legacy/technical-demo/) - interactive local and
  Worker REPLs with validation commands.
- [`legacy/cli/`](legacy/cli/) - `build_index.js`, a small snapshot builder for
  binary embedding files.

There is also a raw, no-bundler browser page at
[`../dist/technical-demo.html`](../dist/technical-demo.html).
