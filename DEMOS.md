# Running the demos

Every command here assumes a fresh clone with dev dependencies installed:

```bash
git clone https://github.com/mcn92/pancake.git
cd pancake
npm install
```

No engine build is needed — the checkout ships prebuilt WASM in `dist/`.
Rebuild only if you are changing the C++ under `src/` (see the README's
"Building from source").

The demos below are grouped by what they need. Start with the first group.

## Works immediately

### Technical proof suite (local engine)

Runs eight falsifiable checks — determinism, deletion exclusion, compaction,
export/import round-trip, self-recall, recall vs brute force, stability:

```bash
npm run demo
```

`npm run demo` generates synthetic vectors, builds an index, and runs the full
validation suite non-interactively (about 8/8 checks, ~99% recall@10). To use
the interactive REPL instead, run it in a terminal (a TTY):

```bash
npm run demo:data
node examples/demo/technical_demo_cli.js       # then type: build, then: validate all
```

Any single command also works one-shot (it builds an index first):

```bash
node examples/demo/technical_demo_cli.js validate all
node examples/demo/technical_demo_cli.js search 1000
```

### Browser Search Artifact demo (client-side semantic docs search)

Semantic search over the Pancake docs, entirely in the browser — a distilled
encoder, a `.pancake-range` index read over HTTP byte ranges, calibrated
abstention, no backend:

```bash
npm run demo:artifact:browser:build
npm run demo:artifact:browser        # serves at http://127.0.0.1:4173
```

Open the printed URL. Try a sample chip, then watch the browser's Network
panel: the range requests on the first query drop toward zero on the next as
the artifact warms into cache. Try `banana pancake recipe` to see abstention
return no results. See `examples/search-artifact-demo/static/README.md` for
the measured warm-cache and static-host findings.

### Node Search Artifact demo (in-process, from a file)

Opens a `.pancake-range` artifact from disk and searches it lazily. Works on
any `.pancake-range` file:

```bash
node examples/search-artifact-demo/demo.js --artifact <path-to.pancake-range>
```

There is a default SIFT1M path, but that 494 MB artifact is not in the repo —
pass `--artifact` with your own file, or build one from a snapshot with
`Pancake.buildRangeArtifactFile(...)`.

## Needs one setup step

### Worker semantic-search integration test

The distilled docs-search Worker (bundled snapshot + int8 encoder + abstention,
zero outbound requests). The test drives it through Miniflare, so bundle the
Worker first with a Wrangler dry-run:

```bash
cd examples/worker-semantic-search
npx wrangler deploy --dry-run --outdir ../../.tmp-test-work/student-worker-distilled
cd ../..
node examples/worker-semantic-search/test_worker.mjs
```

(The test prints these exact commands if the bundle is missing.)

### Reference Worker + catalog demo (live dev server)

Pancake as a retrieval layer next to a live catalog source of truth. Three
terminals:

```bash
# terminal A — mock catalog API
node examples/catalog-demo/mock_catalog_server.mjs

# terminal B — the reference Worker, with admin routes open for the demo
cd examples/worker
npx wrangler dev --port 8787 --log-level error \
  --var ALLOW_INSECURE_ADMIN:1 --var READ_ONLY:0 --var API_KEY:""

# terminal C — build the snapshot, import it, query, then change inventory live
node examples/catalog-demo/build_snapshot.mjs
node examples/catalog-demo/worker_import_snapshot.mjs
node examples/catalog-demo/demo_client.mjs "lightweight waterproof hiking jacket"
curl -X POST http://127.0.0.1:9090/admin/update \
  -H 'Content-Type: application/json' \
  -d '{"id":"jacket-rain-shell","inventory":0}'
node examples/catalog-demo/demo_client.mjs "lightweight waterproof hiking jacket"
```

The `--var` flags open admin routes and enable writes for the local demo. If
you left a `.dev.vars` in `examples/worker/` from earlier work, a committed
`API_KEY`/`READ_ONLY` there can cause a `401 Unauthorized` or read-only
rejection; the `--var` overrides above take precedence, or move the file aside.

When done, stop the dev servers (Ctrl-C in terminals A and B).

## Verification suites

```bash
npm test                 # core (1204) + model loader + sketch conformance (44)
npm run test:browser     # Chromium/Firefox/WebKit consumer smoke (Playwright)
npm run test:worker-web  # workerd entrypoint smoke
npm run test:simd        # SIMD vs scalar output parity
npm run test:fuzz        # malformed-snapshot import fuzzing
```

## Notes

- **npm audit warnings on install** are all in dev-only transitive
  dependencies (the Wrangler and Vite toolchains). The published
  `pancake-wasm` package has zero runtime dependencies, so consumers inherit
  none of them. Running the demos does not require `npm audit fix`.
- **Leftover wrangler processes:** the worker feature suite starts a dev
  server per test and can leave some running. If ports are stuck, clear them
  with `pkill -f "wrangler dev"` (and `pkill -f workerd`).
