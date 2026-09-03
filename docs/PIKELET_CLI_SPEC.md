# Spec: `pikelet` — Scaffold CLI for Pancake Worker Search

**Status:** Draft v2 · **Owner:** TBD · **Target:** v0.1.0 on npm as `pikelet`
**Depends on:** `pikelet-wasm@0.2.0` published to npm (see M0). Registry `0.1.0` predates the
0.2 API contract and cannot run the worker template.

---

## 1. Summary

A one-command scaffold that takes a user's content (local docs folder or website URL), builds a
Pancake search index, and emits a ready-to-deploy Cloudflare Worker with a working search UI.
Success criterion: **a first-time user goes from `npx pikelet create` to a live search URL
over their own content in under 5 minutes**, with no Python and no GPU.

Invocable as `npx pikelet create` (npm resolves this to the `pikelet`
package; this mapping only works cleanly for an unscoped name — see §12, Q1 resolved).

## 2. Goals / Non-goals

### Goals (v1)
- Ingest from a local folder or a single-site URL crawl.
- Support markdown, MDX, HTML, and plain text sources.
- Embed corpus at build time with a local ONNX model (no Python).
- Query-time embedding via Cloudflare Workers AI binding (`--mode workers-ai`, default).
- Generate a complete, deployable Worker project. The runtime is a vendored adaptation of
  `examples/03-edge-docs-search/worker.js`, importing `pikelet-wasm` from the registry
  (the `workerd` export condition resolves to the Workers entrypoint).
- Optional immediate deploy via `wrangler deploy` with the live URL printed at the end.
- Emit a GitHub Action that rebuilds the index and redeploys on content changes.
- Reproducible builds driven entirely by a checked-in `pikelet.config.json` **and a pinned CLI
  version** (the scaffold adds `pikelet` as a devDependency; `reindex` runs the
  local install, never a floating `npx` latest).

### Non-goals (v1) — explicitly out of scope
- Distilled student embedder mode (`--mode student`) — stub the flag, print "coming soon" with a
  link to the manual guide.
- Incremental / delta reindexing (full rebuild every time; acceptable within the bundled-asset
  ceiling, see §10 R5).
- R2-backed snapshot serving for large corpora (the `examples/reference-worker` restore-from-R2 pattern) —
  this is the designated v1.1 path past the bundle ceiling; v1 hard-fails with guidance instead.
- Sitemap ingestion, multi-domain crawling, JS-rendered pages (no headless browser).
- ~~Hybrid BM25 + vector scoring.~~ Shipped since: compiled `.pikelet`
  artifacts carry a lexical index segment (spec/COMPLETE_PROFILE.md §3.8)
  and queries run hybrid retrieval by default — BM25 candidates join the
  vector rerank, result order fused by reciprocal rank
  (`query({ retrieval })` selects a single-signal mode for measurement).
- PDF/docx ingestion.
- Non-Cloudflare deploy targets.

## 3. User experience

### 3.1 Interactive flow

```
$ npx pikelet create

  🥞 pikelet v0.1.0

  ? Project name: › my-docs-search
  ? Content source: › ○ Local folder  ○ Website URL
  ? Path / URL: › ./docs
  ? Deploy to Cloudflare when done? › (Y/n)

  ✓ Ingested 214 files → 1,832 chunks
  ✓ Embedded 1,832 chunks (bge-small-en-v1.5, 384D) in 41s
  ✓ Built index: 2.1 MB snapshot, recall@10 ≥ 99% (sampled)
  ✓ Projected Worker bundle: 3.4 MB compressed (paid-plan limit 10 MB)
  ✓ Scaffolded ./my-docs-search
  ✓ Deployed → https://my-docs-search.username.workers.dev

  Try it:  https://my-docs-search.username.workers.dev/?q=how+do+I+configure+auth
  Rebuild: cd my-docs-search && npm run reindex
```

### 3.2 Non-interactive flags

All prompts must be skippable for CI use:

```
npx pikelet create \
  --name my-docs-search \
  --source ./docs            # or --source https://docs.example.com
  --mode workers-ai          # workers-ai | student (stubbed)
  --model bge-small-en-v1.5  # build-time embedding model id (allowlist, see §12 Q2)
  --max-pages 500            # crawl cap (URL mode)
  --include "**/*.md"        # glob filter (folder mode), repeatable
  --exclude "**/node_modules/**"  # repeatable
  --deploy / --no-deploy
  --yes                      # accept all defaults, never prompt
```

Exit codes: `0` success · `1` user/config error (bad path, unreachable URL) · `2` build failure
(embedding/index/bundle-ceiling) · `3` deploy failure (project still scaffolded; print manual
deploy instructions).

## 4. Architecture

```
pikelet (CLI, Node ≥ 20, ESM)
 ├─ ingest/       folder walker + crawler → RawDoc[]
 ├─ chunk/        heading-aware splitter → Chunk[]
 ├─ embed/        transformers.js (ONNX, WASM backend) → Float32Array[]
 ├─ build/        pikelet-wasm 0.2 index build + export → snapshot.pnck
 ├─ scaffold/     template writer → project directory
 └─ deploy/       wrangler wrapper
```

Package dependencies (keep lean): `pikelet-wasm@^0.2.0`, `@xenova/transformers` (or
`@huggingface/transformers`), `cheerio` (HTML extraction), `undici` (crawl), `prompts` or
`@clack/prompts` (UX), `picocolors`. **No Python, no node-gyp, no native modules** — this is a
hard requirement; see §10 risk R2. The CLI imports `pikelet-wasm` under Node (the `import`
condition resolves to the Node entrypoint, which carries `loadJsonFile`/`loadSnapshotFile`);
the worker template imports it under `workerd`.

### 4.1 Pipeline stages

**Stage 1 — Ingest**

- *Folder mode:* recursively walk `--source`, apply include/exclude globs. Default includes:
  `**/*.{md,mdx,html,txt}`. Default excludes: `node_modules`, `.git`, `dist`, `build`. Globs must
  behave identically on Windows (normalize separators before matching).
- *URL mode:* same-origin BFS crawl from the seed URL. Respect `robots.txt`. Concurrency 4,
  per-request timeout 15s, cap at `--max-pages` (default 500). Skip non-HTML content types.
  Identifiable User-Agent: `pikelet/x.y`.
- HTML extraction: strip `<nav> <header> <footer> <script> <style> <aside>`, prefer
  `<main>`/`<article>` when present, fall back to `<body>`.
- Output: `RawDoc { id, sourcePath | url, title, text, lang? }`.
- Emit warnings (not failures) for unreadable files; fail the run only if 0 docs survive
  ingestion.

**Stage 2 — Chunk**

- Split on heading boundaries (`h1–h3` / `#`–`###`), then greedily pack to a target of
  **~200–320 tokens per chunk** with **~15% overlap** between adjacent chunks within a section.
  Token counts come from **the embedding model's own tokenizer** (already loaded via
  transformers.js) — no word-count heuristics.
- Preserve per-chunk metadata: `{ id, docId, title, headingPath: ["Guide", "Auth", "API keys"],
  url, anchor, text }`.
- Chunks under 25 tokens are merged into their neighbor; code blocks are never split mid-block.
- **Dedupe chunks with byte-identical text** (overlap + repeated boilerplate produce them); keep
  the first occurrence and record dropped duplicates in the build log. This is required for the
  Stage 4 self-recall gate to be meaningful.
- Output: `corpus.json` — array of chunk records. Match the schema already used by
  `examples/03-edge-docs-search/assets/docs-corpus.json` so the worker template consumes it
  unmodified. **If any field must differ, update the worker template, not the schema consumers'
  expectations — the deployed worker.js and corpus must always agree.**

**Stage 3 — Embed (build time)**

- Model: `bge-small-en-v1.5` quantized ONNX (384D) via transformers.js, WASM execution provider.
  Configurable via `--model` from the allowlist (§12 Q2).
- **Prefix policy is part of the contract.** BGE v1.5 retrieval quality depends on the query-side
  instruction prefix; passages are embedded raw. The CLI embeds passages with no prefix and
  records `"prefixPolicy"` in `manifest.json` (e.g.
  `{ "passage": "", "query": "Represent this sentence for searching relevant passages: " }`).
  The worker applies exactly the manifest's query prefix before calling Workers AI. Build-time
  and query-time text preprocessing must never be allowed to drift independently.
- First run downloads model weights (~30 MB) to the platform cache dir (`$XDG_CACHE_HOME` or
  `~/.cache` on Linux/macOS, `%LOCALAPPDATA%` on Windows, subdir
  `pikelet/models`); print download progress. Subsequent runs are offline.
- Batch size 16, normalize embeddings to unit length (cosine).
- Throughput target: ≥ 40 chunks/sec on an M-series laptop; print a progress bar with ETA.

**Stage 4 — Build index**

- Use `pikelet-wasm` 0.2 directly (Node entry):

  ```js
  const index = await Pancake.create({
    dim: manifest.dims,               // 384 for bge-small
    metric: 'cosine',
    quantized: true,
    M: 12,
    efConstruction: 75,
    efSearch: 100,                    // default baked into the snapshot; worker may override
    seed: 108,
    maxElements: Math.ceil(chunkCount * 1.25),
  });
  index.addBatch(vectors);
  const snapshot = index.export();    // envelope format; self-describing
  index.dispose();
  ```

  No deletions happen at build time, so `compact()` is unnecessary (`export()` only requires it
  when `ghostCount > 0`). The envelope snapshot carries its own config; the worker restores it
  with `Pancake.restore(bytes, { maxElements, efSearch })` and needs no other overrides.
- Post-build validation gate (fail build below thresholds): sample 100 chunks, verify
  self-recall@1 = 100% **by text identity** (the top-1 result must be a chunk whose text equals
  the query chunk's text — exact-duplicate ties are impossible after Stage 2 dedupe, but
  quantization ties on near-duplicates are judged by text, not id), and brute-force
  recall@10 ≥ 98% on 50 random held-out queries. Reuse the validation approach from
  `examples/legacy/technical-demo/technical_demo_cli.js`.
- **Bundle-ceiling gate (see §10 R5):** compute the compressed size of
  `worker.js + snapshot.pnck + corpus.json + manifest.json + ui.html` (gzip) and fail with
  guidance if it exceeds the deploy target's script-size limit (~3 MB compressed on the free
  plan, ~10 MB paid). Print the projected size on every build (§3.1).
- Outputs: `assets/snapshot.pnck`, `assets/corpus.json`, `assets/manifest.json` (model id, dims,
  prefix policy, chunk count, build timestamp, config hash, CLI version).

**Stage 5 — Scaffold**

Generated project layout:

```
my-docs-search/
├── worker.js                  # vendored workers-ai variant of the semantic-search worker
├── ui.html                    # search UI, imported by worker.js as a Text module
├── wrangler.toml              # name, nodejs_compat, [ai] binding, rules, guardrail vars
├── pikelet.config.json        # full reproducible build config (see §5)
├── package.json               # pikelet-wasm + pinned pikelet devDependency;
│                              # scripts: dev, deploy, reindex
├── assets/
│   ├── snapshot.pnck
│   ├── corpus.json
│   └── manifest.json
├── .github/workflows/reindex.yml
├── .gitignore
└── README.md                  # generated: how to dev/deploy/reindex/customize, cost note
```

- `npm run dev` → `wrangler dev` · `npm run deploy` → `wrangler deploy` · `npm run reindex` →
  `pikelet rebuild` **from the local devDependency** (reads
  `pikelet.config.json`, reruns stages 1–4 in place). A floating `npx` latest is forbidden: a
  newer CLI with different chunking would silently rebuild a different index under an older
  vendored worker.
- The worker template and `ui.html` are vendored into the CLI package at publish time (not
  fetched from GitHub at runtime) so scaffolds are deterministic per CLI version.
- `wrangler.toml` specifics:
  - `compatibility_flags = ["nodejs_compat"]`, `[ai] binding = "AI"`.
  - `[[rules]]` declares `type = "Data"` for `**/*.pnck` (the demo's existing rule covers only
    `**/*.bin`) and `type = "Text"` for `ui.html`; `corpus.json`/`manifest.json` import as JSON
    modules.
  - **Guardrails on by default:** `READ_ONLY = "1"` (admin/readiness routes fail closed, as in
    the existing worker) and `RATE_LIMIT_RPM = "120"` (reuses the demo worker's in-memory
    limiter). Rate limiting doubles as the Workers AI cost backstop (§10 R4). Public search
    parameters (`k`, `efSearch`) stay clamped exactly as the existing worker clamps them.
- Worker changes vs the existing demo: query embedding applies the manifest prefix and calls the
  Workers AI binding (`env.AI.run('@cf/baai/bge-small-en-v1.5', ...)`) instead of the bundled
  student; everything downstream (Pancake restore, per-query `efSearch` search, result hydration
  from corpus.json, the UI) is reused.
- Startup check (§10 R1): the worker compares its configured AI model id, dims, and prefix
  policy against `manifest.json` and refuses to serve (clear 500 with `code:
  "MANIFEST_MISMATCH"`) on any disagreement.
- AI-binding failure mode: clear 503 JSON error (`{ error, code: "EMBED_UNAVAILABLE" }`), never
  an unhandled exception. Pancake errors surface with their stable `PancakeError` codes, mapped
  to 4xx/5xx the same way the existing example worker maps them.

**Stage 6 — Deploy**

- Check `wrangler whoami`; if unauthenticated, launch `wrangler login` (interactive) or fail
  with instructions (non-interactive).
- Run `wrangler deploy`, parse the URL from output, print it plus one example query URL.
- On deploy failure: exit 3, keep the scaffold, print the manual command. Never delete generated
  work on failure.

### 4.2 Project layout decision

The default scaffold lands **inside the content repository** (e.g. `repo/my-docs-search/` next to
`repo/docs/`): the config's `source.path` is relative to the project directory, and the generated
GitHub Action checks out one repo and watches the real source path. A standalone scaffold repo
(content elsewhere) is supported but the scaffolder then omits the `push.paths` trigger and the
README explains that CI wiring is the user's responsibility. The interactive flow asks; `--yes`
defaults to in-repo when `--source` is a relative path inside a git work tree, standalone
otherwise.

## 5. `pikelet.config.json` schema

The single source of truth; `rebuild` must be a pure function of this file + content + the pinned
CLI version recorded in `package.json`.

```jsonc
{
  "$schema": "https://raw.githubusercontent.com/mcn92/pancake/main/schemas/v1/pikelet.config.schema.json",
  "version": 1,
  "name": "my-docs-search",
  "source": { "type": "folder", "path": "../docs", "include": ["**/*.md"], "exclude": [] },
  // or: { "type": "url", "url": "https://docs.example.com", "maxPages": 500 }
  "chunking": { "targetTokens": 256, "overlapPercent": 15 },
  "embedding": { "mode": "workers-ai", "buildModel": "bge-small-en-v1.5", "dims": 384 },
  "index": { "metric": "cosine", "quantized": true, "M": 16, "efConstruction": 200, "efSearch": 120 },
  "validation": { "minRecallAt10": 0.98 }
}
```

Ship the JSON Schema in the repo under a versioned path (`schemas/v1/`) so old scaffolds keep
resolving even if the schema evolves; validate config on every `rebuild` with actionable error
messages (bad field path + expected type).

## 6. GitHub Action (`reindex.yml`)

```yaml
name: Reindex search
on:
  push:
    branches: [main]
    paths: ['docs/**']        # rewritten by scaffolder to the actual source path (in-repo layout)
  workflow_dispatch:
jobs:
  reindex:
    runs-on: ubuntu-latest
    defaults: { run: { working-directory: my-docs-search } }   # rewritten to the project dir
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 22, cache: npm }
      - uses: actions/cache@v4
        with:
          path: ~/.cache/pikelet/models
          key: cps-model-${{ hashFiles('my-docs-search/pancake.config.json') }}
      - run: npm ci
      - run: npx pikelet rebuild --yes   # resolves to the local pinned devDependency
      - run: npx wrangler deploy
        env:
          CLOUDFLARE_API_TOKEN: ${{ secrets.CLOUDFLARE_API_TOKEN }}
          CLOUDFLARE_ACCOUNT_ID: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
```

`npm ci` + the devDependency pin means `npx` resolves the local install, keeping CI on the exact
CLI version the scaffold was built with. Scaffolder prints post-setup instructions for creating
the two repo secrets.

## 7. Error handling & UX polish

- Every failure message must contain: what failed, why (best guess), and the exact next command
  to run.
- Crawl mode: print a live counter (`crawled 143/500 · queued 61 · skipped 12 non-HTML`).
- Detect and warn (don't fail) on: empty chunks after extraction, duplicate URLs, corpora
  approaching the bundle ceiling (≥ 80% → suggest reducing scope now).
- `--verbose` flag prints per-stage timings; always write a full log to
  `.pikelet/last-build.log` inside the project.
- Ctrl-C at any stage must leave no partial project directory (build in a temp dir, move into
  place atomically on success; `rebuild` writes new assets to temp and swaps).

## 8. Testing & acceptance criteria

**Unit:** chunker (heading paths, overlap, code-block integrity, duplicate-text dedupe), config
validation, HTML extraction against 5 fixture pages (docs-site, blog, SPA-ish fallback,
nav-heavy page, non-English).

**Integration (CI):**
1. Scaffold from a fixture folder of 50 markdown files → assert project builds, snapshot
   restores under the workerd entrypoint, and a known query returns the expected chunk in top 3.
2. Crawl a fixture site served locally (no external network in CI) → same assertions.
3. `rebuild` twice with unchanged content → byte-identical `manifest.json` config hash
   (embeddings may differ only if the model is nondeterministic; document if so).
4. Generated worker passes a smoke test under **miniflare** with the bundled modules and a
   stubbed AI binding — extend the existing harness pattern from
   `examples/03-edge-docs-search/test_worker.mjs` (miniflare is already a declared
   devDependency of the main repo): `/search?q=...` returns 200 + ranked results; AI-binding
   failure returns the `EMBED_UNAVAILABLE` 503 contract; manifest/model mismatch returns the
   `MANIFEST_MISMATCH` 500 contract.

**Acceptance (manual, release gate):**
- Fresh machine (no cache), real docs folder of ≥ 100 files: `npx pikelet create` →
  deployed URL in **< 5 min wall clock** (excluding `wrangler login`).
- Search quality spot check: 10 natural-language queries against the Pancake repo's own docs,
  ≥ 8 return a relevant chunk in the top 3.
- Windows, macOS, Linux all pass the integration suite (transformers.js WASM backend is the
  cross-platform bet — verify, don't assume).

## 9. Milestones

| # | Deliverable | Est. |
|---|---|---|
| M0 | **Publish `pikelet-wasm@0.2.0`**: version bump from 0.1.0, `prepublishOnly` gate (build:all + full test suites + worker-web test), release notes explicitly listing breaking changes vs the published 0.1.0 (mutable-ef and matrix helpers removed; restore/inspectSnapshot/per-query efSearch/PancakeError added). | 0.5 wk |
| M1 | Pipeline core: ingest (folder) + chunk + embed + build, CLI plumbing, config schema, bundle-ceiling gate | 1.5 wk |
| M2 | Scaffolder + adapted worker template (manifest check, prefix policy, guardrail vars) + `rebuild` command with local pin | 1 wk |
| M3 | URL crawl mode + deploy step + GH Action | 1 wk |
| M4 | Test suite, cross-platform validation, error-message polish, docs | 1 wk |
| M5 | Release: npm publish, README section in main repo, demo GIF/recording | 0.5 wk |

## 10. Risks & mitigations

- **R1 — Workers AI model drift:** query-time model must match build-time model family, dims,
  *and prefix policy*. Pin all three in `manifest.json`; the worker refuses to serve
  (`MANIFEST_MISMATCH`) on any disagreement.
- **R2 — Native dependency creep:** any transitive node-gyp dependency breaks the "works
  everywhere" promise (the main repo's `hnswlib-node` benchmark dep already fails to install in
  restricted environments — do not inherit it). Whitelist-audit the dependency tree in CI
  (`npm ls` gate, fail on any install script).
- **R3 — Crawl liability:** users pointing the crawler at sites they don't own. Respect
  robots.txt, same-origin only, default page cap, identifiable User-Agent
  (`pikelet/x.y`), and a note in the generated README.
- **R4 — Workers AI cost surprise:** per-query embedding is billed. Generated README must
  include an honest cost estimate; `RATE_LIMIT_RPM` ships on by default as the backstop; point
  to student mode as the future zero-cost path.
- **R5 — Corpus size vs Worker bundle limit:** the binding constraint is the **compressed
  script-size cap** (~3 MB free / ~10 MB paid), which the `[[rules]]` data modules count
  against — not the 128 MB isolate memory, which is never the first wall for bundled assets.
  Rough budget at 384D quantized with ~256-token chunks: ~0.6 KB/chunk of snapshot (uint8
  vectors, effectively incompressible) + ~0.5 KB/chunk of gzipped corpus JSON ⇒ a practical
  ceiling around **~2K chunks on the free plan, ~8K on paid**. The build computes the real gzip
  size and hard-fails past the limit with guidance (reduce scope / raise plan / wait for the
  R2-backed v1.1 tier). Isolate memory is still checked but is secondary.

## 11. Resolved decisions (was: open questions)

- **Q1 — Naming: `pikelet`, unscoped.** `npx pikelet create` only maps
  cleanly unscoped; the scoped form would require `npm create @pikelet/search` →
  `@pikelet/create-search`, which nobody will guess.
- **Q2 — Model allowlist: the BGE v1.5 family.** `bge-small-en-v1.5` (384D),
  `bge-base-en-v1.5` (768D), `bge-large-en-v1.5` (1024D) all have exact Workers AI counterparts
  (`@cf/baai/bge-{small,base,large}-en-v1.5`), satisfying the identical-model-and-dims
  constraint. v1 ships **small only**; the allowlist plumbing carries dims through
  manifest → worker so adding base/large later is config, not code.
- **Q3 — `rebuild` lives in `pikelet`, pinned as a devDependency of the
  scaffold.** No second CLI; no floating `npx` latest (see §4.1 Stage 5). CI cold-start is a
  non-issue because `npm ci` installs the pinned version anyway.
- **Q4 — UI ships as `ui.html`, imported by worker.js as a Text module.** Single deploy
  artifact, but users restyle without touching worker code.
- **Q5 — Telemetry: none in v1.**

### Still open

- **Q6:** When the R2-backed tier lands (v1.1), does it reuse `examples/reference-worker`'s append-only
  snapshot-key scheme as-is, or simplify to a single fixed key since the scaffold's rebuild flow
  is the only writer?
- **Q7:** Whether `--model` accepts base/large at v1.0 behind a `--experimental` flag or stays
  hard-locked to small until cross-dim testing exists.
