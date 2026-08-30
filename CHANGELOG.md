# Changelog

Release notes for the two published packages in this repository,
`pancake-wasm` (engine + artifact readers/builders) and
`create-pancake-search` (scaffolder, `doctor`, Docusaurus plugin). Both are
pre-1.0: a minor bump may carry breaking changes, and each entry lists them
first.

## Unreleased

### Breaking / compatibility

- **`create-pancake-search` rejects `--runtime complete`** on create and
  rebuild instead of silently building a snapshot project (the generated
  Worker templates only serve snapshot and artifact runtimes; `complete`
  passed validation but was dropped by the config builder). The error
  points at the new command below.
- **`--include`/`--exclude` against a URL source now error** instead of
  being silently ignored (they never applied to crawls). URL sources get
  their own filters: `--include-url`/`--exclude-url`, matched against the
  URL path with `*` as the only wildcard, and aggregate pages such as
  mdBook's `print.html` are excluded from crawls by default (they
  duplicate a whole site's content past exact-text chunk dedupe).
  `--include-url`/`--exclude-url` against a folder source error
  symmetrically.
- **The `.pancake-range` profile is deprecated** (contract §9.2). The
  sketch profile superseded it in every measured regime — depth-1
  execution ~5x faster end-to-end over real networks at roughly a third
  of the artifact size — and the complete `.pancake` profile is the
  product surface. Nothing breaks in this release: readers
  (`PancakeRangeArtifact`, `openRangeArtifactFile`) stay supported for
  existing artifacts, and the scaffold's `--runtime artifact` still
  works. The builders (`buildRangeArtifact[File]`) are marked
  `@deprecated` in the type declarations, the scaffold logs a deprecation
  warning when the artifact runtime builds, and no further range format
  revisions are planned. The Docusaurus plugin's default output still
  uses the profile; its `completeProfile` option is the recommended
  configuration.
- **Crawler fixes from a five-site independent-docs sweep** (Sphinx,
  MkDocs Material, Docusaurus, VitePress, Jekyll): URL filters now gate
  the queue instead of the dequeue, so filtered links no longer consume
  the `--max-pages` budget (a large excluded section — e.g. a versioned
  doc tree — used to starve discovery: docusaurus.io yielded 30 pages of
  a 70-page budget, 69 after the fix); self-recall validation tolerates
  near-duplicate content (original within top 5 passes as degraded with a
  logged hint toward `--exclude-url`, instead of hard-failing the build —
  versioned doc trees put several near-identical chunks at the same point
  in embedding space, where rank 1 among them is arbitrary); crawls that
  discover fewer than 5 pages warn that client-side-rendered navigation
  cannot be followed; the filter-skip log names which filters did the
  skipping.

### Added

- **HTTP range-storage benchmark** (`npm run bench:range-storage`):
  serves an artifact from a loopback server implementing HEAD + byte
  ranges and opens it through `httpRangeSource()` — the same 206 path a
  CDN or object store serves — measuring what the storage model actually
  costs, separate from retrieval relevance. Reports per artifact: open
  transfer (bytes/ranges/time before the first query), first-pass
  per-query bytes, range counts, and latency percentiles, a repeat pass
  isolating cache effects, and RSS/heap deltas; each artifact runs in a
  fresh child process so memory numbers do not accumulate across corpus
  sizes, `--server-delay-ms` injects a fixed per-response delay as a
  rough RTT model, and the server cross-checks that bytes requested by
  the range source equal bytes served. `--json`/`--csv` emit
  machine-readable results. First measurements (kind-3 docs artifacts,
  20 queries, hybrid): open transfer is ~25 MiB in 9 range GETs
  regardless of corpus size — the eager inline-encoder segment dominates
  cold start; first-pass queries cost 40–127 KiB in 5–16 parallel
  ranges (p50 78–100 ms with 25 ms injected delay, so roughly 2–3
  sequential request waves); repeat queries transfer zero bytes — the
  row/record caches absorb everything, p50 ~20 ms of pure compute.
- **Ingestion/anchor conformance suite** (`test/ingestion/`, in npm
  test): fixture corpora covering nested and duplicate headings, explicit
  `{#custom-id}` anchors, headings inside fenced code, Unicode and setext
  headings, inline code/links in headings, snake_case identifiers,
  frontmatter titles/slugs, MDX components around headings, oversized
  section splitting, and HTML pages with authored `id` attributes — each
  asserting the full chain a search result depends on: document URL,
  headingPath, anchor, final href, chunk text. Anchor generation is
  parity-tested against the real github-slugger (the library Docusaurus
  renders with) via a recorded 26-heading battery plus a live comparison
  when the library is installed; two divergences are documented as
  deliberate (multi-space heading runs, JSX tags stripped by MDX).
  Building the suite surfaced and fixed four real defects: the slugger
  was ASCII-only (github-slugger keeps Unicode letters — "Über uns" must
  slug to "über-uns"), dash runs were wrongly collapsed ("C++ & C#" slugs
  to "c--c"), heading underscores were stripped (destroying snake_case
  identifiers like ACTION_QUERY_PARAMS), and setext headings
  (`Title\n===`) were not recognized at all. Compiling nodejs.org's API
  docs surfaced a fifth: minified HTML with unquoted attributes
  (`href=fs.html`, `id=fsreadfilesync…`) was invisible to the link and
  anchor extractors — the crawl found 2 pages of 30.
- **The v1 spec no longer lists hybrid search as a non-goal**:
  docs/CREATE_PANCAKE_SEARCH_SPEC.md's non-goals entry for "Hybrid BM25 +
  vector scoring" is struck through with a pointer to the shipped lexical
  segment (spec §3.8) and the `query({ retrieval })` modes.
- **Section-aware ingestion and chunking.** Markdown/MDX documents are
  parsed into their heading structure before chunking — heading lines are
  recognized only outside code fences, explicit `{#custom-id}` heading ids
  win over derived slugs, and duplicate headings get GitHub-style `-1`
  suffixes — and HTML crawls section on `h1`–`h6` tags, preserving the
  page's own `id` anchors. Chunking keeps a section together when it fits
  (up to 1.6x the target size), subdivides at paragraph boundaries when it
  does not, and merges sub-25-token sections into their neighbor. Every
  chunk now carries the fields the corpus schema always reserved but
  nothing populated: `headingPath` (ancestor headings ending in the
  section's own) and `anchor`, so results deep-link to the exact section
  (`Routing > Dynamic routes > Rest parameters`, `#rest-parameters`)
  instead of pointing at a page. A leading h1 that restates the document
  title stays out of heading paths. Plain-text files keep the flat token
  windows.
- **Retrieval modes and a retrieval-quality bakeoff.** `query()` accepts
  `retrieval: 'hybrid' | 'vector' | 'lexical'` (hybrid remains the
  default; the single-signal modes exist for measurement and debugging —
  abstention scores identically in every mode), and
  `scripts/bakeoff-retrieval.mjs` runs a query set with expected sections
  through all three modes, reporting success@1/@3 and MRR@5 split into
  semantic questions and exact lookups. Measured on a 120-page Astro docs
  crawl (2,123 section chunks, 24 queries): exact lookups score
  success@3 0.67 vector-only vs 1.00 BM25-only vs 0.92 hybrid; semantic
  questions 0.75 / 0.67 / 0.83 — hybrid is the only mode competitive on
  both classes. A 9-doc repo-docs corpus agrees on exact lookups (hybrid
  success@1 0.71 vs 0.57 for either single mode) with semantic within
  noise at that sample size.
- **Hybrid search: complete artifacts carry a BM25 lexical index**
  (segment kind 5, layout `bm25-v1`, spec §3.8) and the reader fuses
  lexical and vector retrieval. The builder writes a static inverted
  index over the corpus records (hash-addressed fixed-width term table,
  varint postings, ~113 KiB for a 132-record docs corpus); `compile`
  includes it by default. At query time the BM25 top matches join the
  sketch's exact rerank as extra candidates — scored by true vector
  distance, which is what rescues known-item lookups the sketch scan's
  top-C misses — and the final result order fuses the distance and BM25
  rankings by reciprocal rank (RRF, constant 60, untuned). Lexical hits
  under a third of the best lexical score are dropped before fusion: idf
  collapses common query terms into a near-tied mass that carries no
  ranking information. Abstention is unaffected — it scores the
  distance-sorted top-k exactly as calibrated; fusion changes result
  order only. Readers that predate the segment kind skip it (spec §3.3)
  and serve vector-only queries from the same file; artifacts without
  the segment behave exactly as before (`info().lexical` reports which).
  `PancakeSketchArtifact.search` gains `extraCandidates` (external row
  ids joining the exact rerank) and `fullRerankOutput` (return every
  reranked candidate, not the top k) to support this. The format is
  lazy-friendly by construction (absolute offsets, hash-sorted
  fixed-width term table) so a range-reading opener for wiki-scale packs
  needs no format revision; the phase-1 reader loads the segment eagerly
  and verifies its digest at open. Covered by 9 new conformance checks
  including a deterministic rescue case (a known-item query whose
  encoder vector is pure noise still surfaces the right record) and
  lexical-segment tamper rejection.
- **The crawler follows the seed's own redirects** — HTTP and meta-refresh,
  bounded at 5 hops, each hop logged — and the final URL defines the crawl
  origin. Sites routinely send their root to a canonical host or localized
  landing page (docs.astro.build's root is an HTTP 200 whose only content
  is `<meta http-equiv="refresh" ...>`); the crawl previously discovered
  one empty page and failed with an error blaming the content. Every other
  fetch keeps the strict redirect-skip, and a contentless meta-refresh
  page found mid-crawl is now followed through the normal frontier filters
  instead of consuming page budget as an empty document. Exercised by a
  new scaffold-e2e check against a local server chaining a 301 into a
  meta refresh.
- **Grounding coverage weighs word informativeness and scores more
  passages**: corpus-common words (chunk document frequency above ~5%,
  shipped as a bloom in `asset.coverage.commonBloom`) count at 1/3 weight
  — a docs corpus's "templates"/"support" no longer ground a query on
  their own, which was letting roadmap-style questions score strong — and
  coverage is the max over the top 5 passages instead of the top 1, so a
  paraphrased query gets more chances to find the passage sharing its
  vocabulary while recombination negatives' source chunks still each
  ground only their own half. Hard-negative CV AUC across the three test
  corpora: docs repro 0.808 → 0.831, wiki 0.918 → 0.924, Astro docs
  crawl 0.877 → 0.887, with recombination-kind AUCs at 0.91–0.98.
- **The abstention model gains a grounding feature** (`self-templates-v3`):
  the fraction of the query's content words present in the top retrieved
  passage's text, fit as a fifth term in the same logistic model. Every
  prior feature measures topic similarity (three distances plus membership
  in a corpus-wide bloom), so "the corpus discusses this area" and "this
  passage answers this question" were indistinguishable by construction —
  a query can retrieve topically-adjacent content at a closer distance
  than genuine controls while none of its terms appear in the returned
  passage. The term is serialized as a supplemental `asset.coverage` field
  rather than a `features[]` entry, so readers that predate it score the
  topic-only model against the same thresholds (a conservative
  degradation) instead of feeding an unknown feature name NaN into the
  logistic — and the reader now degrades any non-finite score to
  `unscored` rather than letting NaN compare false against both
  thresholds and answer everything. Scoring an artifact that carries the
  term hydrates the top-1 record before the verdict (the page is hot for
  result hydration when answered; an abstained query costs one extra
  read); word rules ship in the asset so builder and reader cannot drift.
  Measured on the hard-negative repro corpus: hard-negative CV AUC 0.772
  → 0.808, recombination-kind AUC 0.891, all five same-field near-domain
  probes now abstain outright, and the calibration summary adds
  `cvAucHardByKind`. A/B-verified on a second, structurally opposite
  corpus (250 distinct-topic wiki articles, low vocabulary overlap):
  hard-negative CV AUC 0.853 without the term → 0.907 with it, pooled
  0.903 → 0.937, with paraphrased no-title-word controls staying strong
  under both. Known residual limits: a single-content-word query can be
  spuriously grounded by a topically adjacent passage containing that
  word, and attribute questions about present entities (entity words
  grounded, attribute absent) still score strong under both models.
- **compile's self-calibration now trains against hard negatives**
  (`self-templates-v2`), fixing artifacts that confidently answered
  unanswerable in-domain questions. The v1 calibrator's negatives were all
  easy — off-domain bank and gibberish, low known-token fraction, distant
  retrieval — so the fitted model handed the vocabulary feature a dominant
  weight (+4.1 standardized) and scored any in-domain query answerable:
  measured on a docs corpus, 6 of 8 unanswerable in-domain probes returned
  `strong` at p 0.75–0.94 while the pooled CV AUC read 0.998. v2 adds two
  hard-negative classes with high known-token fractions: held-out
  documents (excluded from calibration searches via `searchFiltered`, then
  asked about — the wiki calibrator's held-out-shard trick made
  corpus-generic) and cross-chunk recombinations (corpus words no single
  document lexically supports). Hard-negative labels are verified by
  retrieval (a held-out topic covered by a near-duplicate document is
  dropped, not mislearned); overlapping off-domain queries are kept as
  eval-only rows and reported instead of silently discarded. The gate now
  also requires hard-negative CV AUC ≥ 0.75 — the pooled number is
  dominated by the easy classes and cannot see this failure — and the
  `weak` threshold rises to the hard-negative ceiling, so overlapped
  queries surface with a caveat instead of full confidence. Same probes
  after the change: 0 of 8 unanswerable queries score `strong` (6 weak,
  2 none), answerable controls unchanged at `strong`. The calibration
  summary adds per-class counts and `cvAucEasy`/`cvAucHard`.
- **`create-pancake-search compile`** builds a complete kind-3 `.pancake`
  artifact from `--source <path|url>` and stops — no project, no Worker,
  no Cloudflare. It chunks and embeds the corpus with the packaged inline
  MiniLM encoder (auto-fetched, digest-pinned, when not present), measures
  the rerank operating point, and writes one file (`--out`, default
  `search.pancake`) openable with `pancake-wasm/complete` on any runtime.
  Takes `--name`, `--max-pages`, `--include`/`--exclude`, `--force`.
  Exercised by the scaffold e2e (compile → open → query, plus the
  `--runtime complete` rejection).
- **compile self-calibrates abstention** (`src/calibrate.mjs`), so compiled
  artifacts answer `matchQuality` strong/weak/none out of the box instead
  of "unscored". Positives are templated title questions plus content-word
  queries, each verified by retrieval before it counts; negatives are a
  built-in off-domain bank (overlap-dropped against the positive median
  d0) and bloom-checked gibberish; the fit is the wiki calibrator's
  logistic model over (d0, margin, mean10, known_frac). Thresholds are
  placed on percentiles (5th of positives, 95th of negatives) rather than
  the wiki calibrator's raw extremes — auto-generated queries put junk in
  both tails and the saturated fit collapses geometric means of them —
  with hard a quarter of the way up the gap: a false abstain hides
  results, a false weak shows them with a caveat, so the weak verdict
  owns the uncertain band. Skips with a logged reason (and ships unscored)
  when positives or negatives run short or the cross-validated AUC misses
  the 0.85 gate — the gate uses a deterministic 5-fold cross-validation
  over the generated queries, since the fit AUC is in-sample; the embedded
  asset records method, query counts, and both numbers (`fitAuc`,
  `cvAuc`). New
  flags: `--calibration <file>` embeds a prebuilt retrieval-signals-v1
  asset, `--skip-calibration` opts out. The Docusaurus plugin default is
  unchanged (auto-calibration is opt-in via `runtime.calibration: 'auto'`
  outside compile).
- **Inline-transformer passage embedding runs on a worker pool** past 32
  chunks (one WASM kernel + weight blob per worker, chunks dealt by index
  so builds stay deterministic — pool and sequential builds return
  bit-identical distances). 571 chunks: 13 min → 3 min on 8 cores.
  `PANCAKE_SEARCH_EMBED_WORKERS` overrides the worker count (0/1 forces
  sequential); environments without worker support fall back to the
  sequential path with a logged reason.

## pancake-wasm 0.5.0 / create-pancake-search 0.5.0 — 2026-08-26

### Breaking / compatibility

- **Sketch artifact format 2 (per-row integrity) is now what the builders
  write by default.** The vectors region is laid out as interleaved blocks
  — a digest page of truncated per-row SHA-256s (16 rows × 16 bytes by
  default) ahead of each block's rows — anchored by a 32-byte-per-block
  page-hash table at the end of the resident prefix, so `residentSha256`
  (and, in a complete artifact, `index.headerSha256` and the identity)
  covers it. `fetchRows` verifies every page against the table and every
  row against its digest slot **on the read that fetches it**, in both the
  full and staged micro tiers; tampered rows fail the query instead of
  silently altering results. Geometry per the 2026-08-25 measurement:
  ~0% query-latency overhead, +4.2% file size. Readers ≤ 0.4.0 reject
  format-2 sketches ("Unsupported sketch artifact version"); the new
  reader opens format 1 and 2 (`verify: false` opts out, reported).
  `exportSketchArtifact(..., { rowIntegrity: false })` still emits
  format 1; `rowsPerBlock` / `rowDigestBytes` tune the geometry.
- Complete artifacts embedding a format-2 sketch report
  `info().indexRowIntegrity: 'per-row-sha256'`; with that, every byte that
  influences a query is authenticated per read and the spec's transitional
  integrity stance applies only to format-1-sketch artifacts.
- New `test/sketch_row_integrity.js` in `npm test`.
- `create-pancake-search` has no behavior changes of its own; generated
  projects now depend on `pancake-wasm@^0.5.0` so the artifacts the
  scaffolder builds (format-2 sketches included) open on the installed
  reader.

## pancake-wasm 0.4.0 / create-pancake-search 0.4.0 — 2026-08-25

### Breaking / compatibility

- **Complete artifact format 2** (`pancake-complete-v2`, header
  `formatVersion` 2) is now what the builder writes by default
  (`buildCorpusSegment`, `PROFILE_V2`): the corpus segment carries one
  SHA-256 per record behind a page table the manifest commits to, so every
  hydrated record is verified on its own range read. Readers ≤ 0.3.0 reject
  format-2 files explicitly ("unsupported .pancake format version 2"); the
  0.4.0 reader opens format 1 and 2 and reports the stance in
  `info().corpusIntegrity`. `create-pancake-search` complete-mode builds
  (`search.pancake`) and `examples/05` compiles now produce format 2;
  released wiki artifacts stay format 1 and remain readable.
- `openPancakeFile` on kind-2 artifacts now runs `options.encodeQuery`
  against the declaration's verification vectors **at open** and refuses
  the open on disagreement (dimension or tolerance); declarations without
  vectors are refused unless `allowUnverifiedEncoder: true`. Previously the
  host encoder was trusted.
- Reader hardening: every artifact-derived read is validated (safe
  integers, overflow, `fileBytes`, source size, per-open `maxReadBytes`
  budget — default 256 MiB, `Infinity` → 2 GiB backstop — and
  `maxRecordBytes` per record, default 16 MiB) and must return exactly the
  bytes requested; u64 fields above 2^53−1, unknown query-interp versions,
  duplicate segment kinds, and inconsistent offsets tables are rejected.
  Unknown segment kinds are skipped (spec 3.3). Files that previously
  opened only because a check was missing may now be refused with a
  specific error.

### Added

- `info().formatVersion`, `profile`, `encoderVerified`, `corpusIntegrity`;
  `record(id)`; `verifyHostEncoder()`, `SUPPORTED_PROFILES`,
  `CORPUS_LAYOUT_V2` exports; `assemblePancakeFile` accepts unknown kinds
  via `kindNumber` and returns `formatVersion`.
- `test/complete_profile.mjs` — deterministic complete-profile reader
  conformance (77 checks: hostile input, verification, integrity, format-1
  compatibility, kind-1 goldens from the committed 03 assets), run by
  `npm test` and therefore CI.
- spec/COMPLETE_PROFILE.md Draft 2.
- `create-pancake-search --help` / `-h` print usage (previously "Missing
  value for --help").

### Fixed (artifact layer hardening, from the 2026-08-21 external review)

- `PancakeRangeArtifact.openFile` / `PancakeSketchArtifact.openFile` close
  their file descriptor when the open is rejected (previously every corrupt
  artifact leaked one fd); `NodeFileRangeSource` releases the fd if `fstat`
  fails after `open`, and refuses reads after `close()`.
- Range and sketch `search()` apply the core engine's query contract:
  plain-array coordinates must be numbers (no `'1'` coercion), every
  component must be finite for every metric (L2 no longer returns
  `distance: NaN`), dimension checked; `k`, `rerank`, `efSearch` must be
  positive integers and `k` / the sketch candidate pool are capped to the
  row count before sizing any allocation (`search(q, 1e9)` over 300 rows no
  longer attempts a 1e9-slot allocation).
- `parseUint8Snapshot` validates `M`/`M0`, bounds each level's adjacency
  count by `M0`/`M`, checks the bytes it claims are present, and range-checks
  neighbor ids — all before allocating that level's edge array.
- `httpRangeSource`: after the one-time full-download fallback, reads are
  served from memory with no further requests (previously every read still
  issued a fetch).
- Complete reader: `id` and `distance` are reserved result names and always
  come from the search (a record's own fields no longer overwrite them);
  `close()` is idempotent (no `EBADF` on a second call) and queries after
  close are refused; `query({ k })` rejects 0 / negative / fractional `k`
  instead of silently defaulting to 5.
- `create-pancake-search` loaders prefer the in-repo `pancake-wasm` when
  running inside the monorepo (a stale installed copy higher in the tree
  no longer shadows the checkout); npm consumers are unaffected.
- Packaging: the native benchmark baselines (`faiss-node`, `hnswlib-node`,
  `usearch`) and the `@emnapi/*` pins are no longer `optionalDependencies`
  of `pancake-wasm` — consumers no longer attempt those native builds; they
  live in `benchmarks/package.json` (`cd benchmarks && npm install`). The
  unused self devDependency on `pancake-wasm` is removed.
- Sketch `search()` with a candidate pool equal to the row count (small
  artifact, or `k`/`rerank` ≥ count) no longer runs the resident selection
  scan — every row is a candidate for the exact rerank — instead of the
  2·count² slot operations the selection loop degenerated to.
- `parseUint8Snapshot` reads version-1 uint8 payloads with their 4-byte
  edges (id only), matching the engine's deserializer; v1 payloads were
  previously misparsed as the v2 {id, distance} layout.
- `rerank: 0` / `efSearch: 0` keep their historical "use the default"
  meaning (`undefined`, `null`, `0` → default; anything else must be a
  positive integer).
- `parseUint8Snapshot` bounds `M`/`M0` to the engine's own envelope
  (`M ∈ [2,128]`, `M ≤ M0 ≤ 256`).
- Complete reader: a kind-3 encoder's WASM buffers are released when the
  open fails after the encoder was created; a digest-page read that fails
  for a transport reason is not cached as a rejection (the next hydration
  retries). `httpRangeSource` refuses a 206 whose `Content-Range` is not the
  requested slice.
- **Format 2 manifests now commit to the embedded sketch's 256-byte header
  (`index.headerSha256`, required)**, and the reader verifies it at open:
  the sketch's metric/dim/count and its `residentSha256`/`vectorsSha256`
  are anchored to the artifact identity instead of living only in the
  segment's own (attacker-rewritable) header. Previously a flipped metric
  word changed search semantics under an unchanged identity. On every
  format the reader also cross-checks the sketch metric against the
  manifest. Format-2 identities change (format 2 was never released).
- `httpRangeSource`: a 206 without a valid `Content-Range` naming the
  requested slice is refused (RFC 9110 requires the header) and the body
  length must match it; the full-download fallback cap is enforced on
  bytes actually received (a chunked 200 aborts as soon as it streams past
  the cap) instead of trusting `Content-Length`/HEAD; new `cacheKeyParam`
  option (`null` for signed-URL hosts keeps the URL untouched, default
  `'r'` keeps the Chromium cache-lock workaround).
- Stable external ids are bounded to the uint32 the v3 snapshot stores:
  `add()` refuses (INDEX_LIMIT) before the engine mutates and `addBatch()`
  preflights the whole batch, instead of assigning ids the snapshot would
  truncate into an unrestorable export.
- `canonicalJson()` follows JSON semantics for unsupported values
  (undefined/function object properties omitted, array entries become
  null) instead of emitting invalid JSON.
- `evaluation()` is typed `Promise<Record<string, unknown> | null>` (it
  returns null with no evaluation segment) and refuses a non-object
  segment at runtime.
- Encoder verification validates the *expected* embeddings too: a
  verification vector whose expectation is malformed (strings, null, NaN,
  wrong length, missing) now fails verification instead of passing
  vacuously — `x - "x"` is NaN and `NaN > maxDiff` is false, so such
  vectors previously compared nothing while still reporting
  `encoderVerified: true`. Applies to kind-2 host verification and kind-3
  inline verification (shared `validateExpectedEmbedding`, exported).
- Lazy rerank rows: new `verifyIndexVectors: true` open option and
  `verifyVectors()` method on the complete reader run the full pass
  against the identity-anchored `vectorsSha256`; `info().vectorsVerified`
  reports the state. Documented prominently (README, spec section 6):
  without the pass, rows that feed reranking are committed but not
  verified per read; per-row commitments are spec section 8 question 6.
- Builder inputs may be plain `Uint8Array`s, as the typings always said:
  `buildCorpusSegment(FromBuffers)`, `buildQueryInterpSegment`, and
  `buildInlineTransformerEncoderSegment` no longer call Buffer-only
  `.copy()` on caller-supplied bytes.
- New `test/artifact_hardening.js` (66 checks) in `npm test`; 41 more checks
  in `test/complete_profile.mjs`.

### Internal

- `pancake-artifact.js` is now a thin entry over `pancake-artifact-common.js`,
  `pancake-artifact-range.js`, `pancake-artifact-sketch.js` (same exports,
  same typings; the three files ship in the tarball).
- `create-pancake-search/src/cli.mjs` is split into `common`, `ingest`,
  `embed`, `complete-build`, `scaffold`, `build` modules; `cli.mjs` keeps
  the commands and re-exports `buildSearchAssets`, `fetchInlineEncoderWeights`,
  `CliError` for the Docusaurus plugin. See the README's "Package layout".

## pancake-wasm 0.3.0 — 2026-08-20

### Breaking / compatibility

- `buildRangeArtifact` now emits `.pancake-range` **format v3** (whole-segment
  SHA-256 digests for the id map, router, and base segments stamped in the
  header). Readers in 0.2.1 and earlier only accept v1/v2 and reject v3 files
  with `SNAPSHOT_INVALID` ("Unsupported Pancake range artifact version").
  The 0.3.0 reader still opens v1/v2 files (structurally validated,
  unverified — they carry no digests).
- Snapshots whose payload declares an unknown future format version are
  rejected at import instead of being parsed as the newest known layout
  (JS and engine). Every version a released writer has emitted still
  imports.
- `Pancake.create()` rejects configurations whose eager arena estimate
  exceeds the 1.5 GiB wasm32 budget with `WASM_ALLOCATION_FAILED` (estimate,
  budget, and backend formula in `details`) instead of aborting the WASM
  instance from inside the engine.
- Types: `SketchArtifactBuildManifest.file` is now optional (absent on the
  bytes-in/bytes-out builder).

### Added

- **`pancake-wasm/complete`** — the one-file `.pancake` reader
  (`openPancakeFile`, `httpRangeSource`, the inline-transformer host and its
  verification-vector API), and **`pancake-wasm/complete/builder`** — Node
  assembly (`assemblePancakeFile`, segment builders,
  `measureRecommendedRerank`, `loadInlineEncoderKernel`). Typed via
  `complete/index.d.ts` / `complete/builder.d.ts`. The PSF1 wire format has one
  definition (`complete/format.mjs`) shared by reader and builder; the encoder
  kernels (`complete/encoder-kernels/`) ship here.
- Kind-3 (inline transformer) declarations are validated against the
  compiled kernel before the first forward (layout, dim, exact blob length),
  and their verification vectors (contract §4.4 mode 1, including a
  >128-token windowed-pool probe) run automatically on open; `verify: false`
  opts out.
- Range artifacts: `integrity` digests, `segmentVerified`, `verify` open
  option, `verifyBaseSegment()`; sketch artifacts: `vectorsVerified`,
  `verifyVectors()`, `clearCache()`, `buildSketchArtifactBytes()` (no
  filesystem).
- Layered read budgets: `maxReadBytes` (open-path reads, default 256 MiB,
  strict) and `maxRangeBytes` (coalesced query-path reads split at 16 MiB
  record/row-aligned boundaries; also accepted by `prefetch()`). The 2 GiB
  backstop remains. Segment verification streams through Node's incremental
  SHA-256 in bounded chunks.

### Changed / fixed

- Complete-artifact `open()` fetches its post-segment-table extents as one
  wave: 8 dependent range rounds down to 5.
- `manifest.corpus.records` is validated before it sizes a read; the
  full-download fallback refuses when the content length is unknown
  (previously an unbounded read on Range-ignoring hosts).
- AVX-512 native builds no longer drop `uint8_dot` to scalar (the kernel's
  AVX2 branch is now taken on AVX-512 targets); the native addon and an ISA
  compile sweep now run in CI.
- Both engine variants rebuilt (`dist/`).

## create-pancake-search 0.3.1 — 2026-08-21

### Fixed

- Generated projects depended on `pancake-wasm: ^0.2.0`, which `npm install`
  resolves to 0.2.1 — a reader that rejects the format-v3 `.pancake-range`
  this version's scaffolder builds. Every `--runtime artifact` project
  generated by 0.3.0 therefore answered `/search` with `SNAPSHOT_INVALID`
  ("Unsupported Pancake range artifact version") until its dependency was
  bumped by hand. The generated `package.json` now inherits this package's
  own `pancake-wasm` range (`^0.3.0`) instead of carrying a second hardcoded
  one. Projects already generated by 0.3.0: change the dependency to
  `^0.3.0` and `npm install`.

## create-pancake-search 0.3.0 — 2026-08-20

Requires `pancake-wasm >= 0.3.0` (uses `pancake-wasm/complete`).

### Breaking / compatibility

- The 24.3 MiB MiniLM encoder weights are no longer in the tarball
  (package size ~170 kB, was ~25.7 MB). The CLI and the Docusaurus plugin
  download them once on first kind-3 use from the `inline-encoder-v1`
  release asset, verify the pinned SHA-256, and write them to the configured
  `weightsPath`; `PANCAKE_ENCODER_WEIGHTS_URL` points at a mirror.
  Custom-named weights are never fetched.
- Kind-3 declarations now state `maxTokens: 128` (the embedded kernel's
  real limit; earlier builds declared the teacher's 256) and
  `longInputs: 'windowed-mean-pool'`, and carry verification vectors.
  Artifacts built by 0.2.0 remain readable; rebuild to get the corrected
  declaration.

### Added

- `create-pancake-search doctor <url>` — probes an artifact hosting URL
  (Range/206, Range with a cache-key query string, negotiated protocol
  h1/h2/h3, ETag, median RTT, artifact magic/identity) and exits 1 on any
  failing check.
- Complete profile (kind 3) output: `runtime.mode: "complete"` /
  `profile: "kind3"` with `embedding.mode: "inline-transformer"` emits a
  single `search.pancake`. Docusaurus plugin option `completeProfile`
  (`enabled`, optional `vectors`, `vocab`, `weights`, `calibration`,
  `model`, `maxTokens`) writes `build/pancake-search/search.pancake` and
  serves it with a bundled kind-3 browser reader; without precomputed
  vectors the build embeds chunks through the packaged encoder.
- Docusaurus plugin `sourcePath` / `sourceRouteBase`: index the site's
  markdown/MDX sources instead of rendered HTML, with front-matter `slug`
  honored and `NN-` sidebar prefixes stripped in result routes.
- Generated Worker README and config schema updated for the new modes.

### Fixed

- `pancake-wasm/artifact` and `pancake-wasm/complete` resolve from the site,
  the plugin, or the monorepo sibling, with an actionable error otherwise
  (npm consumers of the Docusaurus plugin previously failed at bundle time).

## Earlier releases (summary)

- **pancake-wasm 0.2.1 / create-pancake-search 0.2.0 — 2026-08-12.**
  pancake-wasm: hostile-input hardening on deserialize, `packVersion`
  stamping. create-pancake-search: first publish of the Docusaurus plugin and
  `--mode student` (corpus-specific PSTU query encoder bundled in the
  Worker, no Workers AI binding).
- **pancake-wasm 0.2.0 / create-pancake-search 0.1.0 — 2026-07-30.**
  Pancake 0.2 contract: `restore`, `inspectSnapshot`, per-query `efSearch`,
  `PancakeError` / `PANCAKE_ERROR_CODES`; mutable-ef and matrix helpers
  removed; `pancake-wasm/artifact` search-artifact readers/builders.
  create-pancake-search: initial scaffolder (snapshot and artifact
  runtimes).
- **pancake-wasm 0.1.0 — 2026-06-27.** Initial release: HNSW engine in WASM
  (float and uint8 backends) for Node, browsers, and Cloudflare Workers.
