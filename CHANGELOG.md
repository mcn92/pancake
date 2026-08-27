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

### Added

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
  logistic model over (d0, margin, mean10, known_frac) with its threshold
  placement, except that with no weak band the hard threshold hugs the
  negative ceiling — a false abstain hides results, a false weak shows
  them with a caveat. Skips with a logged reason (and ships unscored)
  when positives or negatives run short or the fit misses its 0.85 AUC
  gate; the embedded asset records method, query counts, and AUC. New
  flags: `--calibration <file>` embeds a prebuilt retrieval-signals-v1
  asset, `--skip-calibration` opts out. The Docusaurus plugin default is
  unchanged (auto-calibration is opt-in via `runtime.calibration: 'auto'`
  outside compile).

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
