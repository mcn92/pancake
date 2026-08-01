# Pancake Roadmap

Last updated: 2026-07-31

Distilled from `spec/SEARCH_ARTIFACT_CONTRACT.md` (Draft 1) and the
project's working notes. Update this file when a track changes direction.

## North star

Turn Pancake from a vector-search library into an artifact compiler and
reader. The target is the Complete Search Artifact profile
(`spec/SEARCH_ARTIFACT_CONTRACT.md` §9.4): one immutable, content-addressed
package carrying corpus, index, encoder, evaluation, and calibration, that a
bounded-memory reader can execute over range reads.

The pieces already exist separately: the range-readable index
(`pancake-artifact.js`), the distilled encoder with calibrated abstention
(`examples/worker-semantic-search/`), and the contract that binds them
(`spec/`). The roadmap is mostly about converging them.

## Track A: Range artifact proof path (active)

Settled comparison point: SIFT1M u8, `efSearch=80`, `expansionBatch=8`,
`gap=65536`, parallelism 6, 95.58% recall@10 — in two regimes that must not
be conflated (audit of 2026-08-01):
- warm-amortized (cache accumulating over 1000 queries, the historical
  numbers): 106 requests / 0.535 MiB mean, ~12 miss rounds;
- cold per-query (`--clear-cache-per-query`, the number that matches
  first-visit behavior and the Worker/R2 measurements): 665 requests /
  6.39 MiB mean (p95 9.22 MiB), 23.7 miss rounds, modeled p95 1894 ms at
  10 ms/read.
The structural bottleneck is sequential miss-round depth in both regimes.

Near term:

1. DONE 2026-07-31: docs-scale semantic artifact live at
   pancake-artifact-demo.pages.dev — 208-chunk docs index, local distilled
   encoder, corpus hydration, and calibrated abstention, all client-side.
   Rank-parity verified against the snapshot reader; abstention golden
   probes pass in-browser.
2. DONE 2026-07-31: warm-cache measured across 1/10/100 queries with
   server-side ground truth — cold query pages in the base segment
   (~122 KiB), queries 11–100 generate zero requests; reload re-warm costs
   ~13 KiB with immutable cache headers vs 1.37 MiB without. Numbers in the
   static demo README.
3. Confirm static hosts preserve Range requests and cache behavior. Status
   2026-07-31: Cloudflare Pages ignores Range in its static pipeline
   (full-download fallback added); a bundled Pages Function now restores real
   206 slicing, verified live at pancake-artifact-demo.pages.dev with cache
   headers honored. jsDelivr serves correct ranges (content-range total field
   unreliable but unused). Remaining: GitHub Pages, S3/CloudFront.

Structural work (the real fix for miss-round depth):

4. Geometry results 2026-08-01: the winning geometry is **resident sketch +
   fetch-to-rerank** — a 2:1 pooled u8 sketch of every vector stays resident
   (68.7 MiB at 64D; pooling preserves each row's affine scale/offset), the
   query scans sketches locally, and only the top-C full records are fetched
   in one parallel round for exact rerank. Measured end to end through the
   real RangeArtifact reader over HTTP with 10 ms injected per-request delay
   (`benchmarks/sketch_rerank_e2e.js`, SIFT1M, C=300, ~95.4% recall@10 vs
   traversal's 95.8%):
   - p=6:  sketch 355 ms mean / 528 ms p95 vs traversal 1206 / 2035 ms
   - p=32: sketch 293 ms mean / 347 ms p95 vs traversal 967 / 1293 ms,
     with 244 KiB/query vs 775 KiB (gap=4096 for both)
   Cold-fetch request geometry at C=300 sweeps from 285 requests / 144 KiB
   (gap=0) to 128 requests / 2.5 MiB (gap=65536): rerank fetches want small
   gaps, traversal wants large ones. Remaining sketch wall time is dominated
   by the ~120 ms pure-JS sketch scan — the natural WASM SIMD target.
   Known accounting caveat, found by the e2e test: simulated byte counts
   that ignore coalescing gap filler understate real transfer (the sim's
   pre-fix numbers claimed 144 KiB at gap=65536; reality is 2.5 MiB).
   Audit result 2026-08-01: `range_artifact.js`'s own byte arithmetic is
   correct — the historical 0.535 MiB baseline was warm-amortized (shared
   cache across 1000 queries), not per-query cost. Cold per-query reality is
   665 requests / 6.39 MiB / 23.7 miss rounds, which fully explains the
   Worker/R2 measurements; see the settled-comparison-point note above.
   Cold-vs-cold, sketch C=300 at gap=0 fetches 144 KiB in 1 round versus
   traversal's 6.39 MiB over ~24 rounds.
   Progress 2026-08-01: 4-bit sketches reach router-parity residency
   (38.1 MiB, recall 96.00% vs 96.11% at 8-bit); the engine now exports a
   SIMD pancake_sketch_scan kernel (scan 84.9 ms JS -> 16.7 ms WASM; e2e
   over HTTP @10 ms, p=32: sketch 224 ms mean / 288 ms p95 vs traversal
   981 / 1300 ms at equal recall).
   Real-world validation 2026-08-01: against the July R2 bucket (8-part
   SIFT1M artifact) through a range-proxy Worker over the public internet
   (~200 ms RTT): sketch 1256 ms mean / 1752 ms p95 vs traversal 6317 /
   7741 ms — 5.0x, with traversal reproducing the July Worker/R2 failure
   numbers almost exactly. Sketch wall time is RTT-bound (~6 request waves);
   a client near an edge PoP at 20-50 ms RTT lands at roughly 150-400 ms.
   Progress: the sketch profile is specified (`spec/SKETCH_PROFILE.md`,
   contract §9.3) and implemented — builder and reference reader ship in
   `pancake-artifact.js` on all entrypoints (`Pancake.SketchArtifact`,
   `buildSketchArtifact[File]`, `openSketchArtifactFile`), with conformance
   checks in `test/sketch_profile.js` wired into npm test. SIFT1M validated
   through the product reader: 160 MiB artifact (vs 494 MiB range), 38.1 MiB
   verified resident prefix, 96.00% recall@10 at the header's recommended
   rerank, 42.5 KiB/query fetched. Remaining: browser/e2e exercise of the
   new reader and committed golden fixtures.
   Closed geometry lines (measured, do not reopen without new evidence):
   - cluster-page routing with centroid selection: needs P=128 pages /
     11.8 MiB for 96% — selection, not partition quality, is the bottleneck
     (oracle selection hits 98.1% at P=8);
   - sampled-representative, router-hit, and centroid-minus-radius page
     selectors: all worse than plain centroids in 128D;
   - two-round edge-guided page refinement: +1–3 points over same-budget
     single round — METIS edge-cut minimization keeps candidates' edges
     inside already-fetched pages, defeating graph-guided selection.
5. Traversal-level tweaks (speculative neighbor-of-neighbor fetch, adaptive
   expansion on miss) are moot if the sketch geometry ships: round depth 1
   beats any traversal tuning.

Closed lines (do not reopen without new evidence):

- Worker/R2 direct lazy range reads — quarantined in
  `archive/worker-r2-range-artifact-2026-07-31/`.
- Per-node miss-hot resident sets — no effect on miss rounds.
- Trace-cohort synthetic pages — worse than baseline on bytes, requests, and
  rounds.

## Track B: Contract and formats

1. Write the byte-layout specification for `.pancake-range` v2. It currently
   exists only as code in `pancake-artifact.js`; the contract requires each
   graph profile to declare its traversal semantics explicitly.
2. Package the conformance kit from the fixtures named in contract §5.4
   (golden snapshots, search oracles, abstention goldens, encoder parity).
   Designate the reference reader and build target per profile and record its
   golden results.
3. Next format revision, to meet Draft 1 requirements:
   - canonical manifest identity with range-verifiable chunk commitments
     (current v2 binary has neither);
   - corpus segment with id → byte-range mapping for result hydration
     (corpus is currently an adjacent application asset);
   - encoder / evaluation / calibration segments, seeded from the
     worker-semantic-search assets (`docs-abstention.json`,
     `docs-manifest.json` evaluation block).
4. Resolve the remaining open decisions in contract §10 — chunk-size policy
   and per-metric distance-error bounds first, since they gate the byte
   layout and the conformance kit.

Curation status (2026-07-31): examples/ has a guided README naming the
canonical trio; broken/legacy demos and local copies archived; private-mode
hosting documented; tracked-tree secret scan clean. All demos swept and
verified working (see examples/README.md ordering).

## Track C: Demos and product surface

- `create-pancake-search` (v0.1.0): artifact runtime mode just landed
  (`templates/worker.artifact.js`). Next: the `student` encoder mode
  (currently stubbed), which is the scaffolder's path to encoder-bearing
  artifacts. Decide when to publish to npm.
- `worker-semantic-search` maintenance:
  - always run `test_worker.mjs` after retraining, not just
    `verify_student.mjs` — offline and live distances have diverged before;
  - if data changes move the weak fixture (`docker compose networking`),
    replace it but keep a strict weak fixture — never relax weak to
    "weak or none";
  - consider keeping `docs-student-evaluation.json` out of the Worker bundle
    if size matters;
  - run a fresh secret/state scan on the exact folder or archive before any
    external sharing (Show HN etc.).

## Track D: Core library

The core is stable (1191 tests passing) and positioned honestly. Work here is
maintenance, not features:

- Keep the two HNSW backends separate (decision 2026-07-31, reversing the
  earlier consolidation item). The code-level review found they differ in
  representation, not just storage type — cached edge distances, three
  distance contexts, sum caches, and distinct serialization — so a
  storage-policy template would parameterize more than it deduplicates, on
  the hot path, for zero WASM binary savings. History supports it: 9 of the
  uint8 backend's commits changed its internal representation in ways the
  float backend must not inherit, and the float backend doubles as the
  control when debugging quantization regressions. Dual-maintenance risk is
  mitigated by the recall oracles, golden outputs, and SIMD parity tests.
  Rule of three: abstract when a third backend appears, not before. The
  duplicated wrapper adapter in `src/engine.cpp` / `native/pancake_napi.cpp`
  is the one defensible extraction (pure adapter code, no representational
  divergence), low priority.
- Remove or relocate `src/embedding_model.hpp` (unused by the engine build).
- Performance work on the u8 backend must gate on recall across datasets
  before shipping — nytimes-angular caught the B-heal and querydot
  regressions that SIFT did not.

## Track E: Repo hygiene

- Commit the pending range-artifact work: `pancake-artifact.js`,
  `create-pancake-search/templates/worker.artifact.js`, `package.json`,
  `spec/`, `examples/search-artifact-demo/static/`, `benchmarks/range_artifact*`.
- Clear the root of loose artifacts before any external source browse: the
  29 GB ground-truth tarball, AWS bundles, MNIST files, `commit*.txt`
  drafts, the duplicate `worker-semantic-search` copies and zips, and the
  unshipped clutter in `dist/` (`vectors_100k.bin` is 614 MB).
