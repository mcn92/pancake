# Pancake Roadmap

Last updated: 2026-07-31

Distilled from `spec/SEARCH_ARTIFACT_CONTRACT.md` (Draft 1) and the
project's working notes. Update this file when a track changes direction.

## North star

Turn Pancake from a vector-search library into an artifact compiler and
reader. The target is the Complete Search Artifact profile
(`spec/SEARCH_ARTIFACT_CONTRACT.md` §9.3): one immutable, content-addressed
package carrying corpus, index, encoder, evaluation, and calibration, that a
bounded-memory reader can execute over range reads.

The pieces already exist separately: the range-readable index
(`pancake-artifact.js`), the distilled encoder with calibrated abstention
(`examples/worker-semantic-search/`), and the contract that binds them
(`spec/`). The roadmap is mostly about converging them.

## Track A: Range artifact proof path (active)

Settled comparison point (do not change until something clearly beats it):
SIFT1M u8, `efSearch=80`, `expansionBatch=8`, `gap=65536`, parallelism 6 —
95.58% recall@10, ~106 requests and ~0.54 MiB per query. The structural
bottleneck is sequential base-layer miss-round depth (~12 mean, 15–16 tail),
not bytes.

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

4. Geometry results 2026-08-01 (`benchmarks/range_cluster_page_sim.js`,
   SIFT1M, 1000 queries): the winning geometry is **resident sketch +
   fetch-to-rerank** — a 2:1 pooled u8 sketch of every vector stays resident
   (68.7 MiB at 64D; pooling preserves each row's affine scale/offset), the
   query scans sketches locally, and only the top-C full records are fetched
   in one parallel round for exact rerank. At C=300: 96.11% recall@10
   (baseline 95.58%), 144 KiB/query (baseline 535 KiB), 87 mean coalesced
   requests, round depth 1 (baseline ~12 sequential). Modeled p95 @10ms/read:
   241 ms vs 615 ms at p=6, and ~51 ms at browser-real parallelism (p=32),
   which sequential traversal cannot exploit. Next: 4-bit sketches to cut
   resident bytes to router parity (~40 MiB), then a real reader
   implementation as a new artifact profile.
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

- Consolidate the near-duplicate HNSW backends (`src/float_hnsw.hpp`,
  `src/uint8_float_hnsw.hpp`) behind a storage-policy template, and the
  duplicated wrapper layer in `src/engine.cpp` / `native/pancake_napi.cpp`.
  Every graph fix currently lands in two to four places.
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
