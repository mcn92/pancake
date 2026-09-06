# Complete-artifact performance characterization — 2026-08-26

What the `.pikelet` reader actually costs, measured with `poc/harness.mjs`
on the two real artifacts (both format 2 with v2 sketches), local-file and
over HTTP range requests, against the in-memory HNSW engine on the same
corpora. Raw per-query records: `poc-docs-local.json`, `poc-docs-http.json`,
`poc-wiki-local.json`, `poc-wiki-http.json` (session scratchpad).

## Setups

| | docs artifact | wiki-inline artifact |
| --- | --- | --- |
| records | 208 | 456,153 |
| file | 1.4 MB, kind 1 (student encoder) | 537 MB, kind 3 (inline MiniLM) |
| queries | 10 embedded goldens | 200 eval queries with brute-force ground truth |
| operating point | measured C=30 | measured C=600 |

Engine baseline: `Pancake.restore()` of the same snapshots, in-process WASM.

## Results

### Docs scale (208 records — the `pikelet` regime)

| metric | local file | HTTP (localhost) |
| --- | ---: | ---: |
| open (cold start to ready) | 28 ms, 1.2 MB, 8 reads | 72 ms, 1.2 MB, 8 reads |
| query, cold reader | p50 5.5 ms · 35 KB · 5 reads | p50 9.6 ms · 64 KB · ~2 reads |
| query, warm | p50 0.5 ms · 0 reads | p50 1.1 ms · 0 reads |
| correctness | 10/10 golden labels | 10/10 golden labels |

Engine on the same corpus: restore 308 ms, search p50 0.05 ms. At this
scale the artifact is effectively free either way: it opens **10× faster
than the engine restores**, and queries are single-digit ms. There is no
performance story to worry about below ~10k records.

### Wiki scale (456k records)

| metric | local file | HTTP (localhost) |
| --- | ---: | ---: |
| open | — | 1.24 s, **54 MB**, 7 reads |
| query, cold reader | p50 **351 ms** · 192 KB · 348 reads | p50 **535 ms** · 514 KB · 382 reads |
| query, warm | p50 **251 ms** · 5.7 KB · 10 reads | p50 262 ms · 5.7 KB · 10 reads |
| recall@10 vs brute force | 82.4 % | 82.2 % (60-query subset) |

The decomposition matters more than the totals:

- **The inline encoder is the floor: ~250 ms/query.** Warm queries do
  ~6 KB of I/O in 10 reads and still take 251 ms — that is the kind-3
  MiniLM forward in WASM, not the artifact. Retrieval + rerank I/O adds
  only ~100 ms cold (local) / ~280 ms (HTTP, localhost). A kind-1/kind-2
  artifact at this scale would answer warm queries in ~10–20 ms.
- **Open cost is 54 MB** — 28 MB resident sketch + 25 MB encoder weights
  (kind-3 carries its model). That is the cold-boot price of "no server,
  no model host"; it amortizes across the session and CDN-caches.
- **Requests, not bytes, price the network**: ~380 reads/query cold. On
  localhost that is ~180 ms; from the HTTP replay measurement, at 20 ms
  RTT over h2 the same pattern costs ~170 ms (parallel waves), and
  ~1.5 s on HTTP/1.1's 6-connection limit — host on h2/h3, as `doctor`
  checks.

Engine comparison at wiki scale (extrapolated — no full 456k engine build
exists locally; the 5.6k-chunk sample restores in 580 ms at 3.6 MB and
searches at p50 0.28 ms): the full snapshot is ~190 MB, so an engine
deployment pays **~190 MB transfer + multi-second restore + resident RAM +
an encoder service** before its first ~1 ms query. The artifact pays 54 MB
and ~350 ms/query with the encoder *included*.

### Break-even

Cold-for-cold at wiki scale: engine boot ≈ 190 MB ≈ **370 artifact
queries' worth of egress** (514 KB/query). Sessions shorter than a few
hundred queries are cheaper end-to-end on the artifact even before
counting server cost; sustained high-QPS workloads should download the
snapshot and run the engine — which is exactly the snapshot runtime
`pikelet` also ships.

## How to read this against "HNSW is faster"

In-memory HNSW answers in ~0.05–1 ms and always will; the artifact's
query path is bounded by encoder time (kind 3) and RTT × request-waves
(any kind). They are different points on a residency curve, not
competitors on one axis:

| | resident bytes before 1st query | 1st-query latency | steady-state query |
| --- | ---: | ---: | ---: |
| engine + snapshot (wiki) | ~190 MB + RAM + encoder service | seconds–minutes (transfer) | ~1 ms + encoder |
| artifact, kind 3 (wiki) | 54 MB | ~1.6 s | ~250 ms (encoder-bound) |
| artifact, docs scale | 1.2 MB | ~40 ms | ~1 ms |

Latency is worth measuring — but as *latency at a residency budget*,
cold/warm split, at a declared RTT and protocol. The single-number
comparison against resident HNSW measures the wrong thing.

## Caveats

- HTTP numbers are localhost: request-count effects are real (the reads
  happen), RTT effects come from the separate replay measurement
  (`ROW_COMMITMENT_MEASUREMENT.md`), bandwidth is unpriced.
- The wiki engine row is extrapolated from the 5.6k sample restore.
- Kind-3 encoder time is the WASM kernel on this machine (one thread);
  browsers measure in the same ~200–300 ms class (see `test-browser.mjs`
  timings).
- v2 sketches (per-row verification) are what's measured; verification
  hashing is included in every number above.
