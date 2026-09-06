# Per-row vector commitments: access-pattern measurement

**For:** `spec/COMPLETE_PROFILE.md` section 8, question 6 (closing the
lazy-row integrity gap per read).
**Method:** `measure-row-commitments.mjs` — open the wiki-scale sketch
(456,153 rows × 384 B u8), replay the 200-query eval set at the measured
operating point (C=600) and at C=120, capture the exact candidate-id sets
`fetchRows()` receives, and model — per digest-page geometry — the reads a
verifier would add, under the reader's own run-coalescing. Three id-set
families: **real** (captured), **uniform** (random, same sizes), and
**adversarial** (ids spaced N/C apart: maximal dispersion, zero
coalescing). `measure-row-commitments-http.mjs` then replays the captured
run lists against local HTTP/1.1 (6-connection pool) and HTTP/2 (64
streams) servers with injected per-request latency, turning request counts
into wall-clock. Raw numbers: `row-commitment-results.json`,
`row-commitment-http-results.json`.

## Workload baselines (mean per query)

| workload | candidates | id-gap p50/p90 (rows) | row bytes fetched | row read runs |
| --- | ---: | ---: | ---: | ---: |
| real, C=600 | 600 | 270 / 2,059 | 673 KiB | 423 |
| uniform, C=600 | 600 | 527 / 1,748 | 481 KiB | 567 |
| adversarial, C=600 | 600 | 760 / 760 | 225 KiB | 600 |
| real, C=120 | 120 | 1,393 / 10,078 | 92 KiB | 95 |

Real candidates cluster ~2× tighter than uniform (article chunks are
id-adjacent, queries are topical): fewer requests, more gap bytes.

## Byte/run model (separate-region digest table)

Per-row digests of D bytes in pages of P rows; a resident page-hash table
(32 B/page) anchors the pages and sits in the resident prefix, so
`residentSha256` — and therefore the artifact identity — covers it for
free. Selected points, real workload:

| P × D | digest reads C=600 (gap 0) | extra requests | C=120 (gap 0) | resident anchor | table |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 × 16 B | 121 KiB | ~484 | 25 KiB | 891 KiB | 6.96 MB (4.2%) |
| 32 × 16 B | 232 KiB | ~464 | 50 KiB | 446 KiB | 6.96 MB (4.2%) |
| 256 × 32 B (corpus-v2 defaults) | 4.08 MB | ~364 | 723 KiB | 56 KiB | 13.9 MB (8.3%) |

- Corpus-v2's geometry does not transplant (4–6× the row bytes per query).
- Small pages + truncated digests are cheap in **bytes** (0.18× rows) but
  roughly **double the request count**.
- **Adversarial dispersion stays bounded**: every candidate in its own page
  → digests 150 KiB in 600 runs at C=600 (0.67× that workload's row
  bytes). The scheme degrades linearly, never cliff-like.
- Cross-query page reuse is high (73–85% at C=600): with a page cache the
  6.96 MB table converges toward resident within a session.
- Merkle-per-run proofs are dominated: ≥0.75× row bytes on every query,
  no cache convergence, format and prover complexity.

## Real-HTTP wall-clock (the decisive table)

Replaying each query's actual run lists (geometry 16 × 16 B), mean
ms/query; overhead vs today's rows-only reader:

| scenario | h1, 20 ms RTT | h2, 20 ms | h1, 80 ms | h2, 80 ms |
| --- | ---: | ---: | ---: | ---: |
| rows only (C=600) | 1,569 | 171 | 6,050 | 630 |
| + separate digests, gap 0 | **+102%** | **+94%** | +102% | +93% |
| + separate digests, gap 4K | +70% | +63% | +71% | +62% |
| interleaved digest blocks | **−1%** | **−5%** | −1% | −4% |
| rows only (C=120) | 358 | 46 | 1,413 | 169 |
| + separate digests, gap 0 | +100% | +88% | +100% | +86% |
| interleaved | +0% | +2% | +0% | −0% |

Two findings the byte model alone would have gotten wrong:

1. **Request count dominates wall-clock on both protocols.** Even HTTP/2's
   multiplexing pays ~90% for doubled requests — 64 concurrent streams
   just means half as many round-trip *waves*, and doubling requests
   doubles the waves. Lazily fetching a separate digest table costs ~2×
   query latency at WAN RTT despite adding only 0.2× bytes.
2. **The interleaved layout is wall-clock free.** Each 16-row block
   stores its 256 B digest page inline (`[digests][rows]`), so with the
   reader's 16 KiB coalescing gap a block's digests always ride inside an
   existing row run: **zero additional requests**, ±0% latency in every
   cell. Its cost is bytes — the rerank reads grow from 673 KiB to
   ~2.1 MB at C=600 (gap-swallowed block content) — which is invisible
   when RTT dominates and costs ~230 ms at 50 Mbps when bandwidth does.

## Conclusion for SKETCH_PROFILE's next revision

Two designs survive, and they are complementary rather than competing:

- **Interleaved digest blocks (16 rows × 16-byte truncated SHA-256, 32-byte
  page hashes in the resident prefix): the default.** Per-read
  verification at ~0% latency overhead and ~3× byte inflation on the
  rerank's cold reads; adversarial-safe; +4.2% file size; +891 KiB
  resident anchor, identity-covered via `residentSha256`.
- **Eager table warm for verify-everything hosts:** the same digests are
  readable as one ~7 MB sequential region (they are contiguous per block
  run — or a builder can emit a separate table variant), after which every
  verification is memory-only. Either way this replaces today's 167 MB
  `verifyIndexVectors` full pass at ~24× less transfer.

Rejected by measurement: corpus-v2's 256-row/32-byte geometry (4–6× byte
amplification), lazily-fetched separate-region tables as the default
(~2× wall-clock on h1 *and* h2), and Merkle-per-run proofs (perpetual
0.75× byte cost, no convergence, complexity).

Caveats: loopback underweights bandwidth (byte costs priced from the
static model, not the replay); h1/h2 client models are Node approximations
of browser connection behavior; single corpus (wiki) — though the
adversarial family bounds layouts with no locality at all.
