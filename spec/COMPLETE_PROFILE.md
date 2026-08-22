# Pancake Complete Search Artifact Profile

**Status:** Draft 2 (2026-08-21) — for review, not frozen. Draft 2 adds
format version 2: per-record corpus integrity (section 3.5), the host-encoder
verification obligation for kind 2 (section 3.6), the reader's bounded-read
rules (section 4), and the CI conformance suite (section 5). Format-1 files
remain readable; readers report which integrity stance a file carries.
**Profile of:** the Search Artifact Contract (`SEARCH_ARTIFACT_CONTRACT.md`, section 9.4)
**File extension:** `.pancake`
**Magic:** `PSF1` (`0x31465350`, little-endian u32)

## 1. Purpose

One file that is a search engine for one corpus. A complete artifact carries
all five contract components — corpus, index, encoder, evaluation,
calibration — under a single content-addressed identity, so that a reader
can answer natural-language queries with hydrated, confidence-scored results
using nothing but the file and byte-range reads against it.

The profile composes formats this repository has already frozen rather than
inventing new ones: the index segment is a complete `.pancake-sketch`
artifact embedded verbatim, and the query-interpretation segment carries the
existing student-encoder and calibration formats. The container contributes
identity, addressing, and the corpus layout — the three things the
2026-08-13 composition spike (`examples/05-one-file-search/`) identified as
missing between the components.

## 2. Design decisions (resolved 2026-08-13)

1. **Compile-time identity mapping.** Index row `i` IS corpus record `i`.
   The compiler renumbers when assembling the file; the container carries no
   id map and readers perform no id translation. A producer packing a
   snapshot with a non-identity internal/external id map MUST renumber the
   corpus to match the index's internal order (or rebuild) at compile time.
2. **Embedded sketch artifact as the index segment.** The index segment's
   bytes are a valid `.pancake-sketch` file (SKETCH_PROFILE.md), opened by
   the existing sketch reader at `indexOffset`. Depth-1 execution, staged
   boot, and resident-hash verification are inherited, not re-specified.
3. **One query-interpretation unit.** Encoder and calibration share a
   segment and a version, because calibration may consume the encoder's
   feature stream and hidden state, not only its output vector. They cannot
   version independently.
4. **Two query-interpretation kinds** (added 2026-08-14 for the wiki-scale
   compile): inline student encoders (kind 1) and pinned external encoders
   with verification vectors (kind 2, contract section 4.4 mode 2). The
   product direction favors kind 1; kind 2 exists because real corpora
   (the wiki pack) currently interpret queries with a full teacher model
   the host executes.

## 3. File layout

All integers little-endian. All offsets absolute. Segments begin at
16-byte-aligned offsets; inter-segment padding MUST be zero bytes.

```
[0, 64)       header
[64, ...)     manifest        canonical JSON, manifestBytes long
[...]         segment table   segmentCount x 48-byte entries
[...]         segments        in table order, 16-byte aligned
```

### 3.1 Header

| Offset | Type | Field | Notes |
| ---: | --- | --- | --- |
| 0 | u32 | magic | `0x31465350` (`PSF1`) |
| 4 | u32 | formatVersion | `1` (corpus layout v1, profile `pancake-complete-v1`) or `2` (corpus layout v2, profile `pancake-complete-v2`); readers MUST reject other values |
| 8 | u32 | manifestBytes | length of the canonical manifest JSON (readers MUST reject > 16 MiB) |
| 12 | u32 | segmentCount | number of segment-table entries |
| 16 | u64 | fileBytes | total file size; MUST match |
| 24 | 32 bytes | manifestSha256 | digest of the manifest bytes |
| 56 | 8 bytes | reserved | MUST be zero in v1; readers MUST ignore |

The **artifact identity** (contract section 4.1) is `manifestSha256`. The
manifest commits to every segment's digest, so the identity transitively
commits to all content. `fileBytes` and the header itself are addressing
convenience, not identity: two files with identical manifests and segments
but different segment order have different bytes and the same identity.

### 3.2 Manifest

Canonical JSON (UTF-8, sorted keys, no insignificant whitespace — the byte
serialization the digest is computed over is the one in the file). Required
fields:

```jsonc
{
  "profile": "pancake-complete-v2",                    // "pancake-complete-v1" for format-1 files
  "corpus": {
    "records": 208,
    "provenance": null,                                 // reserved per contract 4.3
    // format 2 only (layout v2, section 3.5); absent on format 1:
    "layout": "records-v2",
    "pageRecords": 256,
    "pages": 1,
    "recordDigest": "sha256",
    "pageTableSha256": "..."                            // digest of the page table bytes
  },
  "dim": 384,
  "metric": "cosine",
  "encoder": { /* identity, preprocessing, dims, normalization — contract 4.4 list */ },
  "segments": [
    { "kind": "index",       "sha256": "...", "bytes": 126400 },
    { "kind": "corpus",      "sha256": "...", "bytes": 167994 },
    { "kind": "query-interp","sha256": "...", "bytes": 1208893 },
    { "kind": "evaluation",  "sha256": "...", "bytes": 190192 }
  ],
  "sampleQueries": [ "..." ]
}
```

Segment order in `segments` MUST match the segment table. Unknown manifest
fields MUST be ignored by readers (and are covered by the identity digest).

### 3.3 Segment table entry (48 bytes)

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | u32 | kind (`1` index, `2` corpus, `3` query-interp, `4` evaluation) |
| 4 | u32 | reserved (zero) |
| 8 | u64 | offset (16-byte aligned) |
| 16 | u64 | bytes |
| 24 | 24 bytes | reserved (zero) |

Exactly one segment of each kind 1–3 is REQUIRED; evaluation (kind 4) is
REQUIRED for conformance but a reader MAY serve queries without reading it.
Unknown kinds MUST be skipped (they are still committed via the manifest).

### 3.4 Index segment (kind 1)

A byte-for-byte valid `.pancake-sketch` artifact (magic `PSA1`), row ids
`[0, count)` binding positionally to corpus records. Readers open it with
the sketch reader against a range source offset by the segment's `offset`;
`staged` open is RECOMMENDED for interactive hosts. The sketch artifact's
internal `residentSha256`/`vectorsSha256` remain valid and SHOULD be
verified per SKETCH_PROFILE.md; the container's segment digest additionally
covers the whole segment.

### 3.5 Corpus segment (kind 2)

The contract's section 4.8 id-to-byte-range requirement, in the simplest
layout that satisfies it:

```
[0, 4)                u32 count
[4, 4 + 8*(count+1))  u64 offsets[count+1]   record i occupies
                                             [offsets[i], offsets[i+1])
                                             relative to segment start
[...]                 records                UTF-8 JSON, one object each
```

Record objects carry at minimum `title`, `text`, `sourcePath`; `preview`,
`anchor`, and `url` are OPTIONAL. Hydrating result id `i` is one range read
of `[offsets[i], offsets[i+1])` — resident cost is the offsets array
(`8*(count+1)` bytes, ~1.6 MB at 200k records). Readers MAY cache records;
cache state MUST NOT change results.

That is **layout v1**, carried by format-1 files: record reads are covered
only by the whole-segment digest, so a lazily read record is not
independently verifiable. **Layout v2** (format 2, manifest
`corpus.layout: "records-v2"`) makes each record read range-verifiable per
contract sections 4.1 and 7:

```
[0, 4)                 u32 count
[4, 8)                 u32 pageRecords (P)        1 <= P <= 65536; 256 by default
[8, 8 + 8*(count+1))   u64 offsets[count+1]       record i = [offsets[i], offsets[i+1])
[A, A + 32*pages)      pageSha256[pages]          pages = ceil(count / P); entry p is the
                                                  SHA-256 of recordSha256[p*P, min(count,(p+1)*P))
[B, B + 32*count)      recordSha256[count]        SHA-256 of each record's bytes
[C, ...)               records                    offsets[0] MUST equal C
```

The manifest commits to the page table (`corpus.pageTableSha256`, together
with `pageRecords`, `pages`, and `recordDigest: "sha256"`), so the
verification chain is: identity → page table (read at open, 32 bytes per
`P` records — 57 KB at 456k records) → one page of record digests (one read
of `32*P` bytes the first time a record in that page is hydrated; readers
SHOULD cache verified pages) → the record. A reader MUST verify the page
table against the manifest at open, and each hydrated record against its
digest (after verifying that digest's page) before returning it; a mismatch
MUST fail the hydration, not degrade it. Per-record reads stay one range
read plus, per page touched, one small read.

Format-2 files MUST use layout v2 and header `formatVersion` 2; format-1
files MUST use layout v1 (no `corpus.layout`) and `formatVersion` 1. A
reader MUST reject a manifest whose layout does not match its header
version.

### 3.6 Query-interpretation segment (kind 3)

```
[0, 4)    u32 version          shared encoder+calibration version
[4, 8)    u32 kind             1 = student-inline-v1, 2 = external-transformers-v1,
                               3 = inline-transformer-v1
[8, 12)   u32 encoderBytes
[12, 16)  u32 calibrationBytes
[16, ...) encoder              per kind, below
[...]     calibration          UTF-8 JSON, per kind
```

**kind 1 — student-inline-v1:** encoder bytes are a distilled student model
(loadStudentModel format); calibration is the student abstention model,
which consumes the encoder's feature stream. Fully self-contained, pure-JS
execution.

**kind 2 — external-transformers-v1:** the contract's section 4.4 mode 2
(pinned external encoder). The encoder bytes are a UTF-8 JSON *declaration*
pinning the model (id, pooling, normalization policy, max tokens) and
carrying verification test vectors — query texts with their expected
embeddings at a declared tolerance — so a host-supplied encoder can be
checked against the artifact before serving. Calibration is a
retrieval-signal abstention model (distance signals plus a corpus-vocabulary
bloom filter, base64-embedded) that scores query text and hits without
touching the encoder internals. A reader without a host encoder for the
declared model MUST surface the artifact as requiring one, not fall back
silently. A reader given a host encoder MUST run it against every
verification vector before serving the first query — dimension equal to
the manifest's, every component within the vector's declared tolerance —
and MUST refuse the open on disagreement (encoder/index skew is a contract
violation, section 4.4). A kind-2 declaration without verification vectors
is incomplete: a reader MUST refuse to serve it as verified and MAY serve
it only when the host explicitly accepts an unverified encoder, reporting
that state (the reference reader: `allowUnverifiedEncoder`,
`info().encoderVerified === false`).

**kind 3 — inline-transformer-v1:** the pinned teacher compiled into the
artifact as data. The encoder bytes are three length-prefixed regions:

```
[0, 4)   u32 declBytes      [4, 8)  u32 vocabBytes   [8, 12) u32 blobBytes
[12, ..) declaration        UTF-8 JSON: model identity, revision, license
                            and attribution (the corpus-provenance rule of
                            contract 4.3 applied to weights), pooling,
                            normalization, max tokens, an optional prefix
                            policy ({passage, query} strings; readers MUST
                            prepend the query prefix before embedding, so
                            queries land in the same space as prefixed
                            passages; absent means empty), and the blob
                            layout constants (V, P, D, F, L, B, H)
[...]    vocab              UTF-8, one WordPiece token per line
[...]    weight blob        block-affine u8 matrices + f32 norm params in
                            the declared deterministic layout
```

Weights are data; the kernels that execute them ship with the READER, never
inside the artifact (a reader MUST NOT execute code from artifact bytes —
contract section 7). Calibration is the same retrieval-signal model as
kind 2. A reader without transformer kernels for the declared layout MUST
surface the artifact as unsupported. The token-embedding table occupies a
contiguous region of the blob; a future revision of this kind will declare
its offsets so readers can leave it lazy (range-read per token id,
sketch-style) instead of resident — v1 readers load the blob whole.

In all kinds, encoder and calibration are loaded together and verified
together under the shared version. A reader that cannot evaluate the
calibration MUST report results as uncalibrated rather than inventing
confidence (contract section 4.7).

### 3.7 Evaluation segment (kind 4)

UTF-8 JSON: golden queries with expected top-k ids and match-quality
labels, recall-vs-C measurements for the embedded sketch geometry, and
teacher/student fidelity metrics. The golden queries double as the
conformance fixtures for readers (section 5).

## 4. Execution semantics

`open(source)`:

1. Read header (64 bytes); check magic, version `1` or `2`, sane counts
   (`manifestBytes` ≤ 16 MiB, 1 ≤ `segmentCount` ≤ 64, `fileBytes` a safe
   integer equal to the source size when the source reports one).
2. Read manifest + segment table; verify `manifestSha256`; parse; verify
   table/manifest agreement (kinds, packed 16-byte-aligned offsets within
   file, byte lengths, one segment per known kind, unknown kinds skipped),
   and the profile/layout against the header version.
3. Read the query-interpretation segment (one read, ~1.2 MB today), verify
   its digest; check its version word; load encoder + calibration; for
   kind 2 with a host encoder, verify the host encoder (section 3.6).
4. Open the index segment with the sketch reader (staged or full); read the
   corpus tables (count + offsets, plus the page table on layout v2) in one
   read; check the count words against the manifest, the offsets
   (monotonic, starting exactly where the tables end, ending inside the
   segment), and on layout v2 the page table's digest.
5. The artifact is now serving. Total cold-open transfer at the reference
   corpus: header + manifest + query-interp + sketch stage-1 — under 2 MB.

Every read in steps 1–4 (and every lazy read below) is issued only after
its `(offset, length)` has been validated — safe non-negative integers, no
overflow, within `fileBytes`, within the source's size when known, and
within the reader's per-read budget (the reference reader: 256 MiB per
open-path read, configurable; 16 MiB per corpus record; 2 GiB absolute) —
and MUST return exactly the bytes requested. A short read is a failure, not
a partial success. u64 fields above `2^53 - 1` are rejected.

`query(text, k)`:

1. Encode: text → vector + feature stream (pre-search abstention MAY answer
   here without touching the index).
2. Search: sketch scan + one parallel rerank fetch round (SKETCH_PROFILE.md
   section 3).
3. Hydrate: one range read per result id via the corpus offsets; on layout
   v2, verify the record against its digest (fetching and verifying that
   digest's page on first use). A record that fails verification fails the
   query rather than being returned.
4. Calibrate: score match quality from hits + feature stream; a `none`
   verdict returns zero results with the score.

Result shape: `{ matchQuality, confidence, results: [{ id, distance,
title, text, sourcePath, ... }] }` — ids, distances, and hydrated records
in one response; returning bare ids does not satisfy this profile.

## 5. Conformance

- **Fixtures:** `test/complete_profile.mjs` (run by `npm test`, so by CI)
  builds every fixture deterministically in-process — no downloads, no
  model weights — and is the reference reader's conformance suite:
  (A) a seeded kind-2 format-2 artifact with a deterministic host encoder,
  covering open/query/hydrate, host-encoder verification, per-record and
  page-table tamper detection, structural rejection of hostile headers,
  tables, and offsets, read budgets, truncation and short reads, unknown
  and duplicate segments, and format-version bounds; (B) the same corpus as
  a format-1 file, which MUST still open and report the transitional
  integrity stance (a record tamper is documented as undetectable there);
  (C) a kind-1 artifact compiled from the committed
  `examples/03-edge-docs-search` assets, whose 10 abstention goldens MUST
  reproduce their labels, whose hydration round-trips the source corpus,
  and whose compile MUST be byte-deterministic. Kind 3 is covered by
  `examples/05-one-file-search/test-inline.mjs` against the released
  wiki-inline artifact (its weight blob is not a CI fixture).
- **Producer:** emits structurally valid files whose manifest digests
  verify, whose index segment passes sketch-profile conformance, whose
  corpus round-trips every record, and whose evaluation bounds hold under
  the reference reader.
- **Reader:** verifies identity before serving (manifest digest; segment
  digests for eagerly-read segments; sketch resident hash), rejects
  unsupported versions/encoders, executes section 4, and passes the golden
  fixtures.
- All artifact bytes are untrusted input: every offset/length/count is
  validated before allocation per the contract's section 7 rules, and reads
  respect the layered budgets the readers already enforce.

## 6. Integrity stance

Format 2 commits to every corpus record individually (section 3.5): a
hydrated record is verified on its own range read through the page table
the manifest commits to, which is contract section 4.1's range-verifiable
chunk commitment sized to the corpus's real read shape (one record). Eager
segments (manifest, query-interp, evaluation, corpus tables) verify whole,
and the index segment inherits the sketch artifact's resident hash.

What remains transitional: the sketch artifact's lazy rerank rows are
still covered only by its whole-segment `vectorsSha256` (verifiable after
the fact via `verifyVectors()`, not per read). Per-row commitments belong
to SKETCH_PROFILE.md's next revision; this profile will inherit them
unchanged.

Format-1 files keep Draft 1's stance — whole-segment digests only, lazy
record reads not independently verifiable — and readers MUST report which
stance a file carries (`info().corpusIntegrity`) rather than presenting
both as equivalent.

## 7. Relationship to existing tooling

- **Compiler:** `create-pancake-search`'s pipeline (ingest → chunk → embed
  → index → distill → calibrate) becomes the frontend; a `compile` step
  assembles its outputs into one `.pancake`. Requires bytes-in/bytes-out
  builder variants (spike lesson) and compile-time renumbering.
- **Hosts:** a static page with the browser reader, a Worker, and Node all
  open the same file; the Worker example becomes one host among three.
- **Existing profiles:** `.pnck`, `.pancake-range`, and `.pancake-sketch`
  remain valid standalone profiles; this container embeds the sketch
  profile and does not deprecate anything.

## 8. Open questions for Draft 2

1. Corpus compression (per-record or segment-level) — record-granular
   range reads argue for per-record; measure before deciding.
2. Whether `sampleQueries` belongs in the manifest or the evaluation
   segment (identity implications of moving it).
3. Signatures over the identity (contract section 10, question 7).
4. A `filters` or metadata-index segment — out of contract today
   (section 8), revisit only with a concrete host need.
5. Browser reader packaging (the encoder runs in plain JS today; confirm
   no Node-only dependencies before freezing the host story).
