# Pancake Complete Search Artifact Profile

**Status:** Draft 1 — for review, not frozen
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
| 4 | u32 | formatVersion | `1` |
| 8 | u32 | manifestBytes | length of the canonical manifest JSON |
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
  "profile": "pancake-complete-v1",
  "corpus": { "records": 208, "provenance": null },   // provenance reserved per contract 4.3
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
silently.

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

1. Read header (64 bytes); check magic, version `=== 1`, sane counts.
2. Read manifest + segment table; verify `manifestSha256`; parse; verify
   table/manifest agreement (kinds, offsets within file, byte lengths).
3. Read the query-interpretation segment (one read, ~1.2 MB today); load
   encoder + calibration.
4. Open the index segment with the sketch reader (staged or full); read the
   corpus offsets array.
5. The artifact is now serving. Total cold-open transfer at the reference
   corpus: header + manifest + query-interp + sketch stage-1 — under 2 MB.

`query(text, k)`:

1. Encode: text → vector + feature stream (pre-search abstention MAY answer
   here without touching the index).
2. Search: sketch scan + one parallel rerank fetch round (SKETCH_PROFILE.md
   section 3).
3. Hydrate: one range read per result id via the corpus offsets.
4. Calibrate: score match quality from hits + feature stream; a `none`
   verdict returns zero results with the score.

Result shape: `{ matchQuality, confidence, results: [{ id, distance,
title, text, sourcePath, ... }] }` — ids, distances, and hydrated records
in one response; returning bare ids does not satisfy this profile.

## 5. Conformance

- **Fixtures:** a committed golden `.pancake` compiled from the
  `examples/03-edge-docs-search` assets, with the reference reader's exact
  results for the evaluation segment's golden queries (ids, distances at
  declared tolerance, match-quality labels). The 10 abstention goldens MUST
  reproduce their labels — already demonstrated component-wise by the
  composition spike.
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

## 6. Integrity stance (transitional)

Draft 1 commits to whole-segment digests via the manifest, plus the sketch
artifact's internal hashes. Individual lazy range reads (corpus records,
sketch rows) are NOT independently verifiable — the same transitional
stance as every current profile. Contract section 4.1's range-verifiable
chunk commitments are the planned v2 upgrade: a per-segment chunk-digest
list in the manifest, sized to the real read shapes (corpus records and
sketch rows), at which point this profile satisfies section 4.1 in full.
Draft 1 states this honestly rather than claiming completeness.

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
