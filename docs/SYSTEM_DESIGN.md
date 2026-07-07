# Pancake — System Design

**Scope:** The full Pancake vector-search system as built today — the C++ HNSW
backends, the WASM C ABI, the JavaScript wrapper, the native benchmarking addon,
serialization, and the Cloudflare Worker reference deployments.
**Last updated:** 2026-06-26
**Status:** Reflects the current source tree (`src/`, `pancake-core.js`,
`native/`, `examples/worker*`). This document was written from a ground-up
re-read of the code.

> Verification note: every mechanism below was read out of the source. Where a
> behavior is subtle (default divergence between layers, deletion state on
> import, ID remap on compaction), the exact code path is named so it can be
> re-checked.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Layered Architecture](#2-layered-architecture)
3. [The Two HNSW Backends](#3-the-two-hnsw-backends)
4. [Quantization and Asymmetric Distance](#4-quantization-and-asymmetric-distance)
5. [SIMD and Distance Kernels](#5-simd-and-distance-kernels)
6. [The C ABI and Handle Table](#6-the-c-abi-and-handle-table)
7. [The JavaScript Wrapper](#7-the-javascript-wrapper)
8. [ID Translation and Compaction](#8-id-translation-and-compaction)
9. [Serialization Formats](#9-serialization-formats)
10. [Build Pipeline (WASM and Native)](#10-build-pipeline-wasm-and-native)
11. [Cloudflare Worker Deployment](#11-cloudflare-worker-deployment)
12. [Configuration Reference](#12-configuration-reference)
13. [Invariants](#13-invariants)
14. [Limitations and Non-Goals](#14-limitations-and-non-goals)
15. [Appendix A: WASM Export Inventory](#appendix-a-wasm-export-inventory)
16. [Appendix B: Serialization Byte Layouts](#appendix-b-serialization-byte-layouts)

---

## 1. Overview

Pancake is an HNSW (Hierarchical Navigable Small World) approximate
nearest-neighbor index compiled from C++ to WebAssembly. The primary artifact is
a single portable WASM module (`dist/engine.wasm`, ~137 KB raw / ~49 KB gzipped,
plus a ~17 KB `engine.js` loader / ~5 KB gzipped) that runs unchanged in Node.js,
browsers, and Cloudflare Workers — no native dependency on the default path.

The engine provides two interchangeable HNSW backends behind one API:

- **`FloatHNSW`** — full-precision float32 storage and distances.
- **`Int8FloatHNSW`** — row-wise int8 quantized storage with *asymmetric* search
  (float32 query against int8 database), roughly 3.5× smaller than float32 at
  1536D with a modest recall-ceiling cost.

A separate native Node N-API addon (`native/`) compiles the *same* two C++
backend classes with AVX2/SSE2 SIMD. It is a benchmarking tool — it is not part
of the shipped package — and exists to separate runtime overhead (WASM vs native)
from graph quality (which is identical because the backend code is shared).

Pancake is an index, not a database and not an embedding model. It stores
vectors and graph edges only; metadata, persistence policy, and embedding are
the caller's responsibility.

---

## 2. Layered Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Application                                                       │
│   Node.js app   │   Cloudflare Worker   │   Browser (ESM import)   │
├─────────┬───────┴───────────┬───────────┴──────────┬──────────────┤
│  JS Wrapper Layer           │                      │              │
│   pancake.js (CJS) / pancake.node.mjs (ESM) / pancake.web.mjs (web)│
│            └──────────── pancake-core.js ──────────┘              │
│            PancakeIndex: marshalling, ID translation,             │
│            export envelope, buffer management                     │
├──────────────────────────────┬────────────────────────────────────┤
│  C ABI (WASM exports)         │                                    │
│   src/engine.cpp — handle table, IndexWrapper dispatch            │
│   _pancake_init / _add / _query / _query_filtered / _export / ... │
├──────────────────────────────┬────────────────────────────────────┤
│  Backend Layer (C++ templates-free, runtime dimension)            │
│   ┌──────────────────┐   ┌────────────────────────────────────┐   │
│   │ FloatHNSW        │   │ Int8FloatHNSW                      │   │
│   │ float_hnsw.hpp   │   │ int8_float_hnsw.hpp               │   │
│   └──────────────────┘   └────────────────────────────────────┘   │
│   Distance kernels: WASM SIMD128 / AVX2 / SSE2 / scalar           │
└──────────────────────────────────────────────────────────────────┘
```

The native addon (`native/pancake_napi.cpp`) replaces the *C ABI* layer with an
N-API binding but reuses the identical backend layer — it `#include`s
`float_hnsw.hpp` and `int8_float_hnsw.hpp` directly.

The same `IndexWrapper` abstraction (`src/engine.cpp:26`) is used by both the
WASM C ABI and the native addon (the native addon defines an equivalent wrapper
pair in `pancake_napi.cpp`). Backend choice is made once, at construction, from
the `quantized` flag, and dispatched through virtual calls thereafter — there is
no per-call branching on a backend type.

---

## 3. The Two HNSW Backends

Both backends are runtime-dimension (no compile-time `DIMS` template
specialization) and implement the same HNSW algorithm with backend-specific
storage and distance kernels. The graph algorithm code (level assignment, greedy
descent, layer beam search, neighbor selection, compaction) is **largely
duplicated** between `float_hnsw.hpp` (~1080 lines) and `int8_float_hnsw.hpp`
(~1740 lines) rather than shared through a common base — a deliberate trade of
DRY-ness for keeping each backend's hot path self-contained and independently
tunable.

### 3.1 Graph parameters

| Parameter | Meaning | Notes |
|-----------|---------|-------|
| `M` | Max neighbors per node on upper layers | |
| `M0` | Max neighbors at layer 0 | Hard-coded `M0 = 2 * M`; not separately configurable. Layer 0 is denser because all searches terminate there. |
| `ef_construction` | Beam width during insert | |
| `ef_search` | Beam width during query | Mutable at runtime via `set_ef_search`. |

### 3.2 Level assignment

Identical in both backends: an exponential distribution
`level = floor(-ln(U) * level_mult)` where `level_mult = 1 / ln(M)` and `U` is
uniform (0,1) from a seeded `std::mt19937`. (`float_hnsw.hpp:115`,
`int8_float_hnsw.hpp:244`.)

### 3.3 Insert

1. Capacity check — returns `UINT32_MAX` if `count == max_elements` (no
   resize). (`float_hnsw.hpp:98`, `int8_float_hnsw.hpp:191`.)
2. Assign sequential internal id `count_++`.
3. Store the vector (float32 stored raw, or normalized-then-quantized for the
   int8 cosine path — see §4).
4. Greedy descent from the entry point through the upper layers to the new
   node's top level (exhaustive per-layer neighbor scan, no beam).
5. For each layer from the node's level down to 0: a beam search of width
   `ef_construction` collects candidates; the neighbor-selection heuristic picks
   up to `M` (or `M0` at layer 0); reciprocal edges are added to neighbors with
   pruning.
6. Update the entry point if the new node's level exceeds the current max.

### 3.4 Neighbor-selection heuristic

Both backends default to `use_heuristic = true` and implement the HNSW
diversity heuristic with **backfill** (the paper's "keep pruned connections"):
a candidate is kept only if it is not closer to an already-selected neighbor than
to the inserting node; if fewer than `M` survive, the closest rejected
candidates are added back until `M` slots are filled. (`float_hnsw.hpp:954`,
`int8_float_hnsw.hpp:1620`.) The float backend caches pairwise candidate
distances during selection to avoid recomputing them; the int8 backend leans on
its closed-form symmetric distance (§4.3) instead.

**Per-node slot capacity is hard.** Layer-0 neighbor slots are pre-allocated at
exactly `M0` entries per node with no headroom between adjacent slots, so adding
a reciprocal edge to a node already holding `M0` neighbors must not append past
`M0`. When a node is saturated, `append_neighbor` runs the diversity heuristic
over the existing neighbors plus the new candidate and re-selects within `M0`
rather than writing a transient `M0+1`th entry (which would corrupt the next
node's slot, or overrun the buffer on the last node). The int8 backend's
`append_edge_with_prune` follows the same rule. (`float_hnsw.hpp:773`,
`int8_float_hnsw.hpp:1028`.)

### 3.5 Delete and compaction

- **Soft delete** sets a per-node flag in a `std::vector<uint8_t> deleted_`
  (one byte per node, not a packed bitset — a deliberate trade of memory for a
  branch-free check). The node stays in the graph and is skipped during
  traversal. (`float_hnsw.hpp:346`, `int8_float_hnsw.hpp:483`.)
- **Compaction** is a rebuild-with-remap: it builds an `old_id → new_id` map over
  survivors, physically moves vector/graph data into the compacted positions,
  remaps every neighbor id, and runs a **backfill pass** that re-expands
  under-connected nodes via their neighbors-of-neighbors and re-applies the
  selection heuristic. Deletion state is reset. The remap is returned to the
  caller via an `out_map` out-parameter. (`float_hnsw.hpp:360`,
  `int8_float_hnsw.hpp:497`.)

The C ABI exposes both a void `compact()` and a `compact_remap()` that surfaces
the `out_map`; in practice the current JS layer calls plain `compact()` and
rebuilds its own id map (§8).

### 3.6 Filtered search

Both backends implement `search_filtered(query, k, filter_bitset, bitset_len)`.
The filter is a bitset over **internal** ids: bit `(id & 7)` of byte `(id >> 3)`.
Filtering happens *inside* the layer-0 traversal, not as a post-filter:
non-matching nodes still navigate the graph (they remain in the candidate queue),
but only matching nodes enter the result heap. The search starts with
`ef = max(ef_search, k*2)` and **dynamically widens** ef (up to ~4× initial) when
too few filtered results have been found, so that restrictive filters still
return `k` results where the graph allows. (`float_hnsw.hpp:216`,
`int8_float_hnsw.hpp:365`.) Effectiveness degrades below ~1% selectivity, where
the graph may lack navigable paths to the target set.

### 3.7 The visited-set generation trick

Rather than clearing a visited bitmap before each search, both backends keep a
`std::vector<uint32_t> visited_list_` and a monotonically increasing
`visited_curr_` counter; a node is "visited this search" iff
`visited_list_[id] == visited_curr_`. Each search just increments the counter;
the array is only zeroed on the rare `uint32_t` wraparound. This avoids an
`O(max_elements)` clear per query. (`float_hnsw.hpp:95`,
`int8_float_hnsw.hpp:187`.)

---

## 4. Quantization and Asymmetric Distance

### 4.1 Row-wise affine quantization (int8 backend)

Each vector is quantized independently to **uint8** using its own min/max
(`int8_float_hnsw.hpp:213`):

```
vmin = min(v),  vmax = max(v)
range = vmax - vmin
if (range < 1e-30) range = 1.0      // degenerate-vector guard (constant vector)
scale = range / 255
q[d]  = clamp(round((v[d] - vmin) / scale), 0, 255)   // round via +0.5
// stored per vector: scale (f32), offset = vmin (f32), q[0..D-1] (uint8)
```

Dequantization is `v[d] ≈ offset + scale * q[d]` (`int8_float_hnsw.hpp:1122`).

**Per-vector storage:** `D` bytes (quantized data) + 4 (scale) + 4 (offset) +
4 (`sum_q`) + 4 (`sum_q2`) = **D + 16 bytes**, versus `4D` for float32. The two
extra `uint32` sums are precomputed statistics used by the symmetric distance
(§4.3). (Fields: `int8_float_hnsw.hpp:1719–1722`.)

### 4.2 Asymmetric search distance

At query time the **query stays float32**; the database vector is dequantized on
the fly during the distance computation. This avoids quantizing the query (which
would compound error and require knowing the query's scale before the distance is
computed) and preserves query-side precision. For cosine, both the stored vectors
(at insert) and the query (at search) are L2-normalized first; distance is
`1 − clamp(dot, −1, 1)`. (`int8_float_hnsw.hpp:199`, `:1184`.)

### 4.3 Closed-form symmetric distance

Graph maintenance (neighbor selection, edge-distance recomputation) needs
node-to-node distances between two *stored* int8 vectors. Instead of
dequantizing both, the int8 backend computes the distance algebraically from the
quantized bytes and the cached `sum_q` / `sum_q2` statistics plus an integer
`int8_dot` (`int8_float_hnsw.hpp:1433`):

```
L2(a,b) = D·(oa−ob)² + 2(oa−ob)(sa·sum_q[a] − sb·sum_q[b])
        + sa²·sum_q2[a] + sb²·sum_q2[b] − 2·sa·sb·dot(a,b)
```

This makes graph construction distance evaluations cheap and is the main reason
the int8 build path stays competitive with float32.

The float backend has no analog — it computes node-to-node distance directly on
the stored float vectors.

---

## 5. SIMD and Distance Kernels

Both backends select a kernel at compile time:

```cpp
#if defined(__wasm_simd128__)            // WASM SIMD128 (the shipped path)
#elif defined(PANCAKE_ENABLE_AVX2_SIMD)  // native, 256-bit
#elif defined(PANCAKE_ENABLE_SSE2_SIMD)  // native, 128-bit
#else                                    // scalar fallback
#endif
```

Priority is WASM SIMD128 → AVX2 → SSE2 → scalar. Each kernel processes the
dimension in SIMD-width chunks with a scalar tail for the remainder.

- **Float L2:** load 4/8 floats, subtract, square, accumulate
  (`float_hnsw.hpp:803`). **Float cosine:** dot-product accumulate, then
  `1 − clamp(dot)` (`float_hnsw.hpp:841`).
- **Int8 asymmetric:** load 16 uint8, widen u8→u16→f32, dequantize in-register
  (`offset + scale·q`) via FMA, subtract the float query, square/dot, accumulate.
  Multiple independent accumulators (`acc0..acc3`) hide FMA latency.
  (`int8_float_hnsw.hpp:1184`, `:1305`.)
- **Relaxed SIMD:** an opt-in WASM build path uses `wasm_f32x4_relaxed_madd`
  (fused multiply-add) where available, falling back to separate mul+add
  otherwise (`int8_float_hnsw.hpp:57`). Relaxed-SIMD reductions are
  non-deterministic across runtimes; this is why it is opt-in (see §10).

The native AVX2 build is the fastest distance path; the shipped WASM SIMD128
build is the portable one. Recall is identical across all of them because they
compute the same distances — only throughput differs.

---

## 6. The C ABI and Handle Table

`src/engine.cpp` (433 lines) is the boundary between WASM and the C++ backends.

### 6.1 Handle table

```cpp
constexpr uint32_t MAX_HANDLES = 64;
constexpr uint32_t INVALID_HANDLE = 0xFFFFFFFF;
struct HandleSlot { IndexWrapper* index = nullptr; size_t dim = 0; };
static HandleSlot g_handles[MAX_HANDLES];
static std::vector<uint8_t> g_export_bufs[MAX_HANDLES];
```

A fixed 64-slot static array. `alloc_handle()` linear-scans for the first free
slot (or returns `INVALID_HANDLE`); `free_handle()` deletes the wrapper and
clears the slot. Multiple independent indexes can coexist in one WASM instance —
e.g., one per tenant — up to 64. State is plain mutable globals, which is safe
under WASM's single-threaded execution model. (`engine.cpp:143–168`.)

### 6.2 Backend dispatch

`IndexWrapper` (`engine.cpp:38`) is an abstract base with virtual `insert`,
`bulk_insert`, `search`, `search_filtered`, `mark_delete`, `compact` (both void
and `out_map` forms), `count`, `ghost_count`, `ghost_ratio`, `memory_bytes`,
`serialize`, `deserialize`, `set_ef_search`, `dimension`. `pancake_init` picks
the concrete wrapper from the `quantized` flag and the metric (`metric == 1` →
cosine, else L2):

```cpp
if (quantized) g_handles[h].index = new Int8FloatHNSWWrapper(dim, i8cfg);
else           g_handles[h].index = new FloatHNSWWrapper(dim, cfg);
```

(`engine.cpp:181`.)

> **Defaults.** The library defaults are `M=16`, `efConstruction=50`,
> `efSearch=100`, and the JavaScript wrapper passes them explicitly. The C ABI
> only falls back to its internal defaults when a direct caller passes
> non-positive values to `pancake_init`; normal JS callers use the canonical
> library defaults.

### 6.3 Result marshalling

`pancake_query` / `pancake_query_filtered` take caller-allocated `uint64_t* ids`
and `float* dists` buffers, run the search, and copy results in — widening the
backend's internal `uint32_t` ids to `uint64_t` on the way out (the BigInt-wide
ABI is why `WASM_BIGINT=1` is set at build time). They return the result count.
(`engine.cpp:223`, `:233`.)

### 6.4 Export buffer ownership

`pancake_export` serializes into the per-handle static `g_export_bufs[h]` and
returns a pointer + size; the pointer is valid only until the next export on that
handle or `pancake_dispose`. The caller must copy promptly. `pancake_import`
returns `0` / `-1`; on failure the existing index is left intact. The wrapper's
`deserialize` builds a fresh backend into a `unique_ptr` and only swaps it in on
success, and `pancake_import` wraps the call in a `try/catch` so a hostile
snapshot that still slips an oversized allocation through the bounds checks
(below) returns `-1` instead of aborting the WASM instance. (`engine.cpp:297`,
`:304`.)

### 6.5 Utilities and lifecycle

Beyond the index API, the ABI exports `emsc_malloc`/`emsc_free` (heap for
marshalling), `dense_matmul`/`sparse_matmul`/`normalize` (SIMD helpers),
`pancake_profile_print`/`pancake_profile_reset` (no-ops unless built with
`PANCAKE_INT8_HNSW_BUILD_PROFILE`), and `pancake_shutdown_all` (frees all
handles). The WASM ABI no longer exposes the earlier experimental `emb_*`
embedding-model path; the public surface is the index API plus the small set of
numeric/test helpers above.

The complete export list is in [Appendix A](#appendix-a-wasm-export-inventory).

---

## 7. The JavaScript Wrapper

### 7.1 Entry points and engine loading

| Entry | File | WASM resolution |
|-------|------|-----------------|
| CJS (Node) | `pancake.js` | `fs.readFileSync('dist/engine.wasm')`, passed as `wasmBinary` |
| ESM (Node) | `pancake.node.mjs` | same, via `node:fs` + `fileURLToPath` |
| Web / Workers | `pancake.web.mjs` | `locateFile()` returns `new URL('./dist/engine.wasm', import.meta.url)`, engine fetches it |

All three call the Emscripten factory (exported as `P`, `MODULARIZE=1`) and then
construct the same `PancakeIndex` from `pancake-core.js`. `Pancake.create()`
allocates the per-index WASM scratch buffers (query vector, result ids, result
distances), calls `_pancake_init`, and returns the wrapper.
(`pancake-core.js:598`.)

### 7.2 PancakeIndex API

| Method | Marshalling |
|--------|-------------|
| `add(vec)` | validates element type (plain-array elements must be numbers, not coerced) + dim + finiteness, `HEAPF32.set` into the query buffer, `_pancake_add`, assigns an external id |
| `addBatch(vecs)` | one `emsc_malloc` for the whole batch, `_pancake_bulk_insert`, records a contiguous id range |
| `search(q,k)` | `_ensureSearchCapacity(k)`, marshal query, `_pancake_query`, translate ids + (for L2) `sqrt` the squared distance |
| `searchFiltered(q,k,allowedIds)` | builds an internal-id bitset from the allowed external-id `Set`, `_pancake_query_filtered` |
| `delete(id)` | external→internal, `_pancake_delete`, record in `_deletedExt` |
| `compact()` | rebuild id maps from survivors (§8) |
| `export()` | guard `ghostCount===0`, prepend v3 envelope (§9.3) |
| `import(data)` | parse + validate envelope, load WASM state, restore id maps |
| `dispose()` | free handle + scratch buffers, idempotent |

Properties `count`, `ghostCount`, `ghostRatio`, `memory` proxy directly to WASM
exports; `dim` is cached. `_setEfSearch(ef)` calls `_pancake_set_ef`.

### 7.3 Buffer management

Search result buffers (`_idPtr`, `_distPtr`) are reused across queries and grown
on demand by `_ensureSearchCapacity` when `k` exceeds the current capacity
(`pancake-core.js:465`). Result ids are read back as a `uint64` (two `uint32`
halves recombined) from the heap and translated to external ids; for L2 the
stored squared distance is `Math.sqrt`-ed before being returned
(`pancake-core.js:573`). `dispose()` frees all three scratch pointers even if the
handle dispose throws, then sets a disposed flag that every method checks.

---

## 8. ID Translation and Compaction

The WASM backend assigns sequential internal ids (0, 1, 2, …) and **reassigns
them on compaction** (gaps from deletes are closed). To give callers stable ids,
`pancake-core.js` keeps a bidirectional map:

- `_extToInt: Map<extId, intId>`
- `_intToExt: Map<intId, extId>`
- `_deletedExt: Set<extId>`
- `_nextExtId`: monotonic external-id counter, never reused

External ids are assigned at insert and never change. On `compact()`
(`pancake-core.js:210`):

1. Collect surviving `{extId, intId}` pairs (skip `_deletedExt`).
2. Sort by **old** internal id.
3. Call `_pancake_compact` — the WASM layer reassigns internal ids `0..n-1` in
   that same surviving order.
4. Clear and rebuild both maps so external id `e` now points at its new internal
   id (its index in the sorted survivor list); clear `_deletedExt`.

The correctness hinge: the JS layer assumes WASM compaction preserves the
**relative order** of survivors when renumbering them `0..n-1`, which is why it
sorts by old internal id before reassigning. It does **not** consume the
`compact_remap` out-map; it reconstructs the mapping itself. (The Worker keeps an
equivalent map and does the same thing — §11.5.)

---

## 9. Serialization Formats

There are three nested layers. From innermost out: the backend blob, the
JS export envelope, and (Worker only) the WRK1 envelope.

### 9.1 Backend blobs

Both backends emit little-endian blobs with a magic, a version, a fixed header,
the vector data, then the per-node graph (level, base edges, upper-level edges).
Byte layouts are in [Appendix B](#appendix-b-serialization-byte-layouts).

- **FloatHNSW:** magic `0x464C4831` ("FLH1"), current version `1`; also reads a
  legacy magic. Stores raw (or normalized) float vectors + graph.
  (`float_hnsw.hpp:513`.)
- **Int8FloatHNSW:** magic `0x49384831` ("I8H1"), current version `2`; reads a
  legacy magic and older versions. Stores scales, offsets, quantized bytes, then
  graph. Version 2 stores edge distances inline; importing an older version
  recomputes them. (`int8_float_hnsw.hpp:660`.)

> **Deletion state does not survive a round-trip.** Neither backend serializes
> the `deleted_` flags — ghosts are written as ordinary vectors and come back
> *live* on import, and `num_deleted_` resets to 0. The contract is: **compact
> before export** if deletes must persist. The JS `export()` enforces this by
> throwing when `ghostCount > 0`; the Worker's persist path compacts first.
> (`float_hnsw.hpp:513`, `int8_float_hnsw.hpp:660`.)

### 9.2 Statistics reconstruction (int8)

The int8 deserializer recomputes `sum_q` / `sum_q2` from the loaded quantized
bytes rather than storing them, and (for pre-v2 blobs) recomputes all edge
distances. So the on-disk format is slightly smaller than the in-memory
footprint.

### 9.3 JS export envelope (v3)

`PancakeIndex.export()` wraps the backend blob with a 32-byte header **plus an
embedded id-mapping table** so external ids survive an export/import cycle. This
is new in v3 — earlier envelopes (v1/v2, 20-byte header, still accepted on
import) carried no mapping. (`pancake-core.js:4`, `:250`.)

```
Offset  Size           Field
0       4              Magic 0x504E434B ("PNCK")
4       4              Envelope version (3)
8       4              Dimension
12      4              Metric (0=L2, 1=Cosine)
16      4              Quantized (0/1)
20      4              nextExtId
24      4              Mapping entry count
28      4              WASM blob byte length
32      8 × count      Mapping table: [intId u32, extId u32] per entry
...     blobLen        Backend blob (PNCK-less, raw pancake_export bytes)
```

`import()` validates magic, version (1/2/3 accepted, ≥4 rejected), and that the
embedded **dim / metric / quantized** match the target index — mismatches throw
before the WASM import runs, preventing silent memory corruption. For v3 it also
checks that the mapping count equals the post-import vector count and that
`nextExtId ≥ count`, then commits the restored maps. A bare backend blob with no
envelope is imported with identity id mappings.

### 9.4 Untrusted-snapshot hardening (backend deserialize)

The JS envelope checks in §9.3 are the *outer* gate. The backend
`deserialize()` in each `*.hpp` parses an attacker-controlled byte buffer
directly (anything reaching `index.import()` or the raw-blob path), so it is
hardened to fail closed rather than corrupt memory or abort the instance:

- **Bounds checks use subtraction, not addition.** `size_t` is 32-bit under
  Emscripten (wasm32), so an `offset + len > data_size` test could wrap and pass
  while the following `memcpy` ran out of bounds. Every block-size check is
  written as `offset > data_size || data_size - offset < len`, and the float
  backend adds an explicit `count_*dims_` multiply-overflow guard before sizing
  the vector store. (`float_hnsw.hpp:560`, `int8_float_hnsw.hpp:720`.)
- **The HNSW level count is capped** at `MAX_DESERIALIZE_LEVEL` (64). An
  unbounded `max_level` read from a snapshot would otherwise drive a
  multi-gigabyte per-node `upper_[i].resize()`, throwing `length_error` /
  `bad_alloc`. 64 levels covers any realistic element count.
  (`int8_float_hnsw.hpp:156`.)
- **Quantization scales are validated** `> 0` (and finite, and `≤ 1e20`). A
  zero/negative stored scale would collapse a vector to a constant and poison
  every distance to it; `insert()` never produces one, so a snapshot may not
  carry one either. (`int8_float_hnsw.hpp:213` for the insert-side guard the
  importer mirrors.)
- **Exceptions are contained** at the `pancake_import` boundary (§6.4): any
  remaining oversized allocation returns `-1` rather than unwinding out of the
  WASM module.

Per-edge neighbor ids are also validated `< count_`, and vector/scale/offset
components are rejected if non-finite (bit-level exponent check, since
`std::isfinite` is unreliable under `-ffast-math`).

### 9.5 WRK1 (Worker envelope)

The Worker wraps the *JS-envelope* export in its own `WRK1` envelope to persist
Worker-specific metadata (init params, `maxElements`, id mapping) as JSON
alongside the bytes. Detailed in §11.3.

---

## 10. Build Pipeline (WASM and Native)

### 10.1 WASM (`build.sh`)

Single translation unit (`src/engine.cpp`) compiled with Emscripten. Key flags:

- **Optimization:** `-O3` (release) / `-O2 -gsource-map` (debug). Speed, not size
  — the binary is small enough that the speed trade wins.
- **SIMD:** `-msimd128` by default. `WASM_RELAXED_SIMD=1` adds `-mrelaxed-simd`
  (opt-in; the default build stays on plain SIMD so the checked-in artifact is
  broadly compatible and bit-reproducible). Also `-mnontrapping-fptoint`,
  `-msign-ext`, `-mbulk-memory`.
- **Runtime:** `MODULARIZE=1 EXPORT_NAME="P"`, `ENVIRONMENT='web,node'`,
  `ALLOW_MEMORY_GROWTH=1`, `INITIAL_MEMORY=16MB`, `STACK_SIZE=64KB`,
  `MALLOC=emmalloc`, `WASM_BIGINT=1` (64-bit ids across the boundary),
  `FILESYSTEM=0`, `DYNAMIC_EXECUTION=0`, `ASSERTIONS=0`,
  `DISABLE_EXCEPTION_CATCHING=0` (C++ exceptions on).
- **Exports:** the 26-function list in Appendix A; runtime methods
  `ccall, cwrap, HEAPF32, HEAPU8, HEAPU32, HEAP32`.
- **Output:** `dist/engine.js` (~17 KB) + `dist/engine.wasm` (~137 KB;
  ~49 KB gzipped).

A post-build `patch_engine.py` rewrites the Emscripten-generated environment
detection, forcing `ENVIRONMENT_IS_NODE = false` in `engine.js` so the modular
factory loads cleanly across Node, browser, and Workers rather than taking
Emscripten's CommonJS auto-path. (`patch_engine.py`.)

**Scalar fallback.** `npm run build:all` (`scripts/build-all.mjs`) builds the
SIMD engine above and then re-runs the same compile with `WASM_SIMD=0` to emit
`dist/engine.scalar.{js,wasm}`, a non-SIMD engine for runtimes without WASM
SIMD. The JS loaders probe `WebAssembly.validate` for SIMD support and pick the
scalar artifact when it is absent. Both engines compile from the same source, so
`prepublishOnly` runs `build:all` first to keep the shipped SIMD and scalar
artifacts in lockstep with the current `src/`.

### 10.2 Native (`native/binding.gyp`)

node-gyp compiles `pancake_napi.cpp` (which includes the same backend headers
from `../src`) with `-O3 -std=c++17 -ffast-math -march=native -mavx2 -msse2
-fno-rtti` and `-DPANCAKE_ENABLE_AVX2_SIMD -DPANCAKE_ENABLE_SSE2_SIMD`
(with macOS/Windows equivalents). Output: `native/build/Release/pancake_native.node`.

### 10.3 Native vs WASM divergence

| | WASM | Native |
|---|---|---|
| Compiler | Emscripten | system clang/gcc/MSVC |
| SIMD | SIMD128 (opt. relaxed) | AVX2 + SSE2 |
| Boundary | C ABI + Emscripten heap | N-API |
| Query result | caller buffers (`uint64` ids) | JS object `{ ids: Uint32Array, distances: Float32Array, count }` |
| Exported surface | 26 C functions | ~15 N-API functions |
| Backends | identical (`src/*.hpp`) | identical (`src/*.hpp`) |

Because the backend code is shared, recall and graph structure are identical;
the native build exists only to measure the runtime-overhead delta.

---

## 11. Cloudflare Worker Deployment

Two reference Workers live under `examples/`. The first (`examples/worker/`) is a
full read/write reference; the second (`examples/worker-semantic-search/`) is a
hardened snapshot-first demo. The mental model for both is **snapshot search at
the edge**, not a durable mutable database inside one isolate.

### 11.1 Request lifecycle

Each isolate holds two module-global references: the Emscripten engine
(`pancake`, lazily initialized once) and the active `index`. They stay warm
across requests within an isolate. On any non-trivial route, if `index` is null
and a bucket is bound, the Worker lazily **restores from R2** before serving
(`/health`, `/readiness`, `/reset_cache`, `/init`, `/import` skip auto-restore).
(`examples/worker/worker.js`, `restoreIndex()`.)

### 11.2 Endpoints

| Endpoint | Method | Body → Response | Admin? |
|----------|--------|-----------------|--------|
| `/health` | GET | — → status, count, memory, restore timings, read_only | no (public) |
| `/readiness` | GET | — → loaded + snapshot availability (no restore) | no (bearer once `API_KEY` set) |
| `/search` | POST | `{query,k?,ef?}` → `{neighbors,distances,latency_ms}` | no (bearer once `API_KEY` set) |
| `/stats` | GET | — → count, memory, ghost stats, dims | no (bearer once `API_KEY` set) |
| `/init` | POST | `{dims,maxElements,M?,efConstruction?,efSearch?,vectors?}` → init result | yes |
| `/add` | POST | `{vector}` → `{id,count}` | yes |
| `/add_batch` | POST | `{vectors}` → `{inserted,ids,count}` | yes |
| `/delete` | POST | `{id}` → ghost stats | yes |
| `/compact` | POST | — → compaction result (awaits persist) | yes |
| `/export` | GET | — → WRK1 binary | yes |
| `/import` | POST | WRK1 binary (`?dims=` fallback) → import result | yes |
| `/reset_cache` | POST | — → drops warm index, forces cold restore | yes |
| `/search_debug` | POST | `{query,k?}` → raw vs translated ids | yes |

Cross-cutting: `/health` always stays public, but admin routes now fail closed
unless `API_KEY` is set or `ALLOW_INSECURE_ADMIN=1` is explicitly enabled for a
local/demo deployment. Once `API_KEY` is set, every route except `/health`
requires the bearer token — including `/search`, `/stats`, and `/readiness`;
without `API_KEY`, those non-admin routes stay open while the admin routes
return `403`. Also: optional per-IP sliding-window rate limiting
(`RATE_LIMIT_RPM`, 60 s window, per-isolate so the effective global limit is
`limit × isolates`); request-body caps for JSON (`MAX_JSON_BYTES`) and binary
snapshot import (`MAX_SNAPSHOT_BYTES`); and opt-in CORS via `ALLOWED_ORIGIN`
(unset => no `Access-Control-Allow-Origin` header).

### 11.3 R2 persistence

Snapshots are written under **timestamped, append-only keys**
(`pancake-index-<13-digit-ms>-<6-digit-seq>.bin`); restore lists the prefix and
picks the lexicographically greatest key (zero-padding makes string order match
time order), with a fallback to a legacy fixed key. Mutating routes
(`/add`, `/add_batch`, `/delete`) schedule a **fire-and-forget** persist via
`ctx.waitUntil`; `/compact` and `/import` **await** the persist. Before export
the Worker compacts if there are ghosts (so the snapshot has no live ghosts).
(`examples/worker/worker.js`, `SNAPSHOT_KEY_PREFIX` / `restoreIndex()` /
`schedulePersist()`.)

The append-only scheme means a slow/late async write cannot clobber a newer
snapshot under a shared key — restore always loads the newest. The durability
boundary is R2; in-memory isolate state is a warm cache only.

The **WRK1 envelope** (`0x57524B31`, version 1) is a 16-byte header
(magic, version, JSON-metadata length, raw-blob length) followed by JSON metadata
then the raw `pancake-core` export bytes. The metadata carries `dims`,
`maxElements`, `nextExtId`, `initParams {M, efC, efS}`, and the full
`[intId, extId]` mapping — everything needed to reconstruct the index *and* its
external-id mapping on cold restore.
(`examples/worker/worker.js`, `encodeWorkerExportEnvelope()`.)

### 11.4 READ_ONLY mode

`READ_ONLY` (`1/true/yes/on`) makes every admin route return `403`. The check is
a single guard after auth/rate-limit: `if (isReadOnly(env) &&
isAdminRoute(path)) return 403`. `/search`, `/stats`, `/health`, `/readiness`
remain available. This is the recommended posture for a public,
snapshot-backed search endpoint: publish the index out-of-band, deploy read-only,
expose only search. (`examples/worker/worker.js`, `isReadOnly()` / `isAdminRoute()`.)

### 11.5 Worker-side id mapping

The Worker maintains its own `_extToInt` / `_intToExt` / `_deletedExt` /
`_nextExtId` (and an optional `_vectors` stash), independent of `pancake-core.js`,
because it drives the WASM exports directly. Its `compact()` mirrors the core
logic (sort survivors by old internal id, `_pancake_compact`, rebuild maps); no
vector re-insertion is required. On cold restore, `_restoreMapping` rehydrates
the maps from the WRK1 metadata (or seeds identity ids for legacy snapshots).
(`examples/worker/worker.js`, `compact()` / `_restoreMapping()`.)

### 11.6 Semantic-search demo differences

`examples/worker-semantic-search/` is snapshot-first and read-oriented: it builds
the index offline, stores three R2 objects (`docs-index.bin`, `docs-corpus.json`,
`docs-manifest.json`), validates the manifest's `dim` against the demo embedder
before deserializing, and serves `/search` (with optional source-filtered search
via `searchFiltered`), `/health`, `/readiness`, `/reset_cache` plus a minimal UI.
It uses the high-level `pancake-core.js` API rather than raw WASM exports, and a
deterministic hash-based embedder so it runs without API keys. It has no
write endpoints.

---

## 12. Configuration Reference

### 12.1 `Pancake.create(opts)` — JS defaults (the effective ones)

| Option | Default | Notes |
|--------|---------|-------|
| `dim` | required | positive integer |
| `maxElements` | `100000` | capacity, pre-allocated |
| `metric` | `'cosine'` | `'cosine'` or `'l2'` |
| `quantized` | `true` | int8 backend when true |
| `M` | `16` | |
| `efConstruction` | `50` | |
| `efSearch` | `100` | mutable via `_setEfSearch` |

Removed options (`compressed`, `varianceSample`) throw if passed.

### 12.2 C ABI fallback defaults (direct-ABI callers only)

`M=16`, `ef_construction=50`, `ef_search=100`, applied only when a non-positive
value reaches `pancake_init`. Normal JS callers already pass these explicitly.

### 12.3 Worker limits and env

| Constant | Value | | Env var | Default | Purpose |
|----------|-------|---|---------|---------|---------|
| `MAX_RESULTS` | 100 | | `API_KEY` | required for admin routes unless `ALLOW_INSECURE_ADMIN=1` | bearer token |
| `MAX_DIMS` | 4096 | | `ALLOWED_ORIGIN` | unset | opt-in CORS |
| `MAX_EF` | 2000 | | `RATE_LIMIT_RPM` | 0 (off) | per-IP/min |
| `MAX_M` | 128 | | `READ_ONLY` | off | reject admin routes |
| `DEFAULT_MAX_ELEMENTS` | 5000 | | `MAX_ELEMENTS_LIMIT` | 5,000,000 | capacity ceiling |
| `DEFAULT_MAX_JSON_BYTES` | 1 MiB | | `MAX_JSON_BYTES` | 1 MiB | JSON body cap |
| `RATE_LIMIT_WINDOW_MS` | 60000 | | | | |
| | | | `ALLOW_INSECURE_ADMIN` | off | local/demo opt-in for unauthenticated admin routes |

### 12.4 Build constants

`MAX_HANDLES=64`, `INVALID_HANDLE=0xFFFFFFFF` (engine.cpp);
`INITIAL_MEMORY=16MB`, `STACK_SIZE=64KB` (build.sh).

---

## 13. Invariants

- **I1 — Handle validity.** A handle is valid from `pancake_init` until
  `pancake_dispose`; range and null checks guard every ABI call. Max 64 live.
- **I2 — Single-threaded.** All access to a WASM instance must be from one
  thread; enforced by the WASM model, not the code. Each Worker isolate /
  worker_thread needs its own instance.
- **I3 — Insert ids are sequential and unique** within a handle (`count_++`),
  until compaction renumbers survivors (transparent through the JS id map).
- **I4 — Capacity is hard.** Insert at `count == max_elements` returns
  `UINT32_MAX` and mutates nothing.
- **I5 — Delete is immediate and permanent** (no undelete); the node is skipped
  from the next search on.
- **I6 — Compact preserves the live set**, rebuilds the graph (quality may shift),
  and renumbers survivors in old-id order.
- **I7 — Deletion state is not serialized.** Import returns all vectors live;
  compact before export to persist deletes. JS `export()` enforces
  `ghostCount===0`.
- **I8 — Envelope validation on import.** dim/metric/quantized mismatches (and,
  for v3, mapping-count / nextExtId inconsistencies) throw before the WASM import.
- **I9 — Export pointer lifetime.** The `pancake_export` pointer is valid only
  until the next export on that handle or dispose; copy immediately.

---

## 14. Limitations and Non-Goals

- **No metadata storage.** The index stores vectors + graph only. *Filtered*
  search exists (`searchFiltered`), but the caller supplies the allowed-id set
  and owns all id→metadata mapping.
- **No server-side persistence in Node.** In-memory only; persistence is
  `export()`/`import()` that the application drives. (The Worker adds R2.)
- **No concurrency within an instance.** Single-threaded by design.
- **No cross-handle / multi-index query merge.** Each handle is independent.
- **No incremental serialization.** `export` writes the whole index.
- **No automatic compaction.** The host (or Worker `/compact`) triggers it.
- **No dimensionality reduction.** Vectors arrive at their final dimension.
- **Worker isolates are independent.** No cross-isolate read-after-write
  consistency; R2 is the only durability boundary and uses last-write-wins by
  newest key.
- **Backend code is duplicated, not shared.** float and int8 backends repeat the
  graph algorithm; a change to HNSW logic must be made in both.
- **Relaxed SIMD is non-deterministic** and therefore opt-in; the default build
  is the reproducible one.

---

## Appendix A: WASM Export Inventory

26 functions exported by `build.sh` (`-s EXPORTED_FUNCTIONS`):

**Index API:** `_pancake_init`, `_pancake_add`, `_pancake_bulk_insert`,
`_pancake_query`, `_pancake_query_filtered`, `_pancake_delete`,
`_pancake_compact`, `_pancake_compact_remap`, `_pancake_count`,
`_pancake_memory`, `_pancake_ghost_count`, `_pancake_ghost_ratio`,
`_pancake_set_ef`, `_pancake_export`, `_pancake_import`, `_pancake_dispose`,
`_pancake_dimension`, `_pancake_shutdown_all`, `_shutdown_all`.

**Utilities:** `_dense_matmul`, `_sparse_matmul`, `_normalize`, `_emsc_malloc`,
`_emsc_free`, `_pancake_profile_print`, `_pancake_profile_reset`.

**Runtime methods:** `ccall`, `cwrap`, `HEAPF32`, `HEAPU8`, `HEAPU32`, `HEAP32`.

Selected signatures:

| Export | Signature → returns |
|--------|---------------------|
| `_pancake_init` | `(dim, max_elem, quantized, metric, M, ef_c, ef_s)` → handle / `0xFFFFFFFF` |
| `_pancake_add` | `(handle, vec_ptr)` → id / `0xFFFFFFFF` |
| `_pancake_bulk_insert` | `(handle, vecs_ptr, n)` → inserted count |
| `_pancake_query` | `(handle, q_ptr, k, ids_ptr, dists_ptr)` → count |
| `_pancake_query_filtered` | `(handle, q_ptr, k, ids_ptr, dists_ptr, bitset_ptr, bitset_len)` → count |
| `_pancake_compact_remap` | `(handle, out_buf, out_capacity)` → pre-compaction count |
| `_pancake_export` | `(handle, out_size_ptr)` → data_ptr / null |
| `_pancake_import` | `(handle, data_ptr, size)` → `0` / `-1` |
| `_pancake_set_ef` | `(handle, ef)` → void |

The native N-API addon exposes an equivalent (smaller) surface; its
`pancake_query` returns `{ ids: Uint32Array, distances: Float32Array, count }`
rather than writing into caller buffers.

---

## Appendix B: Serialization Byte Layouts

### FloatHNSW blob — magic `0x464C4831` ("FLH1"), version 1

```
0   u32   magic 0x464C4831
4   u32   dims
8   u32   version (1)
12  u32   count
16  u32   entry_point
20  u32   max_level
24  u32   M
28  u32   M0
32  u32   metric (0=L2, 1=Cosine)
36  u32   ef_construction
40  count×dims×f32   vectors (raw or normalized)
... per node: u32 level, then per level: u32 edge_count, edge_count×u32 neighbor ids
```

Deletion state not serialized.

### Int8FloatHNSW blob — magic `0x49384831` ("I8H1"), version 2

```
0   u32   magic 0x49384831
4   u32   dims
8   u32   version (2)
12  u32   count
16  u32   entry_point
20  u32   max_level
24  u32   M
28  u32   M0
32  u32   metric
36  u32   ef_construction
40  count×f32   scales
... count×f32   offsets
... count×dims×u8   quantized data
... per node: u32 level, u32 base_edge_count, base edges (u32 neighbor + f32 dist),
              then per upper level: u32 edge_count, edges (u32 neighbor + f32 dist)
```

`sum_q`/`sum_q2` are recomputed on load, not stored. Pre-v2 blobs omit edge
distances and have them recomputed on import. Deletion state not serialized.

### JS export envelope (v3) — magic `0x504E434B` ("PNCK")

See §9.3. 32-byte header + `[intId,extId]` mapping table + backend blob.

### WRK1 Worker envelope — magic `0x57524B31` ("WRK1"), version 1

```
0   u32   magic 0x57524B31
4   u32   version (1)
8   u32   JSON metadata length
12  u32   raw blob length
16  ...   JSON metadata { dims, maxElements, nextExtId, initParams{M,efC,efS}, mapping }
...  ...  raw pancake-core export bytes (itself a PNCK v3 envelope)
```

---

*End of document.*
