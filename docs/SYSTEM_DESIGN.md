# Pancake WASM Vector Search Engine: System Design Document

**Scope:** The complete Pancake 1.0.0 system — C++ core, WASM compilation, JavaScript wrapper, Cloudflare Worker deployment  
**Last Updated:** 2026-04-24

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architectural Rationale](#2-architectural-rationale)
3. [System Invariants](#3-system-invariants)
4. [Concurrency Model](#4-concurrency-model)
5. [Data Flow](#5-data-flow)
6. [WASM Compilation and Runtime](#6-wasm-compilation-and-runtime)
7. [Serialization and Persistence](#7-serialization-and-persistence)
8. [Cloudflare Worker Deployment](#8-cloudflare-worker-deployment)
9. [Quantization](#9-quantization)
10. [Segmented Indexes](#10-segmented-indexes)
11. [SIMD Architecture](#11-simd-architecture)
12. [Limitations and Non-Goals](#12-limitations-and-non-goals)
13. [Appendix A: Serialization Formats](#appendix-a-serialization-formats)
14. [Appendix B: Configuration Reference](#appendix-b-configuration-reference)
15. [Appendix C: WASM Export Inventory](#appendix-c-wasm-export-inventory)

---

## 1. System Overview

Pancake is a vector similarity search engine compiled from C++ to WebAssembly. It runs in browsers, Node.js, and Cloudflare Workers — any environment with a WASM runtime and 128-bit SIMD support. The core is a set of HNSW (Hierarchical Navigable Small World) graph implementations with int8 quantization for memory efficiency.

### 1.1 Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Application Layer                                           │
│  ┌───────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  Node.js App  │  │  Cloudflare       │  │  Browser     │  │
│  │  (pancake.js) │  │  Worker           │  │  (ESM import)│  │
│  └───────┬───────┘  └────────┬─────────┘  └──────┬───────┘  │
├──────────┼───────────────────┼────────────────────┼──────────┤
│  JS Wrapper Layer                                            │
│  ┌───────┴───────────────────┴────────────────────┴───────┐  │
│  │  pancake-core.js                                        │  │
│  │  PancakeIndex class — ID translation, envelope format   │  │
│  └───────────────────────────┬─────────────────────────────┘  │
├──────────────────────────────┼───────────────────────────────┤
│  C ABI Layer                 │                               │
│  ┌───────────────────────────┴─────────────────────────────┐  │
│  │  engine.cpp — handle-based API                          │  │
│  │  pancake_init / pancake_add / pancake_query / ...       │  │
│  │  IndexWrapper → dispatch to backend                     │  │
│  └─────┬────────────────┬──────────────────┬───────────────┘  │
├────────┼────────────────┼──────────────────┼─────────────────┤
│  Backend Layer                                               │
│  ┌─────┴──────┐  ┌─────┴──────────┐  ┌────┴─────────────┐   │
│  │ FloatHNSW  │  │QuantizedHNSW   │  │ Int8FloatHNSW    │   │
│  │ (runtime   │  │ <384> <1536>   │  │ (runtime dim,    │   │
│  │  dim, f32) │  │ (template dim, │  │  int8 asymmetric)│   │
│  │            │  │  SIMD-opt)     │  │                  │   │
│  └────────────┘  └────────────────┘  └──────────────────┘   │
│                          │                                   │
│  ┌───────────────────────┴─────────────────────────────────┐  │
│  │  quantized_simd.hpp — WASM SIMD / SSE2 / scalar        │  │
│  │  AffineVector<DIMS>, distance kernels                   │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 Backend Selection

When `pancake_init(dim, max_elem, quantized, metric, M, ef_c, ef_s)` is called, the C ABI dispatches to the appropriate backend:

| Condition | Backend | Rationale |
|-----------|---------|-----------|
| `quantized && dim == 384` | `QuantizedHNSW<384>` | Template specialization enables compile-time SIMD loop unrolling for 384-dimensional embeddings (all-MiniLM-L6-v2, etc.) |
| `quantized && dim == 1536` | `QuantizedHNSW<1536>` | Same for OpenAI Ada embeddings |
| `quantized && other dim` | `Int8FloatHNSW` | Runtime-dimension int8 with asymmetric search (float32 query vs int8 database) |
| `!quantized` | `FloatHNSW` | Full-precision float32, runtime dimension |

This dispatch is invisible to the caller. The handle returned by `pancake_init` is an opaque `uint32_t` index into a 64-slot table. All subsequent operations use the handle.

### 1.3 Thread Model

WASM is single-threaded. There is no concurrency within the WASM module. The threading model is entirely in the host environment:

- **Node.js:** Single-threaded. All operations are synchronous within the WASM call.
- **Cloudflare Workers:** Each isolate is single-threaded. Concurrent requests are handled by separate isolates. Each isolate has its own WASM instance and its own index.
- **Browser:** Main thread or Web Worker. WASM calls block the calling thread.

This means there are **no mutexes, no atomics, no lock-free data structures** inside the WASM module. The C++ code uses global mutable state (the handle table) which is safe because WASM execution is single-threaded.

---

## 2. Architectural Rationale

### 2.1 Why WASM and Not Native

The target deployment is Cloudflare Workers, which runs V8 isolates — not native processes. The constraints are:

1. **No native code execution.** Workers run JavaScript and WASM only. A native C++ library would require a separate server.
2. **128 MB memory limit.** Workers have a hard memory cap. Int8 quantization reduces per-vector memory by ~4×, making 100K+ vectors feasible within the limit.
3. **50ms CPU time limit per request.** HNSW search must complete within this budget. WASM with SIMD achieves sub-millisecond search latency for typical workloads.
4. **Cold start sensitivity.** Workers may be evicted and restarted. WASM modules compile faster than native binaries load, and the binary is ~100KB gzipped.

WASM also provides portability: the same binary runs in Node.js (for testing and batch operations), browsers (for client-side search), and Workers (for edge deployment).

### 2.2 Why Three HNSW Implementations

A single generic HNSW with runtime dimension and runtime quantization would be simpler, but the performance cost is unacceptable:

1. **Template specialization (`QuantizedHNSW<DIMS>`) enables compile-time loop unrolling.** The SIMD distance kernel processes 16 uint8 elements per iteration. When DIMS is known at compile time, the compiler can unroll the loop exactly `DIMS/16` times with no loop overhead. For DIMS=384, this is 24 iterations — a 2-3× speedup over a runtime loop with bounds checking.

2. **Asymmetric distance (`Int8FloatHNSW`) avoids double quantization.** During search, the query stays as float32 while the database is int8. This avoids quantizing the query (which would add quantization error to the query, compounding with database quantization error). The asymmetric kernel dequantizes database vectors on the fly during distance computation.

3. **Float32 (`FloatHNSW`) exists for workloads where quantization is unacceptable.** Some applications (e.g., exact nearest neighbor verification, high-dimensional spaces where int8 doesn't preserve distances) need full precision. FloatHNSW avoids the overhead of quantization/dequantization.

The cost is code duplication: all three backends implement the same HNSW graph algorithm (insert, search, compact, serialize, deserialize) with different storage and distance functions. This is a maintenance burden but an acceptable trade for a 2-3× performance difference on the hot path.

### 2.3 Why Handle-Based C ABI

The original engine.cpp had 13 global singletons and ~100 C exports (`_i8_init`, `_i8_add`, `_float_init`, `_float_add`, `_p128_init`, `_p128_add`, ...). This was replaced with a handle-based API (`_pancake_init` returns a handle, all operations take a handle) for three reasons:

1. **Extensibility.** Adding a new backend (e.g., product quantization, binary embeddings) required adding a new set of C exports and corresponding JS branching. With handles, adding a backend means adding a new `IndexWrapper` subclass — zero changes to the C ABI or JS wrapper.

2. **Multi-index support.** The old API supported exactly one index per type. With handles, the host can create multiple independent indexes (e.g., one per tenant in a multi-tenant Worker).

3. **JS wrapper simplification.** The old `pancake-core.js` had `if (this._useInt8)` branching on every method call. With handles, every method is a uniform `_pancake_*(handle, ...)` call.

The handle table is a fixed 64-slot array with linear scan for allocation. This is simple, O(1) lookup by index, and sufficient for any realistic use case (no one creates 64 simultaneous HNSW indexes in a single WASM instance).

### 2.4 Why Soft Delete and Rebuild Compaction

HNSW graphs do not support structural node removal. Removing a node would leave dangling edges — neighbors that reference a freed slot. The options are:

1. **Ignore the problem.** Let deleted nodes stay in the graph forever. This wastes memory and slows search (deleted nodes are visited but filtered from results).
2. **Soft-delete + lazy compaction.** Mark nodes as deleted (set a flag). During search, skip deleted nodes. When the ghost ratio exceeds a threshold, rebuild the entire graph from live vectors only.
3. **Edge patching.** Reconnect deleted node's neighbors to each other. This is complex, degrades graph quality (heuristic reconnection is inferior to construction-time edge selection), and cannot be done concurrently with reads.

Pancake uses option 2. The trade-offs:

- **Delete is O(1).** Set a flag and return. No graph modification.
- **Compaction is O(N log N).** Build a fresh HNSW from N live vectors. This is expensive but infrequent (triggered when ghost ratio exceeds 10-20%).
- **Peak memory during compaction is 2×.** The old graph and new graph coexist briefly. Acceptable because compaction is infrequent and segments are bounded.

### 2.5 Why the JavaScript ID Translation Layer

WASM HNSW assigns internal IDs as sequential slot indices: 0, 1, 2, .... When compaction rebuilds the graph, it produces a new graph where the same vectors have new sequential IDs (the gaps from deleted vectors are closed). If the host sees these ID changes, it breaks its own ID-to-data mappings.

`pancake-core.js` maintains a bidirectional map between external IDs (stable, monotonically increasing, never reused) and internal IDs (sequential, reassigned on compact). The host only sees external IDs. After compaction, the JS layer rebuilds the mapping by sorting survivors and assigning new internal IDs.

This translation adds ~5μs per operation (one hashmap lookup). It is simpler and more portable than maintaining stable IDs inside the WASM module, which would require resizable hash tables in C++ and complicate the serialization format.

### 2.6 Why the Export Envelope Format

The raw WASM export (`pancake_export`) produces a backend-specific binary blob (QuantizedHNSW serialization, FloatHNSW serialization, etc.). The JS wrapper prepends a 20-byte envelope:

```
[0-3]   Magic: 0x504E434B ("PNCK")
[4-7]   Version: 2
[8-11]  Dimension
[12-15] Metric: 0=L2, 1=Cosine
[16-19] Quantized: 0=float, 1=int8
[20+]   WASM binary blob
```

This prevents three classes of errors:

1. **Dimension mismatch.** Loading a 128D index into a 384D instance would corrupt memory. The envelope check catches this before the WASM import is called.
2. **Metric mismatch.** An L2 index loaded as cosine would produce nonsensical distances. The envelope check catches this.
3. **Quantization mismatch.** A float32 blob loaded into an int8 instance (or vice versa) would be silently interpreted as garbage. The envelope check catches this.

The 20-byte overhead is negligible relative to index sizes (typically 1MB+).

---

## 3. System Invariants

### Handle and Lifecycle

**I1 — Handle validity.** A handle returned by `pancake_init` is valid until `pancake_dispose` is called with that handle. Using a disposed handle is undefined behavior. Using a handle from one WASM instance in another is undefined behavior.

*Enforced by:* Handle range check (`h < MAX_HANDLES`) and null-pointer check (`g_handles[h].index != nullptr`) at the top of every C ABI function.

**I2 — Single-threaded access.** All operations on a given WASM instance (including all handles created by that instance) must be called from the same thread. Concurrent calls from multiple threads to the same WASM instance are undefined behavior.

*Enforced by:* The WASM execution model (single-threaded). Not enforced by the code itself — the caller is responsible.

**I3 — Dispose-before-reuse.** A handle slot is not reused until `pancake_dispose` is called. The 64-slot table uses linear scan for allocation; a disposed slot is marked free (index pointer set to null) and may be reused by a subsequent `pancake_init`.

*Enforced by:* `free_handle` sets `g_handles[h].index = nullptr`. `alloc_handle` scans for null slots.

### Data Integrity

**I4 — Insert returns unique sequential IDs.** Each call to `pancake_add` returns a unique uint32_t within the handle's scope. IDs are assigned sequentially starting from 0. No ID is ever returned twice (until compaction reassigns IDs, which is transparent through the JS layer).

*Enforced by:* `count_++` in the HNSW insert path. The returned ID is `count_ - 1`.

**I5 — Delete is instant and permanent.** After `pancake_delete(h, id)` returns, the vector at `id` will not appear in subsequent search results. The deletion is permanent — there is no undelete.

*Enforced by:* The HNSW `mark_delete` sets a flag checked during search. The flag is set before the function returns. Since WASM is single-threaded, there is no window where a concurrent search could see the old state.

**I6 — Compact preserves live data.** After `pancake_compact(h)`, the index contains exactly the vectors that were alive (not deleted) before compaction. The graph structure is rebuilt from scratch, so graph quality may differ from the pre-compaction graph, but the set of live vectors is identical.

*Enforced by:* The compact implementation collects all live vectors, then rebuilds the HNSW graph. Vectors are identified by iterating the storage array and skipping entries with deletion flags set.

**I7 — Export-import round-trip.** `pancake_import(h, data, size)` followed by `pancake_count(h)` returns the same count as the index that produced the export. Search results on the imported index are identical to search results on the original index (assuming no mutations between export and import).

*Enforced by:* The serialization format includes all graph state (vectors, edges, levels, deletion flags). Deserialization reconstructs the identical data structure. The CRC32 or magic check validates data integrity.

### Capacity

**I8 — Max elements is enforced.** If `pancake_add` is called when `count == max_elements`, it returns `0xFFFFFFFF` (failure). No vector is inserted and no state is modified.

*Enforced by:* Capacity check at the top of the HNSW `insert` method.

**I9 — Handle table capacity.** At most 64 handles can be active simultaneously. If `pancake_init` is called when all 64 slots are occupied, it returns `0xFFFFFFFF`.

*Enforced by:* `alloc_handle` returns `INVALID_HANDLE` when no free slot exists.

### Envelope

**I10 — Envelope validation on import.** If the import data has a valid envelope (magic == 0x504E434B), the dimension, metric, and quantization mode are validated against the target index. A mismatch throws an error (JS layer) or returns failure (C ABI layer). If the data does not have an envelope (legacy format), it is passed directly to the WASM import without validation.

*Enforced by:* `PancakeIndex.import()` in pancake-core.js checks the first 4 bytes for the magic number and validates fields if present.

---

## 4. Concurrency Model

### 4.1 WASM: No Concurrency

The WASM module is single-threaded. There are no mutexes, no atomics, no concurrent data structures inside the compiled C++. The handle table (`g_handles`) and all per-handle index state are plain mutable globals.

This is safe because:
- WASM execution is sequential within a single instance.
- Each Cloudflare Worker isolate has its own WASM instance.
- Node.js is single-threaded (unless using `worker_threads`, in which case each worker must create its own WASM instance).

### 4.2 Worker Isolate: Request-Level Concurrency

Cloudflare Workers handle concurrent requests by spawning multiple isolates. Each isolate:
- Has its own WASM instance (its own heap, its own handle table).
- Has its own in-memory index (indexes are not shared across isolates).
- Has its own rate-limiting state (per-isolate, not globally accurate).
- Can read and write R2 independently (last-write-wins on conflicts).

This means:
- **No cross-request data races.** Each request runs to completion in a single isolate before the next request starts.
- **No shared index state.** Two concurrent requests to different isolates see different indexes. If one isolate inserts a vector, the other isolate does not see it until it reloads from R2.
- **No durable writes between isolates.** R2 writes are debounced (2-second timer). An isolate may be evicted before its timer fires, losing the write. There is no distributed consensus.

### 4.3 Worker Concurrency Implications

| Scenario | Behavior | Impact |
|----------|----------|--------|
| Two requests insert different vectors | Each isolate inserts independently. Both succeed. | R2 persistence may overwrite one isolate's state with the other's. |
| One request inserts, another searches | If same isolate: search sees insert. If different isolates: search does not see insert. | Eventual consistency via R2 restore. |
| Cold start during traffic | New isolate loads from R2. Existing isolate has newer state. | Staleness until R2 is refreshed. |
| Isolate eviction | In-memory index is lost. Pending R2 write may be lost. | Data loss up to 2 seconds of writes. |

### 4.4 ID Translation Concurrency

The JavaScript ID translation layer in `pancake-core.js` uses three `Map` objects:
- `_intToExt`: internal WASM ID → external user ID
- `_extToInt`: external user ID → internal WASM ID
- `_deletedExt`: set of deleted external IDs

These are plain JavaScript Maps, not thread-safe. This is safe because:
- In Node.js, all calls are single-threaded.
- In Workers, each isolate is single-threaded.
- In browsers, all calls happen on the calling thread (main or worker).

The Worker's `buildIndexWrapper` maintains its own parallel ID maps (`_extToInt`, `_intToExt`, `_deletedExt`, `_vectors`) for the same reasons. These are per-isolate, not shared.

---

## 5. Data Flow

### 5.1 Insert Path (Node.js / Browser)

```
PancakeIndex.add(vec: Float32Array)
  │
  ├─ Validate dimension (vec.length === this._dim)
  ├─ Copy vec into WASM heap (HEAPF32.set)
  │
  ├─ _pancake_add(handle, vecPtr) ──────────── C ABI boundary
  │    │
  │    ├─ IndexWrapper::insert(vec)
  │    │    │
  │    │    ├─ [QuantizedHNSWWrapper<384>]
  │    │    │    ├─ Normalize if cosine
  │    │    │    ├─ quantize_fast<384>(float→uint8)
  │    │    │    ├─ Random level assignment (geometric distribution)
  │    │    │    ├─ Greedy descent from entry_point to target layer
  │    │    │    ├─ Beam search insertion per layer (ef_construction width)
  │    │    │    ├─ Diversity heuristic neighbor selection
  │    │    │    └─ Return internal ID (count_++)
  │    │    │
  │    │    ├─ [FloatHNSWWrapper]
  │    │    │    └─ Same algorithm, float32 distances
  │    │    │
  │    │    └─ [Int8FloatHNSWWrapper]
  │    │         ├─ Row-wise affine quantize: find min/max, scale to [0,255]
  │    │         └─ Same algorithm, asymmetric distances during construction
  │    │
  │    └─ Return internal ID
  │
  ├─ Map internal ID → external ID (this._nextExtId++)
  └─ Return external ID
```

### 5.2 Search Path

```
PancakeIndex.search(query: Float32Array, k: number)
  │
  ├─ Validate dimension and k
  ├─ Ensure result buffer capacity (realloc if k > current capacity)
  ├─ Copy query into WASM heap
  │
  ├─ _pancake_query(handle, queryPtr, k, idsPtr, distsPtr) ── C ABI
  │    │
  │    ├─ IndexWrapper::search(query, k)
  │    │    │
  │    │    ├─ [QuantizedHNSW<384>]
  │    │    │    ├─ Quantize query (cached if same query)
  │    │    │    ├─ Greedy descent from entry_point (upper layers)
  │    │    │    ├─ Beam search on layer 0 (ef_search width)
  │    │    │    │    ├─ Distance: l2_squared_fast<384> (WASM SIMD)
  │    │    │    │    │    └─ 24 iterations of 16-uint8 widening
  │    │    │    │    └─ Skip nodes with deletion flag set
  │    │    │    └─ Return top-k pairs (id, distance)
  │    │    │
  │    │    └─ [FloatHNSW / Int8FloatHNSW: same structure, different kernels]
  │    │
  │    ├─ Write results to ids/dists heap arrays
  │    └─ Return count of results found
  │
  ├─ Read uint64 IDs from WASM heap (DataView)
  ├─ Translate internal IDs → external IDs
  ├─ If L2 metric: sqrt(distance) for user-facing result
  └─ Return SearchResult[]
```

### 5.3 Worker Insert Path (HTTP)

```
POST /add { vector: [0.1, 0.2, ...] }
  │
  ├─ Authentication check (Bearer token)
  ├─ Rate limit check (per-IP sliding window)
  ├─ Parse JSON body
  ├─ Validate vector (array, correct length, all finite numbers)
  │
  ├─ index.add(new Float32Array(body.vector))
  │    ├─ Copy to WASM heap
  │    ├─ _pancake_add(handle, ptr) → internal ID
  │    ├─ Map to external ID
  │    └─ Stash vector copy for compact rebuild
  │
  ├─ schedulePersist(env, ctx)
  │    └─ Debounce 2s → R2 PUT (async, via ctx.waitUntil)
  │
  └─ Return JSON { id, count }
```

### 5.4 Worker Compact Path

The Worker implements a rebuild-based compaction strategy:

```
POST /compact
  │
  ├─ Collect surviving vectors from stash (_vectors Map)
  │    └─ Skip entries in _deletedExt
  │
  ├─ If stash is empty (e.g., post-import):
  │    ├─ Fall back to in-place compaction (_pancake_compact)
  │    ├─ Rebuild ID maps from survivors
  │    └─ Return
  │
  ├─ Dispose old handle (_pancake_dispose)
  ├─ Create new handle (_pancake_init with same params)
  ├─ Re-insert all surviving vectors
  ├─ Rebuild ID maps
  ├─ Clear and repopulate vector stash
  │
  └─ Return JSON { compacted, elapsed_ms, count, memory_bytes }
```

This is more expensive than the WASM-level compaction (which only rebuilds the graph), but produces optimal graph quality because the HNSW is built from scratch with fresh random level assignments.

---

## 6. WASM Compilation and Runtime

### 6.1 Build Pipeline

```
Source: src/engine.cpp + src/*.hpp
  │
  ├─ Emscripten (emcc) with flags:
  │    ├─ -O3 (speed optimization, not -Oz size optimization)
  │    ├─ -msimd128 (WASM 128-bit SIMD)
  │    ├─ -mnontrapping-fptoint, -msign-ext, -mbulk-memory
  │    ├─ -ffast-math -fvectorize -fslp-vectorize
  │    ├─ -fno-rtti (no runtime type info — saves ~10KB)
  │    ├─ MALLOC=emmalloc (lightweight allocator)
  │    ├─ ALLOW_MEMORY_GROWTH=1 (dynamic heap)
  │    ├─ INITIAL_MEMORY=16MB
  │    ├─ STACK_SIZE=64KB
  │    ├─ DISABLE_EXCEPTION_CATCHING=0 (C++ exceptions enabled)
  │    └─ FILESYSTEM=0 (no virtual filesystem)
  │
  ├─ Output: dist/engine.js (19KB) + dist/engine.wasm (232KB)
  │
  └─ patch_engine.py: Fix ENVIRONMENT_IS_NODE detection for Workers
       └─ Replaces `var ENVIRONMENT_IS_NODE = ...` with `false`
```

### 6.2 Why -O3 Instead of -Oz

The original build used `-Oz` (optimize for size). This was changed to `-O3` (optimize for speed) because:

- `-Oz` disables inlining and loop optimizations.
- The insert path has ~300 non-SIMD function calls reachable from the hot `_i8_add` entry point.
- With `-Oz`, inlining was suppressed, causing 2× regression on insert throughput.
- The WASM binary grew from ~200KB to ~230KB — acceptable for the 2× speed improvement.

### 6.3 Memory Model

WASM linear memory starts at 16MB and grows on demand. The host environment (Node.js, V8) manages the backing memory. Key implications:

- **`HEAPF32`, `HEAPU8`, `HEAPU32`** are typed array views into WASM linear memory. After memory growth, these views are invalidated. The Emscripten runtime automatically updates them, but host code must not cache the `.buffer` property across calls that might trigger growth.
- **`emsc_malloc` / `emsc_free`** are exported wrappers around emmalloc. The host uses these to allocate WASM heap space for passing vectors and receiving results.
- **No garbage collection.** WASM memory is manually managed. The JS wrapper calls `emsc_free` for temporary buffers and `pancake_dispose` for index handles. Leaking a handle leaks the entire index.

### 6.4 WASM Features Required

| Feature | Flag | Requirement |
|---------|------|-------------|
| SIMD | `-msimd128` | Required for quantized distance kernels. Without SIMD, quantized search would fall back to scalar (4× slower). |
| Bulk memory | `-mbulk-memory` | Used for efficient `memory.copy` and `memory.fill` during serialization. |
| Sign extension | `-msign-ext` | Enables `i32.extend8_s` etc. for sign-extending int8 values during dequantization. |
| Non-trapping f2i | `-mnontrapping-fptoint` | Avoids traps on NaN/overflow during float-to-int conversion in quantization. |
| BigInt integration | `WASM_BIGINT=1` | Required for passing uint64_t IDs across the WASM boundary. |

**Browser support:** Chrome 91+, Firefox 89+, Safari 16.4+ (all support WASM SIMD). Node.js 16+.

---

## 7. Serialization and Persistence

### 7.1 Three-Layer Serialization

```
Layer 1: JS Envelope (20 bytes)
  ├─ Magic, version, dim, metric, quantized
  │
Layer 2: WASM Binary (pancake_export output)
  ├─ Backend-specific header (40 bytes)
  ├─ Vector data
  ├─ Graph structure (neighbor lists)
  └─ Deletion state
  │
Layer 3: R2 Object (Worker only)
  ├─ Binary body = Layer 2 (no envelope)
  └─ Custom metadata: { dims, count, savedAt }
```

The JS envelope (Layer 1) is added by `PancakeIndex.export()` and stripped by `PancakeIndex.import()`. The Worker's R2 persistence stores Layer 2 directly (no envelope) because the metadata is stored as R2 custom metadata.

### 7.2 Backend Serialization Formats

See Appendix A for byte-level format details. Key properties:

- **All formats are little-endian.** Matches WASM's native byte order.
- **All formats include a magic number.** QuantizedHNSW uses `0x504E434B` ("PNCK") for quantized and `0x504E4346` ("PNCF") for float. FloatHNSW uses `0x464C4831` ("FLH1"). Int8FloatHNSW uses `0x49384831` ("I8H1").
- **All formats include a version field.** Enables forward-compatible format evolution.
- **QuantizedHNSW v3+ serializes deletion state.** Import preserves which vectors were deleted. Int8FloatHNSW and FloatHNSW do not — import resets all vectors to alive.

### 7.3 R2 Persistence Model

The Worker persists indexes to Cloudflare R2 (S3-compatible object storage):

- **Key:** `pancake-index.bin` (single key per bucket)
- **Write strategy:** Debounced. After any mutation (insert, delete, compact), a 2-second timer starts. When it fires, the current index is exported and written to R2 via `ctx.waitUntil`.
- **Read strategy:** Lazy. On cold start, the first non-health request triggers R2 restore. The index is loaded from R2 and ID maps are bootstrapped.
- **Consistency:** Last-write-wins. No versioning, no conflict resolution. If two isolates write concurrently, one write is lost.
- **Durability:** Best-effort. If an isolate is evicted before the 2-second timer fires, the pending write is lost. The maximum data loss window is `PERSIST_DEBOUNCE_MS` (2 seconds) worth of mutations.

---

## 8. Cloudflare Worker Deployment

### 8.1 API Surface

| Endpoint | Method | Body | Response | Auth |
|----------|--------|------|----------|------|
| `/health` | GET | — | `{ status, initialized, count, memory_bytes, dims }` | No |
| `/init` | POST | `{ dims, maxElements, M?, efConstruction?, efSearch?, vectors? }` | `{ status, dims, maxElements, inserted, memory_bytes }` | Yes |
| `/add` | POST | `{ vector: float[] }` | `{ id, count }` | Yes |
| `/add_batch` | POST | `{ vectors: float[][] }` | `{ inserted, ids, count }` | Yes |
| `/delete` | POST | `{ id: number }` | `{ deleted, ghost_count, ghost_ratio }` | Yes |
| `/compact` | POST | — | `{ compacted, elapsed_ms, count, memory_bytes }` | Yes |
| `/search` | POST | `{ query: float[], k?, ef? }` | `{ neighbors, distances, latency_ms }` | Yes |
| `/export` | GET | — | Binary blob | Yes |
| `/import` | POST | Binary blob, `?dims=N` | `{ status, dims, count, memory_bytes }` | Yes |
| `/stats` | GET | — | `{ count, memory_bytes, ghost_count, ghost_ratio, dims }` | Yes |

### 8.2 Validation Limits

| Parameter | Minimum | Maximum | Default |
|-----------|---------|---------|---------|
| `dims` | 1 | 4,096 | Required |
| `maxElements` | 1 | 1,000,000 | Required |
| `M` | 1 | 128 | 8 |
| `efConstruction` | 1 | 2,000 | 150 |
| `efSearch` | 1 | 2,000 | 100 |
| `k` (search) | 1 | 100 | 10 |
| `ef` (search) | 1 | 2,000 | 100 |

### 8.3 Rate Limiting

- **Mechanism:** Per-IP sliding window, in-memory.
- **Configuration:** `env.RATE_LIMIT_RPM` (requests per minute, 0 = disabled).
- **Scope:** Per-isolate. Not globally accurate across concurrent isolates.
- **Cleanup:** Stale entries purged every 60 seconds.
- **Response:** HTTP 429 with `{ error: "Rate limit exceeded. Try again later." }`.

### 8.4 Authentication

- **Mechanism:** Bearer token in `Authorization` header.
- **Configuration:** `env.API_KEY` (if unset, no authentication required).
- **Scope:** `/health` and CORS preflight are always public.
- **Response:** HTTP 401 with `{ error: "Unauthorized" }`.

### 8.5 Worker Memory Budget

Cloudflare Workers have a 128 MB memory limit. Approximate memory usage per vector at M=16:

| Backend | Bytes per vector (128D) | Bytes per vector (384D) | Bytes per vector (1536D) |
|---------|------------------------|------------------------|--------------------------|
| Int8 quantized | ~256 | ~512 | ~1,672 |
| Float32 | ~640 | ~1,664 | ~6,272 |

With int8 quantization at 128D:
- 128 MB budget → ~100K headroom after WASM overhead → **~400K vectors** theoretical max
- Practical limit ~200K vectors (accounting for graph overhead, JS heap, R2 buffers)

---

## 9. Quantization

### 9.1 Row-Wise Affine Quantization

Each vector is independently quantized to uint8 using per-vector scale and offset:

```
For each vector v[0..D-1]:
  min = min(v[0..D-1])
  max = max(v[0..D-1])
  scale = (max - min) / 255
  offset = min
  quantized[d] = round((v[d] - offset) / scale * 255)
  
Dequantize:
  v[d] ≈ offset + quantized[d] * scale
```

**Storage:**

- `QuantizedHNSW<DIMS>`: `AffineVector<DIMS>` struct = `{float scale, float offset, uint8_t data[DIMS]}`, aligned to 16 bytes. Total: 8 + DIMS bytes per vector.
- `Int8FloatHNSW`: Separate arrays for scales, offsets, and quantized data. Same total size but different memory layout (SoA vs AoS).

### 9.2 Quantization Error

Row-wise affine quantization maps the value range [min, max] of each vector to 256 levels. The maximum quantization error per element is:

```
max_error = (max - min) / (2 × 255)
```

For normalized vectors (unit L2 norm), values are typically in [-0.2, 0.2], giving max_error ≈ 0.0008 per dimension. Over D dimensions, the L2 distance error is bounded by:

```
L2_error ≤ sqrt(D) × max_error
```

For D=128: L2_error ≤ 0.009. For D=384: L2_error ≤ 0.016. In practice, recall degradation is <2% for typical embedding distributions (measured on SIFT-1M and NYTimes-256).

### 9.3 Asymmetric Search

During search, the query stays as float32. The distance between a float32 query and a uint8 database vector is computed by dequantizing the database vector on the fly:

```
distance(query, db_vec) = Σ (query[d] - (db_vec.offset + db_vec.data[d] × db_vec.scale))²
```

This avoids quantizing the query (which would compound quantization error) at the cost of dequantization during distance computation. The SIMD kernels perform the dequantization (u8→f32 widening + FMA) in-register, so the overhead is minimal.

---

## 10. Segmented Indexes

The C ABI exports two segmented index implementations alongside the handle-based single-index API. These are legacy APIs that predate the handle system.

### 10.1 Segmented 384D (`si_*`)

A set of `QuantizedHNSW<384>` shards with centroid-based routing.

- **Routing:** Nearest centroid with available capacity. Running mean centroids (no k-means).
- **Query:** Top-N segments by centroid distance → search each → merge results.
- **Global ID mapping:** `g_si_id_map` translates global IDs to (segment_index, local_id) pairs.
- **Centroid update:** Incremental running mean on insert. No periodic refresh.

### 10.2 Segmented 128D (`si128_*`)

Similar to 384D but with periodic k-means reclustering:

- **Reservoir sampling:** Algorithm R collects 5K samples for k-means.
- **Recluster trigger:** Every 10K inserts.
- **Bootstrap phase:** First N inserts distributed round-robin to seed all segments before centroid-based routing begins.
- **K-means:** L2 metric, 10 iterations, run on reservoir sample.

### 10.3 Why These Are Separate

These segmented indexes were built before the handle-based API and before `pancake-core.js`. They use direct global state and are exported as separate C functions (`_si_init`, `_si_add`, etc.). They are not accessible through the `PancakeIndex` JS wrapper.

They exist for specialized use cases (Cloudflare Worker demos, pre-built multi-segment indexes) but are not the recommended API. Future work should migrate them to the handle-based system.

---

## 11. SIMD Architecture

### 11.1 Platform Detection

```cpp
// quantized_simd.hpp, lines 23-41
#if defined(__EMSCRIPTEN__) || defined(__wasm__)
  #define PANCAKE_WASM_BUILD 1
  #if defined(__wasm_simd128__)
    #define PANCAKE_WASM_SIMD 1
  #endif
#endif

#if defined(__AVX2__)
  #define PANCAKE_USE_AVX2 1        // Detected but NOT IMPLEMENTED
#endif

#if defined(__SSE4_1__) || defined(__SSE2__)
  #define PANCAKE_USE_SSE 1
#endif
```

### 11.2 Dispatch Hierarchy

```cpp
template<size_t DIMS>
float l2_squared_fast(const AffineVector<DIMS>& a, const AffineVector<DIMS>& b) {
#ifdef PANCAKE_WASM_SIMD
    return l2_squared_wasm<DIMS>(a, b);       // WASM 128-bit SIMD
#elif defined(PANCAKE_USE_SSE)
    return l2_squared_sse<DIMS>(a, b);        // x86 SSE2
#else
    return l2_squared_scalar<DIMS>(a, b);     // Scalar fallback
#endif
}
```

The same pattern applies to `dot_product_fast`, `compute_norm_fast`, `cosine_distance_fast`, and `quantize_fast`.

### 11.3 WASM SIMD Quantized Distance Kernel

The core distance computation for quantized vectors processes 16 uint8 elements per iteration using WASM 128-bit SIMD:

```
For each group of 16 elements:
  1. Load 16 uint8 from vector A and B          (v128_load)
  2. Widen to 2×8 uint16                        (extend_low_u8x16, extend_high_u8x16)
  3. Widen each uint16×8 to 2×4 uint32         (extend_low_u16x8, extend_high_u16x8)
  4. Convert uint32×4 to float32×4              (convert_f32x4_u32x4)
  5. Dequantize: float = offset + data * scale  (f32x4_add, f32x4_mul)
  6. Compute squared difference                 (f32x4_sub, f32x4_mul)
  7. Accumulate into sum register               (f32x4_add)
```

For DIMS=384, this is 24 iterations (384/16). The inner loop has no branches and no memory dependencies between iterations.

### 11.4 Non-Quantized SIMD

`FloatHNSW` uses WASM SIMD for float32 L2 and cosine distance:

```
For each group of 4 float32 elements:
  1. Load 4 floats from A and B               (v128_load)
  2. Subtract                                  (f32x4_sub)
  3. Multiply (square)                         (f32x4_mul)
  4. Accumulate                                (f32x4_add)
```

This is simpler than the quantized kernel (no widening) but processes only 4 elements per iteration instead of 16.

### 11.5 AVX2 Status

The `PANCAKE_USE_AVX2` flag is detected but **no AVX2 kernels exist**. The codebase uses SSE2 as the highest x86 SIMD level. AVX2 would double throughput (256-bit vs 128-bit lanes) but requires AVX2-capable hardware and a native (non-WASM) build.

---

## 12. Limitations and Non-Goals

### 12.1 Things the System Does Not Do

**No server-side persistence (Node.js).** The Node.js deployment is purely in-memory. There is no WAL, no checkpoint, no disk I/O. If the process exits, the index is lost. The only persistence mechanism is `export()` to a `Uint8Array` that the application can write to disk manually.

**No streaming ingestion.** Unlike `pancake_gt/core/streaming/`, this system has no WAL, no delta segments, no async building, no snapshot management. Every insert is synchronous and immediately visible. There is no concept of ingestion lag or publish delay.

**No concurrent access within a WASM instance.** The system is designed for single-threaded access. Using it from multiple threads (e.g., Node.js `worker_threads`) sharing the same WASM instance is undefined behavior.

**No multi-index queries.** There is no mechanism to search across multiple handles and merge results. Each handle is independent. Cross-index search must be implemented by the host.

**No filtered search.** There is no metadata-based filtering. The system returns the k nearest neighbors by vector distance only. Post-filtering must be implemented by the host (oversample k, then filter).

**No incremental serialization.** `pancake_export` serializes the entire index. There is no mechanism to export deltas or incremental updates. For large indexes, this means exporting and storing megabytes of data on every R2 persist.

**No automatic compaction.** The system does not automatically compact when the ghost ratio exceeds a threshold. The host must call `compact()` explicitly (or the Worker's `/compact` endpoint). The Worker's compact path rebuilds the entire graph from the vector stash, which is correct but expensive.

**No dimension reduction.** Vectors must arrive pre-reduced to the target dimension. The system does not perform PCA, random projection, or any other dimensionality reduction.

### 12.2 Known Weaknesses

**Worker R2 persistence is best-effort.** The 2-second debounce timer means up to 2 seconds of mutations can be lost on isolate eviction. There is no acknowledgment that the R2 write completed. The Worker has no mechanism to detect or recover from R2 write failures other than a `console.error`.

**Worker isolates are independent.** Two concurrent isolates see different index states. There is no coordination, no leader election, no read-after-write consistency across isolates. This is a fundamental limitation of the Cloudflare Workers execution model.

**Serialization format is version-coupled.** An index exported from one version of Pancake may not import correctly into a different version if the serialization format changed. The magic number and version field provide detection, but there is no automatic migration.

**No graph quality monitoring.** There is no metric for HNSW graph quality (e.g., average neighbor connectivity, recall vs brute-force). The only observable metric is `ghostRatio`. An index with poor graph quality (e.g., after many inserts into a compacted graph) will have degraded recall with no warning.

**Segmented indexes use global state.** The `si_*` and `si128_*` functions use global variables, not handles. Only one segmented index of each type can exist per WASM instance. They cannot coexist with handle-based indexes that use the same dimension.

**Int8FloatHNSW does not preserve deletion state on import.** When importing a serialized Int8FloatHNSW index, the deletion bitmap is reset — all vectors come back alive. This differs from QuantizedHNSW (which preserves deletion state in v3+ format). The workaround is to `compact()` before `export()`.

### 12.3 Intentional Non-Optimizations

**No neighbor list packing.** Neighbor lists are stored as `vector<vector<vector<uint32_t>>>` — three levels of indirection per node. A flat packed representation (single contiguous array with offset table) would improve cache locality by 2-4× for search. This is the single largest performance opportunity but would require rewriting all three HNSW backends.

**No batch distance computation.** The SIMD kernels compute distance between one query and one database vector at a time. A batch kernel (one query vs N database vectors) would enable better SIMD utilization (process 4 query-vector pairs in parallel). This would require restructuring the HNSW search loop.

**No prefetching.** The HNSW search loop does not issue prefetch instructions for upcoming neighbor vectors. Since HNSW search is pointer-chasing (follow edge → load neighbor vector → compute distance → follow next edge), cache misses dominate latency at large scale. Software prefetching of the next neighbor's vector data during the current distance computation could reduce latency by 10-20%.

**No NUMA awareness.** The system is single-threaded WASM with no concept of memory topology. In a native deployment, NUMA-aware allocation of segment data would reduce cross-socket memory access latency.

---

## Appendix A: Serialization Formats

### QuantizedHNSW (Magic: 0x504E434B / 0x504E4346)

```
Offset   Size           Field
0        4              Magic (0x504E434B=quantized, 0x504E4346=float)
4        4              DIMS (template parameter)
8        4              Version (current: 3)
12       4              Count
16       4              Entry point
20       4              Max level
24       4              M
28       4              M0 (2×M)
32       4              Metric (0=L2, 1=Cosine)
36       4              ef_construction
40       count×AV       Vectors (AffineVector: scale+offset+data per vec)
         or count×D×4   Float vectors (if magic==PNCF)
?        count×4        Norms (if cosine, v3+)
?        variable       Graph (per-node: level(4), per-level: count(4)+neighbors(count×4))
?        4              num_deleted (v1+)
?        count×1        Deleted flags (v1+)
```

### FloatHNSW (Magic: 0x464C4831)

```
Offset   Size           Field
0        4              Magic (0x464C4831)
4        4              Dims
8        4              Version (0 or 1)
12       4              Count
16       4              Entry point
20       4              Max level
24       4              M
28       4              M0
32       4              Metric (v1+)
36       4              ef_construction (v1+)
40       count×D×4      Vectors (float32)
?        variable       Graph (same format as QuantizedHNSW)
```

Note: FloatHNSW does NOT serialize deletion state. Import resets all vectors to alive.

### Int8FloatHNSW (Magic: 0x49384831)

```
Offset   Size           Field
0        4              Magic (0x49384831)
4        4              Dims
8        4              Version (1)
12       4              Count
16       4              Entry point
20       4              Max level
24       4              M
28       4              M0
32       4              Metric
36       4              ef_construction
40       count×4        Scales (float32)
?        count×4        Offsets (float32)
?        count×D        Quantized data (uint8)
?        variable       Graph (same format)
```

Note: Int8FloatHNSW does NOT serialize deletion state. Import resets all vectors to alive.

### JS Envelope (prepended by pancake-core.js)

```
Offset   Size    Field
0        4       Magic (0x504E434B)
4        4       Version (2)
8        4       Dimension
12       4       Metric (0=L2, 1=Cosine)
16       4       Quantized (0=float, 1=int8)
20       ...     Backend binary (from pancake_export)
```

---

## Appendix B: Configuration Reference

### PancakeIndex.create() Options

| Option | Type | Default | Range | Notes |
|--------|------|---------|-------|-------|
| `dim` | number | Required | 1–100,000 | Vector dimension |
| `maxElements` | number | 100,000 | 1–∞ | Capacity (memory allocated upfront) |
| `metric` | string | `'cosine'` | `'cosine'`, `'l2'` | Distance metric |
| `quantized` | boolean | `true` | — | Int8 quantization (4× memory savings) |
| `M` | number | 16 | 1–∞ | HNSW max neighbors per layer |
| `efConstruction` | number | 200 | 1–∞ | Build beam width (higher = better graph, slower build) |
| `efSearch` | number | 100 | 1–∞ | Search beam width (higher = better recall, slower search) |

### Worker Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_KEY` | (none) | Bearer token for authentication. If unset, no auth. |
| `ALLOWED_ORIGIN` | `*` | CORS `Access-Control-Allow-Origin` header value. |
| `RATE_LIMIT_RPM` | `0` (disabled) | Maximum requests per minute per IP. |

### Build-Time Constants

| Constant | Value | Location |
|----------|-------|----------|
| `MAX_HANDLES` | 64 | engine.cpp |
| `INVALID_HANDLE` | 0xFFFFFFFF | engine.cpp |
| `MAX_RESULTS` | 100 | worker.js |
| `MAX_DIMS` | 4,096 | worker.js |
| `MAX_ELEMENTS` | 1,000,000 | worker.js |
| `MAX_EF` | 2,000 | worker.js |
| `MAX_M` | 128 | worker.js |
| `PERSIST_DEBOUNCE_MS` | 2,000 | worker.js |

---

## Appendix C: WASM Export Inventory

### Handle-Based Index API (16 functions)

| Export | Signature | Returns |
|--------|-----------|---------|
| `_pancake_init` | `(dim, max_elem, quantized, metric, M, ef_c, ef_s) → handle` | uint32 handle or 0xFFFFFFFF |
| `_pancake_add` | `(handle, vec_ptr) → id` | uint32 ID or 0xFFFFFFFF |
| `_pancake_bulk_insert` | `(handle, vecs_ptr, count) → inserted` | int count |
| `_pancake_query` | `(handle, query_ptr, k, ids_ptr, dists_ptr) → found` | int count |
| `_pancake_delete` | `(handle, id) → void` | — |
| `_pancake_compact` | `(handle) → void` | — |
| `_pancake_count` | `(handle) → count` | size_t |
| `_pancake_memory` | `(handle) → bytes` | size_t |
| `_pancake_ghost_count` | `(handle) → count` | size_t |
| `_pancake_ghost_ratio` | `(handle) → ratio` | float |
| `_pancake_set_ef` | `(handle, ef) → void` | — |
| `_pancake_export` | `(handle, out_size_ptr) → data_ptr` | uint8_t* or null |
| `_pancake_import` | `(handle, data_ptr, size) → status` | 0=success, -1=failure |
| `_pancake_dispose` | `(handle) → void` | — |
| `_pancake_dimension` | `(handle) → dim` | int |
| `_pancake_shutdown_all` | `() → void` | — |

### Segmented Index API (26 functions)

| Prefix | Functions |
|--------|-----------|
| `_si_` | init, add, query, delete, count, memory, segment_count, ghost_count, ghost_ratio, compact, set_budget |
| `_si128_` | init, add, query, delete, count, memory, segment_count, ghost_count, compact, set_budget, get_vector_segment, get_query_segments |

### Utility Functions (5 functions)

| Export | Purpose |
|--------|---------|
| `_dense_matmul` | SIMD-accelerated matrix-vector multiply |
| `_sparse_matmul` | Sparse matrix-vector multiply |
| `_normalize` | In-place L2 normalization |
| `_emsc_malloc` | WASM heap allocation |
| `_emsc_free` | WASM heap deallocation |

### Exported Runtime Methods

| Method | Purpose |
|--------|---------|
| `ccall` | Call C function by name with type marshaling |
| `cwrap` | Create JS wrapper for C function |
| `HEAPF32` | Float32 view of WASM heap |
| `HEAPU8` | Uint8 view of WASM heap |
| `HEAPU32` | Uint32 view of WASM heap |
| `HEAP32` | Int32 view of WASM heap |

---

*End of document.*
