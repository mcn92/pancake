#include "float_hnsw.hpp"
#include "int8_float_hnsw.hpp"
#include <unordered_map>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <cstring>
#include <memory>

#if defined(__wasm_simd128__)
    #include <wasm_simd128.h>
#endif

using namespace pancake::wasm;

namespace {

size_t serialized_index_count_hint(const uint8_t* data, size_t size, uint32_t versioned_magic) {
    if (!data || size < 12) return 100000;

    uint32_t magic = 0;
    std::memcpy(&magic, data, sizeof(magic));

    size_t count_offset = (magic == versioned_magic) ? 12 : 8;
    if (size < count_offset + 4) return 100000;

    uint32_t count = 0;
    std::memcpy(&count, data + count_offset, sizeof(count));
    return static_cast<size_t>(count);
}

} // namespace

// =============================================================================
// Handle-based index system
// =============================================================================

class IndexWrapper {
public:
    virtual ~IndexWrapper() = default;
    virtual uint32_t insert(const float* vec) = 0;
    virtual int bulk_insert(const float* vecs, int n) = 0;
    virtual std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k) = 0;
    virtual std::vector<std::pair<uint32_t, float>> search_filtered(const float* query, size_t k, const uint8_t* bitset, size_t bitset_len) = 0;
    virtual void mark_delete(uint32_t id) = 0;
    virtual void compact() = 0;
    virtual void compact(std::vector<uint32_t>& out_map) = 0;
    virtual size_t count() const = 0;
    virtual size_t ghost_count() const = 0;
    virtual float ghost_ratio() const = 0;
    virtual size_t memory_bytes() const = 0;
    virtual std::vector<uint8_t> serialize() const = 0;
    virtual bool deserialize(const uint8_t* data, size_t size) = 0;
    virtual void set_ef_search(size_t ef) = 0;
    virtual size_t dimension() const = 0;
};


class FloatHNSWWrapper : public IndexWrapper {
    std::unique_ptr<FloatHNSW> impl_;
    FloatHNSWConfig cfg_;
    size_t dims_;
public:
    FloatHNSWWrapper(size_t dims, const FloatHNSWConfig& cfg)
        : impl_(std::make_unique<FloatHNSW>(dims, cfg)), cfg_(cfg), dims_(dims) {}

    uint32_t insert(const float* vec) override { return impl_->insert(vec); }
    int bulk_insert(const float* vecs, int n) override {
        int inserted = 0;
        for (int i = 0; i < n; i++) {
            uint32_t id = impl_->insert(vecs + i * dims_);
            if (id != 0xFFFFFFFF) inserted++;
        }
        return inserted;
    }
    std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k) override {
        return impl_->search(query, k);
    }
    std::vector<std::pair<uint32_t, float>> search_filtered(const float* query, size_t k, const uint8_t* bitset, size_t bitset_len) override {
        return impl_->search_filtered(query, k, bitset, bitset_len);
    }
    void mark_delete(uint32_t id) override { impl_->mark_delete(id); }
    void compact() override { impl_->compact(); }
    void compact(std::vector<uint32_t>& out_map) override { impl_->compact(out_map); }
    size_t count() const override { return impl_->count(); }
    size_t ghost_count() const override { return impl_->ghost_count(); }
    float ghost_ratio() const override { return impl_->ghost_ratio(); }
    size_t memory_bytes() const override { return impl_->memory_bytes(); }
    std::vector<uint8_t> serialize() const override { return impl_->serialize(); }
    bool deserialize(const uint8_t* data, size_t size) override {
        size_t cnt = serialized_index_count_hint(data, size, 0x464C4831);
        FloatHNSWConfig cfg = cfg_;
        if (cnt > cfg.max_elements) return false;

        auto next = std::make_unique<FloatHNSW>(dims_, cfg);
        if (!next->deserialize(data, size)) return false;
        impl_ = std::move(next);
        return true;
    }
    void set_ef_search(size_t ef) override { impl_->set_ef(ef); }
    size_t dimension() const override { return dims_; }
};

class Int8FloatHNSWWrapper : public IndexWrapper {
    std::unique_ptr<Int8FloatHNSW> impl_;
    Int8FloatHNSWConfig cfg_;
    size_t dims_;
public:
    Int8FloatHNSWWrapper(size_t dims, const Int8FloatHNSWConfig& cfg)
        : impl_(std::make_unique<Int8FloatHNSW>(dims, cfg)), cfg_(cfg), dims_(dims) {}

    uint32_t insert(const float* vec) override { return impl_->insert(vec); }
    int bulk_insert(const float* vecs, int n) override { return impl_->bulk_insert(vecs, n); }
    std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k) override {
        return impl_->search(query, k);
    }
    std::vector<std::pair<uint32_t, float>> search_filtered(const float* query, size_t k, const uint8_t* bitset, size_t bitset_len) override {
        return impl_->search_filtered(query, k, bitset, bitset_len);
    }
    void mark_delete(uint32_t id) override { impl_->mark_delete(id); }
    void compact() override { impl_->compact(); }
    void compact(std::vector<uint32_t>& out_map) override { impl_->compact(out_map); }
    size_t count() const override { return impl_->count(); }
    size_t ghost_count() const override { return impl_->ghost_count(); }
    float ghost_ratio() const override { return impl_->ghost_ratio(); }
    size_t memory_bytes() const override { return impl_->memory_bytes(); }
    std::vector<uint8_t> serialize() const override { return impl_->serialize(); }
    bool deserialize(const uint8_t* data, size_t size) override {
        size_t cnt = serialized_index_count_hint(data, size, 0x49384831);
        Int8FloatHNSWConfig cfg = cfg_;
        if (cnt > cfg.max_elements) return false;

        auto next = std::make_unique<Int8FloatHNSW>(dims_, cfg);
        if (!next->deserialize(data, size)) return false;
        impl_ = std::move(next);
        return true;
    }
    void set_ef_search(size_t ef) override { impl_->set_ef_search(ef); }
    size_t dimension() const override { return dims_; }
};

// Handle table
constexpr uint32_t MAX_HANDLES = 64;
constexpr uint32_t INVALID_HANDLE = 0xFFFFFFFF;

struct HandleSlot {
    IndexWrapper* index = nullptr;
    size_t dim = 0;
};

static HandleSlot g_handles[MAX_HANDLES];
static std::vector<uint8_t> g_export_bufs[MAX_HANDLES];

static uint32_t alloc_handle() {
    for (uint32_t i = 0; i < MAX_HANDLES; i++) {
        if (!g_handles[i].index) return i;
    }
    return INVALID_HANDLE;
}

static void free_handle(uint32_t h) {
    if (h < MAX_HANDLES && g_handles[h].index) {
        delete g_handles[h].index;
        g_handles[h].index = nullptr;
        g_handles[h].dim = 0;
        g_export_bufs[h].clear();
    }
}


// =============================================================================
// C ABI
// =============================================================================

extern "C" {

// =============================================================================
// Handle-based pancake_* API
// =============================================================================

uint32_t pancake_init(int dim, int max_elem, int quantized, int metric,
                      int M, int ef_c, int ef_s) {
    if (dim <= 0 || max_elem <= 0) return INVALID_HANDLE;

    uint32_t h = alloc_handle();
    if (h == INVALID_HANDLE) return INVALID_HANDLE;

    g_handles[h].dim = static_cast<size_t>(dim);
    bool use_cosine = (metric == 1);

    if (quantized) {
        Int8FloatHNSWConfig i8cfg;
        i8cfg.max_elements = static_cast<size_t>(max_elem);
        i8cfg.M = (M > 0) ? static_cast<size_t>(M) : 16;
        i8cfg.ef_construction = (ef_c > 0) ? static_cast<size_t>(ef_c) : 50;
        i8cfg.ef_search = (ef_s > 0) ? static_cast<size_t>(ef_s) : 100;
        i8cfg.metric = use_cosine ? DistanceMetric::Cosine : DistanceMetric::L2;
        i8cfg.use_heuristic = true;
        g_handles[h].index = new Int8FloatHNSWWrapper(dim, i8cfg);
    } else {
        FloatHNSWConfig cfg;
        cfg.max_elements = static_cast<size_t>(max_elem);
        cfg.M = (M > 0) ? static_cast<size_t>(M) : 16;
        cfg.ef_construction = (ef_c > 0) ? static_cast<size_t>(ef_c) : 50;
        cfg.ef_search = (ef_s > 0) ? static_cast<size_t>(ef_s) : 100;
        cfg.metric = use_cosine ? DistanceMetric::Cosine : DistanceMetric::L2;
        g_handles[h].index = new FloatHNSWWrapper(dim, cfg);
    }

    return h;
}

uint32_t pancake_add(uint32_t h, const float* vec) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return 0xFFFFFFFF;
    return g_handles[h].index->insert(vec);
}

int pancake_bulk_insert(uint32_t h, const float* vecs, int n) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return 0;
    return g_handles[h].index->bulk_insert(vecs, n);
}

int pancake_query(uint32_t h, const float* qv, int k, uint64_t* ids, float* dists) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return 0;
    auto res = g_handles[h].index->search(qv, static_cast<size_t>(k));
    for (size_t j = 0; j < res.size(); ++j) {
        ids[j] = static_cast<uint64_t>(res[j].first);
        dists[j] = res[j].second;
    }
    return static_cast<int>(res.size());
}

int pancake_query_filtered(uint32_t h, const float* qv, int k,
                           uint64_t* ids, float* dists,
                           const uint8_t* bitset, size_t bitset_len) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return 0;
    auto res = g_handles[h].index->search_filtered(qv, static_cast<size_t>(k), bitset, bitset_len);
    for (size_t j = 0; j < res.size(); ++j) {
        ids[j] = static_cast<uint64_t>(res[j].first);
        dists[j] = res[j].second;
    }
    return static_cast<int>(res.size());
}

void pancake_delete(uint32_t h, uint32_t id) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return;
    g_handles[h].index->mark_delete(id);
}

void pancake_compact(uint32_t h) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return;
    g_handles[h].index->compact();
}

// Compact and write the old→new ID remap into caller-allocated buffer.
// out_buf[old_id] = new_id, or 0xFFFFFFFF for deleted vectors.
// Returns the number of entries written (= pre-compaction count).
size_t pancake_compact_remap(uint32_t h, uint32_t* out_buf, size_t out_capacity) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return 0;
    if (!out_buf || out_capacity == 0) return 0;
    std::vector<uint32_t> map;
    g_handles[h].index->compact(map);
    size_t n = std::min(map.size(), out_capacity);
    std::memcpy(out_buf, map.data(), n * sizeof(uint32_t));
    return n;
}

size_t pancake_count(uint32_t h) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return 0;
    return g_handles[h].index->count();
}

size_t pancake_memory(uint32_t h) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return 0;
    return g_handles[h].index->memory_bytes();
}

size_t pancake_ghost_count(uint32_t h) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return 0;
    return g_handles[h].index->ghost_count();
}

float pancake_ghost_ratio(uint32_t h) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return 0.0f;
    return g_handles[h].index->ghost_ratio();
}

void pancake_set_ef(uint32_t h, int ef) {
    if (h >= MAX_HANDLES || !g_handles[h].index || ef <= 0) return;
    g_handles[h].index->set_ef_search(static_cast<size_t>(ef));
}

int pancake_bulk_insert_flat(uint32_t h, const float* vecs, int n) {
    return pancake_bulk_insert(h, vecs, n);
}

uint8_t* pancake_export(uint32_t h, size_t* out_size) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return nullptr;
    g_export_bufs[h] = g_handles[h].index->serialize();
    *out_size = g_export_bufs[h].size();
    return g_export_bufs[h].data();
}

int pancake_import(uint32_t h, const uint8_t* data, size_t size) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return -1;
    // deserialize() parses an untrusted buffer. Bounds and level/scale caps make
    // a hostile snapshot fail closed, but a remaining oversized resize() could
    // still throw; catch it here so import returns an error instead of aborting
    // the whole WASM instance.
    try {
        return g_handles[h].index->deserialize(data, size) ? 0 : -1;
    } catch (...) {
        return -1;
    }
}

void pancake_dispose(uint32_t h) {
    free_handle(h);
}

int pancake_dimension(uint32_t h) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return 0;
    return static_cast<int>(g_handles[h].index->dimension());
}


// =============================================================================
// Utility exports
// =============================================================================

void* emsc_malloc(size_t size) { return malloc(size); }
void emsc_free(void* ptr) { free(ptr); }

void pancake_profile_print(uint32_t range_start, uint32_t range_end) {
#if defined(PANCAKE_INT8_HNSW_BUILD_PROFILE)
    pancake::wasm::g_build_profile.print(range_start, range_end);
#else
    (void)range_start;
    (void)range_end;
#endif
}

void pancake_profile_reset() {
#if defined(PANCAKE_INT8_HNSW_BUILD_PROFILE)
    pancake::wasm::g_build_profile.reset();
#endif
}

// =============================================================================
// Dense/Sparse matmul + normalize (WASM SIMD)
// =============================================================================

void dense_matmul(const float* matrix, const float* vec, const float* bias, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        const float* row = matrix + i * cols;
#if defined(__wasm_simd128__)
        v128_t acc = wasm_f32x4_splat(0.0f);
        int j = 0;
        for (; j + 4 <= cols; j += 4) {
            v128_t m = wasm_v128_load(row + j);
            v128_t v = wasm_v128_load(vec + j);
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(m, v));
        }
        sum = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) +
              wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
        for (; j < cols; j++) sum += row[j] * vec[j];
#else
        for (int j = 0; j < cols; j++) sum += row[j] * vec[j];
#endif
        output[i] = sum + bias[i];
    }
}

void sparse_matmul(const float* matrix, const int* indices, const float* values, int nnz,
                   const float* bias, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) output[i] = bias[i];
    for (int k = 0; k < nnz; k++) {
        int j = indices[k];
        float val = values[k];
        for (int i = 0; i < rows; i++) {
            output[i] += matrix[i * cols + j] * val;
        }
    }
}

void normalize(float* vec, int dim) {
    float norm_sq = 0.0f;
#if defined(__wasm_simd128__)
    v128_t acc = wasm_f32x4_splat(0.0f);
    int i = 0;
    for (; i + 4 <= dim; i += 4) {
        v128_t v = wasm_v128_load(vec + i);
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(v, v));
    }
    norm_sq = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) +
              wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
    for (; i < dim; i++) norm_sq += vec[i] * vec[i];
#else
    for (int i = 0; i < dim; i++) norm_sq += vec[i] * vec[i];
#endif
    if (norm_sq > 0.0f) {
        float inv_norm = 1.0f / sqrtf(norm_sq);
#if defined(__wasm_simd128__)
        v128_t inv = wasm_f32x4_splat(inv_norm);
        int i = 0;
        for (; i + 4 <= dim; i += 4) {
            v128_t v = wasm_v128_load(vec + i);
            wasm_v128_store(vec + i, wasm_f32x4_mul(v, inv));
        }
        for (; i < dim; i++) vec[i] *= inv_norm;
#else
        for (int i = 0; i < dim; i++) vec[i] *= inv_norm;
#endif
    }
}

// =============================================================================
// Global cleanup
// =============================================================================

void pancake_shutdown_all() {
    // Free all handle-based indexes
    for (uint32_t i = 0; i < MAX_HANDLES; i++) {
        free_handle(i);
    }
}

void shutdown_all() {
    pancake_shutdown_all();
}

} // extern "C"
