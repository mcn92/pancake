#include "float_hnsw.hpp"
#include "int8_float_hnsw.hpp"
#if __has_include("embedding_model.hpp")
#include "embedding_model.hpp"
#define PANCAKE_HAS_EMBEDDING 1
#else
#define PANCAKE_HAS_EMBEDDING 0
#endif
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
        size_t cnt = 100000;
        if (size >= 16) {
            uint32_t c; memcpy(&c, data + 12, 4);
            cnt = static_cast<size_t>(c);
        }
        FloatHNSWConfig cfg = cfg_;
        cfg.max_elements = std::max(static_cast<size_t>(cnt * 1.2), static_cast<size_t>(100000));
        impl_ = std::make_unique<FloatHNSW>(dims_, cfg);
        return impl_->deserialize(data, size);
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
        size_t cnt = 100000;
        if (size >= 16) {
            uint32_t c; memcpy(&c, data + 12, 4);
            cnt = static_cast<size_t>(c);
        }
        Int8FloatHNSWConfig cfg = cfg_;
        cfg.max_elements = std::max(static_cast<size_t>(cnt * 1.2), static_cast<size_t>(100000));
        impl_ = std::make_unique<Int8FloatHNSW>(dims_, cfg);
        return impl_->deserialize(data, size);
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
        i8cfg.M = (M > 0) ? static_cast<size_t>(M) : 32;
        i8cfg.ef_construction = (ef_c > 0) ? static_cast<size_t>(ef_c) : 200;
        i8cfg.ef_search = (ef_s > 0) ? static_cast<size_t>(ef_s) : 128;
        i8cfg.metric = use_cosine ? DistanceMetric::Cosine : DistanceMetric::L2;
        i8cfg.use_heuristic = true;
        g_handles[h].index = new Int8FloatHNSWWrapper(dim, i8cfg);
    } else {
        FloatHNSWConfig cfg;
        cfg.max_elements = static_cast<size_t>(max_elem);
        cfg.M = (M > 0) ? static_cast<size_t>(M) : 32;
        cfg.ef_construction = (ef_c > 0) ? static_cast<size_t>(ef_c) : 200;
        cfg.ef_search = (ef_s > 0) ? static_cast<size_t>(ef_s) : 128;
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
    return g_handles[h].index->deserialize(data, size) ? 0 : -1;
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
    pancake::wasm::g_build_profile.print(range_start, range_end);
}

void pancake_profile_reset() {
    pancake::wasm::g_build_profile.reset();
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
// Embedding Model Integration
// =============================================================================

#if PANCAKE_HAS_EMBEDDING

static embedding::EmbeddingModel* g_embedder = nullptr;
static std::vector<float> g_embedding_buf;

int emb_init(int vocab_size) {
    if (g_embedder) { delete g_embedder; g_embedder = nullptr; }
    g_embedder = new embedding::EmbeddingModel(static_cast<size_t>(vocab_size));
    g_embedding_buf.clear();
    return 0;
}

const float* emb_encode(const char* text, size_t* out_size) {
    if (!g_embedder || !text) return nullptr;
    std::string str(text);
    g_embedding_buf = g_embedder->encode(str);
    if (out_size) *out_size = g_embedding_buf.size();
    return g_embedding_buf.data();
}

int emb_encode_batch(const char** texts, int count, float* out_buffer) {
    if (!g_embedder || !texts || !out_buffer || count <= 0) return 0;
    for (int i = 0; i < count; ++i) {
        if (!texts[i]) continue;
        std::string str(texts[i]);
        auto embedding = g_embedder->encode(str);
        std::memcpy(out_buffer + i * 384, embedding.data(), 384 * sizeof(float));
    }
    return count;
}

int emb_dimension() { return embedding::EmbeddingModel::D_MODEL; }

void emb_free() {
    if (g_embedder) { delete g_embedder; g_embedder = nullptr; }
    g_embedding_buf.clear();
}

uint32_t emb_add(const char*) { return 0xFFFFFFFF; }
int emb_search(const char*, int, uint64_t*, float*) { return 0; }

#else

int emb_init(int) { return -1; }
const float* emb_encode(const char*, size_t*) { return nullptr; }
int emb_encode_batch(const char**, int, float*) { return 0; }
int emb_dimension() { return 0; }
void emb_free() {}
uint32_t emb_add(const char*) { return 0xFFFFFFFF; }
int emb_search(const char*, int, uint64_t*, float*) { return 0; }

#endif

// =============================================================================
// Global cleanup
// =============================================================================

void pancake_shutdown_all() {
    // Free all handle-based indexes
    for (uint32_t i = 0; i < MAX_HANDLES; i++) {
        free_handle(i);
    }


#if PANCAKE_HAS_EMBEDDING
    if (g_embedder) { delete g_embedder; g_embedder = nullptr; }
    g_embedding_buf.clear();
#endif
}

void shutdown_all() {
    pancake_shutdown_all();
}

} // extern "C"
