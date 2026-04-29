#include "float_hnsw.hpp"
#include "quantized_hnsw.hpp"
#include "int8_float_hnsw.hpp"
#include "kmeans.hpp"
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
    virtual void mark_delete(uint32_t id) = 0;
    virtual void compact() = 0;
    virtual size_t count() const = 0;
    virtual size_t ghost_count() const = 0;
    virtual float ghost_ratio() const = 0;
    virtual size_t memory_bytes() const = 0;
    virtual std::vector<uint8_t> serialize() const = 0;
    virtual bool deserialize(const uint8_t* data, size_t size) = 0;
    virtual void set_ef_search(size_t ef) = 0;
    virtual size_t dimension() const = 0;
};

template<size_t DIMS>
class QuantizedHNSWWrapper : public IndexWrapper {
    std::unique_ptr<QuantizedHNSW<DIMS>> impl_;
    QuantizedHNSWConfig cfg_;
public:
    QuantizedHNSWWrapper(const QuantizedHNSWConfig& cfg)
        : impl_(std::make_unique<QuantizedHNSW<DIMS>>(cfg)), cfg_(cfg) {}

    uint32_t insert(const float* vec) override { return impl_->insert(vec); }
    int bulk_insert(const float* vecs, int n) override { return impl_->bulk_insert(vecs, n); }
    std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k) override {
        return impl_->search(query, k);
    }
    void mark_delete(uint32_t id) override { impl_->mark_delete(id); }
    void compact() override { impl_->compact(); }
    size_t count() const override { return impl_->count(); }
    size_t ghost_count() const override { return impl_->ghost_count(); }
    float ghost_ratio() const override { return impl_->ghost_ratio(); }
    size_t memory_bytes() const override { return impl_->memory_bytes(); }
    std::vector<uint8_t> serialize() const override { return impl_->serialize(); }
    bool deserialize(const uint8_t* data, size_t size) override {
        // Peek at count from header to size max_elements
        size_t cnt = 100000;
        if (size >= 16) {
            uint32_t c; memcpy(&c, data + 12, 4);
            cnt = static_cast<size_t>(c);
        }
        QuantizedHNSWConfig cfg = cfg_;
        cfg.max_elements = std::max(static_cast<size_t>(cnt * 1.2), static_cast<size_t>(100000));
        impl_ = std::make_unique<QuantizedHNSW<DIMS>>(cfg);
        return impl_->deserialize(data, size);
    }
    void set_ef_search(size_t ef) override { impl_->set_ef_search(ef); }
    size_t dimension() const override { return DIMS; }
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
    void mark_delete(uint32_t id) override { impl_->mark_delete(id); }
    void compact() override { impl_->compact(); }
    size_t count() const override { return impl_->count(); }
    size_t ghost_count() const override { return impl_->ghost_count(); }
    float ghost_ratio() const override { return impl_->ghost_ratio(); }
    size_t memory_bytes() const override { return impl_->memory_bytes(); }
    std::vector<uint8_t> serialize() const override { return impl_->serialize(); }
    bool deserialize(const uint8_t* data, size_t size) override {
        // Header: [0] magic [1] dims [2] version [3] count ...
        // Count is at byte offset 12 (field index 3)
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
    void mark_delete(uint32_t id) override { impl_->mark_delete(id); }
    void compact() override { impl_->compact(); }
    size_t count() const override { return impl_->count(); }
    size_t ghost_count() const override { return impl_->ghost_count(); }
    float ghost_ratio() const override { return impl_->ghost_ratio(); }
    size_t memory_bytes() const override { return impl_->memory_bytes(); }
    std::vector<uint8_t> serialize() const override { return impl_->serialize(); }
    bool deserialize(const uint8_t* data, size_t size) override {
        // Header: [0] magic [1] dims [2] version [3] count ...
        // Count is at byte offset 12 (field index 3)
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
// Segmented Index (384D) -- multiple QuantizedHNSW segments with centroid routing
// =============================================================================

const size_t DIMENSIONS = 384;

struct SegmentInfo {
    QuantizedHNSW<DIMENSIONS>* index;
    float centroid[DIMENSIONS];
    size_t count;
    std::vector<uint32_t> local_to_global;
};

struct SiIdMapping { size_t seg_idx; uint32_t local_id; };

static std::vector<SegmentInfo> g_si_segments;
static std::unordered_map<uint32_t, SiIdMapping> g_si_id_map;
static QuantizedHNSWConfig g_si_hnsw_cfg;
static size_t g_si_budget = 3;
static size_t g_si_max_segments = 8;
static size_t g_si_seg_capacity = 5000;
static uint32_t g_si_next_id = 0;

static void si_cleanup() {
    for (auto& seg : g_si_segments) delete seg.index;
    g_si_segments.clear();
    g_si_id_map.clear();
    g_si_next_id = 0;
}

static size_t si_create_segment() {
    SegmentInfo seg;
    seg.index = new QuantizedHNSW<DIMENSIONS>(g_si_hnsw_cfg);
    for (size_t d = 0; d < DIMENSIONS; d++) seg.centroid[d] = 0.0f;
    seg.count = 0;
    seg.local_to_global.reserve(g_si_seg_capacity);
    g_si_segments.push_back(seg);
    return g_si_segments.size() - 1;
}

static float si_l2(const float* a, const float* b) {
    float sum = 0.0f;
    for (size_t d = 0; d < DIMENSIONS; d++) {
        float diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

static size_t si_route(const float* vec) {
    float best_dist = 1e30f;
    size_t best_seg = 0;
    bool found = false;
    for (size_t i = 0; i < g_si_segments.size(); i++) {
        if (g_si_segments[i].count >= g_si_seg_capacity) continue;
        float dist = (g_si_segments[i].count == 0) ? 1e30f :
                     si_l2(vec, g_si_segments[i].centroid);
        if (!found || dist < best_dist) {
            best_dist = dist;
            best_seg = i;
            found = true;
        }
    }
    if (found) return best_seg;
    if (g_si_segments.size() < g_si_max_segments) return si_create_segment();
    return SIZE_MAX;
}

static void si_update_centroid(size_t seg_idx, const float* vec) {
    auto& seg = g_si_segments[seg_idx];
    float n = static_cast<float>(seg.count);
    float inv_n1 = 1.0f / (n + 1.0f);
    for (size_t d = 0; d < DIMENSIONS; d++) {
        seg.centroid[d] = seg.centroid[d] * (n * inv_n1) + vec[d] * inv_n1;
    }
}

// =============================================================================
// Segmented Index 128D
// =============================================================================

struct Seg128Info {
    QuantizedHNSW<128>* index;
    std::vector<float> centroid;
    size_t count;
    std::vector<uint32_t> local_to_global;
};

struct Si128IdMapping { size_t seg_idx; uint32_t local_id; };

static std::vector<Seg128Info> g_si128_segments;
static std::unordered_map<uint32_t, Si128IdMapping> g_si128_id_map;
static QuantizedHNSWConfig g_si128_hnsw_cfg;
static size_t g_si128_dim = 128;
static size_t g_si128_budget = 3;
static size_t g_si128_max_segments = 8;
static size_t g_si128_seg_capacity = 5000;
static uint32_t g_si128_next_id = 0;

static std::vector<float> g_si128_reservoir;
static size_t g_si128_reservoir_size = 5000;
static size_t g_si128_reservoir_count = 0;
static size_t g_si128_inserts_since_recluster = 0;
static size_t g_si128_recluster_interval = 10000;
static uint32_t g_si128_rng_state = 42;
static size_t g_si128_bootstrap_next_seg = 0;

static void si128_cleanup() {
    for (auto& seg : g_si128_segments) delete seg.index;
    g_si128_segments.clear();
    g_si128_id_map.clear();
    g_si128_next_id = 0;
    g_si128_reservoir_count = 0;
    g_si128_inserts_since_recluster = 0;
    g_si128_bootstrap_next_seg = 0;
}

static size_t si128_create_segment() {
    Seg128Info seg;
    seg.index = new QuantizedHNSW<128>(g_si128_hnsw_cfg);
    seg.centroid.assign(g_si128_dim, 0.0f);
    seg.count = 0;
    seg.local_to_global.reserve(g_si128_seg_capacity);
    g_si128_segments.push_back(std::move(seg));
    return g_si128_segments.size() - 1;
}

static float si128_l2(const float* a, const float* b) {
    float sum = 0.0f;
    for (size_t d = 0; d < g_si128_dim; d++) {
        float diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

static size_t si128_route(const float* vec) {
    if (g_si128_bootstrap_next_seg < g_si128_segments.size()) {
        size_t seg = g_si128_bootstrap_next_seg;
        g_si128_bootstrap_next_seg++;
        return seg;
    }
    float best_dist = 1e30f;
    size_t best_seg = 0;
    bool found = false;
    for (size_t i = 0; i < g_si128_segments.size(); i++) {
        if (g_si128_segments[i].count >= g_si128_seg_capacity) continue;
        float dist = si128_l2(vec, g_si128_segments[i].centroid.data());
        if (!found || dist < best_dist) {
            best_dist = dist;
            best_seg = i;
            found = true;
        }
    }
    if (found) return best_seg;
    return SIZE_MAX;
}

static void si128_update_centroid(size_t seg_idx, const float* vec) {
    auto& seg = g_si128_segments[seg_idx];
    float n = static_cast<float>(seg.count);
    float inv_n1 = 1.0f / (n + 1.0f);
    for (size_t d = 0; d < g_si128_dim; d++) {
        seg.centroid[d] = seg.centroid[d] * (n * inv_n1) + vec[d] * inv_n1;
    }
}

static uint32_t si128_rand() {
    g_si128_rng_state = g_si128_rng_state * 1664525u + 1013904223u;
    return g_si128_rng_state;
}

static void si128_add_to_reservoir(const float* vec) {
    if (g_si128_reservoir.empty()) {
        g_si128_reservoir.resize(g_si128_reservoir_size * g_si128_dim);
        g_si128_reservoir_count = 0;
    }
    size_t total_inserts = g_si128_inserts_since_recluster + 1;
    if (g_si128_reservoir_count < g_si128_reservoir_size) {
        size_t offset = g_si128_reservoir_count * g_si128_dim;
        for (size_t d = 0; d < g_si128_dim; d++) {
            g_si128_reservoir[offset + d] = vec[d];
        }
        g_si128_reservoir_count++;
    } else {
        size_t j = si128_rand() % total_inserts;
        if (j < g_si128_reservoir_size) {
            size_t offset = j * g_si128_dim;
            for (size_t d = 0; d < g_si128_dim; d++) {
                g_si128_reservoir[offset + d] = vec[d];
            }
        }
    }
}

static void si128_recluster() {
    if (g_si128_segments.empty() || g_si128_reservoir.empty()) return;
    size_t num_segments = g_si128_segments.size();
    size_t num_samples = g_si128_reservoir_count;
    if (num_samples < num_segments) return;
    if (num_samples == 0 || g_si128_dim == 0 || num_segments == 0) return;
    if (num_segments > num_samples) return;

    std::vector<int> assignments = pancake::clustering::KMeans::fit(
        g_si128_reservoir.data(), num_samples,
        static_cast<uint32_t>(g_si128_dim), num_segments,
        10, 42, false, pancake::clustering::DistanceMetric::L2
    );
    if (assignments.empty()) return;

    std::vector<std::vector<float>> new_centroids(num_segments,
        std::vector<float>(g_si128_dim, 0.0f));
    std::vector<size_t> counts(num_segments, 0);
    for (size_t i = 0; i < num_samples; i++) {
        int cluster = assignments[i];
        if (cluster < 0 || static_cast<size_t>(cluster) >= num_segments) continue;
        counts[cluster]++;
        for (size_t d = 0; d < g_si128_dim; d++) {
            new_centroids[cluster][d] += g_si128_reservoir[i * g_si128_dim + d];
        }
    }
    for (size_t seg_idx = 0; seg_idx < num_segments; seg_idx++) {
        if (counts[seg_idx] > 0) {
            float inv_count = 1.0f / counts[seg_idx];
            for (size_t d = 0; d < g_si128_dim; d++) {
                g_si128_segments[seg_idx].centroid[d] =
                    new_centroids[seg_idx][d] * inv_count;
            }
        }
    }
    g_si128_inserts_since_recluster = 0;
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

void pancake_delete(uint32_t h, uint32_t id) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return;
    g_handles[h].index->mark_delete(id);
}

void pancake_compact(uint32_t h) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return;
    g_handles[h].index->compact();
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
// Segmented Index 384D exports
// =============================================================================

int si_init(int seg_capacity, int max_segments, int budget,
            int M, int ef_construction, int ef_search) {
    si_cleanup();
    if (seg_capacity <= 0 || max_segments <= 0 || budget <= 0) return -1;
    g_si_seg_capacity = static_cast<size_t>(seg_capacity);
    g_si_max_segments = static_cast<size_t>(max_segments);
    g_si_budget = static_cast<size_t>(budget);
    g_si_hnsw_cfg.max_elements = g_si_seg_capacity;
    g_si_hnsw_cfg.M = static_cast<size_t>(M);
    g_si_hnsw_cfg.ef_construction = static_cast<size_t>(ef_construction);
    g_si_hnsw_cfg.ef_search = static_cast<size_t>(ef_search);
    si_create_segment();
    return 0;
}

uint32_t si_add(const float* vec) {
    if (g_si_segments.empty()) return 0xFFFFFFFF;
    size_t seg_idx = si_route(vec);
    if (seg_idx == SIZE_MAX) return 0xFFFFFFFF;
    auto& seg = g_si_segments[seg_idx];
    if (seg.count >= g_si_seg_capacity) return 0xFFFFFFFF;
    uint32_t local_id = seg.index->insert(vec);
    if (local_id == UINT32_MAX) return 0xFFFFFFFF;
    uint32_t global_id = g_si_next_id++;
    si_update_centroid(seg_idx, vec);
    seg.count++;
    seg.local_to_global.push_back(global_id);
    g_si_id_map[global_id] = {seg_idx, local_id};
    return global_id;
}

int si_query(const float* query, int k, uint64_t* ids, float* dists) {
    if (g_si_segments.empty()) return 0;
    size_t num_segs = g_si_segments.size();
    size_t budget = g_si_budget < num_segs ? g_si_budget : num_segs;
    std::vector<std::pair<float, size_t>> seg_dists(num_segs);
    for (size_t i = 0; i < num_segs; i++) {
        float dist = (g_si_segments[i].count > 0)
            ? si_l2(query, g_si_segments[i].centroid) : 1e30f;
        seg_dists[i] = {dist, i};
    }
    std::partial_sort(seg_dists.begin(), seg_dists.begin() + budget, seg_dists.end());
    std::vector<std::pair<uint64_t, float>> all_results;
    all_results.reserve(budget * k);
    for (size_t b = 0; b < budget; b++) {
        size_t seg_idx = seg_dists[b].second;
        if (g_si_segments[seg_idx].count == 0) continue;
        auto& seg = g_si_segments[seg_idx];
        auto res = seg.index->search(query, static_cast<size_t>(k));
        for (auto& r : res) {
            uint32_t local_id = static_cast<uint32_t>(r.first);
            uint32_t global_id = (local_id < seg.local_to_global.size())
                ? seg.local_to_global[local_id] : local_id;
            all_results.push_back({static_cast<uint64_t>(global_id), r.second});
        }
    }
    std::sort(all_results.begin(), all_results.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    int n = static_cast<int>(all_results.size());
    if (n > k) n = k;
    for (int i = 0; i < n; i++) {
        ids[i] = all_results[i].first;
        dists[i] = all_results[i].second;
    }
    return n;
}

void si_delete(uint32_t global_id) {
    auto it = g_si_id_map.find(global_id);
    if (it == g_si_id_map.end()) return;
    g_si_segments[it->second.seg_idx].index->mark_delete(it->second.local_id);
    g_si_id_map.erase(it);
}

size_t si_count() {
    size_t total = 0;
    for (auto& seg : g_si_segments) total += seg.index->count();
    return total;
}

size_t si_memory() {
    size_t total = 0;
    for (auto& seg : g_si_segments) total += seg.index->memory_bytes();
    total += g_si_id_map.size() * (sizeof(uint32_t) + sizeof(SiIdMapping) + 16);
    return total;
}

size_t si_segment_count() { return g_si_segments.size(); }

size_t si_ghost_count() {
    size_t total = 0;
    for (auto& seg : g_si_segments) total += seg.index->ghost_count();
    return total;
}

float si_ghost_ratio() {
    size_t total = si_count();
    size_t ghosts = si_ghost_count();
    return total > 0 ? static_cast<float>(ghosts) / static_cast<float>(total + ghosts) : 0.0f;
}

void si_compact() {
    for (auto& seg : g_si_segments) {
        if (seg.index->ghost_count() > 0) seg.index->compact();
    }
}

void si_set_budget(int budget) {
    if (budget > 0) g_si_budget = static_cast<size_t>(budget);
}

// =============================================================================
// Segmented Index 128D exports
// =============================================================================

int si128_init(int seg_capacity, int max_segments, int budget,
               int M, int ef_construction, int ef_search) {
    si128_cleanup();
    if (seg_capacity <= 0 || max_segments <= 0 || budget <= 0) return -1;
    g_si128_seg_capacity = static_cast<size_t>(seg_capacity);
    g_si128_max_segments = static_cast<size_t>(max_segments);
    g_si128_budget = static_cast<size_t>(budget);
    g_si128_hnsw_cfg.max_elements = g_si128_seg_capacity;
    g_si128_hnsw_cfg.M = static_cast<size_t>(M);
    g_si128_hnsw_cfg.ef_construction = static_cast<size_t>(ef_construction);
    g_si128_hnsw_cfg.ef_search = static_cast<size_t>(ef_search);
    for (size_t i = 0; i < g_si128_max_segments; i++) {
        si128_create_segment();
    }
    g_si128_bootstrap_next_seg = 0;
    return 0;
}

uint32_t si128_add(const float* vec) {
    if (g_si128_segments.empty()) return 0xFFFFFFFF;
    size_t seg_idx = si128_route(vec);
    if (seg_idx == SIZE_MAX) return 0xFFFFFFFF;
    auto& seg = g_si128_segments[seg_idx];
    if (seg.count >= g_si128_seg_capacity) return 0xFFFFFFFF;
    uint32_t local_id = seg.index->insert(vec);
    if (local_id == UINT32_MAX) return 0xFFFFFFFF;
    uint32_t global_id = g_si128_next_id++;
    si128_update_centroid(seg_idx, vec);
    seg.count++;
    seg.local_to_global.push_back(global_id);
    g_si128_id_map[global_id] = {seg_idx, local_id};
    g_si128_inserts_since_recluster++;
    if (g_si128_inserts_since_recluster >= g_si128_recluster_interval &&
        g_si128_segments.size() > 1) {
        si128_recluster();
    }
    return global_id;
}

int si128_query(const float* query, int k, uint64_t* ids, float* dists) {
    if (g_si128_segments.empty()) return 0;
    size_t num_segs = g_si128_segments.size();
    size_t budget = g_si128_budget < num_segs ? g_si128_budget : num_segs;
    std::vector<std::pair<float, size_t>> seg_dists(num_segs);
    for (size_t i = 0; i < num_segs; i++) {
        float dist = (g_si128_segments[i].count > 0)
            ? si128_l2(query, g_si128_segments[i].centroid.data()) : 1e30f;
        seg_dists[i] = {dist, i};
    }
    std::partial_sort(seg_dists.begin(), seg_dists.begin() + budget, seg_dists.end());
    std::vector<std::pair<uint64_t, float>> all_results;
    all_results.reserve(budget * k);
    for (size_t b = 0; b < budget; b++) {
        size_t seg_idx = seg_dists[b].second;
        if (g_si128_segments[seg_idx].count == 0) continue;
        auto& seg = g_si128_segments[seg_idx];
        auto res = seg.index->search(query, static_cast<size_t>(k));
        for (auto& r : res) {
            uint32_t local_id = static_cast<uint32_t>(r.first);
            uint32_t global_id = (local_id < seg.local_to_global.size())
                ? seg.local_to_global[local_id] : local_id;
            all_results.push_back({static_cast<uint64_t>(global_id), r.second});
        }
    }
    std::sort(all_results.begin(), all_results.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    int n = static_cast<int>(all_results.size());
    if (n > k) n = k;
    for (int i = 0; i < n; i++) {
        ids[i] = all_results[i].first;
        dists[i] = all_results[i].second;
    }
    return n;
}

void si128_delete(uint32_t global_id) {
    auto it = g_si128_id_map.find(global_id);
    if (it == g_si128_id_map.end()) return;
    g_si128_segments[it->second.seg_idx].index->mark_delete(it->second.local_id);
    g_si128_id_map.erase(it);
}

size_t si128_count() {
    size_t total = 0;
    for (auto& seg : g_si128_segments) total += seg.index->count();
    return total;
}

size_t si128_memory() {
    size_t total = 0;
    for (auto& seg : g_si128_segments) total += seg.index->memory_bytes();
    total += g_si128_id_map.size() * (sizeof(uint32_t) + sizeof(Si128IdMapping) + 16);
    return total;
}

size_t si128_segment_count() { return g_si128_segments.size(); }

size_t si128_ghost_count() {
    size_t total = 0;
    for (auto& seg : g_si128_segments) total += seg.index->ghost_count();
    return total;
}

void si128_compact() {
    for (auto& seg : g_si128_segments) {
        if (seg.index->ghost_count() > 0) seg.index->compact();
    }
}

void si128_set_budget(int budget) {
    if (budget > 0) g_si128_budget = static_cast<size_t>(budget);
}

int si128_get_vector_segment(uint32_t global_id) {
    auto it = g_si128_id_map.find(global_id);
    if (it == g_si128_id_map.end()) return -1;
    return static_cast<int>(it->second.seg_idx);
}

int si128_get_query_segments(const float* query, int* out_segments) {
    if (g_si128_segments.empty()) return 0;
    size_t num_segs = g_si128_segments.size();
    size_t budget = g_si128_budget < num_segs ? g_si128_budget : num_segs;
    std::vector<std::pair<float, size_t>> seg_dists(num_segs);
    for (size_t i = 0; i < num_segs; i++) {
        float dist = (g_si128_segments[i].count > 0)
            ? si128_l2(query, g_si128_segments[i].centroid.data()) : 1e30f;
        seg_dists[i] = {dist, i};
    }
    std::partial_sort(seg_dists.begin(), seg_dists.begin() + budget, seg_dists.end());
    for (size_t b = 0; b < budget; b++) {
        out_segments[b] = static_cast<int>(seg_dists[b].second);
    }
    return static_cast<int>(budget);
}

// =============================================================================
// Utility exports
// =============================================================================

void* emsc_malloc(size_t size) { return malloc(size); }
void emsc_free(void* ptr) { free(ptr); }

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

    // Clean up segmented indexes
    si_cleanup();
    si128_cleanup();
    g_si128_reservoir.clear();
    g_si128_reservoir.shrink_to_fit();

#if PANCAKE_HAS_EMBEDDING
    if (g_embedder) { delete g_embedder; g_embedder = nullptr; }
    g_embedding_buf.clear();
#endif
}

void shutdown_all() {
    pancake_shutdown_all();
}

} // extern "C"
