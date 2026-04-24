#include "float_hnsw.hpp"
#include "quantized_hnsw.hpp"
#include "int8_float_hnsw.hpp"
#include "kmeans.hpp"
#include "embedding_model.hpp"
#include <unordered_map>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <cstring>
#include <type_traits>

#if defined(__wasm_simd128__)
    #include <wasm_simd128.h>
#endif

using namespace pancake::wasm;

// =============================================================================
// WARNING:  THREAD SAFETY WARNING
// =============================================================================
// This code uses global mutable state and is NOT thread-safe.
// Safe for: single-threaded WASM, single Web Worker
// UNSAFE for: SharedArrayBuffer + Atomics, pthread-enabled WASM, multiple workers
// If you enable multithreading, wrap all operations in mutexes or redesign with
// thread-local storage.

// =============================================================================
// Original 384D HNSW engine (backward compatibility)
// Kept unchanged to maintain API stability for existing WASM deployments.
// New code should use p384f_ or quantized variants.
// =============================================================================

const size_t DIMENSIONS = 384;
static QuantizedHNSW<DIMENSIONS>* g_idx = nullptr;
static std::vector<uint8_t> g_export_buf;

extern "C" {

// =============================================================================
// Original 384D exports (backward compatible)
// =============================================================================

int pi(int max_elem) {
    if (g_idx) delete g_idx;
    QuantizedHNSWConfig cfg;
    cfg.max_elements = static_cast<size_t>(max_elem);
    cfg.M = 12;
    cfg.ef_construction = 150;
    cfg.ef_search = 250;
    g_idx = new QuantizedHNSW<DIMENSIONS>(cfg);
    return 0;
}

uint32_t pa(const float* v) {
    return g_idx ? g_idx->insert(v) : 0xFFFFFFFF;
}

int bulk_insert(const float* vectors, int n) {
    if (!g_idx) return 0;
    for (int idx = 0; idx < n; ++idx) {
        g_idx->insert(vectors + idx * DIMENSIONS);
    }
    return n;
}

int pq(const float* qv, int k, uint64_t* ids, float* dists) {
    if (!g_idx) return 0;
    auto res = g_idx->search(qv, static_cast<size_t>(k));
    for (size_t j = 0; j < res.size(); ++j) {
        ids[j] = (uint64_t)res[j].first;
        dists[j] = res[j].second;
    }
    return (int)res.size();
}

uint8_t* export_index(size_t* out_size) {
    if (!g_idx) return nullptr;
    g_export_buf = g_idx->serialize();
    *out_size = g_export_buf.size();
    return g_export_buf.data();
}

int import_index(const uint8_t* data, size_t data_size) {
    if (g_idx) { delete g_idx; g_idx = nullptr; }
    // Peek at count from header (offset 12: after magic, dims, version)
    size_t count = 100000;
    if (data_size >= 16) {
        uint32_t c;
        memcpy(&c, data + 12, 4);
        count = static_cast<size_t>(c);
    }
    size_t max_elem = std::max(static_cast<size_t>(count * 1.2), static_cast<size_t>(100000));
    QuantizedHNSWConfig cfg;
    cfg.max_elements = max_elem;
    cfg.M = 12;
    cfg.ef_construction = 150;
    cfg.ef_search = 250;
    g_idx = new QuantizedHNSW<DIMENSIONS>(cfg);
    if (!g_idx->deserialize(data, data_size)) {
        delete g_idx;
        g_idx = nullptr;
        return -1;
    }
    return 0;
}

size_t pc() { return g_idx ? g_idx->count() : 0; }
size_t pm() { return g_idx ? g_idx->memory_bytes() : 0; }

void pd(uint32_t id) {
    if (g_idx) g_idx->mark_delete(id);
}

size_t p_ghost_count() { return g_idx ? g_idx->ghost_count() : 0; }
float p_ghost_ratio() { return g_idx ? g_idx->ghost_ratio() : 0.0f; }

void p_compact() {
    if (g_idx) g_idx->compact();
}

// =============================================================================
// Segmented Index (384D) -- multiple QuantizedHNSW segments with centroid routing
// WHY SEGMENTED: Distributes vectors across smaller HNSW graphs to reduce
// construction latency (O(log n) -> O(log n/k)), improve cache locality, and
// enable embarrassingly parallel search across segments. Tradeoff: slightly
// lower recall if budget < num_segments, but much faster inserts at scale.
// =============================================================================

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
    //  Pre-allocate to avoid reallocation churn during insert
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

    // Find best non-full segment by centroid distance
    for (size_t i = 0; i < g_si_segments.size(); i++) {
        if (g_si_segments[i].count >= g_si_seg_capacity) continue;

        // For empty segments, use infinite distance (lowest priority)
        float dist = (g_si_segments[i].count == 0) ? 1e30f :
                     si_l2(vec, g_si_segments[i].centroid);

        if (!found || dist < best_dist) {
            best_dist = dist;
            best_seg = i;
            found = true;
        }
    }

    if (found) return best_seg;

    // All existing segments full - create new one if allowed
    if (g_si_segments.size() < g_si_max_segments) return si_create_segment();

    // WARNING:  All segments full! Return SIZE_MAX to signal capacity exceeded.
    // Caller must check and reject insert rather than silently overloading segment 0.
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
    if (seg_idx == SIZE_MAX) return 0xFFFFFFFF;  // All segments full
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
// Segmented Index (float32, runtime dim)
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

// K-means clustering state
static std::vector<float> g_si128_reservoir;  // Reservoir sample vectors for k-means
static size_t g_si128_reservoir_size = 5000;  // Max vectors to keep in reservoir
static size_t g_si128_reservoir_count = 0;    // Number of vectors currently in reservoir
static size_t g_si128_inserts_since_recluster = 0;
static size_t g_si128_recluster_interval = 10000;  // Recluster every N inserts
static uint32_t g_si128_rng_state = 42;  // Simple LCG RNG state
static size_t g_si128_bootstrap_next_seg = 0;  // Next segment to seed during bootstrap

static void si128_cleanup() {
    for (auto& seg : g_si128_segments) delete seg.index;
    g_si128_segments.clear();
    g_si128_id_map.clear();
    g_si128_next_id = 0;
    // Don't clear reservoir - keep the pre-allocated buffer
    g_si128_reservoir_count = 0;
    g_si128_inserts_since_recluster = 0;
    g_si128_bootstrap_next_seg = 0;
}

static size_t si128_create_segment() {
    Seg128Info seg;
    seg.index = new QuantizedHNSW<128>(g_si128_hnsw_cfg);
    seg.centroid.assign(g_si128_dim, 0.0f);
    seg.count = 0;
    //  Pre-allocate to avoid reallocation churn during insert
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
    // Bootstrap mode: distribute first N vectors round-robin to seed centroids.
    // WHY NEEDED: Without bootstrap, all vectors route to segment 0 (zero centroid),
    // creating massive imbalance. Round-robin ensures each segment gets initial data
    // for meaningful centroid initialization before switching to distance-based routing.
    if (g_si128_bootstrap_next_seg < g_si128_segments.size()) {
        size_t seg = g_si128_bootstrap_next_seg;
        g_si128_bootstrap_next_seg++;
        return seg;
    }

    // Normal mode: route based on centroid distance
    float best_dist = 1e30f;
    size_t best_seg = 0;
    bool found = false;

    for (size_t i = 0; i < g_si128_segments.size(); i++) {
        if (g_si128_segments[i].count >= g_si128_seg_capacity) continue;

        // Use actual centroid distance (all segments now have data)
        float dist = si128_l2(vec, g_si128_segments[i].centroid.data());

        if (!found || dist < best_dist) {
            best_dist = dist;
            best_seg = i;
            found = true;
        }
    }

    if (found) return best_seg;

    // All segments full!
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

// Simple LCG random number generator for reservoir sampling.
// WHY CUSTOM RNG: std::random_device is non-deterministic, std::mt19937 is
// heavy (2.5KB state), and we need fast, lightweight, reproducible sampling.
// LCG provides sufficient randomness for reservoir sampling at near-zero cost.
static uint32_t si128_rand() {
    g_si128_rng_state = g_si128_rng_state * 1664525u + 1013904223u;
    return g_si128_rng_state;
}

static void si128_add_to_reservoir(const float* vec) {
    // Lazy initialization: defer reservoir allocation until first insert to avoid
    // upfront memory cost for indexes that may never use reclustering (e.g., small
    // datasets or disabled k-means). Saves ~2.5MB for typical 5000-vector reservoir.
    if (g_si128_reservoir.empty()) {
        g_si128_reservoir.resize(g_si128_reservoir_size * g_si128_dim);
        g_si128_reservoir_count = 0;
    }

    size_t total_inserts = g_si128_inserts_since_recluster + 1;

    if (g_si128_reservoir_count < g_si128_reservoir_size) {
        // Reservoir not full yet - just append
        size_t offset = g_si128_reservoir_count * g_si128_dim;
        for (size_t d = 0; d < g_si128_dim; d++) {
            g_si128_reservoir[offset + d] = vec[d];
        }
        g_si128_reservoir_count++;
    } else {
        // Reservoir full - use reservoir sampling algorithm (Algorithm R).
        // WHY: Maintains uniform random sample as data streams in, ensuring every
        // vector has equal probability k/n of being in reservoir regardless of when
        // it arrived. Critical for unbiased centroid estimation in k-means.
        size_t j = si128_rand() % total_inserts;
        if (j < g_si128_reservoir_size) {
            // Replace vector j
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
    // Use actual count of samples in reservoir, not full size!
    size_t num_samples = g_si128_reservoir_count;

    if (num_samples < num_segments) return;  // Not enough samples
    if (num_samples == 0 || g_si128_dim == 0 || num_segments == 0) return;
    if (num_segments > num_samples) return;

    // Run k-means on reservoir samples
    std::vector<int> assignments = pancake::clustering::KMeans::fit(
        g_si128_reservoir.data(),
        num_samples,
        static_cast<uint32_t>(g_si128_dim),
        num_segments,
        10,   // max_iters
        42,   // seed
        false, // verbose
        pancake::clustering::DistanceMetric::L2
    );

    // Check if k-means failed (returns empty on error)
    if (assignments.empty()) return;

    // Compute new centroids from assignments
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

    // Average and update segment centroids
    for (size_t seg_idx = 0; seg_idx < num_segments; seg_idx++) {
        if (counts[seg_idx] > 0) {
            float inv_count = 1.0f / counts[seg_idx];
            for (size_t d = 0; d < g_si128_dim; d++) {
                g_si128_segments[seg_idx].centroid[d] =
                    new_centroids[seg_idx][d] * inv_count;
            }
        }
    }

    // Reset counter
    g_si128_inserts_since_recluster = 0;
}

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

    // Pre-create all segments to enable proper centroid-based routing
    for (size_t i = 0; i < g_si128_max_segments; i++) {
        si128_create_segment();
    }

    g_si128_bootstrap_next_seg = 0;
    return 0;
}

uint32_t si128_add(const float* vec) {
    if (g_si128_segments.empty()) return 0xFFFFFFFF;

    // Add to reservoir for k-means clustering
    // DISABLED: K-means has memory access issues in WASM, using running-mean centroids instead
    // si128_add_to_reservoir(vec);

    size_t seg_idx = si128_route(vec);
    if (seg_idx == SIZE_MAX) return 0xFFFFFFFF;  // All segments full
    auto& seg = g_si128_segments[seg_idx];
    if (seg.count >= g_si128_seg_capacity) return 0xFFFFFFFF;
    uint32_t local_id = seg.index->insert(vec);
    if (local_id == UINT32_MAX) return 0xFFFFFFFF;
    uint32_t global_id = g_si128_next_id++;
    si128_update_centroid(seg_idx, vec);
    seg.count++;
    seg.local_to_global.push_back(global_id);
    g_si128_id_map[global_id] = {seg_idx, local_id};

    // Trigger reclustering if needed
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

// Get which segment a vector is assigned to (-1 if not found)
int si128_get_vector_segment(uint32_t global_id) {
    auto it = g_si128_id_map.find(global_id);
    if (it == g_si128_id_map.end()) return -1;
    return static_cast<int>(it->second.seg_idx);
}

// Get which segments would be searched for a query (returns count, writes segment indices to out_segments)
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
// Canonical 128D Quantized INT8 index (default for Pancake WASM)
// WHY 128D + INT8: Sweet spot for OpenAI/Cohere/Voyage embeddings after Random
// Indexing compression (1536D->128D). Int8 quantization gives 4x memory savings
// with <2% recall drop. Targets Cloudflare Workers 128MB memory limit.
// =============================================================================

static QuantizedHNSW<128>* g_p128_idx = nullptr;

int p128_init(int max_elem, int M, int ef_construction, int ef_search) {
    if (g_p128_idx) { delete g_p128_idx; g_p128_idx = nullptr; }
    QuantizedHNSWConfig cfg;
    cfg.max_elements = static_cast<size_t>(max_elem);
    cfg.M = static_cast<size_t>(M);
    cfg.ef_construction = static_cast<size_t>(ef_construction);
    cfg.ef_search = static_cast<size_t>(ef_search);
    cfg.quantize = true;  // Row-wise affine INT8 quantization
    g_p128_idx = new QuantizedHNSW<128>(cfg);
    return 0;
}

uint32_t p128_add(const float* v) {
    return g_p128_idx ? g_p128_idx->insert(v) : 0xFFFFFFFF;
}

int p128_query(const float* qv, int k, uint64_t* ids, float* dists) {
    if (!g_p128_idx) return 0;
    auto res = g_p128_idx->search(qv, static_cast<size_t>(k));
    for (size_t j = 0; j < res.size(); ++j) {
        ids[j] = (uint64_t)res[j].first;
        dists[j] = res[j].second;
    }
    return (int)res.size();
}

size_t p128_count() { return g_p128_idx ? g_p128_idx->count() : 0; }
size_t p128_memory() { return g_p128_idx ? g_p128_idx->memory_bytes() : 0; }

// =============================================================================
// Int8 Quantized 256D index (NYTimes, compile-time SIMD)
// =============================================================================

static QuantizedHNSW<256>* g_p256_idx = nullptr;

int p256_init(int max_elem, int M, int ef_construction, int ef_search, int metric) {
    if (g_p256_idx) { delete g_p256_idx; g_p256_idx = nullptr; }
    QuantizedHNSWConfig cfg;
    cfg.max_elements = static_cast<size_t>(max_elem);
    cfg.M = static_cast<size_t>(M);
    cfg.ef_construction = static_cast<size_t>(ef_construction);
    cfg.ef_search = static_cast<size_t>(ef_search);
    cfg.quantize = true;
    cfg.metric = static_cast<DistanceMetric>(metric);  // 0=L2, 1=Cosine
    g_p256_idx = new QuantizedHNSW<256>(cfg);
    return 0;
}

uint32_t p256_add(const float* v) {
    return g_p256_idx ? g_p256_idx->insert(v) : 0xFFFFFFFF;
}

int p256_query(const float* qv, int k, uint64_t* ids, float* dists) {
    if (!g_p256_idx) return 0;
    auto res = g_p256_idx->search(qv, static_cast<size_t>(k));
    for (size_t j = 0; j < res.size(); ++j) {
        ids[j] = (uint64_t)res[j].first;
        dists[j] = res[j].second;
    }
    return (int)res.size();
}

size_t p256_count() { return g_p256_idx ? g_p256_idx->count() : 0; }
size_t p256_memory() { return g_p256_idx ? g_p256_idx->memory_bytes() : 0; }
int p256_bulk_insert(const float* vectors, int n) {
    return g_p256_idx ? g_p256_idx->bulk_insert(vectors, n) : 0;
}

// =============================================================================
// Int8 Quantized 1536D index (OpenAI Ada, compile-time SIMD)
// =============================================================================

static QuantizedHNSW<1536>* g_p1536_idx = nullptr;

int p1536_init(int max_elem, int M, int ef_construction, int ef_search, int metric) {
    if (g_p1536_idx) { delete g_p1536_idx; g_p1536_idx = nullptr; }
    QuantizedHNSWConfig cfg;
    cfg.max_elements = static_cast<size_t>(max_elem);
    cfg.M = static_cast<size_t>(M);
    cfg.ef_construction = static_cast<size_t>(ef_construction);
    cfg.ef_search = static_cast<size_t>(ef_search);
    cfg.quantize = true;
    cfg.metric = static_cast<DistanceMetric>(metric);  // 0=L2, 1=Cosine
    g_p1536_idx = new QuantizedHNSW<1536>(cfg);
    return 0;
}

uint32_t p1536_add(const float* v) {
    return g_p1536_idx ? g_p1536_idx->insert(v) : 0xFFFFFFFF;
}

int p1536_query(const float* qv, int k, uint64_t* ids, float* dists) {
    if (!g_p1536_idx) return 0;
    auto res = g_p1536_idx->search(qv, static_cast<size_t>(k));
    for (size_t j = 0; j < res.size(); ++j) {
        ids[j] = (uint64_t)res[j].first;
        dists[j] = res[j].second;
    }
    return (int)res.size();
}

size_t p1536_count() { return g_p1536_idx ? g_p1536_idx->count() : 0; }
size_t p1536_memory() { return g_p1536_idx ? g_p1536_idx->memory_bytes() : 0; }
int p1536_bulk_insert(const float* vectors, int n) {
    return g_p1536_idx ? g_p1536_idx->bulk_insert(vectors, n) : 0;
}

static std::vector<uint8_t> g_p1536_export_buf;

uint8_t* p1536_export_index(size_t* out_size) {
    if (!g_p1536_idx) return nullptr;
    g_p1536_export_buf = g_p1536_idx->serialize();
    *out_size = g_p1536_export_buf.size();
    return g_p1536_export_buf.data();
}

int p1536_import_index(const uint8_t* data, size_t data_size) {
    if (g_p1536_idx) { delete g_p1536_idx; g_p1536_idx = nullptr; }
    size_t count = 1000;
    if (data_size >= 16) {
        uint32_t c; memcpy(&c, data + 12, 4);
        count = static_cast<size_t>(c);
    }
    size_t max_elem = std::max(static_cast<size_t>(count * 1.2), static_cast<size_t>(1000));
    QuantizedHNSWConfig cfg;
    cfg.max_elements = max_elem;
    cfg.M = 8;
    cfg.ef_construction = 100;
    cfg.ef_search = 50;
    cfg.metric = DistanceMetric::Cosine;
    cfg.quantize = true;
    g_p1536_idx = new QuantizedHNSW<1536>(cfg);
    if (!g_p1536_idx->deserialize(data, data_size)) {
        delete g_p1536_idx;
        g_p1536_idx = nullptr;
        return -1;
    }
    return 0;
}

// =============================================================================
// Float32 128D index (optional, for comparison)
// =============================================================================

static FloatHNSW* g_p128f_idx = nullptr;

int p128f_init(int max_elem, int M, int ef_construction, int ef_search) {
    if (g_p128f_idx) { delete g_p128f_idx; g_p128f_idx = nullptr; }
    FloatHNSWConfig cfg;
    cfg.max_elements = static_cast<size_t>(max_elem);
    cfg.M = static_cast<size_t>(M);
    cfg.ef_construction = static_cast<size_t>(ef_construction);
    cfg.ef_search = static_cast<size_t>(ef_search);
    g_p128f_idx = new FloatHNSW(128, cfg);
    return 0;
}

uint32_t p128f_add(const float* v) {
    return g_p128f_idx ? g_p128f_idx->insert(v) : 0xFFFFFFFF;
}

int p128f_query(const float* qv, int k, uint64_t* ids, float* dists) {
    if (!g_p128f_idx) return 0;
    auto res = g_p128f_idx->search(qv, static_cast<size_t>(k));
    for (size_t j = 0; j < res.size(); ++j) {
        ids[j] = (uint64_t)res[j].first;
        dists[j] = res[j].second;
    }
    return (int)res.size();
}

size_t p128f_count() { return g_p128f_idx ? g_p128f_idx->count() : 0; }
size_t p128f_memory() { return g_p128f_idx ? g_p128f_idx->memory_bytes() : 0; }

int p128f_get_neighbors(uint32_t id, uint32_t* out_buf) {
    if (!g_p128f_idx || id >= g_p128f_idx->count()) return 0;
    const auto& nbrs = g_p128f_idx->get_neighbors(id);
    for (size_t i = 0; i < nbrs.size(); i++) out_buf[i] = nbrs[i];
    return static_cast<int>(nbrs.size());
}

// =============================================================================
// Float32 384D index (for semantic embeddings like all-MiniLM-L6-v2)
// =============================================================================

static FloatHNSW* g_p384f_idx = nullptr;

int p384f_init(int max_elem, int M, int ef_construction, int ef_search) {
    if (g_p384f_idx) { delete g_p384f_idx; g_p384f_idx = nullptr; }
    FloatHNSWConfig cfg;
    cfg.max_elements = static_cast<size_t>(max_elem);
    cfg.M = static_cast<size_t>(M);
    cfg.ef_construction = static_cast<size_t>(ef_construction);
    cfg.ef_search = static_cast<size_t>(ef_search);
    g_p384f_idx = new FloatHNSW(384, cfg);
    return 0;
}

uint32_t p384f_add(const float* v) {
    return g_p384f_idx ? g_p384f_idx->insert(v) : 0xFFFFFFFF;
}

int p384f_query(const float* qv, int k, uint64_t* ids, float* dists) {
    if (!g_p384f_idx) return 0;
    auto res = g_p384f_idx->search(qv, static_cast<size_t>(k));
    for (size_t j = 0; j < res.size(); ++j) {
        ids[j] = (uint64_t)res[j].first;
        dists[j] = res[j].second;
    }
    return (int)res.size();
}

size_t p384f_count() { return g_p384f_idx ? g_p384f_idx->count() : 0; }
size_t p384f_memory() { return g_p384f_idx ? g_p384f_idx->memory_bytes() : 0; }

int p384f_get_neighbors(uint32_t id, uint32_t* out_buf) {
    if (!g_p384f_idx || id >= g_p384f_idx->count()) return 0;
    const auto& nbrs = g_p384f_idx->get_neighbors(id);
    for (size_t i = 0; i < nbrs.size(); i++) out_buf[i] = nbrs[i];
    return static_cast<int>(nbrs.size());
}

// =============================================================================
// 256-D Float32 HNSW (NYTimes / general 256D embeddings)
// =============================================================================

static FloatHNSW* g_p256f_idx = nullptr;

// metric: 0 = L2, 1 = cosine/angular
int p256f_init(int max_elem, int M, int ef_construction, int ef_search, int metric) {
    if (g_p256f_idx) { delete g_p256f_idx; g_p256f_idx = nullptr; }
    FloatHNSWConfig cfg;
    cfg.max_elements = static_cast<size_t>(max_elem);
    cfg.M = static_cast<size_t>(M);
    cfg.ef_construction = static_cast<size_t>(ef_construction);
    cfg.ef_search = static_cast<size_t>(ef_search);
    cfg.metric = (metric == 1) ? DistanceMetric::Cosine : DistanceMetric::L2;
    g_p256f_idx = new FloatHNSW(256, cfg);
    return 0;
}

uint32_t p256f_add(const float* v) {
    return g_p256f_idx ? g_p256f_idx->insert(v) : 0xFFFFFFFF;
}

int p256f_query(const float* qv, int k, uint64_t* ids, float* dists) {
    if (!g_p256f_idx) return 0;
    auto res = g_p256f_idx->search(qv, static_cast<size_t>(k));
    for (size_t j = 0; j < res.size(); ++j) {
        ids[j] = (uint64_t)res[j].first;
        dists[j] = res[j].second;
    }
    return (int)res.size();
}

size_t p256f_count() { return g_p256f_idx ? g_p256f_idx->count() : 0; }
size_t p256f_memory() { return g_p256f_idx ? g_p256f_idx->memory_bytes() : 0; }

// Naive (no heuristic) float32 index for comparison
static FloatHNSW* g_naive_idx = nullptr;

int naive_init(int max_elem, int M, int ef_construction, int ef_search) {
    if (g_naive_idx) { delete g_naive_idx; g_naive_idx = nullptr; }
    FloatHNSWConfig cfg;
    cfg.max_elements = static_cast<size_t>(max_elem);
    cfg.M = static_cast<size_t>(M);
    cfg.ef_construction = static_cast<size_t>(ef_construction);
    cfg.ef_search = static_cast<size_t>(ef_search);
    cfg.use_heuristic = false;
    g_naive_idx = new FloatHNSW(128, cfg);
    return 0;
}

uint32_t naive_add(const float* v) {
    return g_naive_idx ? g_naive_idx->insert(v) : 0xFFFFFFFF;
}

int naive_query(const float* qv, int k, uint64_t* ids, float* dists) {
    if (!g_naive_idx) return 0;
    auto res = g_naive_idx->search(qv, static_cast<size_t>(k));
    for (size_t j = 0; j < res.size(); ++j) {
        ids[j] = (uint64_t)res[j].first;
        dists[j] = res[j].second;
    }
    return (int)res.size();
}

size_t naive_count() { return g_naive_idx ? g_naive_idx->count() : 0; }

int naive_get_neighbors(uint32_t id, uint32_t* out_buf) {
    if (!g_naive_idx || id >= g_naive_idx->count()) return 0;
    const auto& nbrs = g_naive_idx->get_neighbors(id);
    for (size_t i = 0; i < nbrs.size(); i++) out_buf[i] = nbrs[i];
    return static_cast<int>(nbrs.size());
}

// =============================================================================
// Backward-compatible short-name wrappers for demos (delegate to 384D pi/pa/pq)
// =============================================================================

int i(int max_elem) {
    return pi(max_elem);
}

uint32_t a(const float* v) {
    return pa(v);
}

int q(const float* qv, int k, uint64_t* ids, float* dists) {
    return pq(qv, k, ids, dists);
}

size_t c() {
    return pc();
}

size_t m() {
    return pm();
}

void d(uint32_t id) {
    pd(id);
}

size_t ghost_count() {
    return p_ghost_count();
}

float ghost_ratio() {
    return p_ghost_ratio();
}

void compact() {
    p_compact();
}

// Explicit malloc/free wrappers (Emscripten's malloc wasn't being exported properly)
void* emsc_malloc(size_t size) {
    return malloc(size);
}

void emsc_free(void* ptr) {
    free(ptr);
}

// =============================================================================
// Generic FloatHNSW (arbitrary dimensions, for high-D benchmarking)
// =============================================================================

static FloatHNSW* g_float_idx = nullptr;
static size_t g_float_dim = 0;
static std::vector<uint8_t> g_float_export_buf;

int float_init(int dim, int max_elem, int M, int ef_construction, int ef_search, int metric) {
    if (g_float_idx) { delete g_float_idx; g_float_idx = nullptr; }
    g_float_dim = static_cast<size_t>(dim);

    FloatHNSWConfig cfg;
    cfg.max_elements = static_cast<size_t>(max_elem);
    cfg.M = static_cast<size_t>(M);
    cfg.ef_construction = static_cast<size_t>(ef_construction);
    cfg.ef_search = static_cast<size_t>(ef_search);
    cfg.metric = (metric == 1) ? DistanceMetric::Cosine : DistanceMetric::L2;

    g_float_idx = new FloatHNSW(g_float_dim, cfg);
    return 0;
}

uint32_t float_add(const float* vec) {
    return g_float_idx ? g_float_idx->insert(vec) : 0xFFFFFFFF;
}

int float_query(const float* qv, int k, uint64_t* ids, float* dists) {
    if (!g_float_idx) return 0;
    auto res = g_float_idx->search(qv, static_cast<size_t>(k));
    for (size_t j = 0; j < res.size(); ++j) {
        ids[j] = (uint64_t)res[j].first;
        dists[j] = res[j].second;
    }
    return (int)res.size();
}

size_t float_count() { return g_float_idx ? g_float_idx->count() : 0; }
size_t float_memory() { return g_float_idx ? g_float_idx->memory_bytes() : 0; }

void float_delete(uint32_t id) {
    if (g_float_idx) g_float_idx->mark_delete(id);
}

size_t float_ghost_count() { return g_float_idx ? g_float_idx->ghost_count() : 0; }
float float_ghost_ratio() { return g_float_idx ? g_float_idx->ghost_ratio() : 0.0f; }

void float_compact() {
    if (g_float_idx) g_float_idx->compact();
}

uint8_t* float_export_index(size_t* out_size) {
    if (!g_float_idx) return nullptr;
    g_float_export_buf = g_float_idx->serialize();
    *out_size = g_float_export_buf.size();
    return g_float_export_buf.data();
}

int float_import_index(const uint8_t* data, size_t data_size) {
    if (g_float_dim == 0) return -1;
    if (g_float_idx) { delete g_float_idx; g_float_idx = nullptr; }
    // Peek at count from header (offset 8: after magic, dims)
    size_t count = 100000;
    if (data_size >= 12) {
        uint32_t c;
        memcpy(&c, data + 8, 4);
        count = static_cast<size_t>(c);
    }
    size_t max_elem = std::max(static_cast<size_t>(count * 1.2), static_cast<size_t>(100000));
    FloatHNSWConfig cfg;
    cfg.max_elements = max_elem;
    g_float_idx = new FloatHNSW(g_float_dim, cfg);
    if (!g_float_idx->deserialize(data, data_size)) {
        delete g_float_idx;
        g_float_idx = nullptr;
        return -1;
    }
    return 0;
}

void float_set_ef(int ef) {
    if (ef <= 0) return;
    if (g_float_idx) g_float_idx->set_ef(static_cast<size_t>(ef));
}

// =============================================================================
// Embedding Model Integration (384-d transformer)
// =============================================================================

static embedding::EmbeddingModel* g_embedder = nullptr;
static std::vector<float> g_embedding_buf;

// Initialize the embedding model
int emb_init(int vocab_size) {
    if (g_embedder) { delete g_embedder; g_embedder = nullptr; }
    g_embedder = new embedding::EmbeddingModel(static_cast<size_t>(vocab_size));
    g_embedding_buf.clear();
    return 0;
}

// Generate embedding for a text (passed as null-terminated string)
const float* emb_encode(const char* text, size_t* out_size) {
    if (!g_embedder || !text) return nullptr;

    std::string str(text);
    g_embedding_buf = g_embedder->encode(str);

    if (out_size) *out_size = g_embedding_buf.size();
    return g_embedding_buf.data();
}

// Generate embeddings for multiple texts
// texts: array of null-terminated strings
// count: number of texts
// out_buffer: pre-allocated buffer for embeddings (count * 384 floats)
int emb_encode_batch(const char** texts, int count, float* out_buffer) {
    if (!g_embedder || !texts || !out_buffer || count <= 0) return 0;

    for (int i = 0; i < count; ++i) {
        if (!texts[i]) continue;

        std::string str(texts[i]);
        auto embedding = g_embedder->encode(str);

        // Copy to output buffer
        std::memcpy(out_buffer + i * 384, embedding.data(), 384 * sizeof(float));
    }

    return count;
}

// Get embedding dimension
int emb_dimension() {
    return embedding::EmbeddingModel::D_MODEL;
}

// Free the embedding model
void emb_free() {
    if (g_embedder) {
        delete g_embedder;
        g_embedder = nullptr;
    }
    g_embedding_buf.clear();
}

// Combined: embed text and add to 384D index
uint32_t emb_add(const char* text) {
    if (!g_embedder || !g_idx || !text) return 0xFFFFFFFF;

    std::string str(text);
    auto embedding = g_embedder->encode(str);

    return g_idx->insert(embedding.data());
}

// Combined: embed query and search 384D index
int emb_search(const char* query, int k, uint64_t* ids, float* dists) {
    if (!g_embedder || !g_idx || !query) return 0;

    std::string str(query);
    auto embedding = g_embedder->encode(str);

    auto res = g_idx->search(embedding.data(), static_cast<size_t>(k));
    for (size_t j = 0; j < res.size(); ++j) {
        ids[j] = (uint64_t)res[j].first;
        dists[j] = res[j].second;
    }
    return (int)res.size();
}

// =============================================================================
// Global cleanup (addresses memory leak at module unload)
// =============================================================================

// =============================================================================
// Dense Matrix Multiplication (WASM-accelerated)
// WHY: JavaScript dense matmul is slow (~60 GFLOPS). WASM with SIMD achieves
// 100-150 GFLOPS, giving 2-3x speedup for embedding models.
// =============================================================================

// Dense matmul: output = matrix * vec + bias
// matrix: row-major, rows x cols
// vec: cols elements
// bias: rows elements
// output: rows elements
void dense_matmul(const float* matrix, const float* vec, const float* bias, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        const float* row = matrix + i * cols;

#if defined(__wasm_simd128__)
        // WASM SIMD: 4-way float32 FMA
        v128_t acc = wasm_f32x4_splat(0.0f);
        int j = 0;
        for (; j + 4 <= cols; j += 4) {
            v128_t m = wasm_v128_load(row + j);
            v128_t v = wasm_v128_load(vec + j);
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(m, v));
        }
        // Horizontal sum
        sum = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) +
              wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
        // Scalar tail
        for (; j < cols; j++) {
            sum += row[j] * vec[j];
        }
#else
        // Scalar fallback
        for (int j = 0; j < cols; j++) {
            sum += row[j] * vec[j];
        }
#endif
        output[i] = sum + bias[i];
    }
}

// Sparse matmul: output = matrix * sparse_vec + bias
// matrix: row-major, rows x cols
// indices: non-zero column indices (length = nnz)
// values: non-zero values (length = nnz)
// bias: rows elements
// output: rows elements
void sparse_matmul(const float* matrix, const int* indices, const float* values, int nnz,
                   const float* bias, float* output, int rows, int cols) {
    // Initialize with bias
    for (int i = 0; i < rows; i++) {
        output[i] = bias[i];
    }

    // Sparse multiply: only touch columns with non-zero entries
    for (int k = 0; k < nnz; k++) {
        int j = indices[k];
        float val = values[k];
        const float* col = matrix + j;  // Column-major access for this operation

        for (int i = 0; i < rows; i++) {
            output[i] += matrix[i * cols + j] * val;
        }
    }
}

// L2 normalize: vec /= ||vec||_2 (in-place)
void normalize(float* vec, int dim) {
    float norm_sq = 0.0f;

#if defined(__wasm_simd128__)
    // SIMD dot product
    v128_t acc = wasm_f32x4_splat(0.0f);
    int i = 0;
    for (; i + 4 <= dim; i += 4) {
        v128_t v = wasm_v128_load(vec + i);
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(v, v));
    }
    norm_sq = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) +
              wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
    // Scalar tail
    for (; i < dim; i++) {
        norm_sq += vec[i] * vec[i];
    }
#else
    // Scalar fallback
    for (int i = 0; i < dim; i++) {
        norm_sq += vec[i] * vec[i];
    }
#endif

    if (norm_sq > 0.0f) {
        float norm = sqrtf(norm_sq);
        float inv_norm = 1.0f / norm;

#if defined(__wasm_simd128__)
        v128_t inv = wasm_f32x4_splat(inv_norm);
        int i = 0;
        for (; i + 4 <= dim; i += 4) {
            v128_t v = wasm_v128_load(vec + i);
            wasm_v128_store(vec + i, wasm_f32x4_mul(v, inv));
        }
        // Scalar tail
        for (; i < dim; i++) {
            vec[i] *= inv_norm;
        }
#else
        // Scalar fallback
        for (int i = 0; i < dim; i++) {
            vec[i] *= inv_norm;
        }
#endif
    }
}

// =============================================================================
// Int8FloatHNSW exports (runtime-dimension int8-quantized HNSW)
// WHY: 4x memory savings vs float32 HNSW with minimal recall loss (~1-2%)
// Stores vectors as row-wise affine uint8: per-vector scale + offset + uint8[dims]
// Search uses asymmetric distance: float32 query vs dequantized int8 database
// =============================================================================

enum class I8Backend { None = 0, Template384, Template1536, Runtime };

static Int8FloatHNSW* g_i8_idx = nullptr;
static size_t g_i8_dim = 0;
static std::vector<uint8_t> g_i8_export_buf;
static I8Backend g_i8_backend = I8Backend::None;
static QuantizedHNSW<384>* g_i8_template_384 = nullptr;
static QuantizedHNSW<1536>* g_i8_template_1536 = nullptr;

// dim = vector dimensionality
// max_elem = capacity
// metric: 0 = L2, 1 = cosine
// M = connectivity (0 = default 32)
// ef_c = construction ef (0 = default 200)
// ef_s = search ef (0 = default 128)
int i8_init(int dim, int max_elem, int metric, int M_param, int ef_c_param, int ef_s_param) {
    if (g_i8_idx) { delete g_i8_idx; g_i8_idx = nullptr; }
    if (g_i8_template_384) { delete g_i8_template_384; g_i8_template_384 = nullptr; }
    if (g_i8_template_1536) { delete g_i8_template_1536; g_i8_template_1536 = nullptr; }
    g_i8_dim = 0;
    g_i8_backend = I8Backend::None;

    if (dim <= 0 || max_elem <= 0) return -1;

    g_i8_dim = static_cast<size_t>(dim);
    bool use_cosine = (metric == 1);

    auto build_cfg = [&](QuantizedHNSWConfig& cfg) {
        cfg.max_elements = static_cast<size_t>(max_elem);
        cfg.M = (M_param > 0) ? static_cast<size_t>(M_param) : 32;
        cfg.ef_construction = (ef_c_param > 0) ? static_cast<size_t>(ef_c_param) : 200;
        cfg.ef_search = (ef_s_param > 0) ? static_cast<size_t>(ef_s_param) : 128;
        cfg.metric = use_cosine ? DistanceMetric::Cosine : DistanceMetric::L2;
        cfg.quantize = true;
    };

    if (g_i8_dim == 384) {
        g_i8_backend = I8Backend::Template384;
        QuantizedHNSWConfig cfg;
        build_cfg(cfg);
        g_i8_template_384 = new QuantizedHNSW<384>(cfg);
        g_i8_dim = 384;
        return 0;
    }
    if (g_i8_dim == 1536) {
        g_i8_backend = I8Backend::Template1536;
        QuantizedHNSWConfig cfg;
        build_cfg(cfg);
        g_i8_template_1536 = new QuantizedHNSW<1536>(cfg);
        g_i8_dim = 1536;
        return 0;
    }

    g_i8_backend = I8Backend::Runtime;
    Int8FloatHNSWConfig cfg;
    cfg.max_elements = static_cast<size_t>(max_elem);
    cfg.M = (M_param > 0) ? static_cast<size_t>(M_param) : 32;
    cfg.ef_construction = (ef_c_param > 0) ? static_cast<size_t>(ef_c_param) : 200;
    cfg.ef_search = (ef_s_param > 0) ? static_cast<size_t>(ef_s_param) : 128;
    cfg.metric = use_cosine ? DistanceMetric::Cosine : DistanceMetric::L2;
    cfg.use_heuristic = true;

    g_i8_idx = new Int8FloatHNSW(g_i8_dim, cfg);
    return 0;
}

static QuantizedHNSW<384>* i8_get_template_384() { return g_i8_template_384; }
static QuantizedHNSW<1536>* i8_get_template_1536() { return g_i8_template_1536; }

uint32_t i8_add(const float* vec) {
    switch (g_i8_backend) {
        case I8Backend::Template384:  return i8_get_template_384()  ? i8_get_template_384()->insert(vec)  : 0xFFFFFFFF;
        case I8Backend::Template1536: return i8_get_template_1536() ? i8_get_template_1536()->insert(vec) : 0xFFFFFFFF;
        case I8Backend::Runtime:      return g_i8_idx ? g_i8_idx->insert(vec) : 0xFFFFFFFF;
        default: return 0xFFFFFFFF;
    }
}

int i8_bulk_insert(const float* vectors, int n) {
    switch (g_i8_backend) {
        case I8Backend::Template384:
            return i8_get_template_384() ? i8_get_template_384()->bulk_insert(vectors, n) : 0;
        case I8Backend::Template1536:
            return i8_get_template_1536() ? i8_get_template_1536()->bulk_insert(vectors, n) : 0;
        case I8Backend::Runtime:
            return g_i8_idx ? g_i8_idx->bulk_insert(vectors, n) : 0;
        default:
            return 0;
    }
}

int i8_query(const float* qv, int k, uint64_t* ids, float* dists) {
    auto fill_results = [&](const auto& res) {
        for (size_t j = 0; j < res.size(); ++j) {
            ids[j] = static_cast<uint64_t>(res[j].first);
            dists[j] = res[j].second;
        }
        return static_cast<int>(res.size());
    };
    switch (g_i8_backend) {
        case I8Backend::Template384:
            if (!i8_get_template_384()) return 0;
            return fill_results(i8_get_template_384()->search(qv, static_cast<size_t>(k)));
        case I8Backend::Template1536:
            if (!i8_get_template_1536()) return 0;
            return fill_results(i8_get_template_1536()->search(qv, static_cast<size_t>(k)));
        case I8Backend::Runtime:
            if (!g_i8_idx) return 0;
            return fill_results(g_i8_idx->search(qv, static_cast<size_t>(k)));
        default:
            return 0;
    }
}

size_t i8_count() {
    switch (g_i8_backend) {
        case I8Backend::Template384:  return i8_get_template_384()  ? i8_get_template_384()->count()  : 0;
        case I8Backend::Template1536: return i8_get_template_1536() ? i8_get_template_1536()->count() : 0;
        case I8Backend::Runtime:      return g_i8_idx ? g_i8_idx->count() : 0;
        default: return 0;
    }
}

size_t i8_memory() {
    switch (g_i8_backend) {
        case I8Backend::Template384:  return i8_get_template_384()  ? i8_get_template_384()->memory_bytes()  : 0;
        case I8Backend::Template1536: return i8_get_template_1536() ? i8_get_template_1536()->memory_bytes() : 0;
        case I8Backend::Runtime:      return g_i8_idx ? g_i8_idx->memory_bytes() : 0;
        default: return 0;
    }
}

void i8_delete(uint32_t id) {
    switch (g_i8_backend) {
        case I8Backend::Template384:  if (i8_get_template_384())  i8_get_template_384()->mark_delete(id);  break;
        case I8Backend::Template1536: if (i8_get_template_1536()) i8_get_template_1536()->mark_delete(id); break;
        case I8Backend::Runtime:      if (g_i8_idx) g_i8_idx->mark_delete(id); break;
        default: break;
    }
}

size_t i8_ghost_count() {
    switch (g_i8_backend) {
        case I8Backend::Template384:  return i8_get_template_384()  ? i8_get_template_384()->ghost_count()  : 0;
        case I8Backend::Template1536: return i8_get_template_1536() ? i8_get_template_1536()->ghost_count() : 0;
        case I8Backend::Runtime:      return g_i8_idx ? g_i8_idx->ghost_count() : 0;
        default: return 0;
    }
}

float i8_ghost_ratio() {
    switch (g_i8_backend) {
        case I8Backend::Template384:  return i8_get_template_384()  ? i8_get_template_384()->ghost_ratio()  : 0.0f;
        case I8Backend::Template1536: return i8_get_template_1536() ? i8_get_template_1536()->ghost_ratio() : 0.0f;
        case I8Backend::Runtime:      return g_i8_idx ? g_i8_idx->ghost_ratio() : 0.0f;
        default: return 0.0f;
    }
}

void i8_compact() {
    switch (g_i8_backend) {
        case I8Backend::Template384:  if (i8_get_template_384())  i8_get_template_384()->compact();  break;
        case I8Backend::Template1536: if (i8_get_template_1536()) i8_get_template_1536()->compact(); break;
        case I8Backend::Runtime:      if (g_i8_idx) g_i8_idx->compact(); break;
        default: break;
    }
}

// =============================================================================
// i8_set_ef -- set ef_search at runtime without reinitializing the index
// =============================================================================
void i8_set_ef(int ef) {
    if (ef <= 0) return;
    switch (g_i8_backend) {
        case I8Backend::Template384:
            if (i8_get_template_384()) i8_get_template_384()->set_ef_search(static_cast<size_t>(ef));
            break;
        case I8Backend::Template1536:
            if (i8_get_template_1536()) i8_get_template_1536()->set_ef_search(static_cast<size_t>(ef));
            break;
        case I8Backend::Runtime:
            if (g_i8_idx) g_i8_idx->set_ef_search(static_cast<size_t>(ef));
            break;
        default: break;
    }
}

uint8_t* i8_export_index(size_t* out_size) {
    switch (g_i8_backend) {
        case I8Backend::Template384:
            if (!i8_get_template_384()) return nullptr;
            g_i8_export_buf = i8_get_template_384()->serialize();
            break;
        case I8Backend::Template1536:
            if (!i8_get_template_1536()) return nullptr;
            g_i8_export_buf = i8_get_template_1536()->serialize();
            break;
        case I8Backend::Runtime:
            if (!g_i8_idx) return nullptr;
            g_i8_export_buf = g_i8_idx->serialize();
            break;
        default:
            return nullptr;
    }
    *out_size = g_i8_export_buf.size();
    return g_i8_export_buf.data();
}

int i8_import_index(const uint8_t* data, size_t data_size, int dim) {
    if (dim <= 0) return -1;
    if (g_i8_idx) { delete g_i8_idx; g_i8_idx = nullptr; }
    if (g_i8_template_384) { delete g_i8_template_384; g_i8_template_384 = nullptr; }
    if (g_i8_template_1536) { delete g_i8_template_1536; g_i8_template_1536 = nullptr; }
    g_i8_dim = static_cast<size_t>(dim);

    // Peek at count from header (offset 8: after magic, dims)
    size_t count = 100000;
    if (data_size >= 12) {
        uint32_t c;
        memcpy(&c, data + 8, 4);
        count = static_cast<size_t>(c);
    }
    size_t max_elem = std::max(static_cast<size_t>(count * 1.2), static_cast<size_t>(100000));

    auto setup_template = [&](auto*& ptr, auto backend_tag) {
        QuantizedHNSWConfig cfg;
        cfg.max_elements = max_elem;
        cfg.M = 8;
        cfg.ef_construction = 100;
        cfg.ef_search = 50;
        cfg.metric = DistanceMetric::Cosine;
        cfg.quantize = true;
        ptr = new std::remove_reference_t<decltype(*ptr)>(cfg);
        g_i8_backend = backend_tag;
    };

    if (g_i8_dim == 384) {
        setup_template(g_i8_template_384, I8Backend::Template384);
        if (!g_i8_template_384->deserialize(data, data_size)) {
            delete g_i8_template_384;
            g_i8_template_384 = nullptr;
            g_i8_backend = I8Backend::None;
            g_i8_dim = 0;
            return -1;
        }
        g_i8_dim = 384;
        return 0;
    }
    if (g_i8_dim == 1536) {
        setup_template(g_i8_template_1536, I8Backend::Template1536);
        if (!g_i8_template_1536->deserialize(data, data_size)) {
            delete g_i8_template_1536;
            g_i8_template_1536 = nullptr;
            g_i8_backend = I8Backend::None;
            g_i8_dim = 0;
            return -1;
        }
        g_i8_dim = 1536;
        return 0;
    }

    g_i8_backend = I8Backend::Runtime;
    Int8FloatHNSWConfig cfg;
    cfg.max_elements = max_elem;
    g_i8_idx = new Int8FloatHNSW(g_i8_dim, cfg);
    if (!g_i8_idx->deserialize(data, data_size)) {
        delete g_i8_idx;
        g_i8_idx = nullptr;
        g_i8_backend = I8Backend::None;
        g_i8_dim = 0;
        return -1;
    }
    return 0;
}

void shutdown_all() {
    // Clean up 384D quantized index
    if (g_idx) { delete g_idx; g_idx = nullptr; }

    // Clean up generic float index
    if (g_float_idx) { delete g_float_idx; g_float_idx = nullptr; }
    g_float_dim = 0;
    g_float_export_buf.clear();

    // Clean up int8 HNSW
    if (g_i8_idx) { delete g_i8_idx; g_i8_idx = nullptr; }
    if (g_i8_template_384) { delete g_i8_template_384; g_i8_template_384 = nullptr; }
    if (g_i8_template_1536) { delete g_i8_template_1536; g_i8_template_1536 = nullptr; }
    g_i8_backend = I8Backend::None;
    g_i8_dim = 0;
    g_i8_export_buf.clear();

    // Clean up segmented indexes
    for (auto& seg : g_si_segments) {
        if (seg.index) delete seg.index;
    }
    g_si_segments.clear();
    g_si_id_map.clear();
    g_si_next_id = 0;

    for (auto& seg : g_si128_segments) {
        if (seg.index) delete seg.index;
    }
    g_si128_segments.clear();
    g_si128_id_map.clear();
    g_si128_next_id = 0;
    g_si128_reservoir.clear();
    g_si128_reservoir.shrink_to_fit();
    g_si128_reservoir_count = 0;

    // Clean up 128D indexes
    if (g_p128_idx) { delete g_p128_idx; g_p128_idx = nullptr; }
    if (g_p128f_idx) { delete g_p128f_idx; g_p128f_idx = nullptr; }

    // Clean up 256D indexes
    if (g_p256_idx) { delete g_p256_idx; g_p256_idx = nullptr; }
    if (g_p256f_idx) { delete g_p256f_idx; g_p256f_idx = nullptr; }

    // Clean up 384D and 1536D indexes
    if (g_p384f_idx) { delete g_p384f_idx; g_p384f_idx = nullptr; }
    if (g_p1536_idx) { delete g_p1536_idx; g_p1536_idx = nullptr; }

    // Clean up naive index
    if (g_naive_idx) { delete g_naive_idx; g_naive_idx = nullptr; }

    // Clean up export buffers
    g_export_buf.clear();

    // Clean up embedding model
    if (g_embedder) { delete g_embedder; g_embedder = nullptr; }
    g_embedding_buf.clear();
}

}
