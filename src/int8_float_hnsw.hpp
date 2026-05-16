#pragma once
/**
 * Int8FloatHNSW -- Runtime-dimension int8-quantized HNSW index.
 *
 * Stores vectors as row-wise affine uint8: per-vector scale + offset + uint8[dims].
 * Insert accepts float32, quantizes internally. Search uses asymmetric distance:
 * float32 query vs dequantized-on-the-fly int8 database vectors.
 *
 * WHY ASYMMETRIC: Quantizing query would require knowing its scale/offset before
 * distance computation, but we don't know those until we scan the query. Asymmetric
 * avoids this chicken-and-egg problem and preserves query precision for better recall.
 *
 * Memory: ~(dims + 8) bytes/vector vs dims*4 bytes/vector for FloatHNSW.
 * At 1024D/100K: ~104 MB vs ~404 MB (3.9x savings).
 */

#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <cstring>
#include <cstdio>
#include <unordered_set>

#if defined(__wasm_simd128__)
    #include <wasm_simd128.h>
    #define INT8_HNSW_WASM_SIMD 1
#elif defined(PANCAKE_ENABLE_SSE2_SIMD) && defined(__SSE2__)
    #include <xmmintrin.h>
    #include <emmintrin.h>
    #define INT8_HNSW_SSE2_SIMD 1
#endif

#ifdef __EMSCRIPTEN__
    #include <emscripten/emscripten.h>
#else
    // Fallback for non-Emscripten builds
    #include <chrono>
    static double emscripten_get_now() {
        using namespace std::chrono;
        return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
    }
#endif

namespace pancake {
namespace wasm {

struct BuildProfile {
    uint64_t inserts = 0;

    double quantize_ms = 0;
    double search_layers_ms = 0;
    double search_base_ms = 0;
    double select_neighbors_ms = 0;
    double connect_ms = 0;

    uint64_t dist_calls = 0;
    uint64_t candidate_pushes = 0;
    uint64_t candidate_pops = 0;
    uint64_t visited_marks = 0;

    void reset() { *this = BuildProfile{}; }

    void print(uint32_t range_start, uint32_t range_end) const {
        double n = static_cast<double>(inserts);
        if (n == 0) { printf("No inserts recorded.\n"); return; }
        printf("=== Build Profile [%u..%u] (%u inserts) ===\n",
               range_start, range_end, (unsigned)inserts);
        printf("  avg insert:         %.3f ms\n", (quantize_ms + search_layers_ms + search_base_ms + select_neighbors_ms + connect_ms) / n);
        printf("  quantize:           %.2f ms total (%.4f ms/ins)\n", quantize_ms, quantize_ms / n);
        printf("  upper layer search: %.2f ms total (%.4f ms/ins)\n", search_layers_ms, search_layers_ms / n);
        printf("  base layer search:  %.2f ms total (%.4f ms/ins)\n", search_base_ms, search_base_ms / n);
        printf("  select neighbors:   %.2f ms total (%.4f ms/ins)\n", select_neighbors_ms, select_neighbors_ms / n);
        printf("  connect/rewire:     %.2f ms total (%.4f ms/ins)\n", connect_ms, connect_ms / n);
        printf("  dist calls/insert:  %.1f\n", (double)dist_calls / n);
        printf("  cand pushes/insert: %.1f\n", (double)candidate_pushes / n);
        printf("  cand pops/insert:   %.1f\n", (double)candidate_pops / n);
        printf("  visited marks/ins:  %.1f\n", (double)visited_marks / n);
        printf("================================================\n");
    }
};

static BuildProfile g_build_profile;

#define PROFILE_BLOCK(field, code)              \
  do {                                          \
    double _t0 = emscripten_get_now();          \
    code;                                       \
    g_build_profile.field += emscripten_get_now() - _t0; \
  } while (0)

// Forward: uses DistanceMetric from float_hnsw.hpp (already included in engine.cpp)

struct Int8FloatHNSWConfig {
    size_t M = 32;
    size_t ef_construction = 200;
    size_t ef_search = 128;
    size_t max_elements = 100000;
    uint32_t seed = 42;
    DistanceMetric metric = DistanceMetric::L2;
    bool use_heuristic = true;
};

class Int8FloatHNSW {
public:
    Int8FloatHNSW(size_t dims, const Int8FloatHNSWConfig& config = {})
        : dims_(dims)
        , metric_(config.metric)
        , M_(config.M)
        , M0_(config.M * 2)
        , ef_construction_(config.ef_construction)
        , ef_search_(config.ef_search)
        , max_elements_(config.max_elements)
        , count_(0)
        , num_deleted_(0)
        , entry_point_(UINT32_MAX)
        , max_level_(0)
        , rng_(config.seed)
        , use_heuristic_(config.use_heuristic)
        , cached_query_(nullptr)
        , cached_insert_(nullptr)
    {
        qdata_.reserve(max_elements_ * dims_);
        scales_.reserve(max_elements_);
        offsets_.reserve(max_elements_);
        sum_q_.reserve(max_elements_);
        sum_q2_.reserve(max_elements_);
        neighbors_.reserve(max_elements_);
        levels_.reserve(max_elements_);
        deleted_.reserve(max_elements_);
        level_mult_ = 1.0 / std::log(static_cast<double>(M_));
        visited_list_.assign(max_elements_, 0);
        visited_curr_ = 0;
        // Scratch buffers for symmetric distance during construction
        scratch_norm_.resize(dims_);
    }

    uint32_t insert(const float* vec) {
        if (count_ >= max_elements_) return UINT32_MAX;

        uint32_t id = static_cast<uint32_t>(count_++);
        g_build_profile.inserts++;

        // Normalize if cosine, then quantize
        const float* src = vec;
        {
            double _qt0 = emscripten_get_now();
            if (metric_ == DistanceMetric::Cosine) {
                float norm_sq = 0.0f;
                for (size_t d = 0; d < dims_; d++) norm_sq += vec[d] * vec[d];
                float inv_norm = (norm_sq > 0.0f) ? 1.0f / std::sqrt(norm_sq) : 0.0f;
                for (size_t d = 0; d < dims_; d++) scratch_norm_[d] = vec[d] * inv_norm;
                src = scratch_norm_.data();
            }

            float vmin = src[0];
            float vmax = src[0];
            for (size_t d = 1; d < dims_; d++) {
                if (src[d] < vmin) vmin = src[d];
                if (src[d] > vmax) vmax = src[d];
            }
            float range = vmax - vmin;
            if (range < 1e-30f) range = 1.0f;
            float scale = range / 255.0f;
            float inv_scale = 255.0f / range;

            scales_.push_back(scale);
            offsets_.push_back(vmin);

            size_t off = qdata_.size();
            qdata_.resize(off + dims_);
            for (size_t d = 0; d < dims_; d++) {
                float q = (src[d] - vmin) * inv_scale + 0.5f;
                if (q < 0.0f) q = 0.0f;
                if (q > 255.0f) q = 255.0f;
                qdata_[off + d] = static_cast<uint8_t>(q);
            }

            uint32_t sq = 0;
            uint32_t sq2 = 0;
            size_t qoff = qdata_.size() - dims_;
            for (size_t d = 0; d < dims_; d++) {
                uint32_t v = qdata_[qoff + d];
                sq += v;
                sq2 += v * v;
            }
            sum_q_.push_back(sq);
            sum_q2_.push_back(sq2);
            g_build_profile.quantize_ms += emscripten_get_now() - _qt0;
        }

        deleted_.push_back(0);

        int level = random_level();
        levels_.push_back(level);
        neighbors_.push_back(std::vector<std::vector<uint32_t>>(level + 1));

        if (entry_point_ == UINT32_MAX) {
            entry_point_ = id;
            max_level_ = level;
            return id;
        }

        // Cache the new vector's float32 data so construction can use
        // asymmetric distance (float vs int8) instead of dequantizing both sides.
        cached_insert_ = src;

        PROFILE_BLOCK(search_layers_ms, {
            uint32_t curr_upper = entry_point_;
            float curr_dist_upper = distance_to_insert(curr_upper);

            for (int l = max_level_; l > level; --l) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    for (uint32_t neighbor : neighbors_[curr_upper][l]) {
                        float d = distance_to_insert(neighbor);
                        if (d < curr_dist_upper) {
                            curr_upper = neighbor;
                            curr_dist_upper = d;
                            changed = true;
                        }
                    }
                }
            }
            insert_entry_ = curr_upper;
        });

        uint32_t curr = insert_entry_;

        for (int l = std::min(level, max_level_); l >= 0; --l) {
            std::vector<std::pair<float, uint32_t>> candidates;
            PROFILE_BLOCK(search_base_ms, {
                candidates = search_layer_insert(id, curr, ef_construction_, l);
            });
            size_t max_n = (l == 0) ? M0_ : M_;
            PROFILE_BLOCK(select_neighbors_ms, {
                select_neighbors_heuristic(id, candidates, max_n, l);
            });
            PROFILE_BLOCK(connect_ms, {
                for (uint32_t neighbor : neighbors_[id][l]) {
                    neighbors_[neighbor][l].push_back(id);
                    if (neighbors_[neighbor][l].size() > max_n) {
                        prune_neighbors(neighbor, l, max_n);
                    }
                }
            });
            if (!candidates.empty()) curr = candidates[0].second;
        }

        cached_insert_ = nullptr;

        if (level > max_level_) {
            entry_point_ = id;
            max_level_ = level;
        }
        return id;
    }

    int bulk_insert(const float* vectors, int n) {
        int inserted = 0;
        for (int i = 0; i < n; i++) {
            uint32_t id = insert(vectors + static_cast<size_t>(i) * dims_);
            if (id != UINT32_MAX) inserted++;
        }
        return inserted;
    }

    std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k) {
        if (count_ == 0) return {};

        if (metric_ == DistanceMetric::Cosine) {
            norm_query_.resize(dims_);
            float norm_sq = 0.0f;
            for (size_t d = 0; d < dims_; d++) norm_sq += query[d] * query[d];
            float inv_norm = (norm_sq > 0.0f) ? 1.0f / std::sqrt(norm_sq) : 0.0f;
            for (size_t d = 0; d < dims_; d++) norm_query_[d] = query[d] * inv_norm;
            cached_query_ = norm_query_.data();
        } else {
            cached_query_ = query;
        }

        uint32_t curr = entry_point_;
        float curr_dist = distance_to_query(curr);

        for (int l = max_level_; l > 0; --l) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (uint32_t neighbor : neighbors_[curr][l]) {
                    if (deleted_[neighbor]) continue;
                    float d = distance_to_query(neighbor);
                    if (d < curr_dist) {
                        curr = neighbor;
                        curr_dist = d;
                        changed = true;
                    }
                }
            }
        }

        auto candidates = search_layer_query(curr, std::max(ef_search_, k), 0);
        std::vector<std::pair<uint32_t, float>> results;
        for (size_t i = 0; i < std::min(k, candidates.size()); ++i) {
            if (!deleted_[candidates[i].second]) {
                results.emplace_back(candidates[i].second, candidates[i].first);
            }
        }
        return results;
    }

    void mark_delete(uint32_t id) {
        if (id >= count_ || deleted_[id] != 0) return;
        deleted_[id] = 1;
        num_deleted_++;
    }

    void set_ef_search(size_t ef) { ef_search_ = ef; }
    
    size_t ghost_count() const { return num_deleted_; }

    float ghost_ratio() const {
        return count_ > 0 ? static_cast<float>(num_deleted_) / count_ : 0.0f;
    }

    // Compact: remove ghosts with stable remapping, then refine edges.
    // The overload taking out_map returns the old→new ID mapping so the
    // caller can remap its own data structures.
    void compact() {
        std::vector<uint32_t> discard;
        compact(discard);
    }

    void compact(std::vector<uint32_t>& out_map) {
        if (num_deleted_ == 0) {
            out_map.resize(count_);
            for (uint32_t i = 0; i < count_; i++) out_map[i] = i;
            return;
        }

        // Phase 1: Build id_map and compact arrays in-place.
        out_map.assign(count_, UINT32_MAX);
        auto& id_map = out_map;
        uint32_t new_id = 0;
        for (uint32_t old_id = 0; old_id < count_; ++old_id) {
            if (!deleted_[old_id]) {
                id_map[old_id] = new_id;
                if (new_id != old_id) {
                    std::memcpy(&qdata_[new_id * dims_], &qdata_[old_id * dims_], dims_);
                    scales_[new_id] = scales_[old_id];
                    offsets_[new_id] = offsets_[old_id];
                    sum_q_[new_id] = sum_q_[old_id];
                    sum_q2_[new_id] = sum_q2_[old_id];
                    neighbors_[new_id] = std::move(neighbors_[old_id]);
                    levels_[new_id] = levels_[old_id];
                }
                new_id++;
            }
        }

        // Phase 2: Remap neighbor IDs and strip ghost references.
        for (uint32_t i = 0; i < new_id; i++) {
            for (int l = 0; l <= levels_[i]; l++) {
                auto& nbrs = neighbors_[i][l];
                size_t write = 0;
                for (size_t r = 0; r < nbrs.size(); r++) {
                    uint32_t mapped = id_map[nbrs[r]];
                    if (mapped != UINT32_MAX) {
                        nbrs[write++] = mapped;
                    }
                }
                nbrs.resize(write);
            }
        }

        // Phase 3: Backfill under-connected nodes.
        // After stripping ghost references, some nodes may have lost neighbors.
        // For each such node, gather neighbors-of-neighbors as candidates and
        // re-run the selection heuristic to restore connectivity. New edges are
        // added bidirectionally (same as the insert path).
        for (uint32_t i = 0; i < new_id; i++) {
            for (int l = 0; l <= levels_[i]; l++) {
                size_t target = (l == 0) ? M0_ : M_;
                auto& nbrs = neighbors_[i][l];
                if (nbrs.size() >= target) continue;

                // Snapshot old neighbors before heuristic overwrites them
                std::vector<uint32_t> old_nbrs(nbrs.begin(), nbrs.end());
                std::unordered_set<uint32_t> old_set(old_nbrs.begin(), old_nbrs.end());

                // Collect candidates: current neighbors + their neighbors (1-hop expansion)
                std::unordered_set<uint32_t> seen;
                seen.insert(i);
                for (uint32_t n : old_nbrs) seen.insert(n);

                std::vector<uint32_t> expansion;
                for (uint32_t n : old_nbrs) {
                    if (n >= new_id) continue;
                    if (l > levels_[n]) continue;
                    for (uint32_t nn : neighbors_[n][l]) {
                        if (nn < new_id && seen.find(nn) == seen.end()) {
                            seen.insert(nn);
                            expansion.push_back(nn);
                        }
                    }
                }

                if (expansion.empty()) continue;

                // Build candidate list: existing neighbors + new candidates, sorted by distance
                std::vector<std::pair<float, uint32_t>> candidates;
                candidates.reserve(old_nbrs.size() + expansion.size());
                for (uint32_t n : old_nbrs) candidates.emplace_back(distance(i, n), n);
                for (uint32_t n : expansion) candidates.emplace_back(distance(i, n), n);
                std::sort(candidates.begin(), candidates.end());

                select_neighbors_heuristic(i, candidates, target, l);

                // Add reverse edges for newly added neighbors
                for (uint32_t n : neighbors_[i][l]) {
                    if (old_set.count(n)) continue; // already connected
                    if (n >= new_id || l > levels_[n]) continue;
                    neighbors_[n][l].push_back(i);
                    if (neighbors_[n][l].size() > target) {
                        prune_neighbors(n, l, target);
                    }
                }
            }
        }

        // Phase 4: Update entry point.
        if (entry_point_ != UINT32_MAX && id_map[entry_point_] != UINT32_MAX) {
            entry_point_ = id_map[entry_point_];
        } else {
            entry_point_ = UINT32_MAX;
            max_level_ = 0;
            for (uint32_t i = 0; i < new_id; i++) {
                if (levels_[i] > max_level_) {
                    max_level_ = levels_[i];
                    entry_point_ = i;
                }
            }
        }

        // Shrink arrays.
        qdata_.resize(new_id * dims_);
        scales_.resize(new_id);
        offsets_.resize(new_id);
        sum_q_.resize(new_id);
        sum_q2_.resize(new_id);
        neighbors_.resize(new_id);
        levels_.resize(new_id);
        deleted_.assign(new_id, 0);
        count_ = new_id;
        num_deleted_ = 0;

    }

    std::vector<uint8_t> serialize() const {
        // Header: magic + dims + version + count + entry + max_level + M + M0 + metric + ef_construction
        // Then: scales[count] + offsets[count] + qdata[count * dims]
        // Then: graph structure
        size_t total_size = 10 * 4;
        total_size += count_ * sizeof(float) * 2;  // scales + offsets
        total_size += count_ * dims_;               // qdata
        for (size_t i = 0; i < count_; ++i) {
            total_size += 4;
            for (int l = 0; l <= levels_[i]; ++l)
                total_size += 4 + neighbors_[i][l].size() * 4;
        }

        std::vector<uint8_t> buffer;
        buffer.reserve(total_size);

        auto push_u32 = [&](uint32_t val) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(&val);
            buffer.insert(buffer.end(), p, p + 4);
        };
        auto push_f32 = [&](float val) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(&val);
            buffer.insert(buffer.end(), p, p + 4);
        };

        // Header: 10 x little-endian u32
        // [0] magic   [1] dims     [2] version  [3] count    [4] entry
        // [5] max_lvl [6] M        [7] M0       [8] metric   [9] ef_construction
        // metric: 0=L2, 1=Cosine (see DistanceMetric enum)
        push_u32(0x49384831);  // "I8H1" magic (v1 format)
        push_u32(static_cast<uint32_t>(dims_));
        push_u32(1);  // Version 1: adds ef_construction
        push_u32(static_cast<uint32_t>(count_));
        push_u32(entry_point_);
        push_u32(static_cast<uint32_t>(max_level_));
        push_u32(static_cast<uint32_t>(M_));
        push_u32(static_cast<uint32_t>(M0_));
        push_u32(static_cast<uint32_t>(metric_));
        push_u32(static_cast<uint32_t>(ef_construction_));

        // Scales and offsets
        for (size_t i = 0; i < count_; ++i) push_f32(scales_[i]);
        for (size_t i = 0; i < count_; ++i) push_f32(offsets_[i]);

        // Quantized data
        buffer.insert(buffer.end(), qdata_.begin(), qdata_.begin() + count_ * dims_);

        // Graph
        for (size_t i = 0; i < count_; ++i) {
            push_u32(static_cast<uint32_t>(levels_[i]));
            for (int l = 0; l <= levels_[i]; ++l) {
                push_u32(static_cast<uint32_t>(neighbors_[i][l].size()));
                for (uint32_t neighbor : neighbors_[i][l])
                    push_u32(neighbor);
            }
        }
        return buffer;
    }

    bool deserialize(const uint8_t* data, size_t data_size) {
        size_t offset = 0;
        auto safe_read_u32 = [&](uint32_t& out) -> bool {
            if (offset + 4 > data_size) return false;
            memcpy(&out, data + offset, 4);
            offset += 4;
            return true;
        };
        auto safe_read_f32 = [&](float& out) -> bool {
            if (offset + 4 > data_size) return false;
            memcpy(&out, data + offset, 4);
            offset += 4;
            return true;
        };

        uint32_t magic, dims_val;
        if (!safe_read_u32(magic)) return false;
        // Accept old "I8HW" (0x49384857) and new "I8H1" (0x49384831) magic
        bool is_v1 = (magic == 0x49384831);
        if (magic != 0x49384857 && !is_v1) return false;
        if (!safe_read_u32(dims_val) || dims_val != static_cast<uint32_t>(dims_)) return false;

        uint32_t version = 0;
        if (is_v1) {
            if (!safe_read_u32(version)) return false;
        }

        uint32_t count_val, entry_val, level_val, m_val, m0_val, metric_val;
        if (!safe_read_u32(count_val)) return false;
        if (!safe_read_u32(entry_val)) return false;
        if (!safe_read_u32(level_val)) return false;
        if (!safe_read_u32(m_val)) return false;
        if (!safe_read_u32(m0_val)) return false;
        if (!safe_read_u32(metric_val)) return false;

        uint32_t ef_construction_val = 200;  // default for old format
        if (version >= 1) {
            if (!safe_read_u32(ef_construction_val)) return false;
        }

        count_ = count_val;
        entry_point_ = entry_val;
        max_level_ = static_cast<int>(level_val);
        M_ = m_val;
        M0_ = m0_val;
        metric_ = static_cast<DistanceMetric>(metric_val);
        ef_construction_ = ef_construction_val;

        if (count_ > max_elements_) { count_ = 0; return false; }

        // Bounds check for scales + offsets (count * 4 bytes each)
        if (offset + count_ * sizeof(float) * 2 > data_size) { count_ = 0; return false; }
        scales_.resize(count_);
        offsets_.resize(count_);
        for (size_t i = 0; i < count_; ++i) {
            if (!safe_read_f32(scales_[i])) { count_ = 0; return false; }
        }
        for (size_t i = 0; i < count_; ++i) {
            if (!safe_read_f32(offsets_[i])) { count_ = 0; return false; }
        }

        // Bounds check for quantized data
        size_t qdata_bytes = count_ * dims_;
        if (offset + qdata_bytes > data_size) { count_ = 0; return false; }
        qdata_.resize(qdata_bytes);
        memcpy(qdata_.data(), data + offset, qdata_bytes);
        offset += qdata_bytes;

        neighbors_.resize(count_);
        levels_.resize(count_);
        for (size_t i = 0; i < count_; ++i) {
            uint32_t lvl;
            if (!safe_read_u32(lvl)) { count_ = 0; return false; }
            levels_[i] = static_cast<int>(lvl);
            neighbors_[i].resize(lvl + 1);
            for (int l = 0; l <= static_cast<int>(lvl); ++l) {
                uint32_t sz;
                if (!safe_read_u32(sz)) { count_ = 0; return false; }
                if (offset + sz * 4 > data_size) { count_ = 0; return false; }
                neighbors_[i][l].resize(sz);
                for (uint32_t j = 0; j < sz; ++j) {
                    if (!safe_read_u32(neighbors_[i][l][j])) { count_ = 0; return false; }
                }
            }
        }

        deleted_.assign(count_, 0);
        num_deleted_ = 0;
        if (max_elements_ > visited_list_.size())
            visited_list_.assign(max_elements_, 0);
        visited_curr_ = 0;

        // Recompute precomputed per-vector statistics from quantized data
        sum_q_.resize(count_);
        sum_q2_.resize(count_);
        for (size_t i = 0; i < count_; ++i) {
            const uint8_t* row = &qdata_[i * dims_];
            uint32_t sq = 0, sq2 = 0;
            for (size_t d = 0; d < dims_; d++) {
                uint32_t v = row[d];
                sq += v;
                sq2 += v * v;
            }
            sum_q_[i] = sq;
            sum_q2_[i] = sq2;
        }

        return true;
    }

    size_t count() const { return count_; }
    size_t dims() const { return dims_; }

    size_t memory_bytes() const {
        // Quantized vector storage: uint8[dims] + float scale + float offset per vector
        size_t total = count_ * dims_;                       // qdata
        total += count_ * sizeof(float) * 2;                 // scales + offsets

        // HNSW graph (logical size, not capacity)
        for (const auto& node_neighbors : neighbors_) {
            for (const auto& level : node_neighbors) {
                total += level.size() * sizeof(uint32_t);
            }
        }

        total += sizeof(*this);

        return total;
    }

    const std::vector<uint32_t>& get_neighbors(uint32_t id) const {
        return neighbors_[id][0];
    }

private:
    // =========================================================================
    // Dequantize vector id into dst buffer
    // =========================================================================
    void dequantize(uint32_t id, float* dst) const {
        const uint8_t* data = &qdata_[id * dims_];
        float s = scales_[id];
        float o = offsets_[id];
        for (size_t d = 0; d < dims_; d++) {
            dst[d] = o + s * static_cast<float>(data[d]);
        }
    }

    // =========================================================================
    // Float-float distance (used after dequantize for symmetric construction)
    // =========================================================================
    float l2_float(const float* a, const float* b) const {
        float sum = 0.0f;
        size_t d = 0;
#ifdef INT8_HNSW_WASM_SIMD
        v128_t acc = wasm_f32x4_splat(0.0f);
        for (; d + 4 <= dims_; d += 4) {
            v128_t diff = wasm_f32x4_sub(wasm_v128_load(a + d), wasm_v128_load(b + d));
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(diff, diff));
        }
        sum = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) +
              wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
#elif defined(INT8_HNSW_SSE2_SIMD)
        __m128 acc = _mm_setzero_ps();
        for (; d + 4 <= dims_; d += 4) {
            __m128 diff = _mm_sub_ps(_mm_loadu_ps(a + d), _mm_loadu_ps(b + d));
            acc = _mm_add_ps(acc, _mm_mul_ps(diff, diff));
        }
        alignas(16) float tmp[4];
        _mm_store_ps(tmp, acc);
        sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif
        for (; d < dims_; d++) {
            float diff = a[d] - b[d];
            sum += diff * diff;
        }
        return sum;
    }

    float cosine_dist_float(const float* a, const float* b) const {
        float dot = 0.0f;
        size_t d = 0;
#ifdef INT8_HNSW_WASM_SIMD
        v128_t acc = wasm_f32x4_splat(0.0f);
        for (; d + 4 <= dims_; d += 4) {
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(a + d), wasm_v128_load(b + d)));
        }
        dot = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) +
              wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
#elif defined(INT8_HNSW_SSE2_SIMD)
        __m128 acc = _mm_setzero_ps();
        for (; d + 4 <= dims_; d += 4) {
            acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(a + d), _mm_loadu_ps(b + d)));
        }
        alignas(16) float tmp[4];
        _mm_store_ps(tmp, acc);
        dot = tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif
        for (; d < dims_; d++) dot += a[d] * b[d];
        return 1.0f - dot;
    }

    // =========================================================================
    // Asymmetric distance: float query vs int8 database vector (search hot path)
    // =========================================================================
    float asymmetric_cosine(const float* query, uint32_t db_id) const {
        const uint8_t* data = &qdata_[db_id * dims_];
        float s = scales_[db_id];
        float o = offsets_[db_id];
        float dot = 0.0f;
        size_t d = 0;

#ifdef INT8_HNSW_WASM_SIMD
        v128_t acc = wasm_f32x4_splat(0.0f);
        v128_t v_scale = wasm_f32x4_splat(s);
        v128_t v_offset = wasm_f32x4_splat(o);

        for (; d + 16 <= dims_; d += 16) {
            v128_t bytes = wasm_v128_load(data + d);

            v128_t u16_lo = wasm_u16x8_extend_low_u8x16(bytes);
            v128_t u16_hi = wasm_u16x8_extend_high_u8x16(bytes);

            v128_t f0 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_low_u16x8(u16_lo));
            v128_t val0 = wasm_f32x4_add(v_offset, wasm_f32x4_mul(f0, v_scale));
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(query + d), val0));

            v128_t f1 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_high_u16x8(u16_lo));
            v128_t val1 = wasm_f32x4_add(v_offset, wasm_f32x4_mul(f1, v_scale));
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(query + d + 4), val1));

            v128_t f2 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_low_u16x8(u16_hi));
            v128_t val2 = wasm_f32x4_add(v_offset, wasm_f32x4_mul(f2, v_scale));
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(query + d + 8), val2));

            v128_t f3 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_high_u16x8(u16_hi));
            v128_t val3 = wasm_f32x4_add(v_offset, wasm_f32x4_mul(f3, v_scale));
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(query + d + 12), val3));
        }

        dot = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) +
              wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
#elif defined(INT8_HNSW_SSE2_SIMD)
        __m128 acc = _mm_setzero_ps();
        __m128 v_scale = _mm_set1_ps(s);
        __m128 v_offset = _mm_set1_ps(o);
        __m128i zero = _mm_setzero_si128();

        for (; d + 16 <= dims_; d += 16) {
            __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + d));
            __m128i u16_lo = _mm_unpacklo_epi8(bytes, zero);
            __m128i u16_hi = _mm_unpackhi_epi8(bytes, zero);

            #define SSE2_ASYM_DOT(u16v, lohi, offset) { \
                __m128 ff = _mm_cvtepi32_ps(_mm_unpack##lohi##_epi16(u16v, zero)); \
                __m128 val = _mm_add_ps(v_offset, _mm_mul_ps(ff, v_scale)); \
                acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(query + d + offset), val)); \
            }
            SSE2_ASYM_DOT(u16_lo, lo, 0);
            SSE2_ASYM_DOT(u16_lo, hi, 4);
            SSE2_ASYM_DOT(u16_hi, lo, 8);
            SSE2_ASYM_DOT(u16_hi, hi, 12);
            #undef SSE2_ASYM_DOT
        }

        alignas(16) float tmp[4];
        _mm_store_ps(tmp, acc);
        dot = tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif
        // Scalar tail
        for (; d < dims_; d++) {
            dot += query[d] * (o + s * static_cast<float>(data[d]));
        }
        return 1.0f - dot;
    }

    float asymmetric_l2(const float* query, uint32_t db_id) const {
        const uint8_t* data = &qdata_[db_id * dims_];
        float s = scales_[db_id];
        float o = offsets_[db_id];
        float sum = 0.0f;
        size_t d = 0;

#ifdef INT8_HNSW_WASM_SIMD
        v128_t acc = wasm_f32x4_splat(0.0f);
        v128_t v_scale = wasm_f32x4_splat(s);
        v128_t v_offset = wasm_f32x4_splat(o);

        for (; d + 16 <= dims_; d += 16) {
            v128_t bytes = wasm_v128_load(data + d);
            v128_t u16_lo = wasm_u16x8_extend_low_u8x16(bytes);
            v128_t u16_hi = wasm_u16x8_extend_high_u8x16(bytes);

            v128_t f0 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_low_u16x8(u16_lo));
            v128_t val0 = wasm_f32x4_add(v_offset, wasm_f32x4_mul(f0, v_scale));
            v128_t diff0 = wasm_f32x4_sub(wasm_v128_load(query + d), val0);
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(diff0, diff0));

            v128_t f1 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_high_u16x8(u16_lo));
            v128_t val1 = wasm_f32x4_add(v_offset, wasm_f32x4_mul(f1, v_scale));
            v128_t diff1 = wasm_f32x4_sub(wasm_v128_load(query + d + 4), val1);
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(diff1, diff1));

            v128_t f2 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_low_u16x8(u16_hi));
            v128_t val2 = wasm_f32x4_add(v_offset, wasm_f32x4_mul(f2, v_scale));
            v128_t diff2 = wasm_f32x4_sub(wasm_v128_load(query + d + 8), val2);
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(diff2, diff2));

            v128_t f3 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_high_u16x8(u16_hi));
            v128_t val3 = wasm_f32x4_add(v_offset, wasm_f32x4_mul(f3, v_scale));
            v128_t diff3 = wasm_f32x4_sub(wasm_v128_load(query + d + 12), val3);
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(diff3, diff3));
        }

        sum = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) +
              wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
#elif defined(INT8_HNSW_SSE2_SIMD)
        __m128 acc = _mm_setzero_ps();
        __m128 v_scale = _mm_set1_ps(s);
        __m128 v_offset = _mm_set1_ps(o);
        __m128i zero = _mm_setzero_si128();

        for (; d + 16 <= dims_; d += 16) {
            __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + d));
            __m128i u16_lo = _mm_unpacklo_epi8(bytes, zero);
            __m128i u16_hi = _mm_unpackhi_epi8(bytes, zero);

            #define SSE2_ASYM_L2(u16v, lohi, offset) { \
                __m128 ff = _mm_cvtepi32_ps(_mm_unpack##lohi##_epi16(u16v, zero)); \
                __m128 val = _mm_add_ps(v_offset, _mm_mul_ps(ff, v_scale)); \
                __m128 diff = _mm_sub_ps(_mm_loadu_ps(query + d + offset), val); \
                acc = _mm_add_ps(acc, _mm_mul_ps(diff, diff)); \
            }
            SSE2_ASYM_L2(u16_lo, lo, 0);
            SSE2_ASYM_L2(u16_lo, hi, 4);
            SSE2_ASYM_L2(u16_hi, lo, 8);
            SSE2_ASYM_L2(u16_hi, hi, 12);
            #undef SSE2_ASYM_L2
        }

        alignas(16) float tmp[4];
        _mm_store_ps(tmp, acc);
        sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif
        for (; d < dims_; d++) {
            float diff = query[d] - (o + s * static_cast<float>(data[d]));
            sum += diff * diff;
        }
        return sum;
    }

    // =========================================================================
    // Fast symmetric int8 distance using precomputed per-vector statistics.
    //
    // For two affine-quantized vectors a, b with per-vector scale/offset:
    //   a[d] = oa + sa * qa[d],  b[d] = ob + sb * qb[d]
    //
    // L2:  ||a-b||² = D*(oa-ob)² + 2*(oa-ob)*(sa*Σqa - sb*Σqb)
    //                + sa²*Σqa² + sb²*Σqb² - 2*sa*sb*Σ(qa·qb)
    //
    // Cosine (pre-normalized): dot(a,b) = D*oa*ob + oa*sb*Σqb
    //                                   + ob*sa*Σqa + sa*sb*Σ(qa·qb)
    //
    // Only Σ(qa·qb) is O(D) and pairwise; everything else is O(1) from
    // precomputed sum_q_ and sum_q2_. The integer dot product is the hot
    // loop and gets SIMD acceleration via i16x8_extmul.
    // =========================================================================

    uint32_t int8_dot(uint32_t a, uint32_t b) const {
        const uint8_t* da = &qdata_[a * dims_];
        const uint8_t* db = &qdata_[b * dims_];
        uint32_t sum = 0;
        size_t d = 0;

#ifdef INT8_HNSW_WASM_SIMD
        // Accumulate in i32x4 to avoid overflow.
        // Each uint8*uint8 <= 65025, and at 1536D we get up to ~99M which fits u32.
        v128_t acc = wasm_i32x4_splat(0);
        for (; d + 16 <= dims_; d += 16) {
            v128_t va = wasm_v128_load(da + d);
            v128_t vb = wasm_v128_load(db + d);

            // Extend u8 → u16, then use unsigned extmul u16→u32
            v128_t lo_a = wasm_u16x8_extend_low_u8x16(va);
            v128_t lo_b = wasm_u16x8_extend_low_u8x16(vb);
            v128_t hi_a = wasm_u16x8_extend_high_u8x16(va);
            v128_t hi_b = wasm_u16x8_extend_high_u8x16(vb);

            // u16 values are 0-255, safe for both signed and unsigned extmul.
            // Use signed extmul (universally available in WASM SIMD).
            acc = wasm_i32x4_add(acc, wasm_i32x4_extmul_low_i16x8(lo_a, lo_b));
            acc = wasm_i32x4_add(acc, wasm_i32x4_extmul_high_i16x8(lo_a, lo_b));
            acc = wasm_i32x4_add(acc, wasm_i32x4_extmul_low_i16x8(hi_a, hi_b));
            acc = wasm_i32x4_add(acc, wasm_i32x4_extmul_high_i16x8(hi_a, hi_b));
        }
        sum = static_cast<uint32_t>(
            wasm_i32x4_extract_lane(acc, 0) + wasm_i32x4_extract_lane(acc, 1) +
            wasm_i32x4_extract_lane(acc, 2) + wasm_i32x4_extract_lane(acc, 3));
#endif
        for (; d < dims_; d++) {
            sum += static_cast<uint32_t>(da[d]) * static_cast<uint32_t>(db[d]);
        }
        return sum;
    }

    float symmetric_l2_i8(uint32_t a, uint32_t b) const {
        float sa = scales_[a], oa = offsets_[a];
        float sb = scales_[b], ob = offsets_[b];
        float D = static_cast<float>(dims_);
        float diff_o = oa - ob;
        float dot_ab = static_cast<float>(int8_dot(a, b));

        return D * diff_o * diff_o
             + 2.0f * diff_o * (sa * static_cast<float>(sum_q_[a]) - sb * static_cast<float>(sum_q_[b]))
             + sa * sa * static_cast<float>(sum_q2_[a])
             + sb * sb * static_cast<float>(sum_q2_[b])
             - 2.0f * sa * sb * dot_ab;
    }

    float symmetric_cosine_i8(uint32_t a, uint32_t b) const {
        float sa = scales_[a], oa = offsets_[a];
        float sb = scales_[b], ob = offsets_[b];
        float D = static_cast<float>(dims_);
        float dot_ab = static_cast<float>(int8_dot(a, b));

        float dot = D * oa * ob
                   + oa * sb * static_cast<float>(sum_q_[b])
                   + ob * sa * static_cast<float>(sum_q_[a])
                   + sa * sb * dot_ab;
        return 1.0f - dot;
    }

    float distance(uint32_t a, uint32_t b) const {
        return (metric_ == DistanceMetric::Cosine)
            ? symmetric_cosine_i8(a, b)
            : symmetric_l2_i8(a, b);
    }

    // Asymmetric: float query vs int8 database (search hot path)
    float distance_to_query(uint32_t id) const {
        return (metric_ == DistanceMetric::Cosine)
            ? asymmetric_cosine(cached_query_, id)
            : asymmetric_l2(cached_query_, id);
    }

    // Asymmetric: float insert-vector vs int8 database (construction hot path)
    float distance_to_insert(uint32_t id) const {
        return (metric_ == DistanceMetric::Cosine)
            ? asymmetric_cosine(cached_insert_, id)
            : asymmetric_l2(cached_insert_, id);
    }

    int random_level() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return static_cast<int>(-std::log(dist(rng_)) * level_mult_);
    }

    // Construction: asymmetric float-insert vs int8-db distance
    // Generic search layer implementation - works with any distance function
    template<typename DistFunc>
    std::vector<std::pair<float, uint32_t>> search_layer_impl(
        DistFunc&& dist_func,
        uint32_t entry,
        size_t ef,
        int level,
        bool skip_deleted
    ) {
        std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, std::greater<>> candidates;
        std::priority_queue<std::pair<float, uint32_t>> results;

        prepare_visited();
        float d = dist_func(entry);
        g_build_profile.dist_calls++;
        candidates.emplace(d, entry);
        g_build_profile.candidate_pushes++;
        results.emplace(d, entry);
        mark_visited(entry);
        g_build_profile.visited_marks++;

        while (!candidates.empty()) {
            auto [curr_dist, curr] = candidates.top();
            candidates.pop();
            g_build_profile.candidate_pops++;
            if (curr_dist > results.top().first && results.size() >= ef) break;
            for (uint32_t neighbor : neighbors_[curr][level]) {
                if (is_visited(neighbor)) continue;
                if (skip_deleted && deleted_[neighbor]) continue;
                mark_visited(neighbor);
                g_build_profile.visited_marks++;
                float nd = dist_func(neighbor);
                g_build_profile.dist_calls++;
                if (nd < results.top().first || results.size() < ef) {
                    candidates.emplace(nd, neighbor);
                    g_build_profile.candidate_pushes++;
                    results.emplace(nd, neighbor);
                    if (results.size() > ef) results.pop();
                }
            }
        }
        std::vector<std::pair<float, uint32_t>> res;
        while (!results.empty()) { res.push_back(results.top()); results.pop(); }
        std::reverse(res.begin(), res.end());
        return res;
    }

    // Search layer during insertion (symmetric distance for graph construction)
    std::vector<std::pair<float, uint32_t>> search_layer_insert(uint32_t query_id, uint32_t entry, size_t ef, int level) {
        return search_layer_impl(
            [this](uint32_t id) { return distance_to_insert(id); },
            entry, ef, level, false  // Don't skip deleted during construction
        );
    }

    // Search layer during query (asymmetric distance for search)
    std::vector<std::pair<float, uint32_t>> search_layer_query(uint32_t entry, size_t ef, int level) {
        return search_layer_impl(
            [this](uint32_t id) { return distance_to_query(id); },
            entry, ef, level, true  // Skip deleted during search
        );
    }

    void select_neighbors_heuristic(uint32_t node, std::vector<std::pair<float, uint32_t>>& candidates, size_t M, int level) {
        std::vector<uint32_t> result;
        result.reserve(std::min(M, candidates.size()));
        std::vector<size_t> selected_indices;
        selected_indices.reserve(std::min(M, candidates.size()));
        // Pairwise distance cache — uses symmetric int8 distance directly
        // (integer dot product + precomputed scalar stats), no dequantization.
        const float uncached = std::numeric_limits<float>::lowest();
        std::vector<float> pair_cache(candidates.size() * candidates.size(), uncached);
        auto candidate_distance = [&](size_t a_idx, size_t b_idx) -> float {
            float& cached = pair_cache[a_idx * candidates.size() + b_idx];
            if (cached != uncached) return cached;
            float d = distance(candidates[a_idx].second, candidates[b_idx].second);
            cached = d;
            pair_cache[b_idx * candidates.size() + a_idx] = d;
            return d;
        };
        if (use_heuristic_) {
            size_t best_rejected = candidates.size();
            for (size_t ci = 0; ci < candidates.size(); ci++) {
                if (result.size() >= M) break;
                auto& cand = candidates[ci];
                bool keep = true;
                for (size_t sel_idx : selected_indices) {
                    if (candidate_distance(ci, sel_idx) < cand.first) { keep = false; break; }
                }
                if (keep) {
                    result.push_back(cand.second);
                    selected_indices.push_back(ci);
                } else if (best_rejected == candidates.size()) {
                    best_rejected = ci;
                }
            }
            // Backfill from closest rejected candidates.
            // WHY: Backfill ensures node reaches minimum connectivity even when
            // heuristic rejects candidates (prevents disconnected nodes).
            for (size_t ci = best_rejected; ci < candidates.size() && result.size() < M; ci++) {
                bool already = false;
                for (uint32_t sel : result) {
                    if (sel == candidates[ci].second) { already = true; break; }
                }
                if (!already) result.push_back(candidates[ci].second);
            }
        } else {
            for (auto& cand : candidates) {
                if (result.size() >= M) break;
                result.push_back(cand.second);
            }
        }
        neighbors_[node][level] = std::move(result);
    }

    void prune_neighbors(uint32_t node, int level, size_t M) {
        auto& list = neighbors_[node][level];
        std::vector<std::pair<float, uint32_t>> candidates;
        candidates.reserve(list.size());
        for (uint32_t n : list) {
            candidates.emplace_back(distance(node, n), n);
        }
        std::sort(candidates.begin(), candidates.end());
        select_neighbors_heuristic(node, candidates, M, level);
    }

    void prepare_visited() {
        visited_curr_++;
        // Overflow guard: reset all visited markers on counter wraparound
        if (visited_curr_ == 0) { std::fill(visited_list_.begin(), visited_list_.end(), 0); visited_curr_ = 1; }
    }
    bool is_visited(uint32_t id) const { return visited_list_[id] == visited_curr_; }
    void mark_visited(uint32_t id) { visited_list_[id] = visited_curr_; }

    size_t dims_;
    DistanceMetric metric_;
    size_t M_, M0_, ef_construction_, ef_search_, max_elements_, count_;
    size_t num_deleted_;
    uint32_t entry_point_;
    int max_level_;
    double level_mult_;
    std::mt19937 rng_;
    bool use_heuristic_;

    // Quantized vector storage
    std::vector<uint8_t> qdata_;     // count * dims bytes
    std::vector<float> scales_;      // per-vector scale
    std::vector<float> offsets_;     // per-vector offset (min_val)

    // Precomputed per-vector statistics for fast symmetric int8 distance.
    // Avoids dequantization in the heuristic: pairwise L2/cosine reduces to
    // one integer dot product + O(1) scalar arithmetic from these cached sums.
    std::vector<uint32_t> sum_q_;    // sum of uint8 values per vector
    std::vector<uint32_t> sum_q2_;   // sum of uint8² values per vector

    const float* cached_query_;
    const float* cached_insert_;  // float32 vector being inserted (avoids dequantize during construction)
    uint32_t insert_entry_;       // temp: entry point found by upper layer search
    std::vector<float> norm_query_;
    mutable std::vector<float> scratch_norm_;  // normalization buffer for cosine metric inserts

    std::vector<std::vector<std::vector<uint32_t>>> neighbors_;
    std::vector<int> levels_;
    std::vector<uint8_t> deleted_;
    std::vector<uint32_t> visited_list_;
    uint32_t visited_curr_;
};

} // namespace wasm
} // namespace pancake
