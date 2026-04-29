#pragma once
/**
 * FloatHNSW -- Runtime-dimension float32 HNSW index.
 * No quantization. Designed for use with Random Indexing where RI
 * provides the compression and the user picks the target dimension.
 *
 * WHY RUNTIME-DIMENSION: Template-based HNSW (QuantizedHNSW<128>) requires
 * compile-time dimension, forcing separate builds per dimension. Runtime-dim
 * supports arbitrary dimensions at the cost of ~15% slower distance computation
 * (no loop unrolling). Worth the tradeoff for user-configurable runtime dims.
 */

#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <cstring>

#if defined(__wasm_simd128__)
    #include <wasm_simd128.h>
    #define FLOAT_HNSW_WASM_SIMD 1
#endif

namespace pancake {
namespace wasm {

enum class DistanceMetric : uint8_t { L2 = 0, Cosine = 1 };

struct FloatHNSWConfig {
    size_t M = 32;               // Max neighbors per layer (higher = better recall, more memory)
    size_t ef_construction = 200; // Construction beam width (higher = better quality, slower build)
    size_t ef_search = 128;      // Search beam width (higher = better recall, slower search)
    size_t max_elements = 100000;
    uint32_t seed = 42;
    DistanceMetric metric = DistanceMetric::L2;
    bool use_heuristic = true;   // Diversity heuristic (rejects clustered neighbors).
                                  // WHY: Prevents "hub" nodes, improves graph connectivity.
                                  // Disable for benchmarking against naive HNSW implementations.
};

class FloatHNSW {
public:
    FloatHNSW(size_t dims, const FloatHNSWConfig& config = {})
        : dims_(dims)
        , metric_(config.metric)
        , M_(config.M)
        , M0_(config.M * 2)  // WHY 2x: Layer 0 needs denser connections since all searches
                               // start there. Doubles max neighbors to improve entry point quality.
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
    {
        vectors_.reserve(max_elements_ * dims_);
        neighbors_.reserve(max_elements_);
        levels_.reserve(max_elements_);
        deleted_.reserve(max_elements_);
        level_mult_ = 1.0 / std::log(static_cast<double>(M_));
        visited_list_.assign(max_elements_, 0);
        visited_curr_ = 0;
    }

    uint32_t insert(const float* vec) {
        if (count_ >= max_elements_) return UINT32_MAX;

        uint32_t id = static_cast<uint32_t>(count_++);
        if (metric_ == DistanceMetric::Cosine) {
            // Normalize before storing so distance = 1 - dot(a,b) works
            size_t off = vectors_.size();
            vectors_.resize(off + dims_);
            float norm_sq = 0.0f;
            for (size_t d = 0; d < dims_; d++) norm_sq += vec[d] * vec[d];
            float inv_norm = (norm_sq > 0.0f) ? 1.0f / std::sqrt(norm_sq) : 0.0f;
            for (size_t d = 0; d < dims_; d++) vectors_[off + d] = vec[d] * inv_norm;
        } else {
            vectors_.insert(vectors_.end(), vec, vec + dims_);
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

        uint32_t curr = entry_point_;
        float curr_dist = distance(id, curr);

        for (int l = max_level_; l > level; --l) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (uint32_t neighbor : neighbors_[curr][l]) {
                    float d = distance(id, neighbor);
                    if (d < curr_dist) {
                        curr = neighbor;
                        curr_dist = d;
                        changed = true;
                    }
                }
            }
        }

        for (int l = std::min(level, max_level_); l >= 0; --l) {
            auto candidates = search_layer(id, curr, ef_construction_, l);
            size_t max_n = (l == 0) ? M0_ : M_;
            select_neighbors_heuristic(id, candidates, max_n, l);
            for (uint32_t neighbor : neighbors_[id][l]) {
                neighbors_[neighbor][l].push_back(id);
                if (neighbors_[neighbor][l].size() > max_n) {
                    prune_neighbors(neighbor, l, max_n);
                }
            }
            if (!candidates.empty()) curr = candidates[0].second;
        }

        if (level > max_level_) {
            entry_point_ = id;
            max_level_ = level;
        }
        return id;
    }

    std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k) {
        if (count_ == 0) return {};

        if (metric_ == DistanceMetric::Cosine) {
            // Normalize query into scratch buffer
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

    void set_ef(size_t ef) { ef_search_ = ef; }

    void mark_delete(uint32_t id) {
        if (id >= count_ || deleted_[id] != 0) return;
        deleted_[id] = 1;
        num_deleted_++;
    }

    size_t ghost_count() const { return num_deleted_; }

    float ghost_ratio() const {
        return count_ > 0 ? static_cast<float>(num_deleted_) / count_ : 0.0f;
    }

    // Compact: remove ghosts with stable remapping, then refine edges.
    // Cheaper than a full rebuild — preserves graph skeleton.
    void compact() {
        if (num_deleted_ == 0) return;

        // Phase 1: Build id_map and compact arrays in-place.
        std::vector<uint32_t> id_map(count_, UINT32_MAX);
        uint32_t new_id = 0;
        for (uint32_t old_id = 0; old_id < count_; ++old_id) {
            if (!deleted_[old_id]) {
                id_map[old_id] = new_id;
                if (new_id != old_id) {
                    std::memcpy(&vectors_[new_id * dims_], &vectors_[old_id * dims_], dims_ * sizeof(float));
                    neighbors_[new_id] = std::move(neighbors_[old_id]);
                    levels_[new_id] = levels_[old_id];
                }
                new_id++;
            }
        }

        // Phase 2: Remap all neighbor IDs and strip ghost references.
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

        // Phase 3: Update entry point.
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
        vectors_.resize(new_id * dims_);
        neighbors_.resize(new_id);
        levels_.resize(new_id);
        deleted_.assign(new_id, 0);
        count_ = new_id;
        num_deleted_ = 0;

    }

    std::vector<uint8_t> serialize() const {
        size_t total_size = 10 * 4 + count_ * dims_ * sizeof(float);
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

        // Header: 10 x little-endian u32
        // [0] magic   [1] dims     [2] version  [3] count    [4] entry
        // [5] max_lvl [6] M        [7] M0       [8] metric   [9] ef_construction
        // metric: 0=L2, 1=Cosine (see DistanceMetric enum)
        push_u32(0x464C4831);  // "FLH1" magic (v1 format)
        push_u32(static_cast<uint32_t>(dims_));
        push_u32(1);  // Version 1: adds ef_construction
        push_u32(static_cast<uint32_t>(count_));
        push_u32(entry_point_);
        push_u32(static_cast<uint32_t>(max_level_));
        push_u32(static_cast<uint32_t>(M_));
        push_u32(static_cast<uint32_t>(M0_));
        push_u32(static_cast<uint32_t>(metric_));
        push_u32(static_cast<uint32_t>(ef_construction_));

        size_t vec_bytes = count_ * dims_ * sizeof(float);
        const uint8_t* vec_ptr = reinterpret_cast<const uint8_t*>(vectors_.data());
        buffer.insert(buffer.end(), vec_ptr, vec_ptr + vec_bytes);

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

        uint32_t magic, dims_val;
        if (!safe_read_u32(magic)) return false;
        // Accept old "FLHW" (0x464C4857) and new "FLH1" (0x464C4831) magic
        bool is_v1 = (magic == 0x464C4831);
        if (magic != 0x464C4857 && !is_v1) return false;
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

        size_t vec_bytes = count_ * dims_ * sizeof(float);
        if (offset + vec_bytes > data_size) { count_ = 0; return false; }
        vectors_.resize(count_ * dims_);
        memcpy(vectors_.data(), data + offset, vec_bytes);
        offset += vec_bytes;

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
        return true;
    }

    size_t count() const { return count_; }
    size_t dims() const { return dims_; }
    size_t memory_bytes() const {
        // Vector storage (logical size)
        size_t total = count_ * dims_ * sizeof(float);

        // HNSW graph structure (logical size, not capacity)
        for (const auto& node_neighbors : neighbors_) {
            for (const auto& level : node_neighbors) {
                total += level.size() * sizeof(uint32_t);
            }
        }

        // Class overhead
        total += sizeof(*this);

        return total;
    }

    // Return layer-0 neighbor list for a node (for graph quality metrics)
    const std::vector<uint32_t>& get_neighbors(uint32_t id) const {
        return neighbors_[id][0];
    }

private:
    float l2(const float* a, const float* b) const {
        float sum = 0.0f;
        size_t d = 0;
#ifdef FLOAT_HNSW_WASM_SIMD
        v128_t acc = wasm_f32x4_splat(0.0f);
        for (; d + 4 <= dims_; d += 4) {
            v128_t diff = wasm_f32x4_sub(wasm_v128_load(a + d), wasm_v128_load(b + d));
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(diff, diff));
        }
        sum = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) +
              wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
#elif defined(__SSE2__)
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

    // For pre-normalized vectors: 1 - dot(a,b), range [0, 2]
    float cosine_dist(const float* a, const float* b) const {
        float dot = 0.0f;
        size_t d = 0;
#ifdef FLOAT_HNSW_WASM_SIMD
        v128_t acc = wasm_f32x4_splat(0.0f);
        for (; d + 4 <= dims_; d += 4) {
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(a + d), wasm_v128_load(b + d)));
        }
        dot = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) +
              wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
#elif defined(__SSE2__)
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

    float distance(uint32_t a, uint32_t b) const {
        const float* va = &vectors_[a * dims_];
        const float* vb = &vectors_[b * dims_];
        return (metric_ == DistanceMetric::Cosine) ? cosine_dist(va, vb) : l2(va, vb);
    }

    float distance_to_query(uint32_t id) const {
        const float* v = &vectors_[id * dims_];
        return (metric_ == DistanceMetric::Cosine) ? cosine_dist(cached_query_, v) : l2(cached_query_, v);
    }

    int random_level() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return static_cast<int>(-std::log(dist(rng_)) * level_mult_);
    }

    // Uses vector ID for distance computation (used during graph construction)
    std::vector<std::pair<float, uint32_t>> search_layer(uint32_t query_id, uint32_t entry, size_t ef, int level) {
        std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, std::greater<>> candidates;
        std::priority_queue<std::pair<float, uint32_t>> results;

        prepare_visited();
        float d = distance(query_id, entry);
        candidates.emplace(d, entry);
        results.emplace(d, entry);
        mark_visited(entry);

        while (!candidates.empty()) {
            auto [curr_dist, curr] = candidates.top();
            candidates.pop();
            if (curr_dist > results.top().first && results.size() >= ef) break;
            for (uint32_t neighbor : neighbors_[curr][level]) {
                if (is_visited(neighbor)) continue;
                mark_visited(neighbor);
                float nd = distance(query_id, neighbor);
                if (nd < results.top().first || results.size() < ef) {
                    candidates.emplace(nd, neighbor);
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

    // Uses cached query pointer for distance computation (used during search)
    std::vector<std::pair<float, uint32_t>> search_layer_query(uint32_t entry, size_t ef, int level) {
        std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, std::greater<>> candidates;
        std::priority_queue<std::pair<float, uint32_t>> results;

        prepare_visited();
        float d = distance_to_query(entry);
        candidates.emplace(d, entry);
        results.emplace(d, entry);
        mark_visited(entry);

        while (!candidates.empty()) {
            auto [curr_dist, curr] = candidates.top();
            candidates.pop();
            if (curr_dist > results.top().first && results.size() >= ef) break;
            for (uint32_t neighbor : neighbors_[curr][level]) {
                if (is_visited(neighbor) || deleted_[neighbor]) continue;
                mark_visited(neighbor);
                float nd = distance_to_query(neighbor);
                if (nd < results.top().first || results.size() < ef) {
                    candidates.emplace(nd, neighbor);
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

    void select_neighbors_heuristic(uint32_t node, std::vector<std::pair<float, uint32_t>>& candidates, size_t M, int level) {
        std::vector<uint32_t> result;
        if (use_heuristic_) {
            // Pairwise distance cache — avoids recomputing the same float L2/cosine
            // distance when a candidate is compared against multiple selected neighbors.
            const size_t nc = candidates.size();
            const float uncached = std::numeric_limits<float>::lowest();
            std::vector<float> pair_cache(nc * nc, uncached);
            auto cached_distance = [&](size_t a_idx, size_t b_idx) -> float {
                float& cached = pair_cache[a_idx * nc + b_idx];
                if (cached != uncached) return cached;
                float d = distance(candidates[a_idx].second, candidates[b_idx].second);
                cached = d;
                pair_cache[b_idx * nc + a_idx] = d;
                return d;
            };

            std::vector<size_t> selected_indices;
            selected_indices.reserve(std::min(M, nc));
            size_t best_rejected = nc;
            for (size_t ci = 0; ci < nc; ci++) {
                if (result.size() >= M) break;
                auto& cand = candidates[ci];
                bool keep = true;
                for (size_t sel_idx : selected_indices) {
                    if (cached_distance(ci, sel_idx) < cand.first) { keep = false; break; }
                }
                if (keep) {
                    result.push_back(cand.second);
                    selected_indices.push_back(ci);
                } else if (best_rejected == nc) {
                    best_rejected = ci;
                }
            }
            // Backfill from closest rejected candidates.
            for (size_t ci = best_rejected; ci < nc && result.size() < M; ci++) {
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
        for (uint32_t n : list) candidates.emplace_back(distance(node, n), n);
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
    std::vector<float> vectors_;
    const float* cached_query_;
    std::vector<float> norm_query_;  // scratch buffer for normalized query (cosine)
    std::vector<std::vector<std::vector<uint32_t>>> neighbors_;
    std::vector<int> levels_;
    std::vector<uint8_t> deleted_;
    std::vector<uint32_t> visited_list_;
    uint32_t visited_curr_;
};

} // namespace wasm
} // namespace pancake
