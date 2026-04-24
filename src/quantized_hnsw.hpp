#pragma once
/**
 * QuantizedHNSW -- Compile-time-dimension HNSW with optional int8 quantization.
 *
 * WHY COMPILE-TIME DIMENSION: Template specialization enables full loop unrolling
 * in quantized_simd.hpp, giving ~4-8x distance computation speedup over runtime-dim.
 * Trade-off: requires a separate template instantiation per dimension (128, 384, etc).
 *
 * WHY OPTIONAL FLOAT MODE: quantize=false provides a float32 fallback for
 * debugging and comparison against the quantized path.
 *
 * Storage: quantize=true uses AffineVector<DIMS> (uint8[DIMS] + scale + offset).
 * quantize=false uses float[DIMS]. Same graph structure either way.
 */

#include "quantized_simd.hpp"
#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <cstring>

namespace pancake {
namespace wasm {

struct QuantizedHNSWConfig {
    size_t M = 32;                      // Bidirectional links per node (higher = better recall, more memory)
    size_t ef_construction = 200;       // Beam width during graph construction (higher = better graph quality, slower insert)
    size_t ef_search = 128;             // Beam width during search (higher = better recall, slower query)
    size_t max_elements = 100000;       // Maximum capacity (pre-allocate for this many vectors)
    uint32_t seed = 42;                 // Random seed for level assignment (deterministic builds)
    bool quantize = true;               // false = float32 storage (no quantization)
    DistanceMetric metric = DistanceMetric::L2;  // Distance metric (L2 or Cosine)
};

template<size_t DIMS>
class QuantizedHNSW {
public:
    explicit QuantizedHNSW(const QuantizedHNSWConfig& config = {})
        : config_(config)
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
        , use_quantization_(config.quantize)
        , metric_(config.metric)
        , cached_query_f_(nullptr)
    {
        if (use_quantization_) {
            vectors_.reserve(max_elements_);
        } else {
            float_vectors_.reserve(max_elements_ * DIMS);
        }
        neighbors_.reserve(max_elements_);
        levels_.reserve(max_elements_);
        deleted_.reserve(max_elements_);
        if (metric_ == DistanceMetric::Cosine) {
            norms_.reserve(max_elements_);  // Precompute norms for cosine
        }
        level_mult_ = 1.0 / std::log(static_cast<double>(M_));
        visited_list_.assign(max_elements_, 0);
        visited_curr_ = 0;
    }

    uint32_t insert(const float* vec) {
        if (count_ >= max_elements_) return UINT32_MAX;

        uint32_t id = static_cast<uint32_t>(count_++);

        if (use_quantization_) {
            // For cosine: compute norm from ORIGINAL vector before quantization
            if (metric_ == DistanceMetric::Cosine) {
                float norm = 0.0f;
                for (size_t i = 0; i < DIMS; i++) norm += vec[i] * vec[i];
                norms_.push_back(std::sqrt(norm));
            }
            simd::AffineVector<DIMS> qvec;
            simd::quantize_fast<DIMS>(vec, qvec);
            vectors_.push_back(qvec);
        } else {
            float_vectors_.insert(float_vectors_.end(), vec, vec + DIMS);
            if (metric_ == DistanceMetric::Cosine) {
                float norm = 0.0f;
                for (size_t i = 0; i < DIMS; i++) norm += vec[i] * vec[i];
                norms_.push_back(std::sqrt(norm));
            }
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

        // Greedy descent
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

        // Beam search descent with Diversity Heuristic
        for (int l = std::min(level, max_level_); l >= 0; --l) {
            auto candidates = search_layer(id, curr, ef_construction_, l);
            size_t max_n = (l == 0) ? M0_ : M_;

            // Connect New Node to Neighbors
            select_neighbors_heuristic(id, candidates, max_n, l);

            // Connect Neighbors to New Node (Back-links)
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

    // Bulk insert: all inserts happen inside WASM, no JS boundary crossing per vector
    int bulk_insert(const float* vectors, int n) {
        int inserted = 0;
        for (int i = 0; i < n; i++) {
            uint32_t id = insert(vectors + i * DIMS);
            if (id != UINT32_MAX) inserted++;
        }
        return inserted;
    }

    std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k) {
        if (count_ == 0) return {};

        // Cache query for distance_to_current_query()
        cached_query_f_ = query;
        if (use_quantization_) {
            simd::quantize_fast<DIMS>(query, cached_query_q_);
        }
        // For cosine: compute norm from ORIGINAL query vector
        if (metric_ == DistanceMetric::Cosine) {
            float norm = 0.0f;
            for (size_t i = 0; i < DIMS; i++) norm += query[i] * query[i];
            cached_query_norm_ = std::sqrt(norm);
        }

        uint32_t curr = entry_point_;
        float curr_dist = distance_to_current_query(curr);

        for (int l = max_level_; l > 0; --l) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (uint32_t neighbor : neighbors_[curr][l]) {
                    if (deleted_[neighbor]) continue;
                    float d = distance_to_current_query(neighbor);
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
            // Skip deleted nodes in results
            if (!deleted_[candidates[i].second]) {
                results.emplace_back(candidates[i].second, candidates[i].first);
            }
        }
        return results;
    }

    void set_ef_search(size_t ef) { ef_search_ = ef; }

    // Mark node as deleted (becomes a ghost)
    void mark_delete(uint32_t id) {
        if (id >= count_ || deleted_[id] != 0) return;
        deleted_[id] = 1;
        num_deleted_++;
    }

    // Get ghost count
    size_t ghost_count() const { return num_deleted_; }

    // Get ghost ratio
    float ghost_ratio() const {
        return count_ > 0 ? static_cast<float>(num_deleted_) / count_ : 0.0f;
    }

    // Compact: rebuild index without ghosts (original implementation)
    void compact() {
        if (num_deleted_ == 0) return;

        std::vector<uint32_t> id_map(count_, UINT32_MAX);
        uint32_t new_id = 0;
        for (uint32_t old_id = 0; old_id < count_; ++old_id) {
            if (!deleted_[old_id]) {
                id_map[old_id] = new_id++;
            }
        }

        std::vector<int> new_levels;
        std::vector<std::vector<std::vector<uint32_t>>> new_neighbors;
        new_levels.reserve(new_id);
        new_neighbors.reserve(new_id);

        if (use_quantization_) {
            std::vector<simd::AffineVector<DIMS>> new_vectors;
            new_vectors.reserve(new_id);
            for (uint32_t old_id = 0; old_id < count_; ++old_id) {
                if (deleted_[old_id]) continue;
                new_vectors.push_back(vectors_[old_id]);
                new_levels.push_back(levels_[old_id]);
                std::vector<std::vector<uint32_t>> remapped(levels_[old_id] + 1);
                for (int l = 0; l <= levels_[old_id]; ++l) {
                    for (uint32_t old_n : neighbors_[old_id][l]) {
                        if (!deleted_[old_n]) remapped[l].push_back(id_map[old_n]);
                    }
                }
                new_neighbors.push_back(std::move(remapped));
            }
            vectors_ = std::move(new_vectors);
        } else {
            std::vector<float> new_float_vectors;
            new_float_vectors.reserve(new_id * DIMS);
            for (uint32_t old_id = 0; old_id < count_; ++old_id) {
                if (deleted_[old_id]) continue;
                const float* src = &float_vectors_[old_id * DIMS];
                new_float_vectors.insert(new_float_vectors.end(), src, src + DIMS);
                new_levels.push_back(levels_[old_id]);
                std::vector<std::vector<uint32_t>> remapped(levels_[old_id] + 1);
                for (int l = 0; l <= levels_[old_id]; ++l) {
                    for (uint32_t old_n : neighbors_[old_id][l]) {
                        if (!deleted_[old_n]) remapped[l].push_back(id_map[old_n]);
                    }
                }
                new_neighbors.push_back(std::move(remapped));
            }
            float_vectors_ = std::move(new_float_vectors);
        }

        uint32_t new_entry = UINT32_MAX;
        if (entry_point_ != UINT32_MAX && id_map[entry_point_] != UINT32_MAX) {
            new_entry = id_map[entry_point_];
        }

        if (!norms_.empty()) {
            std::vector<float> new_norms;
            new_norms.reserve(new_id);
            for (uint32_t old_id = 0; old_id < count_; ++old_id) {
                if (!deleted_[old_id]) new_norms.push_back(norms_[old_id]);
            }
            norms_ = std::move(new_norms);
        }

        levels_ = std::move(new_levels);
        neighbors_ = std::move(new_neighbors);
        deleted_.assign(new_id, 0);

        if (new_entry != UINT32_MAX) {
            entry_point_ = new_entry;
            max_level_ = levels_[new_entry];
        } else if (new_id > 0) {
            entry_point_ = 0;
            max_level_ = levels_[0];
            for (uint32_t i = 1; i < new_id; ++i) {
                if (levels_[i] > max_level_) {
                    entry_point_ = i;
                    max_level_ = levels_[i];
                }
            }
        } else {
            entry_point_ = UINT32_MAX;
            max_level_ = 0;
        }

        count_ = new_id;
        num_deleted_ = 0;
    }

    // Serialization
    std::vector<uint8_t> serialize() const {
        static constexpr uint32_t kFormatVersion = 3;

        size_t total_size = 10 * 4;  // Header: magic + dims + version + count + entry + max_level + M + M0 + metric + ef_construction
        if (use_quantization_) {
            total_size += count_ * sizeof(simd::AffineVector<DIMS>);
        } else {
            total_size += count_ * DIMS * sizeof(float);
        }
        if (metric_ == DistanceMetric::Cosine) {
            total_size += count_ * sizeof(float);  // norms
        }
        for (size_t i = 0; i < count_; ++i) {
            total_size += 4;
            for (int l = 0; l <= levels_[i]; ++l) {
                total_size += 4 + neighbors_[i][l].size() * 4;
            }
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
        push_u32(use_quantization_ ? 0x504E434B : 0x504E4346);  // "PNCK" or "PNCF"
        push_u32(static_cast<uint32_t>(DIMS));
        push_u32(kFormatVersion);
        push_u32(static_cast<uint32_t>(count_));
        push_u32(entry_point_);
        push_u32(static_cast<uint32_t>(max_level_));
        push_u32(static_cast<uint32_t>(M_));
        push_u32(static_cast<uint32_t>(M0_));
        push_u32(static_cast<uint32_t>(metric_));
        push_u32(static_cast<uint32_t>(ef_construction_));

        if (use_quantization_) {
            size_t vec_bytes = count_ * sizeof(simd::AffineVector<DIMS>);
            const uint8_t* vec_ptr = reinterpret_cast<const uint8_t*>(vectors_.data());
            buffer.insert(buffer.end(), vec_ptr, vec_ptr + vec_bytes);
        } else {
            size_t vec_bytes = count_ * DIMS * sizeof(float);
            const uint8_t* vec_ptr = reinterpret_cast<const uint8_t*>(float_vectors_.data());
            buffer.insert(buffer.end(), vec_ptr, vec_ptr + vec_bytes);
        }
        if (metric_ == DistanceMetric::Cosine) {
            const uint8_t* norm_ptr = reinterpret_cast<const uint8_t*>(norms_.data());
            buffer.insert(buffer.end(), norm_ptr, norm_ptr + count_ * sizeof(float));
        }

        for (size_t i = 0; i < count_; ++i) {
            push_u32(static_cast<uint32_t>(levels_[i]));
            for (int l = 0; l <= levels_[i]; ++l) {
                push_u32(static_cast<uint32_t>(neighbors_[i][l].size()));
                for (uint32_t neighbor : neighbors_[i][l]) {
                    push_u32(neighbor);
                }
            }
        }

        // Serialize deletion state
        push_u32(static_cast<uint32_t>(num_deleted_));
        for (size_t i = 0; i < count_; ++i) {
            buffer.push_back(deleted_[i]);
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

        uint32_t magic;
        if (!safe_read_u32(magic)) return false;
        if (magic != 0x504E434B && magic != 0x504E4346) return false;
        bool is_quantized = (magic == 0x504E434B);
        if (is_quantized != use_quantization_) return false;

        uint32_t dims;
        if (!safe_read_u32(dims) || dims != DIMS) return false;

        uint32_t version;
        if (!safe_read_u32(version)) return false;

        uint32_t count_val, entry_val, level_val, m_val, m0_val;
        if (!safe_read_u32(count_val)) return false;
        if (!safe_read_u32(entry_val)) return false;
        if (!safe_read_u32(level_val)) return false;
        if (!safe_read_u32(m_val)) return false;
        if (!safe_read_u32(m0_val)) return false;

        uint32_t metric_val = 0;        // default: L2
        uint32_t ef_construction_val = 200;  // default for old format
        if (version >= 2) {
            if (!safe_read_u32(metric_val)) return false;
            if (!safe_read_u32(ef_construction_val)) return false;
        }

        count_ = count_val;
        entry_point_ = entry_val;
        max_level_ = static_cast<int>(level_val);
        M_ = m_val;
        M0_ = m0_val;
        if (metric_val > 1) {
            count_ = 0;
            return false;  // Unknown metric encoding
        }
        metric_ = static_cast<DistanceMetric>(metric_val);
        ef_construction_ = ef_construction_val;

        if (count_ > max_elements_) {
            count_ = 0;
            return false;
        }

        if (use_quantization_) {
            size_t vec_bytes = count_ * sizeof(simd::AffineVector<DIMS>);
            if (offset + vec_bytes > data_size) { count_ = 0; return false; }
            vectors_.resize(count_);
            memcpy(vectors_.data(), data + offset, vec_bytes);
            offset += vec_bytes;
        } else {
            size_t vec_bytes = count_ * DIMS * sizeof(float);
            if (offset + vec_bytes > data_size) { count_ = 0; return false; }
            float_vectors_.resize(count_ * DIMS);
            memcpy(float_vectors_.data(), data + offset, vec_bytes);
            offset += vec_bytes;
        }
        if (metric_ == DistanceMetric::Cosine) {
            norms_.resize(count_);
            if (version >= 3) {
                size_t norm_bytes = count_ * sizeof(float);
                if (offset + norm_bytes > data_size) { count_ = 0; return false; }
                memcpy(norms_.data(), data + offset, norm_bytes);
                offset += norm_bytes;
            } else {
                recompute_norms();
            }
        } else {
            norms_.clear();
        }

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

        // Deserialize deletion state (only in version 1+)
        if (version >= 1) {
            uint32_t num_del;
            if (!safe_read_u32(num_del)) { count_ = 0; return false; }
            num_deleted_ = num_del;
            if (offset + count_ > data_size) { count_ = 0; return false; }
            deleted_.resize(count_);
            for (size_t i = 0; i < count_; ++i) {
                deleted_[i] = data[offset++];
            }
        } else {
            // Old format (version 0) - no deletion state
            deleted_.assign(count_, 0);
            num_deleted_ = 0;
        }

        if (max_elements_ > visited_list_.size()) {
            visited_list_.assign(max_elements_, 0);
        }
        visited_curr_ = 0;
        return true;
    }

    size_t count() const { return count_; }

    size_t memory_bytes() const {
        // Vector storage (logical size, not capacity)
        size_t total = 0;
        if (use_quantization_) {
            total = count_ * sizeof(simd::AffineVector<DIMS>);
        } else {
            total = count_ * DIMS * sizeof(float);
        }

        // HNSW graph structure (neighbors_) - use actual size not capacity
        for (const auto& node_neighbors : neighbors_) {
            for (const auto& level : node_neighbors) {
                total += level.size() * sizeof(uint32_t);
            }
        }

        // Class overhead
        total += sizeof(*this);

        return total;
    }

private:
    float float_l2(const float* a, const float* b) const {
        float sum = 0.0f;
        for (size_t d = 0; d < DIMS; d++) {
            float diff = a[d] - b[d];
            sum += diff * diff;
        }
        return sum;
    }

    float float_cosine(const float* a, const float* b) const {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t d = 0; d < DIMS; d++) {
            dot += a[d] * b[d];
            norm_a += a[d] * a[d];
            norm_b += b[d] * b[d];
        }
        float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        return (denom > 1e-8f) ? (1.0f - dot / denom) : 1.0f;
    }

    __attribute__((always_inline)) inline float distance(uint32_t a, uint32_t b) const {
        if (use_quantization_) {
            if (metric_ == DistanceMetric::Cosine) {
                float dot = simd::dot_product_fast<DIMS>(vectors_[a], vectors_[b]);
                return 1.0f - dot / (norms_[a] * norms_[b]);
            }
            return simd::l2_squared_fast<DIMS>(vectors_[a], vectors_[b]);
        }
        if (metric_ == DistanceMetric::Cosine) {
            float dot = 0.0f;
            for (size_t i = 0; i < DIMS; i++) {
                dot += float_vectors_[a * DIMS + i] * float_vectors_[b * DIMS + i];
            }
            return 1.0f - dot / (norms_[a] * norms_[b]);
        }
        return float_l2(&float_vectors_[a * DIMS], &float_vectors_[b * DIMS]);
    }

    __attribute__((always_inline)) inline float distance_to_current_query(uint32_t id) const {
        if (use_quantization_) {
            if (metric_ == DistanceMetric::Cosine) {
                float dot = simd::dot_product_fast<DIMS>(cached_query_q_, vectors_[id]);
                return 1.0f - dot / (cached_query_norm_ * norms_[id]);
            }
            return simd::l2_squared_fast<DIMS>(cached_query_q_, vectors_[id]);
        }
        if (metric_ == DistanceMetric::Cosine) {
            float dot = 0.0f;
            for (size_t i = 0; i < DIMS; i++) {
                dot += cached_query_f_[i] * float_vectors_[id * DIMS + i];
            }
            return 1.0f - dot / (cached_query_norm_ * norms_[id]);
        }
        return float_l2(cached_query_f_, &float_vectors_[id * DIMS]);
    }

    int random_level() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return static_cast<int>(-std::log(dist(rng_)) * level_mult_);
    }

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
        candidates.emplace(d, entry);
        results.emplace(d, entry);
        mark_visited(entry);

        while (!candidates.empty()) {
            auto [curr_dist, curr] = candidates.top();
            candidates.pop();
            if (curr_dist > results.top().first && results.size() >= ef) break;

            for (uint32_t neighbor : neighbors_[curr][level]) {
                if (is_visited(neighbor)) continue;
                if (skip_deleted && deleted_[neighbor]) continue;
                mark_visited(neighbor);
                float nd = dist_func(neighbor);
                if (nd < results.top().first || results.size() < ef) {
                    candidates.emplace(nd, neighbor);
                    results.emplace(nd, neighbor);
                    if (results.size() > ef) results.pop();
                }
            }
        }
        std::vector<std::pair<float, uint32_t>> res;
        while(!results.empty()) { res.push_back(results.top()); results.pop(); }
        std::reverse(res.begin(), res.end());
        return res;
    }

    // Search layer during construction (symmetric distance between two indexed vectors)
    std::vector<std::pair<float, uint32_t>> search_layer(uint32_t query_id, uint32_t entry, size_t ef, int level) {
        return search_layer_impl(
            [this, query_id](uint32_t id) { return distance(query_id, id); },
            entry, ef, level, false  // Don't skip deleted during construction
        );
    }

    // Search layer during query (asymmetric distance from cached query to indexed vectors)
    std::vector<std::pair<float, uint32_t>> search_layer_query(uint32_t entry, size_t ef, int level) {
        return search_layer_impl(
            [this](uint32_t id) { return distance_to_current_query(id); },
            entry, ef, level, true  // Skip deleted during search
        );
    }

    void select_neighbors_heuristic(uint32_t node, std::vector<std::pair<float, uint32_t>>& candidates, size_t M, int level) {
        std::vector<uint32_t> result;
        result.reserve(std::min(M, candidates.size()));
        std::vector<size_t> selected_indices;
        selected_indices.reserve(std::min(M, candidates.size()));
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
        size_t best_rejected = candidates.size();
        for (size_t ci = 0; ci < candidates.size(); ci++) {
            if (result.size() >= M) break;
            auto& cand = candidates[ci];
            bool keep = true;
            for (size_t sel_idx : selected_indices) {
                if (candidate_distance(ci, sel_idx) < cand.first) {
                    keep = false; break;
                }
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
        neighbors_[node][level] = std::move(result);
    }

    void prune_neighbors(uint32_t node, int level, size_t M) {
        auto& list = neighbors_[node][level];
        std::vector<std::pair<float, uint32_t>> candidates;
        candidates.reserve(list.size());
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

    void recompute_norms() {
        if (metric_ != DistanceMetric::Cosine) return;
        norms_.resize(count_);
        if (use_quantization_) {
            for (size_t i = 0; i < count_; ++i) {
                const auto& vec = vectors_[i];
                float norm = 0.0f;
                for (size_t d = 0; d < DIMS; ++d) {
                    float val = vec.offset + vec.data[d] * vec.scale;
                    norm += val * val;
                }
                norms_[i] = std::sqrt(norm);
            }
        } else {
            for (size_t i = 0; i < count_; ++i) {
                float norm = 0.0f;
                const float* base = &float_vectors_[i * DIMS];
                for (size_t d = 0; d < DIMS; ++d) {
                    float v = base[d];
                    norm += v * v;
                }
                norms_[i] = std::sqrt(norm);
            }
        }
    }

    QuantizedHNSWConfig config_;
    size_t M_, M0_, ef_construction_, ef_search_, max_elements_, count_;
    size_t num_deleted_;
    uint32_t entry_point_;
    int max_level_;
    double level_mult_;
    std::mt19937 rng_;
    bool use_quantization_;
    DistanceMetric metric_;

    // Quantized storage (used when quantize=true)
    std::vector<simd::AffineVector<DIMS>> vectors_;

    // Float storage (used when quantize=false)
    std::vector<float> float_vectors_;

    // Precomputed norms for cosine distance
    std::vector<float> norms_;

    // Cached query state (set during search())
    simd::AffineVector<DIMS> cached_query_q_;
    const float* cached_query_f_;
    mutable float cached_query_norm_;

    std::vector<std::vector<std::vector<uint32_t>>> neighbors_;
    std::vector<int> levels_;
    std::vector<uint8_t> deleted_;
    std::vector<uint32_t> visited_list_;
    uint32_t visited_curr_;
};

} // namespace wasm
} // namespace pancake
