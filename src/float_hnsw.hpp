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
#include <cstdint>
#include <unordered_set>

#if defined(__wasm_simd128__)
    #include <wasm_simd128.h>
    #define FLOAT_HNSW_WASM_SIMD 1
#elif defined(PANCAKE_ENABLE_AVX2_SIMD) && defined(__AVX2__)
    #include <immintrin.h>
    #define FLOAT_HNSW_AVX2_SIMD 1
#elif defined(PANCAKE_ENABLE_SSE2_SIMD) && defined(__SSE2__)
    #include <xmmintrin.h>
    #include <emmintrin.h>
    #define FLOAT_HNSW_SSE2_SIMD 1
#endif

#ifdef __EMSCRIPTEN__
    #include <emscripten/emscripten.h>
#elif !defined(PANCAKE_EMSCRIPTEN_GET_NOW_DEFINED)
    #define PANCAKE_EMSCRIPTEN_GET_NOW_DEFINED
    #include <chrono>
    static double emscripten_get_now() {
        using namespace std::chrono;
        return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
    }
#endif

namespace pancake {
namespace wasm {

enum class DistanceMetric : uint8_t { L2 = 0, Cosine = 1 };

struct FloatHNSWConfig {
    size_t M = 16;               // Max neighbors per layer (higher = better recall, more memory)
    size_t ef_construction = 50; // Construction beam width (higher = better quality, slower build)
    size_t ef_search = 100;      // Search beam width (higher = better recall, slower search)
    size_t max_elements = 100000;
    uint32_t seed = 42;
    DistanceMetric metric = DistanceMetric::L2;
    bool use_heuristic = true;   // Diversity heuristic (rejects clustered neighbors).
                                  // WHY: Prevents "hub" nodes, improves graph connectivity.
                                  // Disable for benchmarking against naive HNSW implementations.
};

class FloatHNSW {
public:
    // Upper bound on the HNSW level count accepted from an untrusted snapshot.
    // Real graphs never approach this; it caps attacker-controlled allocations
    // during deserialize(). 64 levels covers >10^18 elements at any sane M.
    static constexpr uint32_t MAX_DESERIALIZE_LEVEL = 64;

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
        base_neighbors_.resize(max_elements_ * M0_);
        base_sizes_.assign(max_elements_, 0);
        upper_.reserve(max_elements_);
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
        base_sizes_[id] = 0;
        upper_.push_back(std::vector<std::vector<uint32_t>>(level > 0 ? static_cast<size_t>(level) : 0));

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
                for_each_edge(curr, l, [&](uint32_t neighbor) {
                    float d = distance(id, neighbor);
                    if (d < curr_dist) {
                        curr = neighbor;
                        curr_dist = d;
                        changed = true;
                    }
                });
            }
        }

        for (int l = std::min(level, max_level_); l >= 0; --l) {
            auto candidates = search_layer(id, curr, ef_construction_, l);
            size_t max_n = (l == 0) ? M0_ : M_;
            select_neighbors_heuristic(id, candidates, max_n, l);
            for_each_edge(id, l, [&](uint32_t neighbor) {
                append_neighbor(neighbor, l, id);
                if (level_edge_count(neighbor, l) > max_n) {
                    prune_neighbors(neighbor, l, max_n);
                }
            });
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
                for_each_edge(curr, l, [&](uint32_t neighbor) {
                    if (deleted_[neighbor]) return;
                    float d = distance_to_query(neighbor);
                    if (d < curr_dist) {
                        curr = neighbor;
                        curr_dist = d;
                        changed = true;
                    }
                });
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

    // Filtered search with in-traversal filtering.
    // Only results where the corresponding bit is set in filter_bitset enter the
    // result queue. Non-matching nodes still participate in navigation (they stay
    // in the candidate queue) but don't consume result slots. This keeps the
    // lower bound tight to actual filtered results, allowing the search to explore
    // further before terminating.
    // Uses dynamic ef expansion: doubles ef within the same traversal when fewer
    // than k filtered results are found, capped at 4x initial ef.
    // filter_bitset layout: bit (id & 7) of byte (id >> 3).
    std::vector<std::pair<uint32_t, float>> search_filtered(
        const float* query, size_t k,
        const uint8_t* filter_bitset, size_t bitset_len
    ) {
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

        // Upper-level greedy traversal (no filtering — just navigate)
        for (int l = max_level_; l > 0; --l) {
            bool changed = true;
            while (changed) {
                changed = false;
                for_each_edge(curr, l, [&](uint32_t neighbor) {
                    if (deleted_[neighbor]) return;
                    float d = distance_to_query(neighbor);
                    if (d < curr_dist) {
                        curr = neighbor;
                        curr_dist = d;
                        changed = true;
                    }
                });
            }
        }

        // Layer 0: in-traversal filtered search with dynamic ef expansion
        size_t initial_ef = std::max(ef_search_, k * 2);
        size_t max_ef = initial_ef * 4;
        size_t current_ef = initial_ef;

        std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, std::greater<>> candidates;
        std::priority_queue<std::pair<float, uint32_t>> results; // max-heap: worst filtered result on top

        prepare_visited();
        float d = distance_to_query(curr);
        candidates.emplace(d, curr);
        mark_visited(curr);

        size_t filtered_count = 0;
        float lower_bound = std::numeric_limits<float>::max();

        // Check if entry point passes filter
        if (!deleted_[curr]) {
            size_t byte_idx = curr >> 3;
            if (byte_idx < bitset_len && (filter_bitset[byte_idx] & (1u << (curr & 7)))) {
                results.emplace(d, curr);
                filtered_count++;
                lower_bound = d;
            }
        }

        while (!candidates.empty()) {
            auto [cand_dist, cand_id] = candidates.top();

            // Termination: best candidate is worse than worst filtered result and we have enough
            if (filtered_count >= k && cand_dist > lower_bound) {
                break;
            }

            // Dynamic ef expansion: if we haven't found k filtered results and the
            // candidate queue is exhausted relative to current_ef, widen the beam.
            if (filtered_count < k && cand_dist > lower_bound && results.size() >= current_ef) {
                if (current_ef >= max_ef) break;
                current_ef = std::min(current_ef * 2, max_ef);
            }

            candidates.pop();

            bool stop_expansion = false;
            for_each_edge(cand_id, 0, [&](uint32_t neighbor) {
                if (stop_expansion) return;
                if (is_visited(neighbor)) return;
                mark_visited(neighbor);

                if (deleted_[neighbor]) return;

                float nd = distance_to_query(neighbor);

                // Add to candidate queue if promising (for navigation)
                bool dominated = (results.size() >= current_ef && nd > lower_bound);
                if (!dominated) {
                    candidates.emplace(nd, neighbor);

                    // Check filter — only matching nodes enter the result queue
                    size_t byte_idx = neighbor >> 3;
                    if (byte_idx < bitset_len && (filter_bitset[byte_idx] & (1u << (neighbor & 7)))) {
                        results.emplace(nd, neighbor);
                        filtered_count++;

                        if (results.size() > current_ef) {
                            results.pop();
                            filtered_count--;
                        }
                        if (!results.empty()) {
                            lower_bound = results.top().first;
                        }
                    }
                }

                if (filtered_count >= k && candidates.empty()) stop_expansion = true;
            });
        }

        // Extract filtered results sorted by distance (closest first).
        // results is a max-heap (worst on top), so drain fully and reverse.
        std::vector<std::pair<uint32_t, float>> out;
        while (!results.empty()) {
            auto [dist, id] = results.top();
            results.pop();
            out.emplace_back(id, dist);
        }
        std::reverse(out.begin(), out.end());
        if (out.size() > k) out.resize(k);
        return out;
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
                    std::memcpy(&vectors_[new_id * dims_], &vectors_[old_id * dims_], dims_ * sizeof(float));
                    base_sizes_[new_id] = base_sizes_[old_id];
                    if (base_sizes_[old_id] != 0) {
                        std::memcpy(base_slot(new_id), base_slot(old_id), static_cast<size_t>(base_sizes_[old_id]) * sizeof(uint32_t));
                    }
                    upper_[new_id] = std::move(upper_[old_id]);
                    levels_[new_id] = levels_[old_id];
                }
                new_id++;
            }
        }

        // Phase 2: Remap all neighbor IDs and strip ghost references.
        for (uint32_t i = 0; i < new_id; i++) {
            for (int l = 0; l <= levels_[i]; l++) {
                std::vector<uint32_t> nbrs = copy_level_edges(i, l);
                std::vector<uint32_t> remapped;
                remapped.reserve(nbrs.size());
                for (uint32_t neighbor : nbrs) {
                    uint32_t mapped = id_map[neighbor];
                    if (mapped != UINT32_MAX) {
                        remapped.push_back(mapped);
                    }
                }
                write_level_edges(i, l, remapped);
            }
        }

        // Phase 3: Backfill under-connected nodes.
        for (uint32_t i = 0; i < new_id; i++) {
            for (int l = 0; l <= levels_[i]; l++) {
                size_t target = (l == 0) ? M0_ : M_;
                if (level_edge_count(i, l) >= target) continue;

                std::vector<uint32_t> old_nbrs = copy_level_edges(i, l);
                std::unordered_set<uint32_t> old_set(old_nbrs.begin(), old_nbrs.end());

                std::unordered_set<uint32_t> seen;
                seen.insert(i);
                for (uint32_t n : old_nbrs) seen.insert(n);

                std::vector<uint32_t> expansion;
                for (uint32_t n : old_nbrs) {
                    if (n >= new_id) continue;
                    if (l > levels_[n]) continue;
                    for_each_edge(n, l, [&](uint32_t nn) {
                        if (nn < new_id && seen.find(nn) == seen.end()) {
                            seen.insert(nn);
                            expansion.push_back(nn);
                        }
                    });
                }

                if (expansion.empty()) continue;

                std::vector<std::pair<float, uint32_t>> candidates;
                candidates.reserve(old_nbrs.size() + expansion.size());
                for (uint32_t n : old_nbrs) candidates.emplace_back(distance(i, n), n);
                for (uint32_t n : expansion) candidates.emplace_back(distance(i, n), n);
                std::sort(candidates.begin(), candidates.end());

                select_neighbors_heuristic(i, candidates, target, l);

                for_each_edge(i, l, [&](uint32_t n) {
                    if (old_set.count(n)) return;
                    if (n >= new_id || l > levels_[n]) return;
                    append_neighbor(n, l, i);
                    if (level_edge_count(n, l) > target) {
                        prune_neighbors(n, l, target);
                    }
                });
            }
        }

        // Phase 3b: if a survivor lost all base-layer neighbors, reconnect it
        // against the live pool so compaction cannot serialize an unreachable node.
        if (new_id > 1) {
            for (uint32_t i = 0; i < new_id; i++) {
                if (level_edge_count(i, 0) != 0) continue;

                std::vector<std::pair<float, uint32_t>> candidates;
                candidates.reserve(new_id - 1);
                for (uint32_t j = 0; j < new_id; ++j) {
                    if (j == i) continue;
                    candidates.emplace_back(distance(i, j), j);
                }
                std::sort(candidates.begin(), candidates.end());
                select_neighbors_heuristic(i, candidates, M0_, 0);

                uint32_t anchor = UINT32_MAX;
                float anchor_dist = std::numeric_limits<float>::max();
                for_each_edge(i, 0, [&](uint32_t n) {
                    append_neighbor(n, 0, i);
                    if (level_edge_count(n, 0) > M0_) {
                        prune_neighbors(n, 0, M0_);
                    }
                    if (has_neighbor(n, 0, i)) {
                        anchor = UINT32_MAX;
                        return;
                    }
                    float d = distance(i, n);
                    if (d < anchor_dist) {
                        anchor = n;
                        anchor_dist = d;
                    }
                });
                if (anchor != UINT32_MAX) {
                    ensure_base_neighbor(anchor, i);
                }
            }
        }

        // Phase 4: Recompute a valid entry point / max-level pair from survivors.
        entry_point_ = UINT32_MAX;
        max_level_ = 0;
        for (uint32_t i = 0; i < new_id; i++) {
            if (entry_point_ == UINT32_MAX || levels_[i] > max_level_) {
                max_level_ = levels_[i];
                entry_point_ = i;
            }
        }
        if (entry_point_ == UINT32_MAX && new_id > 0) entry_point_ = 0;

        // Shrink arrays.
        vectors_.resize(new_id * dims_);
        upper_.resize(new_id);
        levels_.resize(new_id);
        deleted_.assign(new_id, 0);
        count_ = new_id;
        num_deleted_ = 0;

    }

    std::vector<uint8_t> serialize() const {
        size_t total_size = 10 * 4 + count_ * dims_ * sizeof(float);
        for (size_t i = 0; i < count_; ++i) {
            total_size += 4;
            for (int l = 0; l <= levels_[i]; ++l) {
                total_size += 4 + level_edge_count(static_cast<uint32_t>(i), l) * 4;
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
                std::vector<uint32_t> edges = copy_level_edges(static_cast<uint32_t>(i), l);
                push_u32(static_cast<uint32_t>(edges.size()));
                for (uint32_t neighbor : edges) push_u32(neighbor);
            }
        }
        return buffer;
    }

    bool deserialize(const uint8_t* data, size_t data_size) {
        size_t offset = 0;
        auto fail = [&]() -> bool {
            count_ = 0;
            entry_point_ = UINT32_MAX;
            max_level_ = 0;
            vectors_.clear();
            std::fill(base_sizes_.begin(), base_sizes_.end(), 0);
            upper_.clear();
            levels_.clear();
            deleted_.clear();
            num_deleted_ = 0;
            return false;
        };
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

        if (count_ > max_elements_) return fail();
        if (metric_val > static_cast<uint32_t>(DistanceMetric::Cosine)) return fail();
        if (M_ <= 1 || M_ > 128 || M0_ == 0 || M0_ > 256) return fail();
        if (ef_construction_ == 0 || ef_construction_ > 4096) return fail();
        // Cap the level count read from an untrusted snapshot. Without this an
        // attacker-controlled level_val drives a multi-gigabyte upper_[i].resize()
        // below, throwing std::length_error/bad_alloc and aborting the instance.
        if (level_val > MAX_DESERIALIZE_LEVEL) return fail();
        if (count_ == 0) {
            if (entry_point_ != UINT32_MAX) return fail();
            max_level_ = 0;
        } else if (entry_point_ >= count_) {
            return fail();
        }
        level_mult_ = 1.0 / std::log(static_cast<double>(M_));

        // count_ <= max_elements_ and dims_ is trusted, but compute the byte size
        // and the end offset without 32-bit wraparound (size_t is 32-bit on wasm32):
        // check the multiply and use subtraction for the bounds test.
        if (dims_ != 0 && count_ > (SIZE_MAX / sizeof(float)) / dims_) return fail();
        size_t vec_count = count_ * dims_;
        size_t vec_bytes = vec_count * sizeof(float);
        if (offset > data_size || data_size - offset < vec_bytes) return fail();
        vectors_.resize(vec_count);
        memcpy(vectors_.data(), data + offset, vec_bytes);
        offset += vec_bytes;

        // Reject non-finite vector components (NaN, Inf).
        // Bit-level check required — std::isfinite is unreliable under -ffast-math.
        for (size_t i = 0; i < vec_count; ++i) {
            uint32_t bits;
            memcpy(&bits, &vectors_[i], 4);
            if (((bits >> 23) & 0xFF) == 0xFF) return fail();
        }

        base_neighbors_.assign(max_elements_ * M0_, 0);
        base_sizes_.assign(max_elements_, 0);
        upper_.resize(count_);
        levels_.resize(count_);
        int observed_max_level = 0;
        for (size_t i = 0; i < count_; ++i) {
            uint32_t lvl;
            if (!safe_read_u32(lvl)) return fail();
            if (lvl > static_cast<uint32_t>(max_level_)) return fail();
            levels_[i] = static_cast<int>(lvl);
            observed_max_level = std::max(observed_max_level, levels_[i]);
            upper_[i].resize(lvl > 0 ? lvl : 0);
            for (int l = 0; l <= static_cast<int>(lvl); ++l) {
                uint32_t sz;
                size_t max_neighbors = (l == 0) ? M0_ : M_;
                if (!safe_read_u32(sz)) return fail();
                if (sz > max_neighbors) return fail();
                if (offset > data_size || data_size - offset < static_cast<size_t>(sz) * 4) return fail();
                if (l == 0) {
                    base_sizes_[i] = static_cast<uint16_t>(sz);
                    uint32_t* base = base_slot(static_cast<uint32_t>(i));
                    for (uint32_t j = 0; j < sz; ++j) {
                        if (!safe_read_u32(base[j])) return fail();
                        if (base[j] >= count_) return fail();
                    }
                } else {
                    auto& edges = upper_[i][static_cast<size_t>(l - 1)];
                    edges.resize(sz);
                    for (uint32_t j = 0; j < sz; ++j) {
                        if (!safe_read_u32(edges[j])) return fail();
                        if (edges[j] >= count_) return fail();
                    }
                }
            }
        }
        if (count_ > 0 && observed_max_level != max_level_) return fail();

        // NOTE: this zeroes all soft-delete state on load. Ghost entries were
        // serialized as regular data, so they come back live. Callers must
        // compact before serializing if deletes need to survive a round-trip.
        // (Worker persist path enforces this in exportBinary(); see worker.js.)
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
        total += static_cast<size_t>(count_) * M0_ * sizeof(uint32_t);
        total += base_sizes_.size() * sizeof(uint16_t);
        for (const auto& node_levels : upper_) {
            for (const auto& level : node_levels) total += level.size() * sizeof(uint32_t);
        }

        // Class overhead
        total += sizeof(*this);

        return total;
    }

    // Return layer-0 neighbor list for a node (for graph quality metrics)
    std::vector<uint32_t> get_neighbors(uint32_t id) const {
        return copy_level_edges(id, 0);
    }

private:
    uint32_t* base_slot(uint32_t id) {
        return base_neighbors_.data() + static_cast<size_t>(id) * M0_;
    }

    const uint32_t* base_slot(uint32_t id) const {
        return base_neighbors_.data() + static_cast<size_t>(id) * M0_;
    }

    std::vector<uint32_t>& upper_edges(uint32_t id, int level) {
        return upper_[id][static_cast<size_t>(level - 1)];
    }

    const std::vector<uint32_t>& upper_edges(uint32_t id, int level) const {
        return upper_[id][static_cast<size_t>(level - 1)];
    }

    size_t level_edge_count(uint32_t id, int level) const {
        return (level == 0) ? base_sizes_[id] : upper_edges(id, level).size();
    }

    std::vector<uint32_t> copy_level_edges(uint32_t id, int level) const {
        if (level == 0) {
            const uint32_t* edges = base_slot(id);
            return std::vector<uint32_t>(edges, edges + base_sizes_[id]);
        }
        return upper_edges(id, level);
    }

    template<typename Fn>
    void for_each_edge(uint32_t id, int level, Fn&& fn) const {
        if (level == 0) {
            const uint32_t* edges = base_slot(id);
            uint16_t sz = base_sizes_[id];
            for (uint16_t i = 0; i < sz; ++i) fn(edges[i]);
            return;
        }
        for (uint32_t edge : upper_edges(id, level)) fn(edge);
    }

    void write_level_edges(uint32_t node, int level, const std::vector<uint32_t>& edges) {
        if (level == 0) {
            uint16_t sz = static_cast<uint16_t>(edges.size());
            base_sizes_[node] = sz;
            uint32_t* slot = base_slot(node);
            for (uint16_t i = 0; i < sz; ++i) slot[i] = edges[i];
            return;
        }
        upper_edges(node, level) = edges;
    }

    void append_neighbor(uint32_t node, int level, uint32_t neighbor) {
        if (level == 0) {
            // Each base slot holds exactly M0_ entries with no headroom between
            // slots, so appending past M0_ would corrupt the next node's slot
            // (or overrun the buffer for the last node). When the node is already
            // saturated, run the heuristic over the existing edges plus the new
            // candidate and re-select within M0_, rather than the (unsafe) eager
            // append-then-prune the caller would otherwise rely on.
            uint16_t sz = base_sizes_[node];
            if (sz >= M0_) {
                const uint32_t* slot = base_slot(node);
                for (uint16_t i = 0; i < sz; ++i) {
                    if (slot[i] == neighbor) return;  // already linked
                }
                std::vector<std::pair<float, uint32_t>> candidates;
                candidates.reserve(static_cast<size_t>(sz) + 1);
                for (uint16_t i = 0; i < sz; ++i) {
                    candidates.emplace_back(distance(node, slot[i]), slot[i]);
                }
                candidates.emplace_back(distance(node, neighbor), neighbor);
                std::sort(candidates.begin(), candidates.end());
                select_neighbors_heuristic(node, candidates, M0_, 0);
                return;
            }
            base_slot(node)[base_sizes_[node]++] = neighbor;
            return;
        }
        upper_edges(node, level).push_back(neighbor);
    }

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
#elif defined(FLOAT_HNSW_AVX2_SIMD)
        __m256 acc = _mm256_setzero_ps();
        for (; d + 8 <= dims_; d += 8) {
            __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, acc);
        sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
#elif defined(FLOAT_HNSW_SSE2_SIMD)
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
#elif defined(FLOAT_HNSW_AVX2_SIMD)
        __m256 acc = _mm256_setzero_ps();
        for (; d + 8 <= dims_; d += 8) {
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d)));
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, acc);
        dot = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
#elif defined(FLOAT_HNSW_SSE2_SIMD)
        __m128 acc = _mm_setzero_ps();
        for (; d + 4 <= dims_; d += 4) {
            acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(a + d), _mm_loadu_ps(b + d)));
        }
        alignas(16) float tmp[4];
        _mm_store_ps(tmp, acc);
        dot = tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif
        for (; d < dims_; d++) dot += a[d] * b[d];
        if (dot > 1.0f) dot = 1.0f;
        else if (dot < -1.0f) dot = -1.0f;
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
            for_each_edge(curr, level, [&](uint32_t neighbor) {
                if (is_visited(neighbor)) return;
                mark_visited(neighbor);
                float nd = distance(query_id, neighbor);
                if (nd < results.top().first || results.size() < ef) {
                    candidates.emplace(nd, neighbor);
                    results.emplace(nd, neighbor);
                    if (results.size() > ef) results.pop();
                }
            });
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
            for_each_edge(curr, level, [&](uint32_t neighbor) {
                if (is_visited(neighbor) || deleted_[neighbor]) return;
                mark_visited(neighbor);
                float nd = distance_to_query(neighbor);
                if (nd < results.top().first || results.size() < ef) {
                    candidates.emplace(nd, neighbor);
                    results.emplace(nd, neighbor);
                    if (results.size() > ef) results.pop();
                }
            });
        }
        std::vector<std::pair<float, uint32_t>> res;
        while (!results.empty()) { res.push_back(results.top()); results.pop(); }
        std::reverse(res.begin(), res.end());
        return res;
    }

    void select_neighbors_heuristic(uint32_t node, std::vector<std::pair<float, uint32_t>>& candidates, size_t M, int level) {
        std::vector<uint32_t> result;
        if (use_heuristic_) {
            const size_t nc = candidates.size();
            const size_t selected_capacity = std::min(M, nc);
            const float uncached = std::numeric_limits<float>::lowest();
            std::vector<float> selected_cache(nc * selected_capacity, uncached);
            std::vector<size_t> selected_indices;
            selected_indices.reserve(selected_capacity);
            auto cached_distance = [&](size_t cand_idx, size_t selected_pos) -> float {
                float& cached = selected_cache[cand_idx * selected_capacity + selected_pos];
                if (cached != uncached) return cached;
                size_t selected_idx = selected_indices[selected_pos];
                float d = distance(candidates[cand_idx].second, candidates[selected_idx].second);
                cached = d;
                return d;
            };

            size_t best_rejected = nc;
            for (size_t ci = 0; ci < nc; ci++) {
                if (result.size() >= M) break;
                auto& cand = candidates[ci];
                bool keep = true;
                for (size_t sel_pos = 0; sel_pos < selected_indices.size(); sel_pos++) {
                    if (cached_distance(ci, sel_pos) < cand.first) { keep = false; break; }
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
        write_level_edges(node, level, result);
    }

    void prune_neighbors(uint32_t node, int level, size_t M) {
        std::vector<std::pair<float, uint32_t>> candidates;
        for_each_edge(node, level, [&](uint32_t n) {
            candidates.emplace_back(distance(node, n), n);
        });
        std::sort(candidates.begin(), candidates.end());
        select_neighbors_heuristic(node, candidates, M, level);
    }

    bool has_neighbor(uint32_t node, int level, uint32_t neighbor) const {
        std::vector<uint32_t> nbrs = copy_level_edges(node, level);
        for (uint32_t n : nbrs) {
            if (n == neighbor) return true;
        }
        return false;
    }

    void ensure_base_neighbor(uint32_t node, uint32_t neighbor) {
        std::vector<uint32_t> nbrs = copy_level_edges(node, 0);
        for (uint32_t n : nbrs) {
            if (n == neighbor) return;
        }

        nbrs.push_back(neighbor);
        if (nbrs.size() <= M0_) {
            write_level_edges(node, 0, nbrs);
            return;
        }

        write_level_edges(node, 0, nbrs);
        prune_neighbors(node, 0, M0_);
        if (has_neighbor(node, 0, neighbor)) return;

        size_t worst = 0;
        float worst_dist = distance(node, nbrs[0]);
        for (size_t i = 1; i < nbrs.size(); ++i) {
            float d = distance(node, nbrs[i]);
            if (d > worst_dist) {
                worst = i;
                worst_dist = d;
            }
        }
        nbrs[worst] = neighbor;
        write_level_edges(node, 0, nbrs);
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
    std::vector<uint32_t> base_neighbors_;
    std::vector<uint16_t> base_sizes_;
    std::vector<std::vector<std::vector<uint32_t>>> upper_;
    std::vector<int> levels_;
    std::vector<uint8_t> deleted_;
    std::vector<uint32_t> visited_list_;
    uint32_t visited_curr_;
};

} // namespace wasm
} // namespace pancake
