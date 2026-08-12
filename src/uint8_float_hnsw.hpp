#pragma once
/**
 * Uint8FloatHNSW -- Runtime-dimension uint8-quantized HNSW index.
 *
 * Stores vectors as row-wise affine uint8: per-vector scale + offset + uint8[dims].
 * Insert accepts float32, quantizes internally. Search uses asymmetric distance:
 * float32 query vs dequantized-on-the-fly uint8 database vectors.
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
#include "float_hnsw.hpp"
#include <cstdint>

#if defined(__wasm_simd128__)
    #include <wasm_simd128.h>
    #define UINT8_HNSW_WASM_SIMD 1
#elif defined(PANCAKE_ENABLE_AVX512_SIMD) && defined(__AVX512F__) && defined(__AVX512BW__)
    #include <immintrin.h>
    #define UINT8_HNSW_AVX512_SIMD 1
#elif defined(PANCAKE_ENABLE_AVX2_SIMD) && defined(__AVX2__)
    #include <immintrin.h>
    #define UINT8_HNSW_AVX2_SIMD 1
#elif defined(PANCAKE_ENABLE_SSE2_SIMD) && defined(__SSE2__)
    #include <xmmintrin.h>
    #include <emmintrin.h>
    #define UINT8_HNSW_SSE2_SIMD 1
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

// Fused multiply-add helper: returns acc + a*b.
// Uses WASM relaxed-SIMD FMA when available (compile with -mrelaxed-simd),
// otherwise falls back to separate mul+add. Recall-neutral either way.
#if defined(UINT8_HNSW_WASM_SIMD)
  #if defined(__wasm_relaxed_simd__)
    #define WFMA(acc, a, b) wasm_f32x4_relaxed_madd((a), (b), (acc))
  #else
    #define WFMA(acc, a, b) wasm_f32x4_add((acc), wasm_f32x4_mul((a), (b)))
  #endif
#endif

namespace pancake {
namespace wasm {

#if defined(PANCAKE_UINT8_HNSW_BUILD_PROFILE)
struct BuildProfile {
    uint64_t inserts = 0;

    double quantize_ms = 0;
    double search_layers_ms = 0;
    double search_base_ms = 0;
    double select_neighbors_ms = 0;
    double connect_ms = 0;
    double prune_sort_ms = 0;
    double prune_select_ms = 0;

    uint64_t dist_calls = 0;
    uint64_t candidate_pushes = 0;
    uint64_t candidate_pops = 0;
    uint64_t visited_marks = 0;
    uint64_t base_append_fast = 0;
    uint64_t base_prune_calls = 0;
    uint64_t upper_prune_calls = 0;
    uint64_t base_prune_candidates = 0;
    uint64_t upper_prune_candidates = 0;

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
        printf("    prune sort:       %.2f ms total (%.4f ms/ins)\n", prune_sort_ms, prune_sort_ms / n);
        printf("    prune heuristic:  %.2f ms total (%.4f ms/ins)\n", prune_select_ms, prune_select_ms / n);
        printf("  dist calls/insert:  %.1f\n", (double)dist_calls / n);
        printf("  cand pushes/insert: %.1f\n", (double)candidate_pushes / n);
        printf("  cand pops/insert:   %.1f\n", (double)candidate_pops / n);
        printf("  visited marks/ins:  %.1f\n", (double)visited_marks / n);
        printf("  base fast appends:  %.1f\n", (double)base_append_fast / n);
        printf("  base prune calls:   %.1f (avg cand %.1f)\n",
               (double)base_prune_calls / n,
               base_prune_calls ? (double)base_prune_candidates / base_prune_calls : 0.0);
        printf("  upper prune calls:  %.1f (avg cand %.1f)\n",
               (double)upper_prune_calls / n,
               upper_prune_calls ? (double)upper_prune_candidates / upper_prune_calls : 0.0);
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
#define PROFILE_COUNT(code) do { code; } while (0)
#else
#define PROFILE_BLOCK(field, code) do { code; } while (0)
#define PROFILE_COUNT(code) do {} while (0)
#endif

struct Uint8FloatHNSWConfig {
    size_t M = 12;
    size_t ef_construction = 75;
    size_t ef_search = 100;
    size_t max_elements = 100000;
    uint32_t seed = 108;
    DistanceMetric metric = DistanceMetric::L2;
    bool use_heuristic = true;
};

class Uint8FloatHNSW {
public:
    struct Edge {
        uint32_t neighbor;
        float dist;
    };
    static_assert(sizeof(Edge) == 8, "Edge must pack tight");

    struct UpperLevel {
        std::vector<Edge> edges;
    };

    // Upper bound on the HNSW level count accepted from an untrusted snapshot;
    // caps attacker-controlled allocations during deserialize(). See float_hnsw.
    static constexpr uint32_t MAX_DESERIALIZE_LEVEL = 64;

    Uint8FloatHNSW(size_t dims, const Uint8FloatHNSWConfig& config = {})
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
        base_edges_.resize(max_elements_ * M0_);
        base_sizes_.assign(max_elements_, 0);
        upper_.reserve(max_elements_);
        levels_.reserve(max_elements_);
        deleted_.reserve(max_elements_);
        level_mult_ = 1.0 / std::log(static_cast<double>(M_));
        visited_list_.assign(max_elements_, 0);
        visited_curr_ = 0;
        scratch_norm_.resize(dims_);
    }

    uint32_t insert(const float* vec) {
        if (count_ >= max_elements_) return UINT32_MAX;

        const float* src = vec;
        if (metric_ == DistanceMetric::Cosine) {
            if (!normalize_cosine_vector(vec, scratch_norm_.data(), dims_)) return UINT32_MAX;
            src = scratch_norm_.data();
        }

        uint32_t id = static_cast<uint32_t>(count_++);
        PROFILE_COUNT(g_build_profile.inserts++);

        PROFILE_BLOCK(quantize_ms, {
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
        });

        deleted_.push_back(0);

        int level = random_level();
        levels_.push_back(level);
        base_sizes_[id] = 0;
        upper_.push_back(std::vector<UpperLevel>(level > 0 ? static_cast<size_t>(level) : 0));

        if (entry_point_ == UINT32_MAX) {
            entry_point_ = id;
            max_level_ = level;
            return id;
        }

        cached_insert_ = src;

        PROFILE_BLOCK(search_layers_ms, {
            uint32_t curr_upper = entry_point_;
            float curr_dist_upper = distance_to_insert(curr_upper);

            for (int l = max_level_; l > level; --l) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    for_each_edge(curr_upper, l, [&](const Edge& edge) {
                        float d = distance_to_insert(edge.neighbor);
                        if (d < curr_dist_upper) {
                            curr_upper = edge.neighbor;
                            curr_dist_upper = d;
                            changed = true;
                        }
                    });
                }
            }
            insert_entry_ = curr_upper;
        });

        uint32_t curr = insert_entry_;
        for (int l = std::min(level, max_level_); l >= 0; --l) {
            std::vector<std::pair<float, uint32_t>> candidates;
            PROFILE_BLOCK(search_base_ms, {
                candidates = search_layer_insert(curr, ef_construction_, l);
            });
            size_t max_n = (l == 0) ? M0_ : M_;
            PROFILE_BLOCK(select_neighbors_ms, {
                // Insert traversal stays asymmetric, but stored edges and the
                // graph-maintenance heuristic use symmetric node-to-node
                // distances. Recompute before selection so Edge.dist has one
                // durable meaning across insert, prune, compact, and import.
                for (auto& candidate : candidates) {
                    candidate.first = distance(id, candidate.second);
                }
                std::sort(candidates.begin(), candidates.end());
                select_neighbors_heuristic(id, candidates, max_n, l);
            });
            PROFILE_BLOCK(connect_ms, {
                for_each_edge(id, l, [&](const Edge& edge) {
                    append_edge_with_prune(edge.neighbor, l, Edge{id, edge.dist}, max_n);
                });
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

    std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k, size_t ef_search) {
        if (count_ == 0) return {};

        if (metric_ == DistanceMetric::Cosine) {
            norm_query_.resize(dims_);
            if (!normalize_cosine_vector(query, norm_query_.data(), dims_)) return {};
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
                for_each_edge(curr, l, [&](const Edge& edge) {
                    float d = distance_to_query(edge.neighbor);
                    if (d < curr_dist) {
                        curr = edge.neighbor;
                        curr_dist = d;
                        changed = true;
                    }
                });
            }
        }

        auto candidates = search_layer_query(curr, std::max(ef_search, k), 0);
        std::vector<std::pair<uint32_t, float>> results;
        for (size_t i = 0; i < std::min(k, candidates.size()); ++i) {
            if (!deleted_[candidates[i].second]) {
                results.emplace_back(candidates[i].second, candidates[i].first);
            }
        }
        return results;
    }

    std::vector<std::pair<uint32_t, float>> search_filtered(
        const float* query, size_t k,
        const uint8_t* filter_bitset, size_t bitset_len,
        size_t ef_search
    ) {
        if (count_ == 0) return {};

        if (metric_ == DistanceMetric::Cosine) {
            norm_query_.resize(dims_);
            if (!normalize_cosine_vector(query, norm_query_.data(), dims_)) return {};
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
                for_each_edge(curr, l, [&](const Edge& edge) {
                    float d = distance_to_query(edge.neighbor);
                    if (d < curr_dist) {
                        curr = edge.neighbor;
                        curr_dist = d;
                        changed = true;
                    }
                });
            }
        }

        size_t initial_ef = std::max(ef_search, k * 2);
        size_t max_ef = initial_ef * 4;
        size_t current_ef = initial_ef;

        std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, std::greater<>> candidates;
        std::priority_queue<std::pair<float, uint32_t>> results;

        prepare_visited();
        float d = distance_to_query(curr);
        candidates.emplace(d, curr);
        mark_visited(curr);

        size_t filtered_count = 0;
        float lower_bound = std::numeric_limits<float>::max();

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

            if (filtered_count >= k && cand_dist > lower_bound) {
                break;
            }

            if (filtered_count < k && cand_dist > lower_bound && results.size() >= current_ef) {
                if (current_ef >= max_ef) break;
                current_ef = std::min(current_ef * 2, max_ef);
            }

            candidates.pop();

            const Edge* edges = base_slot(cand_id);
            uint16_t edge_count = base_sizes_[cand_id];
            for (uint16_t ei = 0; ei < edge_count; ++ei) {
                uint32_t neighbor = edges[ei].neighbor;
                if (is_visited(neighbor)) continue;
                mark_visited(neighbor);

                float nd = distance_to_query(neighbor);
                bool dominated = (results.size() >= current_ef && nd > lower_bound);
                if (!dominated) {
                    candidates.emplace(nd, neighbor);

                    size_t byte_idx = neighbor >> 3;
                    if (!deleted_[neighbor] && byte_idx < bitset_len && (filter_bitset[byte_idx] & (1u << (neighbor & 7)))) {
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

                if (filtered_count >= k && candidates.empty()) break;
            }
        }

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

    void mark_delete(uint32_t id) {
        if (id >= count_ || deleted_[id] != 0) return;
        deleted_[id] = 1;
        num_deleted_++;
    }

    void set_ef_search(size_t ef) { ef_search_ = ef; }

    std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k) {
        return search(query, k, ef_search_);
    }

    std::vector<std::pair<uint32_t, float>> search_filtered(
        const float* query, size_t k,
        const uint8_t* filter_bitset, size_t bitset_len
    ) {
        return search_filtered(query, k, filter_bitset, bitset_len, ef_search_);
    }

    size_t ghost_count() const { return num_deleted_; }

    float ghost_ratio() const {
        return count_ > 0 ? static_cast<float>(num_deleted_) / count_ : 0.0f;
    }

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

        // A graph with at least half its nodes deleted needs fresh topology,
        // not just edge repair. Dequantize the stored live vectors and rebuild
        // in old-ID order so callers receive the same stable remapping contract.
        if (num_deleted_ >= (count_ + 1) / 2) {
            out_map.assign(count_, UINT32_MAX);
            uint32_t live_id = 0;
            for (uint32_t old_id = 0; old_id < count_; ++old_id) {
                if (deleted_[old_id]) continue;
                if (live_id != old_id) {
                    std::memcpy(
                        &qdata_[static_cast<size_t>(live_id) * dims_],
                        &qdata_[static_cast<size_t>(old_id) * dims_],
                        dims_
                    );
                    scales_[live_id] = scales_[old_id];
                    offsets_[live_id] = offsets_[old_id];
                    sum_q_[live_id] = sum_q_[old_id];
                    sum_q2_[live_id] = sum_q2_[old_id];
                }
                out_map[old_id] = live_id++;
            }

            qdata_.resize(static_cast<size_t>(live_id) * dims_);
            scales_.resize(live_id);
            offsets_.resize(live_id);
            sum_q_.resize(live_id);
            sum_q2_.resize(live_id);
            qdata_.shrink_to_fit();
            scales_.shrink_to_fit();
            offsets_.shrink_to_fit();
            sum_q_.shrink_to_fit();
            sum_q2_.shrink_to_fit();

            // Free the hollow graph before constructing the replacement so
            // the allocator can reuse its blocks instead of growing the WASM
            // heap to hold both complete indexes at once.
            std::vector<Edge>().swap(base_edges_);
            std::vector<uint16_t>().swap(base_sizes_);
            decltype(upper_)().swap(upper_);
            std::vector<int>().swap(levels_);
            std::vector<uint8_t>().swap(deleted_);
            std::vector<uint32_t>().swap(visited_list_);

            Uint8FloatHNSWConfig config;
            config.M = M_;
            config.ef_construction = ef_construction_;
            config.ef_search = ef_search_;
            config.max_elements = max_elements_;
            config.metric = metric_;
            config.use_heuristic = use_heuristic_;
            Uint8FloatHNSW rebuilt(dims_, config);

            std::vector<float> restored(dims_);
            for (uint32_t new_id = 0; new_id < live_id; ++new_id) {
                dequantize(new_id, restored.data());
                const uint32_t inserted = rebuilt.insert(restored.data());
                if (inserted == UINT32_MAX) {
                    out_map.clear();
                    return;
                }
            }
            *this = std::move(rebuilt);
            return;
        }

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
                    base_sizes_[new_id] = base_sizes_[old_id];
                    if (base_sizes_[old_id] != 0) {
                        std::memcpy(base_slot(new_id), base_slot(old_id), static_cast<size_t>(base_sizes_[old_id]) * sizeof(Edge));
                    }
                    upper_[new_id] = std::move(upper_[old_id]);
                    levels_[new_id] = levels_[old_id];
                }
                new_id++;
            }
        }

        for (uint32_t i = 0; i < new_id; i++) {
            Edge* base = base_slot(i);
            uint16_t base_write = 0;
            for (uint16_t r = 0; r < base_sizes_[i]; ++r) {
                uint32_t mapped = id_map[base[r].neighbor];
                if (mapped != UINT32_MAX) {
                    base[base_write++] = Edge{mapped, base[r].dist};
                }
            }
            base_sizes_[i] = base_write;

            for (int l = 1; l <= levels_[i]; ++l) {
                auto& edges = upper_edges(i, l);
                size_t write = 0;
                for (size_t r = 0; r < edges.size(); ++r) {
                    uint32_t mapped = id_map[edges[r].neighbor];
                    if (mapped != UINT32_MAX) {
                        edges[write++] = Edge{mapped, edges[r].dist};
                    }
                }
                edges.resize(write);
            }
        }

        for (uint32_t i = 0; i < new_id; i++) {
            for (int l = 0; l <= levels_[i]; l++) {
                size_t target = (l == 0) ? M0_ : M_;
                if (level_edge_count(i, l) >= target) continue;

                std::vector<Edge> old_nbrs = copy_level_edges(i, l);
                std::unordered_set<uint32_t> old_set;
                old_set.reserve(old_nbrs.size());
                for (const Edge& edge : old_nbrs) old_set.insert(edge.neighbor);

                std::unordered_set<uint32_t> seen;
                seen.insert(i);
                for (const Edge& edge : old_nbrs) seen.insert(edge.neighbor);

                std::vector<uint32_t> expansion;
                for (const Edge& edge : old_nbrs) {
                    uint32_t n = edge.neighbor;
                    if (n >= new_id) continue;
                    if (l > levels_[n]) continue;
                    for_each_edge(n, l, [&](const Edge& expanded) {
                        uint32_t nn = expanded.neighbor;
                        if (nn < new_id && seen.find(nn) == seen.end()) {
                            seen.insert(nn);
                            expansion.push_back(nn);
                        }
                    });
                }

                if (expansion.empty()) continue;

                std::vector<std::pair<float, uint32_t>> candidates;
                candidates.reserve(old_nbrs.size() + expansion.size());
                for (const Edge& edge : old_nbrs) candidates.emplace_back(edge.dist, edge.neighbor);
                for (uint32_t n : expansion) candidates.emplace_back(distance(i, n), n);
                std::sort(candidates.begin(), candidates.end());

                select_neighbors_heuristic(i, candidates, target, l);

                std::vector<Edge> new_edges = copy_level_edges(i, l);
                for (const Edge& edge : new_edges) {
                    if (old_set.count(edge.neighbor)) continue;
                    if (edge.neighbor >= new_id || l > levels_[edge.neighbor]) continue;
                    append_edge_with_prune(edge.neighbor, l, Edge{i, edge.dist}, target);
                }
            }
        }

        // If local repair still leaves a survivor isolated at the base layer,
        // reconnect it against the live pool before finalizing compaction.
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

                std::vector<Edge> new_edges = copy_level_edges(i, 0);
                uint32_t anchor = UINT32_MAX;
                float anchor_dist = std::numeric_limits<float>::max();
                for (const Edge& edge : new_edges) {
                    append_edge_with_prune(edge.neighbor, 0, Edge{i, edge.dist}, M0_);
                    if (has_level_neighbor(edge.neighbor, 0, i)) {
                        anchor = UINT32_MAX;
                        break;
                    }
                    if (edge.dist < anchor_dist) {
                        anchor = edge.neighbor;
                        anchor_dist = edge.dist;
                    }
                }
                if (anchor != UINT32_MAX) {
                    ensure_base_neighbor(anchor, Edge{i, anchor_dist});
                }
            }
        }

        entry_point_ = UINT32_MAX;
        max_level_ = 0;
        for (uint32_t i = 0; i < new_id; i++) {
            if (entry_point_ == UINT32_MAX || levels_[i] > max_level_) {
                max_level_ = levels_[i];
                entry_point_ = i;
            }
        }
        if (entry_point_ == UINT32_MAX && new_id > 0) entry_point_ = 0;

        qdata_.resize(new_id * dims_);
        scales_.resize(new_id);
        offsets_.resize(new_id);
        sum_q_.resize(new_id);
        sum_q2_.resize(new_id);
        upper_.resize(new_id);
        levels_.resize(new_id);
        deleted_.assign(new_id, 0);
        count_ = new_id;
        num_deleted_ = 0;
    }

    std::vector<uint8_t> serialize() const {
        size_t total_size = 10 * 4;
        total_size += count_ * sizeof(float) * 2;
        total_size += count_ * dims_;
        for (size_t i = 0; i < count_; ++i) {
            total_size += 4;
            total_size += 4 + static_cast<size_t>(base_sizes_[i]) * sizeof(Edge);
            for (int l = 1; l <= levels_[i]; ++l) {
                total_size += 4 + upper_edges(i, l).size() * sizeof(Edge);
            }
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

        push_u32(0x49384831);
        push_u32(static_cast<uint32_t>(dims_));
        push_u32(2);
        push_u32(static_cast<uint32_t>(count_));
        push_u32(entry_point_);
        push_u32(static_cast<uint32_t>(max_level_));
        push_u32(static_cast<uint32_t>(M_));
        push_u32(static_cast<uint32_t>(M0_));
        push_u32(static_cast<uint32_t>(metric_));
        push_u32(static_cast<uint32_t>(ef_construction_));

        for (size_t i = 0; i < count_; ++i) push_f32(scales_[i]);
        for (size_t i = 0; i < count_; ++i) push_f32(offsets_[i]);

        buffer.insert(buffer.end(), qdata_.begin(), qdata_.begin() + count_ * dims_);

        for (size_t i = 0; i < count_; ++i) {
            push_u32(static_cast<uint32_t>(levels_[i]));
            push_u32(static_cast<uint32_t>(base_sizes_[i]));
            const Edge* base = base_slot(static_cast<uint32_t>(i));
            for (uint16_t ei = 0; ei < base_sizes_[i]; ++ei) {
                push_u32(base[ei].neighbor);
                push_f32(base[ei].dist);
            }
            for (int l = 1; l <= levels_[i]; ++l) {
                const auto& edges = upper_edges(static_cast<uint32_t>(i), l);
                push_u32(static_cast<uint32_t>(edges.size()));
                for (const Edge& edge : edges) {
                    push_u32(edge.neighbor);
                    push_f32(edge.dist);
                }
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
            qdata_.clear();
            scales_.clear();
            offsets_.clear();
            sum_q_.clear();
            sum_q2_.clear();
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
        auto safe_read_f32 = [&](float& out) -> bool {
            if (offset + 4 > data_size) return false;
            memcpy(&out, data + offset, 4);
            offset += 4;
            return true;
        };

        uint32_t magic, dims_val;
        if (!safe_read_u32(magic)) return false;
        bool has_version = (magic == 0x49384831);
        if (magic != 0x49384857 && !has_version) return false;
        if (!safe_read_u32(dims_val) || dims_val != static_cast<uint32_t>(dims_)) return false;

        uint32_t version = 0;
        if (has_version) {
            if (!safe_read_u32(version)) return false;
            // Reject unknown future versions instead of parsing them as v2:
            // a changed layout must fail closed, not misparse (contract §6).
            if (version > 2) return false;
        }

        uint32_t count_val, entry_val, level_val, m_val, m0_val, metric_val;
        if (!safe_read_u32(count_val)) return false;
        if (!safe_read_u32(entry_val)) return false;
        if (!safe_read_u32(level_val)) return false;
        if (!safe_read_u32(m_val)) return false;
        if (!safe_read_u32(m0_val)) return false;
        if (!safe_read_u32(metric_val)) return false;

        uint32_t ef_construction_val = 200;
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
        if (M_ <= 1 || M_ > 128 || M0_ != M_ * 2) return fail();
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

        // count_ <= max_elements_, so count_*sizeof(float)*2 cannot overflow a
        // 32-bit size_t; use subtraction so a near-SIZE_MAX offset can't wrap.
        if (offset > data_size || data_size - offset < count_ * sizeof(float) * 2) return fail();
        scales_.resize(count_);
        offsets_.resize(count_);
        for (size_t i = 0; i < count_; ++i) {
            if (!safe_read_f32(scales_[i])) return fail();
            uint32_t bits;
            memcpy(&bits, &scales_[i], 4);
            if (((bits >> 23) & 0xFF) == 0xFF) return fail();
            // A zero/negative scale collapses a vector to a constant and poisons
            // every distance to it; insert() never produces one (it guards
            // range < 1e-30f), so reject it from snapshots too.
            if (scales_[i] <= 0.0f || scales_[i] > 1e20f) return fail();
        }
        for (size_t i = 0; i < count_; ++i) {
            if (!safe_read_f32(offsets_[i])) return fail();
            uint32_t bits;
            memcpy(&bits, &offsets_[i], 4);
            if (((bits >> 23) & 0xFF) == 0xFF) return fail();
            if (offsets_[i] > 1e20f || offsets_[i] < -1e20f) return fail();
        }

        // count_ <= max_elements_ and dims_ is trusted, so count_*dims_ stays
        // within 32-bit size_t; use subtraction to avoid offset+len wraparound.
        size_t qdata_bytes = count_ * dims_;
        if (offset > data_size || data_size - offset < qdata_bytes) return fail();
        qdata_.resize(qdata_bytes);
        memcpy(qdata_.data(), data + offset, qdata_bytes);
        offset += qdata_bytes;

        base_edges_.assign(max_elements_ * M0_, Edge{0, 0.0f});
        base_sizes_.assign(max_elements_, 0);
        upper_.clear();
        upper_.resize(count_);
        levels_.resize(count_);
        int observed_max_level = 0;
        bool recompute_edge_distances = (version < 2);

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

                if (version >= 2) {
                    if (offset > data_size || data_size - offset < static_cast<size_t>(sz) * sizeof(Edge)) return fail();
                } else {
                    if (offset > data_size || data_size - offset < static_cast<size_t>(sz) * 4) return fail();
                }

                if (l == 0) {
                    base_sizes_[i] = static_cast<uint16_t>(sz);
                    Edge* base = base_slot(static_cast<uint32_t>(i));
                    for (uint32_t j = 0; j < sz; ++j) {
                        if (!safe_read_u32(base[j].neighbor)) return fail();
                        if (base[j].neighbor >= count_) return fail();
                        if (version >= 2) {
                            if (!safe_read_f32(base[j].dist)) return fail();
                            uint32_t bits;
                            memcpy(&bits, &base[j].dist, 4);
                            if (((bits >> 23) & 0xFF) == 0xFF) return fail();
                        } else {
                            base[j].dist = 0.0f;
                        }
                    }
                } else {
                    auto& edges = upper_edges(static_cast<uint32_t>(i), l);
                    edges.resize(sz);
                    for (uint32_t j = 0; j < sz; ++j) {
                        if (!safe_read_u32(edges[j].neighbor)) return fail();
                        if (edges[j].neighbor >= count_) return fail();
                        if (version >= 2) {
                            if (!safe_read_f32(edges[j].dist)) return fail();
                            uint32_t bits;
                            memcpy(&bits, &edges[j].dist, 4);
                            if (((bits >> 23) & 0xFF) == 0xFF) return fail();
                        } else {
                            edges[j].dist = 0.0f;
                        }
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
        if (max_elements_ > visited_list_.size()) visited_list_.assign(max_elements_, 0);
        // Safe to reset to 0 here: search paths always call prepare_visited()
        // before the first is_visited() check, which bumps this generation to 1.
        visited_curr_ = 0;

        sum_q_.resize(count_);
        sum_q2_.resize(count_);
        for (size_t i = 0; i < count_; ++i) {
            const uint8_t* row = &qdata_[i * dims_];
            uint32_t sq = 0;
            uint32_t sq2 = 0;
            for (size_t d = 0; d < dims_; d++) {
                uint32_t v = row[d];
                sq += v;
                sq2 += v * v;
            }
            sum_q_[i] = sq;
            sum_q2_[i] = sq2;
        }

        if (recompute_edge_distances) {
            for (uint32_t i = 0; i < count_; ++i) {
                Edge* base = base_slot(i);
                for (uint16_t ei = 0; ei < base_sizes_[i]; ++ei) {
                    base[ei].dist = distance(i, base[ei].neighbor);
                }
                for (int l = 1; l <= levels_[i]; ++l) {
                    auto& edges = upper_edges(i, l);
                    for (Edge& edge : edges) {
                        edge.dist = distance(i, edge.neighbor);
                    }
                }
            }
        }

        return true;
    }

    size_t count() const { return count_; }
    size_t dims() const { return dims_; }

    size_t memory_bytes() const {
        size_t total = count_ * dims_;
        total += count_ * sizeof(float) * 2;
        total += base_edges_.size() * sizeof(Edge);
        total += base_sizes_.size() * sizeof(uint16_t);
        for (const auto& node_levels : upper_) {
            for (const auto& level : node_levels) {
                total += level.edges.size() * sizeof(Edge);
            }
        }
        total += sizeof(*this);
        return total;
    }

    std::vector<uint32_t> get_neighbors(uint32_t id) const {
        std::vector<uint32_t> out;
        out.reserve(base_sizes_[id]);
        const Edge* edges = base_slot(id);
        for (uint16_t i = 0; i < base_sizes_[id]; ++i) out.push_back(edges[i].neighbor);
        return out;
    }

private:
    Edge* base_slot(uint32_t id) {
        return base_edges_.data() + static_cast<size_t>(id) * M0_;
    }

    const Edge* base_slot(uint32_t id) const {
        return base_edges_.data() + static_cast<size_t>(id) * M0_;
    }

    std::vector<Edge>& upper_edges(uint32_t id, int level) {
        return upper_[id][static_cast<size_t>(level - 1)].edges;
    }

    const std::vector<Edge>& upper_edges(uint32_t id, int level) const {
        return upper_[id][static_cast<size_t>(level - 1)].edges;
    }

    size_t level_edge_count(uint32_t id, int level) const {
        return (level == 0) ? base_sizes_[id] : upper_edges(id, level).size();
    }

    std::vector<Edge> copy_level_edges(uint32_t id, int level) const {
        if (level == 0) {
            const Edge* edges = base_slot(id);
            return std::vector<Edge>(edges, edges + base_sizes_[id]);
        }
        return upper_edges(id, level);
    }

    bool has_level_neighbor(uint32_t id, int level, uint32_t neighbor) const {
        if (level == 0) {
            const Edge* edges = base_slot(id);
            for (uint16_t i = 0; i < base_sizes_[id]; ++i) {
                if (edges[i].neighbor == neighbor) return true;
            }
            return false;
        }
        for (const Edge& edge : upper_edges(id, level)) {
            if (edge.neighbor == neighbor) return true;
        }
        return false;
    }

    template<typename Fn>
    void for_each_edge(uint32_t id, int level, Fn&& fn) const {
        if (level == 0) {
            const Edge* edges = base_slot(id);
            uint16_t sz = base_sizes_[id];
            for (uint16_t i = 0; i < sz; ++i) fn(edges[i]);
            return;
        }
        for (const Edge& edge : upper_edges(id, level)) fn(edge);
    }

    void write_level_edges(uint32_t node, int level, const std::vector<Edge>& edges) {
        if (level == 0) {
            // Base-layer storage is a fixed-width slot. Keep this boundary safe
            // even if a future caller accidentally supplies an oversized list.
            uint16_t sz = static_cast<uint16_t>(std::min(edges.size(), M0_));
            base_sizes_[node] = sz;
            Edge* slot = base_slot(node);
            for (uint16_t i = 0; i < sz; ++i) slot[i] = edges[i];
            return;
        }
        upper_edges(node, level) = edges;
    }

    void append_edge_with_prune(uint32_t node, int level, const Edge& edge, size_t M) {
        if (level == 0) {
            uint16_t& sz = base_sizes_[node];
            Edge* slot = base_slot(node);
            if (sz < M0_) {
                PROFILE_COUNT(g_build_profile.base_append_fast++);
                slot[sz++] = edge;
                return;
            }

            // Fast reject: when the base layer is already full, a new reciprocal
            // edge that is no better than the current worst edge and is already
            // dominated by one of the kept neighbors cannot survive the full
            // heuristic rerun. In that case skip the allocation/sort/heuristic
            // path entirely.
            float worst_dist = slot[0].dist;
            bool duplicate = false;
            for (uint16_t i = 0; i < sz; ++i) {
                worst_dist = std::max(worst_dist, slot[i].dist);
                if (slot[i].neighbor == edge.neighbor) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) return;

            if (edge.dist >= worst_dist) {
                for (uint16_t i = 0; i < sz; ++i) {
                    if (distance(edge.neighbor, slot[i].neighbor) < edge.dist) {
                        return;
                    }
                }
            }

            std::vector<std::pair<float, uint32_t>> candidates;
            candidates.reserve(static_cast<size_t>(sz) + 1);
            for (uint16_t i = 0; i < sz; ++i) {
                candidates.emplace_back(slot[i].dist, slot[i].neighbor);
            }
            candidates.emplace_back(edge.dist, edge.neighbor);
            PROFILE_COUNT({
                g_build_profile.base_prune_calls++;
                g_build_profile.base_prune_candidates += candidates.size();
            });
#if defined(PANCAKE_UINT8_HNSW_BUILD_PROFILE)
            double sort_t0 = emscripten_get_now();
            std::sort(candidates.begin(), candidates.end());
            g_build_profile.prune_sort_ms += emscripten_get_now() - sort_t0;
            double select_t0 = emscripten_get_now();
            select_neighbors_heuristic(node, candidates, M, level);
            g_build_profile.prune_select_ms += emscripten_get_now() - select_t0;
#else
            std::sort(candidates.begin(), candidates.end());
            select_neighbors_heuristic(node, candidates, M, level);
#endif
            return;
        }

        auto& edges = upper_edges(node, level);
        edges.push_back(edge);
        if (edges.size() > M) {
            PROFILE_COUNT({
                g_build_profile.upper_prune_calls++;
                g_build_profile.upper_prune_candidates += edges.size();
            });
            prune_neighbors(node, level, M);
        }
    }

    void ensure_base_neighbor(uint32_t node, const Edge& edge) {
        uint16_t& sz = base_sizes_[node];
        Edge* slot = base_slot(node);
        for (uint16_t i = 0; i < sz; ++i) {
            if (slot[i].neighbor == edge.neighbor) {
                slot[i].dist = edge.dist;
                return;
            }
        }

        append_edge_with_prune(node, 0, edge, M0_);
        if (has_level_neighbor(node, 0, edge.neighbor)) return;

        if (sz < M0_) {
            slot[sz++] = edge;
            return;
        }

        uint16_t worst = 0;
        for (uint16_t i = 1; i < sz; ++i) {
            if (slot[i].dist > slot[worst].dist) worst = i;
        }
        slot[worst] = edge;
    }

    void dequantize(uint32_t id, float* dst) const {
        const uint8_t* data = &qdata_[id * dims_];
        float s = scales_[id];
        float o = offsets_[id];
        for (size_t d = 0; d < dims_; d++) {
            dst[d] = o + s * static_cast<float>(data[d]);
        }
    }

    float asymmetric_cosine(const float* query, uint32_t db_id) const {
        const uint8_t* data = &qdata_[db_id * dims_];
        float s = scales_[db_id];
        float o = offsets_[db_id];
        float dot = 0.0f;
        size_t d = 0;

#ifdef UINT8_HNSW_WASM_SIMD
        v128_t acc0 = wasm_f32x4_splat(0.0f);
        v128_t acc1 = wasm_f32x4_splat(0.0f);
        v128_t acc2 = wasm_f32x4_splat(0.0f);
        v128_t acc3 = wasm_f32x4_splat(0.0f);
        v128_t v_scale = wasm_f32x4_splat(s);
        v128_t v_offset = wasm_f32x4_splat(o);

        for (; d + 16 <= dims_; d += 16) {
            v128_t bytes = wasm_v128_load(data + d);

            v128_t u16_lo = wasm_u16x8_extend_low_u8x16(bytes);
            v128_t u16_hi = wasm_u16x8_extend_high_u8x16(bytes);

            v128_t f0 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_low_u16x8(u16_lo));
            v128_t val0 = WFMA(v_offset, f0, v_scale);
            acc0 = WFMA(acc0, wasm_v128_load(query + d), val0);

            v128_t f1 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_high_u16x8(u16_lo));
            v128_t val1 = WFMA(v_offset, f1, v_scale);
            acc1 = WFMA(acc1, wasm_v128_load(query + d + 4), val1);

            v128_t f2 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_low_u16x8(u16_hi));
            v128_t val2 = WFMA(v_offset, f2, v_scale);
            acc2 = WFMA(acc2, wasm_v128_load(query + d + 8), val2);

            v128_t f3 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_high_u16x8(u16_hi));
            v128_t val3 = WFMA(v_offset, f3, v_scale);
            acc3 = WFMA(acc3, wasm_v128_load(query + d + 12), val3);
        }

        v128_t acc = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1),
                                    wasm_f32x4_add(acc2, acc3));
        dot = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) +
              wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
#elif defined(UINT8_HNSW_AVX512_SIMD)
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        __m512 v_scale = _mm512_set1_ps(s);
        __m512 v_offset = _mm512_set1_ps(o);

        for (; d + 64 <= dims_; d += 64) {
            __m512 ff0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + d))));
            __m512 val0 = _mm512_add_ps(v_offset, _mm512_mul_ps(ff0, v_scale));
            acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(_mm512_loadu_ps(query + d), val0));

            __m512 ff1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + d + 16))));
            __m512 val1 = _mm512_add_ps(v_offset, _mm512_mul_ps(ff1, v_scale));
            acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(_mm512_loadu_ps(query + d + 16), val1));

            __m512 ff2 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + d + 32))));
            __m512 val2 = _mm512_add_ps(v_offset, _mm512_mul_ps(ff2, v_scale));
            acc2 = _mm512_add_ps(acc2, _mm512_mul_ps(_mm512_loadu_ps(query + d + 32), val2));

            __m512 ff3 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + d + 48))));
            __m512 val3 = _mm512_add_ps(v_offset, _mm512_mul_ps(ff3, v_scale));
            acc3 = _mm512_add_ps(acc3, _mm512_mul_ps(_mm512_loadu_ps(query + d + 48), val3));
        }

        __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        alignas(64) float tmp[16];
        _mm512_store_ps(tmp, acc);
        for (int i = 0; i < 16; ++i) dot += tmp[i];
#elif defined(UINT8_HNSW_AVX2_SIMD)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 v_scale = _mm256_set1_ps(s);
        __m256 v_offset = _mm256_set1_ps(o);

        for (; d + 32 <= dims_; d += 32) {
            __m128i bytes0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + d));
            __m256i u16_0 = _mm256_cvtepu8_epi16(bytes0);
            __m256i i32_0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(u16_0));
            __m256i i32_1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(u16_0, 1));

            __m256 ff0 = _mm256_cvtepi32_ps(i32_0);
            __m256 val0 = _mm256_add_ps(v_offset, _mm256_mul_ps(ff0, v_scale));
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(query + d), val0));

            __m256 ff1 = _mm256_cvtepi32_ps(i32_1);
            __m256 val1 = _mm256_add_ps(v_offset, _mm256_mul_ps(ff1, v_scale));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_loadu_ps(query + d + 8), val1));

            __m128i bytes1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + d + 16));
            __m256i u16_1 = _mm256_cvtepu8_epi16(bytes1);
            __m256i i32_2 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(u16_1));
            __m256i i32_3 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(u16_1, 1));

            __m256 ff2 = _mm256_cvtepi32_ps(i32_2);
            __m256 val2 = _mm256_add_ps(v_offset, _mm256_mul_ps(ff2, v_scale));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_loadu_ps(query + d + 16), val2));

            __m256 ff3 = _mm256_cvtepi32_ps(i32_3);
            __m256 val3 = _mm256_add_ps(v_offset, _mm256_mul_ps(ff3, v_scale));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_loadu_ps(query + d + 24), val3));
        }

        __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, acc);
        dot = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
#elif defined(UINT8_HNSW_SSE2_SIMD)
        __m128 acc0 = _mm_setzero_ps();
        __m128 acc1 = _mm_setzero_ps();
        __m128 acc2 = _mm_setzero_ps();
        __m128 acc3 = _mm_setzero_ps();
        __m128 v_scale = _mm_set1_ps(s);
        __m128 v_offset = _mm_set1_ps(o);
        __m128i zero = _mm_setzero_si128();

        for (; d + 16 <= dims_; d += 16) {
            __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + d));
            __m128i u16_lo = _mm_unpacklo_epi8(bytes, zero);
            __m128i u16_hi = _mm_unpackhi_epi8(bytes, zero);

            #define SSE2_ASYM_DOT(u16v, lohi, offset, accN) { \
                __m128 ff = _mm_cvtepi32_ps(_mm_unpack##lohi##_epi16(u16v, zero)); \
                __m128 val = _mm_add_ps(v_offset, _mm_mul_ps(ff, v_scale)); \
                accN = _mm_add_ps(accN, _mm_mul_ps(_mm_loadu_ps(query + d + offset), val)); \
            }
            SSE2_ASYM_DOT(u16_lo, lo, 0,  acc0);
            SSE2_ASYM_DOT(u16_lo, hi, 4,  acc1);
            SSE2_ASYM_DOT(u16_hi, lo, 8,  acc2);
            SSE2_ASYM_DOT(u16_hi, hi, 12, acc3);
            #undef SSE2_ASYM_DOT
        }

        __m128 acc = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
        alignas(16) float tmp[4];
        _mm_store_ps(tmp, acc);
        dot = tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif
        for (; d < dims_; d++) {
            dot += query[d] * (o + s * static_cast<float>(data[d]));
        }
        if (dot > 1.0f) dot = 1.0f;
        else if (dot < -1.0f) dot = -1.0f;
        return 1.0f - dot;
    }

    float asymmetric_l2(const float* query, uint32_t db_id) const {
        const uint8_t* data = &qdata_[db_id * dims_];
        float s = scales_[db_id];
        float o = offsets_[db_id];
        float sum = 0.0f;
        size_t d = 0;

#ifdef UINT8_HNSW_WASM_SIMD
        v128_t acc0 = wasm_f32x4_splat(0.0f);
        v128_t acc1 = wasm_f32x4_splat(0.0f);
        v128_t acc2 = wasm_f32x4_splat(0.0f);
        v128_t acc3 = wasm_f32x4_splat(0.0f);
        v128_t v_scale = wasm_f32x4_splat(s);
        v128_t v_offset = wasm_f32x4_splat(o);

        for (; d + 16 <= dims_; d += 16) {
            v128_t bytes = wasm_v128_load(data + d);
            v128_t u16_lo = wasm_u16x8_extend_low_u8x16(bytes);
            v128_t u16_hi = wasm_u16x8_extend_high_u8x16(bytes);

            v128_t f0 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_low_u16x8(u16_lo));
            v128_t val0 = WFMA(v_offset, f0, v_scale);
            v128_t diff0 = wasm_f32x4_sub(wasm_v128_load(query + d), val0);
            acc0 = WFMA(acc0, diff0, diff0);

            v128_t f1 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_high_u16x8(u16_lo));
            v128_t val1 = WFMA(v_offset, f1, v_scale);
            v128_t diff1 = wasm_f32x4_sub(wasm_v128_load(query + d + 4), val1);
            acc1 = WFMA(acc1, diff1, diff1);

            v128_t f2 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_low_u16x8(u16_hi));
            v128_t val2 = WFMA(v_offset, f2, v_scale);
            v128_t diff2 = wasm_f32x4_sub(wasm_v128_load(query + d + 8), val2);
            acc2 = WFMA(acc2, diff2, diff2);

            v128_t f3 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_high_u16x8(u16_hi));
            v128_t val3 = WFMA(v_offset, f3, v_scale);
            v128_t diff3 = wasm_f32x4_sub(wasm_v128_load(query + d + 12), val3);
            acc3 = WFMA(acc3, diff3, diff3);
        }

        v128_t acc = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1),
                                    wasm_f32x4_add(acc2, acc3));
        sum = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) +
              wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
#elif defined(UINT8_HNSW_AVX512_SIMD)
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        __m512 v_scale = _mm512_set1_ps(s);
        __m512 v_offset = _mm512_set1_ps(o);

        for (; d + 64 <= dims_; d += 64) {
            __m512 ff0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + d))));
            __m512 val0 = _mm512_add_ps(v_offset, _mm512_mul_ps(ff0, v_scale));
            __m512 diff0 = _mm512_sub_ps(_mm512_loadu_ps(query + d), val0);
            acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(diff0, diff0));

            __m512 ff1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + d + 16))));
            __m512 val1 = _mm512_add_ps(v_offset, _mm512_mul_ps(ff1, v_scale));
            __m512 diff1 = _mm512_sub_ps(_mm512_loadu_ps(query + d + 16), val1);
            acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(diff1, diff1));

            __m512 ff2 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + d + 32))));
            __m512 val2 = _mm512_add_ps(v_offset, _mm512_mul_ps(ff2, v_scale));
            __m512 diff2 = _mm512_sub_ps(_mm512_loadu_ps(query + d + 32), val2);
            acc2 = _mm512_add_ps(acc2, _mm512_mul_ps(diff2, diff2));

            __m512 ff3 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + d + 48))));
            __m512 val3 = _mm512_add_ps(v_offset, _mm512_mul_ps(ff3, v_scale));
            __m512 diff3 = _mm512_sub_ps(_mm512_loadu_ps(query + d + 48), val3);
            acc3 = _mm512_add_ps(acc3, _mm512_mul_ps(diff3, diff3));
        }

        __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        alignas(64) float tmp[16];
        _mm512_store_ps(tmp, acc);
        for (int i = 0; i < 16; ++i) sum += tmp[i];
#elif defined(UINT8_HNSW_AVX2_SIMD)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 v_scale = _mm256_set1_ps(s);
        __m256 v_offset = _mm256_set1_ps(o);

        for (; d + 32 <= dims_; d += 32) {
            __m128i bytes0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + d));
            __m256i u16_0 = _mm256_cvtepu8_epi16(bytes0);
            __m256i i32_0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(u16_0));
            __m256i i32_1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(u16_0, 1));

            __m256 ff0 = _mm256_cvtepi32_ps(i32_0);
            __m256 val0 = _mm256_add_ps(v_offset, _mm256_mul_ps(ff0, v_scale));
            __m256 diff0 = _mm256_sub_ps(_mm256_loadu_ps(query + d), val0);
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(diff0, diff0));

            __m256 ff1 = _mm256_cvtepi32_ps(i32_1);
            __m256 val1 = _mm256_add_ps(v_offset, _mm256_mul_ps(ff1, v_scale));
            __m256 diff1 = _mm256_sub_ps(_mm256_loadu_ps(query + d + 8), val1);
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(diff1, diff1));

            __m128i bytes1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + d + 16));
            __m256i u16_1 = _mm256_cvtepu8_epi16(bytes1);
            __m256i i32_2 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(u16_1));
            __m256i i32_3 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(u16_1, 1));

            __m256 ff2 = _mm256_cvtepi32_ps(i32_2);
            __m256 val2 = _mm256_add_ps(v_offset, _mm256_mul_ps(ff2, v_scale));
            __m256 diff2 = _mm256_sub_ps(_mm256_loadu_ps(query + d + 16), val2);
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(diff2, diff2));

            __m256 ff3 = _mm256_cvtepi32_ps(i32_3);
            __m256 val3 = _mm256_add_ps(v_offset, _mm256_mul_ps(ff3, v_scale));
            __m256 diff3 = _mm256_sub_ps(_mm256_loadu_ps(query + d + 24), val3);
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(diff3, diff3));
        }

        __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, acc);
        sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
#elif defined(UINT8_HNSW_SSE2_SIMD)
        __m128 acc0 = _mm_setzero_ps();
        __m128 acc1 = _mm_setzero_ps();
        __m128 acc2 = _mm_setzero_ps();
        __m128 acc3 = _mm_setzero_ps();
        __m128 v_scale = _mm_set1_ps(s);
        __m128 v_offset = _mm_set1_ps(o);
        __m128i zero = _mm_setzero_si128();

        for (; d + 16 <= dims_; d += 16) {
            __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + d));
            __m128i u16_lo = _mm_unpacklo_epi8(bytes, zero);
            __m128i u16_hi = _mm_unpackhi_epi8(bytes, zero);

            #define SSE2_ASYM_L2(u16v, lohi, offset, accN) { \
                __m128 ff = _mm_cvtepi32_ps(_mm_unpack##lohi##_epi16(u16v, zero)); \
                __m128 val = _mm_add_ps(v_offset, _mm_mul_ps(ff, v_scale)); \
                __m128 diff = _mm_sub_ps(_mm_loadu_ps(query + d + offset), val); \
                accN = _mm_add_ps(accN, _mm_mul_ps(diff, diff)); \
            }
            SSE2_ASYM_L2(u16_lo, lo, 0,  acc0);
            SSE2_ASYM_L2(u16_lo, hi, 4,  acc1);
            SSE2_ASYM_L2(u16_hi, lo, 8,  acc2);
            SSE2_ASYM_L2(u16_hi, hi, 12, acc3);
            #undef SSE2_ASYM_L2
        }

        __m128 acc = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
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

    uint32_t uint8_dot(uint32_t a, uint32_t b) const {
        const uint8_t* da = &qdata_[a * dims_];
        const uint8_t* db = &qdata_[b * dims_];
        uint32_t sum = 0;
        size_t d = 0;

#ifdef UINT8_HNSW_WASM_SIMD
        v128_t acc = wasm_i32x4_splat(0);
        for (; d + 16 <= dims_; d += 16) {
            v128_t va = wasm_v128_load(da + d);
            v128_t vb = wasm_v128_load(db + d);

            v128_t lo_a = wasm_u16x8_extend_low_u8x16(va);
            v128_t lo_b = wasm_u16x8_extend_low_u8x16(vb);
            v128_t hi_a = wasm_u16x8_extend_high_u8x16(va);
            v128_t hi_b = wasm_u16x8_extend_high_u8x16(vb);

            acc = wasm_i32x4_add(acc, wasm_i32x4_extmul_low_i16x8(lo_a, lo_b));
            acc = wasm_i32x4_add(acc, wasm_i32x4_extmul_high_i16x8(lo_a, lo_b));
            acc = wasm_i32x4_add(acc, wasm_i32x4_extmul_low_i16x8(hi_a, hi_b));
            acc = wasm_i32x4_add(acc, wasm_i32x4_extmul_high_i16x8(hi_a, hi_b));
        }
        sum = static_cast<uint32_t>(
            wasm_i32x4_extract_lane(acc, 0) + wasm_i32x4_extract_lane(acc, 1) +
            wasm_i32x4_extract_lane(acc, 2) + wasm_i32x4_extract_lane(acc, 3));
#elif defined(UINT8_HNSW_AVX2_SIMD)
        __m256i acc = _mm256_setzero_si256();
        for (; d + 32 <= dims_; d += 32) {
            __m128i va0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(da + d));
            __m128i vb0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(db + d));
            __m256i u16_a0 = _mm256_cvtepu8_epi16(va0);
            __m256i u16_b0 = _mm256_cvtepu8_epi16(vb0);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(u16_a0, u16_b0));

            __m128i va1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(da + d + 16));
            __m128i vb1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(db + d + 16));
            __m256i u16_a1 = _mm256_cvtepu8_epi16(va1);
            __m256i u16_b1 = _mm256_cvtepu8_epi16(vb1);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(u16_a1, u16_b1));
        }
        alignas(32) uint32_t tmp[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), acc);
        sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
#elif defined(UINT8_HNSW_SSE2_SIMD)
        __m128i acc = _mm_setzero_si128();
        __m128i zero = _mm_setzero_si128();
        for (; d + 16 <= dims_; d += 16) {
            __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(da + d));
            __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(db + d));

            __m128i lo_a = _mm_unpacklo_epi8(va, zero);
            __m128i lo_b = _mm_unpacklo_epi8(vb, zero);
            __m128i hi_a = _mm_unpackhi_epi8(va, zero);
            __m128i hi_b = _mm_unpackhi_epi8(vb, zero);

            // _mm_madd_epi16: multiply pairs and horizontally add adjacent
            // results into 32-bit lanes — one instruction for 8 u16 muls.
            acc = _mm_add_epi32(acc, _mm_madd_epi16(lo_a, lo_b));
            acc = _mm_add_epi32(acc, _mm_madd_epi16(hi_a, hi_b));
        }
        // Horizontal sum of 4 x i32 lanes
        __m128i hi64 = _mm_shuffle_epi32(acc, _MM_SHUFFLE(1, 0, 3, 2));
        acc = _mm_add_epi32(acc, hi64);
        __m128i hi32 = _mm_shuffle_epi32(acc, _MM_SHUFFLE(0, 0, 0, 1));
        acc = _mm_add_epi32(acc, hi32);
        sum = static_cast<uint32_t>(_mm_cvtsi128_si32(acc));
#endif
        for (; d < dims_; d++) {
            sum += static_cast<uint32_t>(da[d]) * static_cast<uint32_t>(db[d]);
        }
        return sum;
    }

    float symmetric_l2_u8(uint32_t a, uint32_t b) const {
        float sa = scales_[a], oa = offsets_[a];
        float sb = scales_[b], ob = offsets_[b];
        float D = static_cast<float>(dims_);
        float diff_o = oa - ob;
        float dot_ab = static_cast<float>(uint8_dot(a, b));

        return D * diff_o * diff_o
             + 2.0f * diff_o * (sa * static_cast<float>(sum_q_[a]) - sb * static_cast<float>(sum_q_[b]))
             + sa * sa * static_cast<float>(sum_q2_[a])
             + sb * sb * static_cast<float>(sum_q2_[b])
             - 2.0f * sa * sb * dot_ab;
    }

    float symmetric_cosine_u8(uint32_t a, uint32_t b) const {
        float sa = scales_[a], oa = offsets_[a];
        float sb = scales_[b], ob = offsets_[b];
        float D = static_cast<float>(dims_);
        float dot_ab = static_cast<float>(uint8_dot(a, b));

        float dot = D * oa * ob
                  + oa * sb * static_cast<float>(sum_q_[b])
                  + ob * sa * static_cast<float>(sum_q_[a])
                  + sa * sb * dot_ab;
        if (dot > 1.0f) dot = 1.0f;
        else if (dot < -1.0f) dot = -1.0f;
        return 1.0f - dot;
    }

    float distance(uint32_t a, uint32_t b) const {
        return (metric_ == DistanceMetric::Cosine)
            ? symmetric_cosine_u8(a, b)
            : symmetric_l2_u8(a, b);
    }

    float distance_to_query(uint32_t id) const {
        return (metric_ == DistanceMetric::Cosine)
            ? asymmetric_cosine(cached_query_, id)
            : asymmetric_l2(cached_query_, id);
    }

    float distance_to_insert(uint32_t id) const {
        return (metric_ == DistanceMetric::Cosine)
            ? asymmetric_cosine(cached_insert_, id)
            : asymmetric_l2(cached_insert_, id);
    }

    int random_level() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double sample = dist(rng_);
        if (sample <= 0.0) sample = std::numeric_limits<double>::min();
        return static_cast<int>(-std::log(sample) * level_mult_);
    }

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
        if (!skip_deleted) PROFILE_COUNT(g_build_profile.dist_calls++);
        candidates.emplace(d, entry);
        PROFILE_COUNT(g_build_profile.candidate_pushes++);
        // A soft-deleted entry point stays navigable (candidate queue) but must
        // never occupy a result slot, mirroring how deleted neighbors are treated.
        if (!(skip_deleted && deleted_[entry])) results.emplace(d, entry);
        mark_visited(entry);
        PROFILE_COUNT(g_build_profile.visited_marks++);

        while (!candidates.empty()) {
            auto [curr_dist, curr] = candidates.top();
            candidates.pop();
            PROFILE_COUNT(g_build_profile.candidate_pops++);
            if (results.size() >= ef && curr_dist > results.top().first) break;
            for_each_edge(curr, level, [&](const Edge& edge) {
                uint32_t neighbor = edge.neighbor;
                if (is_visited(neighbor)) return;
                mark_visited(neighbor);
                PROFILE_COUNT(g_build_profile.visited_marks++);
                float nd = dist_func(neighbor);
                if (!skip_deleted) PROFILE_COUNT(g_build_profile.dist_calls++);
                if (results.size() < ef || nd < results.top().first) {
                    candidates.emplace(nd, neighbor);
                    PROFILE_COUNT(g_build_profile.candidate_pushes++);
                    if (!(skip_deleted && deleted_[neighbor])) {
                        results.emplace(nd, neighbor);
                        if (results.size() > ef) results.pop();
                    }
                }
            });
        }

        std::vector<std::pair<float, uint32_t>> res;
        while (!results.empty()) { res.push_back(results.top()); results.pop(); }
        std::reverse(res.begin(), res.end());
        return res;
    }

    std::vector<std::pair<float, uint32_t>> search_layer_insert(uint32_t entry, size_t ef, int level) {
        return search_layer_impl(
            [this](uint32_t id) { return distance_to_insert(id); },
            entry, ef, level, false
        );
    }

    std::vector<std::pair<float, uint32_t>> search_layer_query(uint32_t entry, size_t ef, int level) {
        return search_layer_impl(
            [this](uint32_t id) { return distance_to_query(id); },
            entry, ef, level, true
        );
    }

    void select_neighbors_heuristic(uint32_t node, std::vector<std::pair<float, uint32_t>>& candidates, size_t M, int level) {
        std::vector<Edge> result;
        result.reserve(std::min(M, candidates.size()));

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
                const auto& cand = candidates[ci];
                bool keep = true;
                for (size_t sel_pos = 0; sel_pos < selected_indices.size(); sel_pos++) {
                    if (cached_distance(ci, sel_pos) < cand.first) {
                        keep = false;
                        break;
                    }
                }
                if (keep) {
                    result.push_back(Edge{cand.second, cand.first});
                    selected_indices.push_back(ci);
                } else if (best_rejected == nc) {
                    best_rejected = ci;
                }
            }
            for (size_t ci = best_rejected; ci < nc && result.size() < M; ci++) {
                bool already = false;
                for (const Edge& sel : result) {
                    if (sel.neighbor == candidates[ci].second) {
                        already = true;
                        break;
                    }
                }
                if (!already) result.push_back(Edge{candidates[ci].second, candidates[ci].first});
            }
        } else {
            for (const auto& cand : candidates) {
                if (result.size() >= M) break;
                result.push_back(Edge{cand.second, cand.first});
            }
        }

        write_level_edges(node, level, result);
    }

    void prune_neighbors(uint32_t node, int level, size_t M) {
        std::vector<Edge> edges = copy_level_edges(node, level);
        std::vector<std::pair<float, uint32_t>> candidates;
        candidates.reserve(edges.size());
        for (const Edge& edge : edges) {
            candidates.emplace_back(edge.dist, edge.neighbor);
        }
#if defined(PANCAKE_UINT8_HNSW_BUILD_PROFILE)
        double sort_t0 = emscripten_get_now();
        std::sort(candidates.begin(), candidates.end());
        g_build_profile.prune_sort_ms += emscripten_get_now() - sort_t0;
        double select_t0 = emscripten_get_now();
        select_neighbors_heuristic(node, candidates, M, level);
        g_build_profile.prune_select_ms += emscripten_get_now() - select_t0;
#else
        std::sort(candidates.begin(), candidates.end());
        select_neighbors_heuristic(node, candidates, M, level);
#endif
    }

    void prepare_visited() {
        visited_curr_++;
        if (visited_curr_ == 0) {
            std::fill(visited_list_.begin(), visited_list_.end(), 0);
            visited_curr_ = 1;
        }
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

    std::vector<uint8_t> qdata_;
    std::vector<float> scales_;
    std::vector<float> offsets_;
    std::vector<uint32_t> sum_q_;
    std::vector<uint32_t> sum_q2_;

    const float* cached_query_;
    const float* cached_insert_;
    uint32_t insert_entry_;
    std::vector<float> norm_query_;
    mutable std::vector<float> scratch_norm_;
    std::vector<Edge> base_edges_;
    std::vector<uint16_t> base_sizes_;
    std::vector<std::vector<UpperLevel>> upper_;
    std::vector<int> levels_;
    std::vector<uint8_t> deleted_;
    std::vector<uint32_t> visited_list_;
    uint32_t visited_curr_;
};

} // namespace wasm
} // namespace pancake
