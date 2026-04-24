#pragma once
/**
 * Quantized Vector Engine for Cloudflare Workers (WASM SIMD)
 *
 * Row-Wise Affine Quantization: float32 -> uint8
 * Memory: 512 bytes -> 136 bytes (3.76x compression)
 *
 * Designed for 128MB memory limit, 50ms CPU limit
 */

#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

// 1. Force WASM mode if we are in Emscripten
#if defined(__EMSCRIPTEN__) || defined(__wasm__)
  #ifndef PANCAKE_WASM_BUILD
    #define PANCAKE_WASM_BUILD 1
  #endif
#endif

// 2. Only include what is necessary for the platform
#ifdef PANCAKE_WASM_BUILD
  #include <wasm_simd128.h>
  #include <emscripten.h> // Useful for EMSCRIPTEN_KEEPALIVE
#else
  // This block MUST be invisible to the WASM compiler
  #include <x86intrin.h>
  #include <immintrin.h>
#endif

namespace pancake {
namespace wasm {

// =============================================================================
// Affine Quantized Vector (128D optimized)
// =============================================================================

struct alignas(16) AffineVector128 {
    float scale;      // (max - min) / 255
    float offset;     // min value
    uint8_t data[128]; // Quantized dimensions

    static constexpr size_t size() { return sizeof(float) * 2 + 128; }
};

// Dynamic dimension version
struct AffineVectorDyn {
    float scale;
    float offset;
    size_t dim;
    uint8_t data[];  // Flexible array member

    size_t size() const { return sizeof(float) * 2 + sizeof(size_t) + dim; }
};

// =============================================================================
// Quantization Functions
// =============================================================================

// Quantize float32 vector to uint8 with affine encoding
inline void quantize_128d(const float* input, AffineVector128& output) {
    // Find min/max
    float min_val = input[0];
    float max_val = input[0];

#ifdef PANCAKE_WASM_SIMD
    // SIMD min/max finding
    v128_t vmin = wasm_v128_load(input);
    v128_t vmax = vmin;

    for (size_t i = 4; i < 128; i += 4) {
        v128_t v = wasm_v128_load(input + i);
        vmin = wasm_f32x4_min(vmin, v);
        vmax = wasm_f32x4_max(vmax, v);
    }

    // Reduce to scalar
    float mins[4], maxs[4];
    wasm_v128_store(mins, vmin);
    wasm_v128_store(maxs, vmax);
    min_val = std::min({mins[0], mins[1], mins[2], mins[3]});
    max_val = std::max({maxs[0], maxs[1], maxs[2], maxs[3]});
#else
    for (size_t i = 1; i < 128; ++i) {
        min_val = std::min(min_val, input[i]);
        max_val = std::max(max_val, input[i]);
    }
#endif

    // Handle edge case
    if (max_val == min_val) {
        max_val = min_val + 1.0f;
    }

    float range = max_val - min_val;
    output.scale = range / 255.0f;
    output.offset = min_val;

    // Quantize: q = round(255 * (x - min) / range)
    float inv_scale = 255.0f / range;

#ifdef PANCAKE_WASM_SIMD
    v128_t v_offset = wasm_f32x4_splat(min_val);
    v128_t v_inv_scale = wasm_f32x4_splat(inv_scale);
    v128_t v_255 = wasm_f32x4_splat(255.0f);
    v128_t v_0 = wasm_f32x4_splat(0.0f);

    for (size_t i = 0; i < 128; i += 4) {
        v128_t v = wasm_v128_load(input + i);
        v128_t normalized = wasm_f32x4_mul(wasm_f32x4_sub(v, v_offset), v_inv_scale);

        // Clamp to [0, 255]
        normalized = wasm_f32x4_max(v_0, wasm_f32x4_min(v_255, normalized));

        // Round and convert to int
        v128_t rounded = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_add(normalized, wasm_f32x4_splat(0.5f)));

        // Extract and store as uint8
        output.data[i]   = (uint8_t)wasm_i32x4_extract_lane(rounded, 0);
        output.data[i+1] = (uint8_t)wasm_i32x4_extract_lane(rounded, 1);
        output.data[i+2] = (uint8_t)wasm_i32x4_extract_lane(rounded, 2);
        output.data[i+3] = (uint8_t)wasm_i32x4_extract_lane(rounded, 3);
    }
#else
    for (size_t i = 0; i < 128; ++i) {
        float normalized = (input[i] - min_val) * inv_scale;
        output.data[i] = (uint8_t)std::min(255.0f, std::max(0.0f, std::round(normalized)));
    }
#endif
}

// Dequantize back to float32
inline void dequantize_128d(const AffineVector128& input, float* output) {
    float scale = input.scale;
    float offset = input.offset;

#ifdef PANCAKE_WASM_SIMD
    v128_t v_scale = wasm_f32x4_splat(scale);
    v128_t v_offset = wasm_f32x4_splat(offset);

    for (size_t i = 0; i < 128; i += 4) {
        v128_t q = wasm_i32x4_make(input.data[i], input.data[i+1],
                                    input.data[i+2], input.data[i+3]);
        v128_t qf = wasm_f32x4_convert_i32x4(q);
        v128_t val = wasm_f32x4_add(v_offset, wasm_f32x4_mul(qf, v_scale));
        wasm_v128_store(output + i, val);
    }
#else
    for (size_t i = 0; i < 128; ++i) {
        output[i] = offset + input.data[i] * scale;
    }
#endif
}

// =============================================================================
// WASM SIMD Distance Functions
// =============================================================================

#ifdef PANCAKE_WASM_SIMD

// Optimized L2 Squared Distance for 128D quantized vectors
// Processes 16 bytes at a time using proper SIMD widening
inline float l2_squared_128d(const AffineVector128& query, const AffineVector128& target) {
    float s1 = query.scale;
    float s2 = target.scale;
    float b1 = query.offset;
    float b2 = target.offset;

    v128_t sum = wasm_f32x4_splat(0.0f);
    v128_t v_s1 = wasm_f32x4_splat(s1);
    v128_t v_s2 = wasm_f32x4_splat(s2);
    v128_t v_b1 = wasm_f32x4_splat(b1);
    v128_t v_b2 = wasm_f32x4_splat(b2);

    // Process 16 dimensions per iteration (128 / 8 = 16 iterations for full vector)
    for (size_t i = 0; i < 128; i += 16) {
        // Load 16 bytes from each vector
        v128_t q1_u8 = wasm_v128_load(&query.data[i]);
        v128_t q2_u8 = wasm_v128_load(&target.data[i]);

        // Widen u8x16 -> u16x8 (low half)
        v128_t q1_u16_lo = wasm_u16x8_extend_low_u8x16(q1_u8);
        v128_t q2_u16_lo = wasm_u16x8_extend_low_u8x16(q2_u8);

        // Widen u8x16 -> u16x8 (high half)
        v128_t q1_u16_hi = wasm_u16x8_extend_high_u8x16(q1_u8);
        v128_t q2_u16_hi = wasm_u16x8_extend_high_u8x16(q2_u8);

        // Process low 8 dimensions (4 + 4)
        {
            // u16x8 -> u32x4 (low)
            v128_t q1_i32 = wasm_u32x4_extend_low_u16x8(q1_u16_lo);
            v128_t q2_i32 = wasm_u32x4_extend_low_u16x8(q2_u16_lo);

            // Convert to float
            v128_t q1_f = wasm_f32x4_convert_i32x4(q1_i32);
            v128_t q2_f = wasm_f32x4_convert_i32x4(q2_i32);

            // Dequantize: val = offset + q * scale
            v128_t v1 = wasm_f32x4_add(v_b1, wasm_f32x4_mul(q1_f, v_s1));
            v128_t v2 = wasm_f32x4_add(v_b2, wasm_f32x4_mul(q2_f, v_s2));

            // Difference squared
            v128_t diff = wasm_f32x4_sub(v1, v2);
            sum = wasm_f32x4_add(sum, wasm_f32x4_mul(diff, diff));
        }
        {
            // u16x8 -> u32x4 (high of low)
            v128_t q1_i32 = wasm_u32x4_extend_high_u16x8(q1_u16_lo);
            v128_t q2_i32 = wasm_u32x4_extend_high_u16x8(q2_u16_lo);

            v128_t q1_f = wasm_f32x4_convert_i32x4(q1_i32);
            v128_t q2_f = wasm_f32x4_convert_i32x4(q2_i32);

            v128_t v1 = wasm_f32x4_add(v_b1, wasm_f32x4_mul(q1_f, v_s1));
            v128_t v2 = wasm_f32x4_add(v_b2, wasm_f32x4_mul(q2_f, v_s2));

            v128_t diff = wasm_f32x4_sub(v1, v2);
            sum = wasm_f32x4_add(sum, wasm_f32x4_mul(diff, diff));
        }

        // Process high 8 dimensions (4 + 4)
        {
            v128_t q1_i32 = wasm_u32x4_extend_low_u16x8(q1_u16_hi);
            v128_t q2_i32 = wasm_u32x4_extend_low_u16x8(q2_u16_hi);

            v128_t q1_f = wasm_f32x4_convert_i32x4(q1_i32);
            v128_t q2_f = wasm_f32x4_convert_i32x4(q2_i32);

            v128_t v1 = wasm_f32x4_add(v_b1, wasm_f32x4_mul(q1_f, v_s1));
            v128_t v2 = wasm_f32x4_add(v_b2, wasm_f32x4_mul(q2_f, v_s2));

            v128_t diff = wasm_f32x4_sub(v1, v2);
            sum = wasm_f32x4_add(sum, wasm_f32x4_mul(diff, diff));
        }
        {
            v128_t q1_i32 = wasm_u32x4_extend_high_u16x8(q1_u16_hi);
            v128_t q2_i32 = wasm_u32x4_extend_high_u16x8(q2_u16_hi);

            v128_t q1_f = wasm_f32x4_convert_i32x4(q1_i32);
            v128_t q2_f = wasm_f32x4_convert_i32x4(q2_i32);

            v128_t v1 = wasm_f32x4_add(v_b1, wasm_f32x4_mul(q1_f, v_s1));
            v128_t v2 = wasm_f32x4_add(v_b2, wasm_f32x4_mul(q2_f, v_s2));

            v128_t diff = wasm_f32x4_sub(v1, v2);
            sum = wasm_f32x4_add(sum, wasm_f32x4_mul(diff, diff));
        }
    }

    // Horizontal sum of 4 float lanes
    return wasm_f32x4_extract_lane(sum, 0) +
           wasm_f32x4_extract_lane(sum, 1) +
           wasm_f32x4_extract_lane(sum, 2) +
           wasm_f32x4_extract_lane(sum, 3);
}

#else

// Scalar fallback
inline float l2_squared_128d(const AffineVector128& query, const AffineVector128& target) {
    float s1 = query.scale;
    float s2 = target.scale;
    float b1 = query.offset;
    float b2 = target.offset;

    float sum = 0.0f;
    for (size_t i = 0; i < 128; ++i) {
        float v1 = b1 + query.data[i] * s1;
        float v2 = b2 + target.data[i] * s2;
        float diff = v1 - v2;
        sum += diff * diff;
    }
    return sum;
}

#endif // PANCAKE_WASM_SIMD

// =============================================================================
// Quantized HNSW Index (Memory-Optimized for Cloudflare Workers)
// =============================================================================

class QuantizedIndex {
public:
    explicit QuantizedIndex(size_t max_elements, size_t M = 16, size_t ef_construction = 200)
        : max_elements_(max_elements)
        , M_(M)
        , ef_construction_(ef_construction)
        , ef_search_(50)
        , count_(0)
    {
        vectors_.reserve(max_elements);
        // HNSW graph storage would go here
    }

    // Insert a float vector (will be quantized)
    uint64_t insert(const float* vec) {
        if (count_ >= max_elements_) {
            return UINT64_MAX; // Full
        }

        AffineVector128 qvec;
        quantize_128d(vec, qvec);

        uint64_t id = count_++;
        vectors_.push_back(qvec);

        // TODO: Build HNSW connections

        return id;
    }

    // Search with float query (will be quantized)
    std::vector<std::pair<uint64_t, float>> search(const float* query, size_t k) {
        AffineVector128 qquery;
        quantize_128d(query, qquery);

        // Brute force for now - HNSW would be much faster
        std::vector<std::pair<float, uint64_t>> candidates;
        candidates.reserve(count_);

        for (size_t i = 0; i < count_; ++i) {
            float dist = l2_squared_128d(qquery, vectors_[i]);
            candidates.emplace_back(dist, i);
        }

        // Partial sort to get top-k
        if (candidates.size() > k) {
            std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end());
            candidates.resize(k);
        } else {
            std::sort(candidates.begin(), candidates.end());
        }

        std::vector<std::pair<uint64_t, float>> results;
        results.reserve(k);
        for (const auto& [dist, id] : candidates) {
            results.emplace_back(id, dist);
        }
        return results;
    }

    size_t count() const { return count_; }
    size_t memory_bytes() const {
        return count_ * AffineVector128::size() + sizeof(*this);
    }

private:
    size_t max_elements_;
    size_t M_;
    size_t ef_construction_;
    size_t ef_search_;
    size_t count_;
    std::vector<AffineVector128> vectors_;
};

// =============================================================================
// C API for WASM exports
// =============================================================================

extern "C" {

// Global index instance (for Durable Object)
static QuantizedIndex* g_index = nullptr;

// Initialize the index
int init(size_t max_elements) {
    if (g_index) delete g_index;
    g_index = new QuantizedIndex(max_elements);
    return 0;
}

// Insert a vector (128 floats)
uint64_t insert(const float* vec) {
    if (!g_index) return UINT64_MAX;
    return g_index->insert(vec);
}

// Search for k nearest neighbors
// Results written to out_ids and out_dists (must be pre-allocated with k elements)
int search(const float* query, size_t k, uint64_t* out_ids, float* out_dists) {
    if (!g_index) return -1;

    auto results = g_index->search(query, k);
    for (size_t i = 0; i < results.size(); ++i) {
        out_ids[i] = results[i].first;
        out_dists[i] = results[i].second;
    }
    return (int)results.size();
}

// Get index stats
size_t get_count() {
    return g_index ? g_index->count() : 0;
}

size_t get_memory() {
    return g_index ? g_index->memory_bytes() : 0;
}

} // extern "C"

} // namespace wasm
} // namespace pancake
