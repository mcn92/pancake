#pragma once
/**
 * Optimized SIMD Distance for Quantized Vectors (Pro Template Version)
 *
 * WHY CROSS-PLATFORM SIMD: Distance computation is the hot path in HNSW search,
 * consuming 70-80% of CPU time. SIMD provides ~4-8x speedup by processing
 * 16 uint8 values in parallel. WASM SIMD for Cloudflare Workers, SSE/AVX for
 * native builds ensures consistent performance across deployment targets.
 *
 * Support for arbitrary dimensions (must be multiple of 16 for SIMD).
 */

#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <initializer_list>

// =============================================================================
// Platform Detection
// =============================================================================

#if defined(__EMSCRIPTEN__) || defined(__wasm__) || defined(__wasm32__)
    #define PANCAKE_WASM_BUILD 1
    #if defined(__wasm_simd128__)
        #include <wasm_simd128.h>
        #define PANCAKE_WASM_SIMD 1
    #endif
#elif !defined(PANCAKE_WASM_BUILD)
    #if defined(_MSC_VER)
        #include <intrin.h>
    #elif defined(__GNUC__) || defined(__clang__)
        #include <x86intrin.h>
    #endif
    #if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
        #define PANCAKE_USE_AVX2 1
    #endif
    #if defined(__SSE4_1__) || defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
        #define PANCAKE_USE_SSE 1
    #endif
#endif

namespace pancake {
namespace wasm {
namespace simd {

// =============================================================================
// Quantized Vector Structure (Templated)
// =============================================================================

template<size_t DIMS>
struct alignas(16) AffineVector {
    float scale;
    float offset;
    uint8_t data[DIMS];

    static constexpr size_t dims() { return DIMS; }
    static constexpr size_t size() { return sizeof(float) * 2 + DIMS; }
};

// Typedef for your existing code compatibility
using AffineVector128 = AffineVector<128>;
using AffineVector256 = AffineVector<256>;
using AffineVector384 = AffineVector<384>;

// =============================================================================
// Scalar Implementation
// =============================================================================

template<size_t DIMS>
inline float l2_squared_scalar(const AffineVector<DIMS>& query, const AffineVector<DIMS>& target) {
    float s1 = query.scale;
    float s2 = target.scale;
    float b1 = query.offset;
    float b2 = target.offset;

    float sum = 0.0f;
    for (size_t i = 0; i < DIMS; ++i) {
        float v1 = b1 + query.data[i] * s1;
        float v2 = b2 + target.data[i] * s2;
        float diff = v1 - v2;
        sum += diff * diff;
    }
    return sum;
}

template<size_t DIMS>
inline float dot_product_scalar(const AffineVector<DIMS>& query, const AffineVector<DIMS>& target) {
    float s1 = query.scale;
    float s2 = target.scale;
    float b1 = query.offset;
    float b2 = target.offset;

    float dot = 0.0f;
    for (size_t i = 0; i < DIMS; ++i) {
        float v1 = b1 + query.data[i] * s1;
        float v2 = b2 + target.data[i] * s2;
        dot += v1 * v2;
    }
    return dot;
}

template<size_t DIMS>
inline float compute_norm_scalar(const AffineVector<DIMS>& vec) {
    float s = vec.scale;
    float b = vec.offset;

    float norm = 0.0f;
    for (size_t i = 0; i < DIMS; ++i) {
        float v = b + vec.data[i] * s;
        norm += v * v;
    }
    return std::sqrt(norm);
}

template<size_t DIMS>
inline float cosine_distance_scalar(const AffineVector<DIMS>& query, const AffineVector<DIMS>& target) {
    float s1 = query.scale;
    float s2 = target.scale;
    float b1 = query.offset;
    float b2 = target.offset;

    float dot = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    for (size_t i = 0; i < DIMS; ++i) {
        float v1 = b1 + query.data[i] * s1;
        float v2 = b2 + target.data[i] * s2;
        dot += v1 * v2;
        norm1 += v1 * v1;
        norm2 += v2 * v2;
    }
    float denom = std::sqrt(norm1) * std::sqrt(norm2);
    return (denom > 1e-8f) ? (1.0f - dot / denom) : 1.0f;
}

template<size_t DIMS>
inline void quantize_scalar(const float* input, AffineVector<DIMS>& output) {
    float min_val = input[0];
    float max_val = input[0];
    for (size_t i = 1; i < DIMS; ++i) {
        min_val = std::min(min_val, input[i]);
        max_val = std::max(max_val, input[i]);
    }

    if (max_val == min_val) max_val = min_val + 1.0f;

    float range = max_val - min_val;
    output.scale = range / 255.0f;
    output.offset = min_val;

    float inv_scale = 255.0f / range;
    for (size_t i = 0; i < DIMS; ++i) {
        float norm = (input[i] - min_val) * inv_scale;
        output.data[i] = (uint8_t)std::min(255.0f, std::max(0.0f, norm + 0.5f));
    }
}

// =============================================================================
// WASM SIMD Implementation
// =============================================================================

#ifdef PANCAKE_WASM_SIMD

template<size_t DIMS>
inline float l2_squared_wasm(const AffineVector<DIMS>& query, const AffineVector<DIMS>& target) {
    v128_t sum = wasm_f32x4_splat(0.0f);
    v128_t v_s1 = wasm_f32x4_splat(query.scale);
    v128_t v_s2 = wasm_f32x4_splat(target.scale);
    v128_t v_b1 = wasm_f32x4_splat(query.offset);
    v128_t v_b2 = wasm_f32x4_splat(target.offset);

    for (size_t i = 0; i < DIMS; i += 16) {
        v128_t q1_u8 = wasm_v128_load(&query.data[i]);
        v128_t q2_u8 = wasm_v128_load(&target.data[i]);

        v128_t q1_u16_lo = wasm_u16x8_extend_low_u8x16(q1_u8);
        v128_t q2_u16_lo = wasm_u16x8_extend_low_u8x16(q2_u8);
        v128_t q1_u16_hi = wasm_u16x8_extend_high_u8x16(q1_u8);
        v128_t q2_u16_hi = wasm_u16x8_extend_high_u8x16(q2_u8);

        // Macro to process 4 dimensions: widen u16->u32->f32, dequantize (offset + q*scale),
        // compute squared difference, accumulate to sum. Used 4 times per 16-byte chunk
        // (low/high of low, low/high of high) to process all 16 dimensions. Macro avoids
        // code duplication while allowing token pasting for low/high selection.
        #define PROCESS_4_TEMPLATE(q1_u16, q2_u16, low_or_high) { \
            v128_t q1_i32 = wasm_u32x4_extend_##low_or_high##_u16x8(q1_u16); \
            v128_t q2_i32 = wasm_u32x4_extend_##low_or_high##_u16x8(q2_u16); \
            v128_t q1_f = wasm_f32x4_convert_i32x4(q1_i32); \
            v128_t q2_f = wasm_f32x4_convert_i32x4(q2_i32); \
            v128_t v1 = wasm_f32x4_add(v_b1, wasm_f32x4_mul(q1_f, v_s1)); \
            v128_t v2 = wasm_f32x4_add(v_b2, wasm_f32x4_mul(q2_f, v_s2)); \
            v128_t diff = wasm_f32x4_sub(v1, v2); \
            sum = wasm_f32x4_add(sum, wasm_f32x4_mul(diff, diff)); \
        }

        PROCESS_4_TEMPLATE(q1_u16_lo, q2_u16_lo, low);
        PROCESS_4_TEMPLATE(q1_u16_lo, q2_u16_lo, high);
        PROCESS_4_TEMPLATE(q1_u16_hi, q2_u16_hi, low);
        PROCESS_4_TEMPLATE(q1_u16_hi, q2_u16_hi, high);
        #undef PROCESS_4_TEMPLATE
    }

    return wasm_f32x4_extract_lane(sum, 0) + wasm_f32x4_extract_lane(sum, 1) +
           wasm_f32x4_extract_lane(sum, 2) + wasm_f32x4_extract_lane(sum, 3);
}

// Compute dot product only (for optimized cosine with precomputed norms)
template<size_t DIMS>
inline float dot_product_wasm(const AffineVector<DIMS>& query, const AffineVector<DIMS>& target) {
    v128_t dot = wasm_f32x4_splat(0.0f);
    v128_t v_s1 = wasm_f32x4_splat(query.scale);
    v128_t v_s2 = wasm_f32x4_splat(target.scale);
    v128_t v_b1 = wasm_f32x4_splat(query.offset);
    v128_t v_b2 = wasm_f32x4_splat(target.offset);

    for (size_t i = 0; i < DIMS; i += 16) {
        v128_t q1_u8 = wasm_v128_load(&query.data[i]);
        v128_t q2_u8 = wasm_v128_load(&target.data[i]);

        v128_t q1_u16_lo = wasm_u16x8_extend_low_u8x16(q1_u8);
        v128_t q2_u16_lo = wasm_u16x8_extend_low_u8x16(q2_u8);
        v128_t q1_u16_hi = wasm_u16x8_extend_high_u8x16(q1_u8);
        v128_t q2_u16_hi = wasm_u16x8_extend_high_u8x16(q2_u8);

        #define PROCESS_4_DOT(q1_u16, q2_u16, low_or_high) { \
            v128_t q1_i32 = wasm_u32x4_extend_##low_or_high##_u16x8(q1_u16); \
            v128_t q2_i32 = wasm_u32x4_extend_##low_or_high##_u16x8(q2_u16); \
            v128_t q1_f = wasm_f32x4_convert_i32x4(q1_i32); \
            v128_t q2_f = wasm_f32x4_convert_i32x4(q2_i32); \
            v128_t v1 = wasm_f32x4_add(v_b1, wasm_f32x4_mul(q1_f, v_s1)); \
            v128_t v2 = wasm_f32x4_add(v_b2, wasm_f32x4_mul(q2_f, v_s2)); \
            dot = wasm_f32x4_add(dot, wasm_f32x4_mul(v1, v2)); \
        }

        PROCESS_4_DOT(q1_u16_lo, q2_u16_lo, low);
        PROCESS_4_DOT(q1_u16_lo, q2_u16_lo, high);
        PROCESS_4_DOT(q1_u16_hi, q2_u16_hi, low);
        PROCESS_4_DOT(q1_u16_hi, q2_u16_hi, high);
        #undef PROCESS_4_DOT
    }

    return wasm_f32x4_extract_lane(dot, 0) + wasm_f32x4_extract_lane(dot, 1) +
           wasm_f32x4_extract_lane(dot, 2) + wasm_f32x4_extract_lane(dot, 3);
}

// Compute norm only (for precomputing at insert time)
template<size_t DIMS>
inline float compute_norm_wasm(const AffineVector<DIMS>& vec) {
    v128_t norm = wasm_f32x4_splat(0.0f);
    v128_t v_s = wasm_f32x4_splat(vec.scale);
    v128_t v_b = wasm_f32x4_splat(vec.offset);

    for (size_t i = 0; i < DIMS; i += 16) {
        v128_t q_u8 = wasm_v128_load(&vec.data[i]);
        v128_t q_u16_lo = wasm_u16x8_extend_low_u8x16(q_u8);
        v128_t q_u16_hi = wasm_u16x8_extend_high_u8x16(q_u8);

        #define PROCESS_4_NORM(q_u16, low_or_high) { \
            v128_t q_i32 = wasm_u32x4_extend_##low_or_high##_u16x8(q_u16); \
            v128_t q_f = wasm_f32x4_convert_i32x4(q_i32); \
            v128_t v = wasm_f32x4_add(v_b, wasm_f32x4_mul(q_f, v_s)); \
            norm = wasm_f32x4_add(norm, wasm_f32x4_mul(v, v)); \
        }

        PROCESS_4_NORM(q_u16_lo, low);
        PROCESS_4_NORM(q_u16_lo, high);
        PROCESS_4_NORM(q_u16_hi, low);
        PROCESS_4_NORM(q_u16_hi, high);
        #undef PROCESS_4_NORM
    }

    float norm_sum = wasm_f32x4_extract_lane(norm, 0) + wasm_f32x4_extract_lane(norm, 1) +
                     wasm_f32x4_extract_lane(norm, 2) + wasm_f32x4_extract_lane(norm, 3);
    return std::sqrt(norm_sum);
}

template<size_t DIMS>
inline float cosine_distance_wasm(const AffineVector<DIMS>& query, const AffineVector<DIMS>& target) {
    v128_t dot = wasm_f32x4_splat(0.0f);
    v128_t norm1 = wasm_f32x4_splat(0.0f);
    v128_t norm2 = wasm_f32x4_splat(0.0f);
    v128_t v_s1 = wasm_f32x4_splat(query.scale);
    v128_t v_s2 = wasm_f32x4_splat(target.scale);
    v128_t v_b1 = wasm_f32x4_splat(query.offset);
    v128_t v_b2 = wasm_f32x4_splat(target.offset);

    for (size_t i = 0; i < DIMS; i += 16) {
        v128_t q1_u8 = wasm_v128_load(&query.data[i]);
        v128_t q2_u8 = wasm_v128_load(&target.data[i]);

        v128_t q1_u16_lo = wasm_u16x8_extend_low_u8x16(q1_u8);
        v128_t q2_u16_lo = wasm_u16x8_extend_low_u8x16(q2_u8);
        v128_t q1_u16_hi = wasm_u16x8_extend_high_u8x16(q1_u8);
        v128_t q2_u16_hi = wasm_u16x8_extend_high_u8x16(q2_u8);

        #define PROCESS_4_COS(q1_u16, q2_u16, low_or_high) { \
            v128_t q1_i32 = wasm_u32x4_extend_##low_or_high##_u16x8(q1_u16); \
            v128_t q2_i32 = wasm_u32x4_extend_##low_or_high##_u16x8(q2_u16); \
            v128_t q1_f = wasm_f32x4_convert_i32x4(q1_i32); \
            v128_t q2_f = wasm_f32x4_convert_i32x4(q2_i32); \
            v128_t v1 = wasm_f32x4_add(v_b1, wasm_f32x4_mul(q1_f, v_s1)); \
            v128_t v2 = wasm_f32x4_add(v_b2, wasm_f32x4_mul(q2_f, v_s2)); \
            dot = wasm_f32x4_add(dot, wasm_f32x4_mul(v1, v2)); \
            norm1 = wasm_f32x4_add(norm1, wasm_f32x4_mul(v1, v1)); \
            norm2 = wasm_f32x4_add(norm2, wasm_f32x4_mul(v2, v2)); \
        }

        PROCESS_4_COS(q1_u16_lo, q2_u16_lo, low);
        PROCESS_4_COS(q1_u16_lo, q2_u16_lo, high);
        PROCESS_4_COS(q1_u16_hi, q2_u16_hi, low);
        PROCESS_4_COS(q1_u16_hi, q2_u16_hi, high);
        #undef PROCESS_4_COS
    }

    float dot_sum = wasm_f32x4_extract_lane(dot, 0) + wasm_f32x4_extract_lane(dot, 1) +
                    wasm_f32x4_extract_lane(dot, 2) + wasm_f32x4_extract_lane(dot, 3);
    float norm1_sum = wasm_f32x4_extract_lane(norm1, 0) + wasm_f32x4_extract_lane(norm1, 1) +
                      wasm_f32x4_extract_lane(norm1, 2) + wasm_f32x4_extract_lane(norm1, 3);
    float norm2_sum = wasm_f32x4_extract_lane(norm2, 0) + wasm_f32x4_extract_lane(norm2, 1) +
                      wasm_f32x4_extract_lane(norm2, 2) + wasm_f32x4_extract_lane(norm2, 3);

    float denom = std::sqrt(norm1_sum) * std::sqrt(norm2_sum);
    return (denom > 1e-8f) ? (1.0f - dot_sum / denom) : 1.0f;
}

template<size_t DIMS>
inline void quantize_wasm(const float* input, AffineVector<DIMS>& output) {
    v128_t vmin = wasm_v128_load(input);
    v128_t vmax = vmin;
    for (size_t i = 4; i < DIMS; i += 4) {
        v128_t v = wasm_v128_load(input + i);
        vmin = wasm_f32x4_min(vmin, v);
        vmax = wasm_f32x4_max(vmax, v);
    }

    float mins[4], maxs[4];
    wasm_v128_store(mins, vmin);
    wasm_v128_store(maxs, vmax);
    float min_val = std::min({mins[0], mins[1], mins[2], mins[3]});
    float max_val = std::max({maxs[0], maxs[1], maxs[2], maxs[3]});

    if (max_val == min_val) max_val = min_val + 1.0f;
    float range = max_val - min_val;
    output.scale = range / 255.0f;
    output.offset = min_val;

    float inv_scale = 255.0f / range;
    v128_t v_offset = wasm_f32x4_splat(min_val);
    v128_t v_inv_scale = wasm_f32x4_splat(inv_scale);
    v128_t v_half = wasm_f32x4_splat(0.5f);
    v128_t v_255 = wasm_f32x4_splat(255.0f);
    v128_t v_0 = wasm_f32x4_splat(0.0f);

    for (size_t i = 0; i < DIMS; i += 4) {
        v128_t v = wasm_v128_load(input + i);
        v128_t norm = wasm_f32x4_mul(wasm_f32x4_sub(v, v_offset), v_inv_scale);
        norm = wasm_f32x4_add(norm, v_half);
        norm = wasm_f32x4_max(v_0, wasm_f32x4_min(v_255, norm));
        v128_t qi = wasm_i32x4_trunc_sat_f32x4(norm);

        output.data[i]   = (uint8_t)wasm_i32x4_extract_lane(qi, 0);
        output.data[i+1] = (uint8_t)wasm_i32x4_extract_lane(qi, 1);
        output.data[i+2] = (uint8_t)wasm_i32x4_extract_lane(qi, 2);
        output.data[i+3] = (uint8_t)wasm_i32x4_extract_lane(qi, 3);
    }
}

#endif

// =============================================================================
// SSE2 Implementation (native x86/x64)
// =============================================================================

#if defined(PANCAKE_USE_SSE) && !defined(PANCAKE_WASM_BUILD)

// Helper: widen 16 uint8s to 4 groups of 4 floats via SSE2
// Writes 4 __m128 values to f[0..3] from 16 bytes at src
#define SSE2_WIDEN_U8_TO_4xF32(src, f0, f1, f2, f3) { \
    __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src)); \
    __m128i zero = _mm_setzero_si128(); \
    __m128i u16_lo = _mm_unpacklo_epi8(bytes, zero); \
    __m128i u16_hi = _mm_unpackhi_epi8(bytes, zero); \
    f0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(u16_lo, zero)); \
    f1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(u16_lo, zero)); \
    f2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(u16_hi, zero)); \
    f3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(u16_hi, zero)); \
}

template<size_t DIMS>
inline float l2_squared_sse(const AffineVector<DIMS>& query, const AffineVector<DIMS>& target) {
    __m128 sum = _mm_setzero_ps();
    __m128 v_s1 = _mm_set1_ps(query.scale);
    __m128 v_s2 = _mm_set1_ps(target.scale);
    __m128 v_b1 = _mm_set1_ps(query.offset);
    __m128 v_b2 = _mm_set1_ps(target.offset);

    for (size_t i = 0; i < DIMS; i += 16) {
        __m128 q0, q1, q2, q3, t0, t1, t2, t3;
        SSE2_WIDEN_U8_TO_4xF32(&query.data[i], q0, q1, q2, q3);
        SSE2_WIDEN_U8_TO_4xF32(&target.data[i], t0, t1, t2, t3);

        #define SSE2_L2_ACCUM(qf, tf) { \
            __m128 v1 = _mm_add_ps(v_b1, _mm_mul_ps(qf, v_s1)); \
            __m128 v2 = _mm_add_ps(v_b2, _mm_mul_ps(tf, v_s2)); \
            __m128 diff = _mm_sub_ps(v1, v2); \
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); \
        }
        SSE2_L2_ACCUM(q0, t0);
        SSE2_L2_ACCUM(q1, t1);
        SSE2_L2_ACCUM(q2, t2);
        SSE2_L2_ACCUM(q3, t3);
        #undef SSE2_L2_ACCUM
    }

    alignas(16) float tmp[4];
    _mm_store_ps(tmp, sum);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

template<size_t DIMS>
inline float dot_product_sse(const AffineVector<DIMS>& query, const AffineVector<DIMS>& target) {
    __m128 dot = _mm_setzero_ps();
    __m128 v_s1 = _mm_set1_ps(query.scale);
    __m128 v_s2 = _mm_set1_ps(target.scale);
    __m128 v_b1 = _mm_set1_ps(query.offset);
    __m128 v_b2 = _mm_set1_ps(target.offset);

    for (size_t i = 0; i < DIMS; i += 16) {
        __m128 q0, q1, q2, q3, t0, t1, t2, t3;
        SSE2_WIDEN_U8_TO_4xF32(&query.data[i], q0, q1, q2, q3);
        SSE2_WIDEN_U8_TO_4xF32(&target.data[i], t0, t1, t2, t3);

        #define SSE2_DOT_ACCUM(qf, tf) { \
            __m128 v1 = _mm_add_ps(v_b1, _mm_mul_ps(qf, v_s1)); \
            __m128 v2 = _mm_add_ps(v_b2, _mm_mul_ps(tf, v_s2)); \
            dot = _mm_add_ps(dot, _mm_mul_ps(v1, v2)); \
        }
        SSE2_DOT_ACCUM(q0, t0);
        SSE2_DOT_ACCUM(q1, t1);
        SSE2_DOT_ACCUM(q2, t2);
        SSE2_DOT_ACCUM(q3, t3);
        #undef SSE2_DOT_ACCUM
    }

    alignas(16) float tmp[4];
    _mm_store_ps(tmp, dot);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

template<size_t DIMS>
inline float compute_norm_sse(const AffineVector<DIMS>& vec) {
    __m128 norm = _mm_setzero_ps();
    __m128 v_s = _mm_set1_ps(vec.scale);
    __m128 v_b = _mm_set1_ps(vec.offset);

    for (size_t i = 0; i < DIMS; i += 16) {
        __m128 f0, f1, f2, f3;
        SSE2_WIDEN_U8_TO_4xF32(&vec.data[i], f0, f1, f2, f3);

        #define SSE2_NORM_ACCUM(ff) { \
            __m128 v = _mm_add_ps(v_b, _mm_mul_ps(ff, v_s)); \
            norm = _mm_add_ps(norm, _mm_mul_ps(v, v)); \
        }
        SSE2_NORM_ACCUM(f0);
        SSE2_NORM_ACCUM(f1);
        SSE2_NORM_ACCUM(f2);
        SSE2_NORM_ACCUM(f3);
        #undef SSE2_NORM_ACCUM
    }

    alignas(16) float tmp[4];
    _mm_store_ps(tmp, norm);
    return std::sqrt(tmp[0] + tmp[1] + tmp[2] + tmp[3]);
}

template<size_t DIMS>
inline float cosine_distance_sse(const AffineVector<DIMS>& query, const AffineVector<DIMS>& target) {
    __m128 dot = _mm_setzero_ps();
    __m128 norm1 = _mm_setzero_ps();
    __m128 norm2 = _mm_setzero_ps();
    __m128 v_s1 = _mm_set1_ps(query.scale);
    __m128 v_s2 = _mm_set1_ps(target.scale);
    __m128 v_b1 = _mm_set1_ps(query.offset);
    __m128 v_b2 = _mm_set1_ps(target.offset);

    for (size_t i = 0; i < DIMS; i += 16) {
        __m128 q0, q1, q2, q3, t0, t1, t2, t3;
        SSE2_WIDEN_U8_TO_4xF32(&query.data[i], q0, q1, q2, q3);
        SSE2_WIDEN_U8_TO_4xF32(&target.data[i], t0, t1, t2, t3);

        #define SSE2_COS_ACCUM(qf, tf) { \
            __m128 v1 = _mm_add_ps(v_b1, _mm_mul_ps(qf, v_s1)); \
            __m128 v2 = _mm_add_ps(v_b2, _mm_mul_ps(tf, v_s2)); \
            dot = _mm_add_ps(dot, _mm_mul_ps(v1, v2)); \
            norm1 = _mm_add_ps(norm1, _mm_mul_ps(v1, v1)); \
            norm2 = _mm_add_ps(norm2, _mm_mul_ps(v2, v2)); \
        }
        SSE2_COS_ACCUM(q0, t0);
        SSE2_COS_ACCUM(q1, t1);
        SSE2_COS_ACCUM(q2, t2);
        SSE2_COS_ACCUM(q3, t3);
        #undef SSE2_COS_ACCUM
    }

    alignas(16) float tmp_d[4], tmp_n1[4], tmp_n2[4];
    _mm_store_ps(tmp_d, dot);
    _mm_store_ps(tmp_n1, norm1);
    _mm_store_ps(tmp_n2, norm2);

    float dot_sum = tmp_d[0] + tmp_d[1] + tmp_d[2] + tmp_d[3];
    float n1_sum = tmp_n1[0] + tmp_n1[1] + tmp_n1[2] + tmp_n1[3];
    float n2_sum = tmp_n2[0] + tmp_n2[1] + tmp_n2[2] + tmp_n2[3];

    float denom = std::sqrt(n1_sum) * std::sqrt(n2_sum);
    return (denom > 1e-8f) ? (1.0f - dot_sum / denom) : 1.0f;
}

template<size_t DIMS>
inline void quantize_sse(const float* input, AffineVector<DIMS>& output) {
    // Find min/max with SSE
    __m128 vmin = _mm_loadu_ps(input);
    __m128 vmax = vmin;
    for (size_t i = 4; i < DIMS; i += 4) {
        __m128 v = _mm_loadu_ps(input + i);
        vmin = _mm_min_ps(vmin, v);
        vmax = _mm_max_ps(vmax, v);
    }

    alignas(16) float mins[4], maxs[4];
    _mm_store_ps(mins, vmin);
    _mm_store_ps(maxs, vmax);
    float min_val = std::min({mins[0], mins[1], mins[2], mins[3]});
    float max_val = std::max({maxs[0], maxs[1], maxs[2], maxs[3]});

    if (max_val == min_val) max_val = min_val + 1.0f;
    float range = max_val - min_val;
    output.scale = range / 255.0f;
    output.offset = min_val;

    float inv_scale = 255.0f / range;
    __m128 v_offset = _mm_set1_ps(min_val);
    __m128 v_inv_scale = _mm_set1_ps(inv_scale);
    __m128 v_half = _mm_set1_ps(0.5f);
    __m128 v_255 = _mm_set1_ps(255.0f);
    __m128 v_0 = _mm_setzero_ps();

    for (size_t i = 0; i < DIMS; i += 4) {
        __m128 v = _mm_loadu_ps(input + i);
        __m128 norm = _mm_mul_ps(_mm_sub_ps(v, v_offset), v_inv_scale);
        norm = _mm_add_ps(norm, v_half);
        norm = _mm_max_ps(v_0, _mm_min_ps(v_255, norm));
        __m128i qi = _mm_cvttps_epi32(norm);

        alignas(16) int32_t lanes[4];
        _mm_store_si128(reinterpret_cast<__m128i*>(lanes), qi);
        output.data[i]   = static_cast<uint8_t>(lanes[0]);
        output.data[i+1] = static_cast<uint8_t>(lanes[1]);
        output.data[i+2] = static_cast<uint8_t>(lanes[2]);
        output.data[i+3] = static_cast<uint8_t>(lanes[3]);
    }
}

#undef SSE2_WIDEN_U8_TO_4xF32

#endif // PANCAKE_USE_SSE

// =============================================================================
// Wrapper Selection
// =============================================================================

template<size_t DIMS>
inline float l2_squared_fast(const AffineVector<DIMS>& query, const AffineVector<DIMS>& target) {
#if defined(PANCAKE_WASM_SIMD)
    return l2_squared_wasm<DIMS>(query, target);
#elif defined(PANCAKE_USE_SSE)
    return l2_squared_sse<DIMS>(query, target);
#else
    return l2_squared_scalar<DIMS>(query, target);
#endif
}

template<size_t DIMS>
inline float cosine_distance_fast(const AffineVector<DIMS>& query, const AffineVector<DIMS>& target) {
#if defined(PANCAKE_WASM_SIMD)
    return cosine_distance_wasm<DIMS>(query, target);
#elif defined(PANCAKE_USE_SSE)
    return cosine_distance_sse<DIMS>(query, target);
#else
    return cosine_distance_scalar<DIMS>(query, target);
#endif
}

template<size_t DIMS>
inline void quantize_fast(const float* input, AffineVector<DIMS>& output) {
#if defined(PANCAKE_WASM_SIMD)
    return quantize_wasm<DIMS>(input, output);
#elif defined(PANCAKE_USE_SSE)
    return quantize_sse<DIMS>(input, output);
#else
    return quantize_scalar<DIMS>(input, output);
#endif
}

template<size_t DIMS>
inline float dot_product_fast(const AffineVector<DIMS>& query, const AffineVector<DIMS>& target) {
#if defined(PANCAKE_WASM_SIMD)
    return dot_product_wasm<DIMS>(query, target);
#elif defined(PANCAKE_USE_SSE)
    return dot_product_sse<DIMS>(query, target);
#else
    return dot_product_scalar<DIMS>(query, target);
#endif
}

template<size_t DIMS>
inline float compute_norm_fast(const AffineVector<DIMS>& vec) {
#if defined(PANCAKE_WASM_SIMD)
    return compute_norm_wasm<DIMS>(vec);
#elif defined(PANCAKE_USE_SSE)
    return compute_norm_sse<DIMS>(vec);
#else
    return compute_norm_scalar<DIMS>(vec);
#endif
}

} // namespace simd
} // namespace wasm
} // namespace pancake