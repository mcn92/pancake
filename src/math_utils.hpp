#pragma once
/**
 * Shared Math Utilities
 *
 * Common mathematical operations used across multiple modules.
 * Centralized here to avoid duplication.
 */

#include <cmath>
#include <algorithm>

namespace pancake {
namespace wasm {
namespace math {

/**
 * L2-normalize a vector in-place.
 *
 * Divides each element by the L2 norm (Euclidean length).
 * Zero vectors are left unchanged to avoid division by zero.
 *
 * @param vec Pointer to float array
 * @param dim Vector dimension
 */
inline void normalize_vector(float* vec, size_t dim) {
    float norm_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        norm_sq += vec[i] * vec[i];
    }
    if (norm_sq < 1e-12f) return; // Avoid division by zero

    float inv_norm = 1.0f / std::sqrt(norm_sq);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] *= inv_norm;
    }
}

/**
 * Affine int8 quantization/dequantization for vectors.
 *
 * WHY IN-PLACE QUANTIZE/DEQUANTIZE: Keeping quantization close to the active
 * search representation avoids redundant copies and keeps the runtime kernels
 * simple. Row-wise affine quantization works best when values are reasonably
 * bounded and locally uniform, which matches the current int8 backend.
 *
 * This is a lossy round-trip that simulates int8 storage without actually
 * storing the quantized values. Useful for:
 * - Testing quantization quality before committing to int8 index
 * - Matching training-time quantization during inference
 * - Simulating memory-constrained scenarios
 *
 * @param vec Pointer to float array (modified in-place)
 * @param dim Vector dimension
 */
inline void quantize_dequantize_int8(float* vec, size_t dim) {
    // Find min/max for affine scaling
    float vmin = vec[0], vmax = vec[0];
    for (size_t i = 1; i < dim; ++i) {
        if (vec[i] < vmin) vmin = vec[i];
        if (vec[i] > vmax) vmax = vec[i];
    }

    float range = vmax - vmin;
    if (range < 1e-30f) return; // Constant vector, nothing to quantize

    // Quantize to [0, 255] and dequantize back
    float scale = range / 255.0f;
    float inv_scale = 255.0f / range;

    for (size_t i = 0; i < dim; ++i) {
        // Quantize: map to uint8
        float q = std::round((vec[i] - vmin) * inv_scale);
        q = std::max(0.0f, std::min(255.0f, q));

        // Dequantize: map back to float
        vec[i] = q * scale + vmin;
    }
}

} // namespace math
} // namespace wasm
} // namespace pancake
