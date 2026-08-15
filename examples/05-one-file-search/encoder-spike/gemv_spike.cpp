// Fused-dequant GEMV spike for the inline-transformer encoder direction
// (query-interp kind 3). The pancake asymmetric-distance move applied to
// weight matrices: weights live as block-affine u8 (per-block scale/offset),
// activations stay f32, and dequantization happens inside the dot product —
// dot(dequant(block), x) == scale * dot(u8block, x) + offset * sum(xblock),
// so a float weight tensor is never materialized. Standalone WASM build,
// not part of the engine.

#include <stdint.h>
#include <stddef.h>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

extern "C" {

void gemv_u8_scalar(const uint8_t* a, const float* scales, const float* offsets,
                    int rows, int cols, int block, const float* x,
                    const float* xsums, float* y);

// Per-block sums of the activation vector, reused across every row.
void block_sums(const float* x, int cols, int block, float* sums) {
    for (int b = 0; b < cols / block; b++) {
        float s = 0.f;
        for (int c = 0; c < block; c++) s += x[b * block + c];
        sums[b] = s;
    }
}

// y[rows] = A * x, A stored as u8 rows with per-block affine params.
// block must be a multiple of 16.
void gemv_u8_fused(const uint8_t* a, const float* scales, const float* offsets,
                   int rows, int cols, int block, const float* x,
                   const float* xsums, float* y) {
    const int nblocks = cols / block;
#ifdef __wasm_simd128__
    for (int r = 0; r < rows; r++) {
        const uint8_t* arow = a + (size_t)r * cols;
        const float* srow = scales + (size_t)r * nblocks;
        const float* orow = offsets + (size_t)r * nblocks;
        float acc = 0.f;
        for (int b = 0; b < nblocks; b++) {
            const uint8_t* ab = arow + b * block;
            const float* xb = x + b * block;
            v128_t vacc = wasm_f32x4_const_splat(0.f);
            for (int c = 0; c < block; c += 16) {
                const v128_t bytes = wasm_v128_load(ab + c);
                const v128_t lo16 = wasm_u16x8_extend_low_u8x16(bytes);
                const v128_t hi16 = wasm_u16x8_extend_high_u8x16(bytes);
                const v128_t f0 = wasm_f32x4_convert_u32x4(wasm_u32x4_extend_low_u16x8(lo16));
                const v128_t f1 = wasm_f32x4_convert_u32x4(wasm_u32x4_extend_high_u16x8(lo16));
                const v128_t f2 = wasm_f32x4_convert_u32x4(wasm_u32x4_extend_low_u16x8(hi16));
                const v128_t f3 = wasm_f32x4_convert_u32x4(wasm_u32x4_extend_high_u16x8(hi16));
                vacc = wasm_f32x4_add(vacc, wasm_f32x4_mul(f0, wasm_v128_load(xb + c)));
                vacc = wasm_f32x4_add(vacc, wasm_f32x4_mul(f1, wasm_v128_load(xb + c + 4)));
                vacc = wasm_f32x4_add(vacc, wasm_f32x4_mul(f2, wasm_v128_load(xb + c + 8)));
                vacc = wasm_f32x4_add(vacc, wasm_f32x4_mul(f3, wasm_v128_load(xb + c + 12)));
            }
            const float dot = wasm_f32x4_extract_lane(vacc, 0) + wasm_f32x4_extract_lane(vacc, 1)
                + wasm_f32x4_extract_lane(vacc, 2) + wasm_f32x4_extract_lane(vacc, 3);
            acc += srow[b] * dot + orow[b] * xsums[b];
        }
        y[r] = acc;
    }
#else
    gemv_u8_scalar(a, scales, offsets, rows, cols, block, x, xsums, y);
#endif
}

// Scalar reference of the same fused computation.
void gemv_u8_scalar(const uint8_t* a, const float* scales, const float* offsets,
                    int rows, int cols, int block, const float* x,
                    const float* xsums, float* y) {
    const int nblocks = cols / block;
    for (int r = 0; r < rows; r++) {
        const uint8_t* arow = a + (size_t)r * cols;
        float acc = 0.f;
        for (int b = 0; b < nblocks; b++) {
            float dot = 0.f;
            for (int c = 0; c < block; c++) dot += (float)arow[b * block + c] * x[b * block + c];
            acc += scales[(size_t)r * nblocks + b] * dot
                + offsets[(size_t)r * nblocks + b] * xsums[b];
        }
        y[r] = acc;
    }
}

// f32 GEMV baseline: what a non-quantized encoder would pay (and 4x the
// weight bytes / memory traffic).
void gemv_f32(const float* a, int rows, int cols, const float* x, float* y) {
#ifdef __wasm_simd128__
    for (int r = 0; r < rows; r++) {
        const float* arow = a + (size_t)r * cols;
        v128_t vacc = wasm_f32x4_const_splat(0.f);
        for (int c = 0; c < cols; c += 4) {
            vacc = wasm_f32x4_add(vacc, wasm_f32x4_mul(wasm_v128_load(arow + c), wasm_v128_load(x + c)));
        }
        y[r] = wasm_f32x4_extract_lane(vacc, 0) + wasm_f32x4_extract_lane(vacc, 1)
            + wasm_f32x4_extract_lane(vacc, 2) + wasm_f32x4_extract_lane(vacc, 3);
    }
#else
    for (int r = 0; r < rows; r++) {
        const float* arow = a + (size_t)r * cols;
        float acc = 0.f;
        for (int c = 0; c < cols; c++) acc += arow[c] * x[c];
        y[r] = acc;
    }
#endif
}

} // extern "C"
