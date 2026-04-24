#!/bin/bash
set -e

EMCC="${EMCC:-python3 $HOME/emsdk/upstream/emscripten/emcc.py}"

if [[ "${DEBUG_SYMBOLS}" == "1" ]]; then
    BUILD_DESC="Debug"
    OPT_FLAGS="-O2"
    DEBUG_FLAGS="-gsource-map -g --source-map-base http://localhost/"
else
    # CHANGED: -Oz (size) -> -O3 (speed). -Oz disables inlining and several loop
    # optimizations; on insert-path code with ~300 non-SIMD function calls
    # reachable from _i8_add, this was costing roughly 2x in build throughput.
    BUILD_DESC="Optimized"
    OPT_FLAGS="-O3"
    DEBUG_FLAGS="-g0"
fi

echo "=============================================="
echo "Building Pancake WASM Engine (${BUILD_DESC})"
echo "=============================================="

mkdir -p dist

echo "Compiling with WASM SIMD + Memory Access (exceptions enabled)..."
$EMCC $OPT_FLAGS \
    $DEBUG_FLAGS \
    --closure 0 \
    -s WASM=1 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=16777216 \
    -s MALLOC=emmalloc \
    -s WASM_BIGINT=1 \
    -s ALLOW_TABLE_GROWTH=0 \
    -s DYNAMIC_EXECUTION=0 \
    -s EMULATE_FUNCTION_POINTER_CASTS=0 \
    -s SUPPORT_LONGJMP=0 \
    -msimd128 \
    -mnontrapping-fptoint \
    -msign-ext \
    -mbulk-memory \
    -D PANCAKE_WASM_BUILD=1 \
    -fno-merge-all-constants \
    -s EXPORTED_FUNCTIONS='["_i","_a","_q","_c","_m","_d","_ghost_count","_ghost_ratio","_compact","_pi","_pa","_pq","_pc","_pm","_pd","_p_ghost_count","_p_ghost_ratio","_p_compact","_export_index","_import_index","_bulk_insert","_float_init","_float_add","_float_query","_float_count","_float_memory","_float_delete","_float_ghost_count","_float_ghost_ratio","_float_compact","_float_export_index","_float_import_index","_float_set_ef","_si_init","_si_add","_si_query","_si_delete","_si_count","_si_memory","_si_segment_count","_si_ghost_count","_si_ghost_ratio","_si_compact","_si_set_budget","_si128_init","_si128_add","_si128_query","_si128_delete","_si128_count","_si128_memory","_si128_segment_count","_si128_ghost_count","_si128_compact","_si128_set_budget","_si128_get_vector_segment","_si128_get_query_segments","_p128_init","_p128_add","_p128_query","_p128_count","_p128_memory","_p256_init","_p256_add","_p256_query","_p256_count","_p256_memory","_p256_bulk_insert","_p1536_init","_p1536_add","_p1536_query","_p1536_count","_p1536_memory","_p1536_bulk_insert","_p1536_export_index","_p1536_import_index","_p128f_init","_p128f_add","_p128f_query","_p128f_count","_p128f_memory","_p128f_get_neighbors","_p384f_init","_p384f_add","_p384f_query","_p384f_count","_p384f_memory","_p384f_get_neighbors","_p256f_init","_p256f_add","_p256f_query","_p256f_count","_p256f_memory","_naive_init","_naive_add","_naive_query","_naive_count","_naive_get_neighbors","_i8_init","_i8_add","_i8_bulk_insert","_i8_set_ef","_i8_query","_i8_count","_i8_memory","_i8_delete","_i8_ghost_count","_i8_ghost_ratio","_i8_compact","_i8_export_index","_i8_import_index","_dense_matmul","_sparse_matmul","_normalize","_emb_init","_emb_encode","_emb_encode_batch","_emb_dimension","_emb_free","_emb_add","_emb_search","_emsc_malloc","_emsc_free","_shutdown_all"]' \
    -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPF32","HEAPU8","HEAPU32","HEAP32"]' \
    -s MODULARIZE=1 \
    -s EXPORT_NAME="P" \
    -s ENVIRONMENT='web,node' \
    -s FILESYSTEM=0 \
    -s ASSERTIONS=0 \
    -s DISABLE_EXCEPTION_CATCHING=0 \
    -s SINGLE_FILE=0 \
    -s STACK_SIZE=65536 \
    -fno-rtti \
    -ffast-math \
    -fvectorize \
    -fslp-vectorize \
    --no-entry \
    -Isrc \
    -o dist/engine.js \
    src/engine.cpp

echo ""
echo "Applying engine.js patches..."
python3 patch_engine.py

echo ""
echo "Build complete!"
ls -lh dist/engine.js dist/engine.wasm