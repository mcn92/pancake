#!/bin/bash
set -e

EMCC="${EMCC:-python3 $HOME/emsdk/upstream/emscripten/emcc.py}"
OUT_BASENAME="${OUT_BASENAME:-engine}"
PATCH_ENGINE_JS="${PATCH_ENGINE_JS:-1}"
WASM_SIMD="${WASM_SIMD:-1}"
WASM_RELAXED_SIMD="${WASM_RELAXED_SIMD:-0}"
INT8_HNSW_PROFILE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            INT8_HNSW_PROFILE=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./build.sh [--profile]"
            exit 1
            ;;
    esac
done

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

SIMD_FLAGS=""
SIMD_DESC="scalar"
NODE_WASM_FLAGS=""
if [[ "${WASM_SIMD}" == "1" ]]; then
    SIMD_FLAGS="-msimd128"
    SIMD_DESC="WASM SIMD"
    if [[ "${WASM_RELAXED_SIMD}" == "1" ]]; then
        SIMD_FLAGS="${SIMD_FLAGS} -mrelaxed-simd"
        SIMD_DESC="WASM SIMD + relaxed SIMD"
        if node --v8-options 2>/dev/null | grep -q -- "--experimental-wasm-relaxed-simd"; then
            NODE_WASM_FLAGS="--experimental-wasm-relaxed-simd"
        fi
    fi
fi

PROFILE_FLAGS=""
PROFILE_DESC="profiling off"
if [[ "${INT8_HNSW_PROFILE}" == "1" ]]; then
    PROFILE_FLAGS="-D PANCAKE_INT8_HNSW_BUILD_PROFILE=1"
    PROFILE_DESC="profiling on"
fi

# NOTE: -ffast-math below implies -ffinite-math-only, which makes std::isfinite,
# std::isnan, and std::isinf always return true. Any float validation in C++ must
# use bit-level IEEE 754 checks instead (see deserialization in *_hnsw.hpp).
echo "Compiling with ${SIMD_DESC} + Memory Access (${PROFILE_DESC}, exceptions enabled)..."
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
    $SIMD_FLAGS \
    -mnontrapping-fptoint \
    -msign-ext \
    -mbulk-memory \
    $PROFILE_FLAGS \
    -D PANCAKE_WASM_BUILD=1 \
    -fno-merge-all-constants \
    -s EXPORTED_FUNCTIONS='["_pancake_init","_pancake_add","_pancake_bulk_insert","_pancake_query","_pancake_query_filtered","_pancake_delete","_pancake_compact","_pancake_compact_remap","_pancake_count","_pancake_memory","_pancake_ghost_count","_pancake_ghost_ratio","_pancake_set_ef","_pancake_export","_pancake_import","_pancake_dispose","_pancake_dimension","_pancake_shutdown_all","_shutdown_all","_dense_matmul","_sparse_matmul","_normalize","_emsc_malloc","_emsc_free","_pancake_profile_print","_pancake_profile_reset"]' \
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
    -o "dist/${OUT_BASENAME}.js" \
    src/engine.cpp

if [[ "${PATCH_ENGINE_JS}" == "1" && "${OUT_BASENAME}" == "engine" ]]; then
    echo ""
    echo "Applying engine.js patches..."
    python3 patch_engine.py
fi

echo ""
echo "Build complete!"
ls -lh "dist/${OUT_BASENAME}.js" "dist/${OUT_BASENAME}.wasm"
echo "Running test..."
node ${NODE_WASM_FLAGS} run_tests.js
