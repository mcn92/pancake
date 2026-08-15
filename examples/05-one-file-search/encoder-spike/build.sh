#!/bin/bash
# Build the spike kernel to a standalone WASM module (not part of dist/).
set -e
cd "$(dirname "$0")"
source ~/emsdk/emsdk_env.sh >/dev/null 2>&1
emcc gemv_spike.cpp -O3 -msimd128 \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=node \
  -sEXPORTED_FUNCTIONS=_gemv_u8_fused,_gemv_u8_scalar,_gemv_f32,_block_sums,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32 \
  -sALLOW_MEMORY_GROWTH=1 \
  -o spike.mjs
echo "built spike.mjs + spike.wasm"
