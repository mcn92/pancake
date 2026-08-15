#!/bin/bash
# Build the six-layer encoder to a standalone WASM module (not part of dist/).
set -e
cd "$(dirname "$0")"
source ~/emsdk/emsdk_env.sh >/dev/null 2>&1
emcc encoder.cpp -O3 -msimd128 \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=web,node \
  -sEXPORTED_FUNCTIONS=_encoder_forward,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32,HEAP32 \
  -sALLOW_MEMORY_GROWTH=1 \
  -o encoder.mjs
echo "built encoder.mjs + encoder.wasm"
