#!/bin/bash
# Build the six-layer encoder to a standalone WASM module (not part of dist/).
# encoder.mjs/encoder.wasm target browsers (web,worker); encoder.node.mjs/
# encoder.node.wasm target the Node CLI (web,node). Both pairs are copied into
# create-pancake-search/src/encoder-kernels/ so the spike and the shipped
# package can never diverge.
set -e
cd "$(dirname "$0")"
source ~/emsdk/emsdk_env.sh >/dev/null 2>&1
COMMON_FLAGS="-O3 -msimd128 -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORTED_FUNCTIONS=_encoder_forward,_malloc,_free -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32,HEAP32 -sALLOW_MEMORY_GROWTH=1"
emcc encoder.cpp $COMMON_FLAGS -sENVIRONMENT=web,worker -o encoder.mjs
emcc encoder.cpp $COMMON_FLAGS -sENVIRONMENT=web,node -o encoder.node.mjs
KERNELS_DIR="../../../create-pancake-search/src/encoder-kernels"
cp encoder.mjs encoder.wasm encoder.node.mjs encoder.node.wasm "$KERNELS_DIR/"
echo "built encoder.mjs/encoder.wasm (web,worker) and encoder.node.mjs/encoder.node.wasm (web,node); synced to $KERNELS_DIR"
