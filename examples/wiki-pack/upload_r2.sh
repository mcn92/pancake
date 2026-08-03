#!/bin/bash
set -x
M=node_modules/@huggingface/transformers/.cache/Xenova/all-MiniLM-L6-v2
O=node_modules/onnxruntime-web/dist
npx wrangler r2 object put pancake-wiki-pack/pack/corpus.bin --file data-perm/corpus.bin --content-type application/octet-stream --remote
npx wrangler r2 object put pancake-wiki-pack/pack/wiki.pancake-sketch --file data-perm/wiki.pancake-sketch --content-type application/octet-stream --remote
npx wrangler r2 object put pancake-wiki-pack/pack/corpus-offsets.u32 --file data-perm/corpus-offsets.u32 --content-type application/octet-stream --remote
npx wrangler r2 object put pancake-wiki-pack/pack/wiki-vocab.bloom --file data-perm/wiki-vocab.bloom --content-type application/octet-stream --remote
npx wrangler r2 object put pancake-wiki-pack/pack/wiki-abstention.json --file data-perm/wiki-abstention.json --content-type application/json --remote
npx wrangler r2 object put pancake-wiki-pack/pack/wiki-abstention-probes.json --file data-perm/wiki-abstention-probes.json --content-type application/json --remote
npx wrangler r2 object put pancake-wiki-pack/models/Xenova/all-MiniLM-L6-v2/config.json --file $M/config.json --content-type application/json --remote
npx wrangler r2 object put pancake-wiki-pack/models/Xenova/all-MiniLM-L6-v2/tokenizer.json --file $M/tokenizer.json --content-type application/json --remote
npx wrangler r2 object put pancake-wiki-pack/models/Xenova/all-MiniLM-L6-v2/tokenizer_config.json --file $M/tokenizer_config.json --content-type application/json --remote
npx wrangler r2 object put pancake-wiki-pack/models/Xenova/all-MiniLM-L6-v2/onnx/model_fp16.onnx --file $M/onnx/model_fp16.onnx --content-type application/octet-stream --remote
npx wrangler r2 object put pancake-wiki-pack/ort/ort-wasm-simd-threaded.jsep.wasm --file $O/ort-wasm-simd-threaded.jsep.wasm --content-type application/wasm --remote
npx wrangler r2 object put pancake-wiki-pack/ort/ort-wasm-simd-threaded.wasm --file $O/ort-wasm-simd-threaded.wasm --content-type application/wasm --remote
npx wrangler r2 object put pancake-wiki-pack/ort/ort-wasm-simd-threaded.jsep.mjs --file $O/ort-wasm-simd-threaded.jsep.mjs --content-type text/javascript --remote
npx wrangler r2 object put pancake-wiki-pack/ort/ort-wasm-simd-threaded.mjs --file $O/ort-wasm-simd-threaded.mjs --content-type text/javascript --remote
echo ALL_UPLOADS_DONE
