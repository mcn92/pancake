import { defineConfig } from 'vite';

export default defineConfig({
    base: './',
    build: {
        // getBigUint64 and top-level await need a modern floor; every
        // browser with WASM SIMD (the engine's own floor) clears it.
        target: 'es2022',
    },
});
