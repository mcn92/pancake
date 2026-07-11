'use strict';

const assert = require('assert');
const path = require('path');
const { pathToFileURL } = require('url');

async function verifyEntry(loadApi, label) {
    const originalCompile = WebAssembly.compile;
    let compileCount = 0;
    WebAssembly.compile = async function countedCompile(source) {
        compileCount++;
        return originalCompile.call(WebAssembly, source);
    };

    let indexes = [];
    try {
        const Pancake = await loadApi();
        indexes = await Promise.all(Array.from({ length: 3 }, () => Pancake.create({
            dim: 4,
            maxElements: 4,
            metric: 'l2',
            quantized: false,
        })));

        assert.strictEqual(compileCount, 1, `${label}: concurrent create() compiles WASM once`);
        assert.notStrictEqual(indexes[0]._e, indexes[1]._e, `${label}: Emscripten modules are isolated`);
        assert.notStrictEqual(indexes[0]._e.HEAPU8.buffer, indexes[1]._e.HEAPU8.buffer,
            `${label}: WASM heaps are isolated`);

        indexes[0].add(new Float32Array([1, 0, 0, 0]));
        assert.strictEqual(indexes[0].count, 1, `${label}: first index mutates independently`);
        assert.strictEqual(indexes[1].count, 0, `${label}: second index remains empty`);

        indexes[0].dispose();
        indexes[0] = null;
        indexes[1].add(new Float32Array([0, 1, 0, 0]));
        assert.strictEqual(indexes[1].count, 1, `${label}: disposing one index leaves another usable`);
    } finally {
        WebAssembly.compile = originalCompile;
        for (const index of indexes) {
            if (index) index.dispose();
        }
    }
}

async function main() {
    await verifyEntry(async () => {
        const url = pathToFileURL(path.join(__dirname, '..', 'pancake.node.mjs')).href;
        return (await import(`${url}?model-c-loader-test`)).default;
    }, 'Node ESM');

    await verifyEntry(async () => {
        const entry = require.resolve('../pancake.js');
        delete require.cache[entry];
        return require(entry);
    }, 'Node CJS');

    console.log('Model C loader checks passed.');
}

main().catch((error) => {
    console.error(error.stack || error);
    process.exit(1);
});
