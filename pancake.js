'use strict';

const loadEngine = require('./dist/engine.js');
const createPancakeApi = require('./pancake-core.js');
const _path = require('path');
const _fs = require('fs');

function loadNodeEngine() {
    const wasmBinary = _fs.readFileSync(_path.join(__dirname, 'dist', 'engine.wasm'));
    return loadEngine({
        wasmBinary: wasmBinary.buffer.slice(
            wasmBinary.byteOffset,
            wasmBinary.byteOffset + wasmBinary.byteLength
        )
    });
}

const Pancake = createPancakeApi(loadNodeEngine);

module.exports = Pancake;