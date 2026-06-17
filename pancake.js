'use strict';

const loadEngine = require('./dist/engine.js');
const loadScalarEngine = require('./dist/engine.scalar.js');
const createPancakeApi = require('./pancake-core.js');
const _path = require('path');
const _fs = require('fs');
let _engineVariant = null;

function isSimdLoadFailure(error) {
    const text = String(error && (error.stack || error.message || error)).toLowerCase();
    return text.includes('simd') || text.includes('v128');
}

function readWasmBinary(fileName) {
    const wasmBinary = _fs.readFileSync(_path.join(__dirname, 'dist', fileName));
    return wasmBinary.buffer.slice(
        wasmBinary.byteOffset,
        wasmBinary.byteOffset + wasmBinary.byteLength
    );
}

async function loadNodeEngine() {
    if (_engineVariant === 'scalar') {
        return loadScalarEngine({
            wasmBinary: readWasmBinary('engine.scalar.wasm')
        });
    }
    try {
        const engine = await loadEngine({
            wasmBinary: readWasmBinary('engine.wasm')
        });
        _engineVariant = 'simd';
        return engine;
    } catch (error) {
        if (!isSimdLoadFailure(error)) throw error;
        const engine = await loadScalarEngine({
            wasmBinary: readWasmBinary('engine.scalar.wasm')
        });
        _engineVariant = 'scalar';
        return engine;
    }
}

const Pancake = createPancakeApi(loadNodeEngine);

module.exports = Pancake;
module.exports.default = Pancake;
