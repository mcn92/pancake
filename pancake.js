'use strict';

const loadEngine = require('./dist/engine.js');
const loadScalarEngine = require('./dist/engine.scalar.js');
const createPancakeApi = require('./pancake-core.js');
const _path = require('path');
const _fs = require('fs');
let _engineVariant = null;

function readWasmBinary(fileName) {
    try {
        const wasmBinary = _fs.readFileSync(_path.join(__dirname, 'dist', fileName));
        return wasmBinary.buffer.slice(
            wasmBinary.byteOffset,
            wasmBinary.byteOffset + wasmBinary.byteLength
        );
    } catch (error) {
        const message = error && error.message ? error.message : String(error);
        throw new Error(`Failed to load Pancake WASM binary (${fileName}): ${message}`);
    }
}

function makeLoadError(message, error) {
    const detail = error && error.message ? error.message : String(error);
    return new Error(`${message}: ${detail}`);
}

function hasSimdSupport(binary) {
    try {
        return WebAssembly.validate(binary);
    } catch {
        return false;
    }
}

async function loadNodeEngine() {
    if (_engineVariant === null) {
        _engineVariant = hasSimdSupport(readWasmBinary('engine.wasm')) ? 'simd' : 'scalar';
    }

    if (_engineVariant === 'scalar') {
        try {
            return await loadScalarEngine({
                wasmBinary: readWasmBinary('engine.scalar.wasm')
            });
        } catch (error) {
            throw makeLoadError('Pancake failed to load the scalar WASM engine', error);
        }
    }

    try {
        return await loadEngine({
            wasmBinary: readWasmBinary('engine.wasm')
        });
    } catch (simdError) {
        try {
            const engine = await loadScalarEngine({
                wasmBinary: readWasmBinary('engine.scalar.wasm')
            });
            _engineVariant = 'scalar';
            return engine;
        } catch (scalarError) {
            throw makeLoadError(
                `Pancake failed to load either the SIMD or scalar WASM engine (SIMD error: ${simdError && simdError.message ? simdError.message : String(simdError)})`,
                scalarError
            );
        }
    }
}

const Pancake = createPancakeApi(loadNodeEngine);

module.exports = Pancake;
module.exports.default = Pancake;
