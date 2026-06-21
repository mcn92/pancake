import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import loadEngine from './dist/engine.js';
import loadScalarEngine from './dist/engine.scalar.js';
import createPancakeApi from './pancake-core.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
let engineVariant = null;

function readWasmBinary(fileName) {
  try {
    const wasmBinary = readFileSync(path.join(__dirname, 'dist', fileName));
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
  if (engineVariant === null) {
    engineVariant = hasSimdSupport(readWasmBinary('engine.wasm')) ? 'simd' : 'scalar';
  }

  if (engineVariant === 'scalar') {
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
      engineVariant = 'scalar';
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

export default Pancake;
