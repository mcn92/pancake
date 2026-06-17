import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import loadEngine from './dist/engine.js';
import loadScalarEngine from './dist/engine.scalar.js';
import createPancakeApi from './pancake-core.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
let engineVariant = null;

function isSimdLoadFailure(error) {
  const text = String(error && (error.stack || error.message || error)).toLowerCase();
  return text.includes('simd') || text.includes('v128');
}

function readWasmBinary(fileName) {
  const wasmBinary = readFileSync(path.join(__dirname, 'dist', fileName));
  return wasmBinary.buffer.slice(
    wasmBinary.byteOffset,
    wasmBinary.byteOffset + wasmBinary.byteLength
  );
}

async function loadNodeEngine() {
  if (engineVariant === 'scalar') {
    return loadScalarEngine({
      wasmBinary: readWasmBinary('engine.scalar.wasm')
    });
  }
  try {
    const engine = await loadEngine({
      wasmBinary: readWasmBinary('engine.wasm')
    });
    engineVariant = 'simd';
    return engine;
  } catch (error) {
    if (!isSimdLoadFailure(error)) throw error;
    const engine = await loadScalarEngine({
      wasmBinary: readWasmBinary('engine.scalar.wasm')
    });
    engineVariant = 'scalar';
    return engine;
  }
}

const Pancake = createPancakeApi(loadNodeEngine);

export default Pancake;
