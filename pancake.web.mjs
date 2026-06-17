import loadEngine from './dist/engine.js';
import loadScalarEngine from './dist/engine.scalar.js';
import createPancakeApi from './pancake-core.js';
let engineVariant = null;

function isSimdLoadFailure(error) {
  const text = String(error && (error.stack || error.message || error)).toLowerCase();
  return text.includes('simd') || text.includes('v128');
}

async function loadWebEngine() {
  const wasmUrl = new URL('./dist/engine.wasm', import.meta.url);
  const scalarWasmUrl = new URL('./dist/engine.scalar.wasm', import.meta.url);

  if (engineVariant === 'scalar') {
    return loadScalarEngine({
      locateFile(path) {
        if (path === 'engine.scalar.wasm') {
          return scalarWasmUrl.href;
        }
        return path;
      }
    });
  }

  try {
    const engine = await loadEngine({
      locateFile(path) {
        if (path === 'engine.wasm') {
          return wasmUrl.href;
        }
        return path;
      }
    });
    engineVariant = 'simd';
    return engine;
  } catch (error) {
    if (!isSimdLoadFailure(error)) throw error;
    const engine = await loadScalarEngine({
      locateFile(path) {
        if (path === 'engine.scalar.wasm') {
          return scalarWasmUrl.href;
        }
        return path;
      }
    });
    engineVariant = 'scalar';
    return engine;
  }
}

const Pancake = createPancakeApi(loadWebEngine);

export default Pancake;
