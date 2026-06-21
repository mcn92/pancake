import loadEngine from './dist/engine.js';
import loadScalarEngine from './dist/engine.scalar.js';
import createPancakeApi from './pancake-core.js';
let engineVariant = null;

function makeLoadError(message, error) {
  const detail = error && error.message ? error.message : String(error);
  return new Error(`${message}: ${detail}`);
}

async function detectEngineVariant(wasmUrl) {
  try {
    const response = await fetch(wasmUrl, { credentials: 'same-origin' });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`.trim());
    }
    const binary = await response.arrayBuffer();
    return WebAssembly.validate(binary) ? 'simd' : 'scalar';
  } catch (error) {
    throw makeLoadError('Failed to probe Pancake SIMD WASM support', error);
  }
}

async function loadWebEngine() {
  const wasmUrl = new URL('./dist/engine.wasm', import.meta.url);
  const scalarWasmUrl = new URL('./dist/engine.scalar.wasm', import.meta.url);

  if (engineVariant === null) {
    engineVariant = await detectEngineVariant(wasmUrl);
  }

  if (engineVariant === 'scalar') {
    try {
      return await loadScalarEngine({
        locateFile(path) {
          if (path === 'engine.scalar.wasm') {
            return scalarWasmUrl.href;
          }
          return path;
        }
      });
    } catch (error) {
      throw makeLoadError('Pancake failed to load the scalar WASM engine', error);
    }
  }

  try {
    return await loadEngine({
      locateFile(path) {
        if (path === 'engine.wasm') {
          return wasmUrl.href;
        }
        return path;
      }
    });
  } catch (simdError) {
    try {
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
    } catch (scalarError) {
      throw makeLoadError(
        `Pancake failed to load either the SIMD or scalar WASM engine (SIMD error: ${simdError && simdError.message ? simdError.message : String(simdError)})`,
        scalarError
      );
    }
  }
}

const Pancake = createPancakeApi(loadWebEngine);

export default Pancake;
