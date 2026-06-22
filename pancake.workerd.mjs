import loadEngine from './dist/engine.js';
import loadScalarEngine from './dist/engine.scalar.js';
import simdWasmAsset from './dist/engine.wasm';
import scalarWasmAsset from './dist/engine.scalar.wasm';
import createPancakeApi from './pancake-core.js';

let engineVariant = null;

function makeLoadError(message, error) {
  const detail = error && error.message ? error.message : String(error);
  return new Error(`${message}: ${detail}`);
}

function toWasmSource(asset) {
  if (asset instanceof WebAssembly.Module) return asset;
  if (asset instanceof ArrayBuffer) return asset;
  if (ArrayBuffer.isView(asset)) {
    return asset.buffer.slice(asset.byteOffset, asset.byteOffset + asset.byteLength);
  }
  throw new Error(`Unsupported WASM asset type: ${typeof asset}`);
}

function instantiateEngineWithAsset(factory, asset) {
  return factory({
    instantiateWasm(imports, successCallback) {
      WebAssembly.instantiate(toWasmSource(asset), imports)
        .then((result) => {
          const instance = result instanceof WebAssembly.Instance ? result : result.instance;
          const module = result instanceof WebAssembly.Instance ? undefined : result.module;
          successCallback(instance, module);
        })
        .catch((err) => {
          throw err;
        });
      return {};
    }
  });
}

async function loadWorkerdEngine() {
  if (engineVariant === 'scalar') {
    try {
      return await instantiateEngineWithAsset(loadScalarEngine, scalarWasmAsset);
    } catch (error) {
      throw makeLoadError('Pancake failed to load the scalar WASM engine', error);
    }
  }

  try {
    return await instantiateEngineWithAsset(loadEngine, simdWasmAsset);
  } catch (simdError) {
    try {
      const engine = await instantiateEngineWithAsset(loadScalarEngine, scalarWasmAsset);
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

const Pancake = createPancakeApi(loadWorkerdEngine);

function unsupportedNodeFileHelper(name) {
  return async function unsupported() {
    throw new Error(`${name}() is only available in the Node.js entrypoints`);
  };
}

Pancake.loadSnapshotFile = unsupportedNodeFileHelper('loadSnapshotFile');
Pancake.loadJsonFile = unsupportedNodeFileHelper('loadJsonFile');

export default Pancake;
