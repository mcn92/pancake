import loadEngine from './dist/engine.js';
import loadScalarEngine from './dist/engine.scalar.js';
import simdWasmAsset from './dist/engine.wasm';
import scalarWasmAsset from './dist/engine.scalar.wasm';
import createPancakeApi from './pancake-core.js';
import errorContract from './pancake-errors.js';
const { PancakeError, PANCAKE_ERROR_CODES, pancakeError } = errorContract;

let engineVariant = null;

function makeLoadError(message, error) {
  const detail = error && error.message ? error.message : String(error);
  return pancakeError(PANCAKE_ERROR_CODES.WASM_LOAD_FAILED, `${message}: ${detail}`, undefined, error);
}

function toWasmSource(asset) {
  if (asset instanceof WebAssembly.Module) return asset;
  if (asset instanceof ArrayBuffer) return asset;
  if (ArrayBuffer.isView(asset)) {
    return asset.buffer.slice(asset.byteOffset, asset.byteOffset + asset.byteLength);
  }
  throw pancakeError(PANCAKE_ERROR_CODES.WASM_LOAD_FAILED, `Unsupported WASM asset type: ${typeof asset}`, { assetType: typeof asset });
}

async function instantiateEngineWithAsset(factory, asset) {
  const compiled = asset instanceof WebAssembly.Module
    ? asset
    : await WebAssembly.compile(toWasmSource(asset));

  return factory({
    instantiateWasm(imports, successCallback) {
      const instance = new WebAssembly.Instance(compiled, imports);
      successCallback(instance, compiled);
      return instance.exports;
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
export { PancakeError, PANCAKE_ERROR_CODES };

function unsupportedNodeFileHelper(name) {
  return async function unsupported() {
    throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, `${name}() is only available in the Node.js entrypoints`);
  };
}

Pancake.loadSnapshotFile = unsupportedNodeFileHelper('loadSnapshotFile');
Pancake.loadJsonFile = unsupportedNodeFileHelper('loadJsonFile');

export default Pancake;
