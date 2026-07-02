import loadEngine from './dist/engine.js';
import loadScalarEngine from './dist/engine.scalar.js';
import simdWasmAsset from './dist/engine.wasm?url';
import scalarWasmAsset from './dist/engine.scalar.wasm?url';
import createPancakeApi from './pancake-core.js';
let engineVariant = null;

function makeLoadError(message, error) {
  const detail = error && error.message ? error.message : String(error);
  return new Error(`${message}: ${detail}`);
}

function isUrlLikeWasmAsset(asset) {
  return typeof asset === 'string' || asset instanceof URL;
}

function toWasmSource(asset) {
  if (asset instanceof WebAssembly.Module) return asset;
  if (asset instanceof ArrayBuffer) return asset;
  if (ArrayBuffer.isView(asset)) {
    return asset.buffer.slice(asset.byteOffset, asset.byteOffset + asset.byteLength);
  }
  throw new Error(`Unsupported WASM asset type: ${typeof asset}`);
}

async function instantiateEngineWithAsset(factory, expectedFileName, asset) {
  if (isUrlLikeWasmAsset(asset)) {
    const href = asset instanceof URL ? asset.href : asset;
    return factory({
      locateFile(path) {
        if (path === expectedFileName) {
          return href;
        }
        return path;
      }
    });
  }

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

async function detectEngineVariant(simdAsset) {
  if (!isUrlLikeWasmAsset(simdAsset)) {
    if (simdAsset instanceof WebAssembly.Module) {
      return 'simd';
    }
    return WebAssembly.validate(toWasmSource(simdAsset)) ? 'simd' : 'scalar';
  }

  try {
    const href = simdAsset instanceof URL ? simdAsset.href : simdAsset;
    const response = await fetch(href, { credentials: 'same-origin' });
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
  if (engineVariant === null) {
    engineVariant = await detectEngineVariant(simdWasmAsset);
  }

  if (engineVariant === 'scalar') {
    try {
      return await instantiateEngineWithAsset(loadScalarEngine, 'engine.scalar.wasm', scalarWasmAsset);
    } catch (error) {
      throw makeLoadError('Pancake failed to load the scalar WASM engine', error);
    }
  }

  try {
    return await instantiateEngineWithAsset(loadEngine, 'engine.wasm', simdWasmAsset);
  } catch (simdError) {
    try {
      const engine = await instantiateEngineWithAsset(loadScalarEngine, 'engine.scalar.wasm', scalarWasmAsset);
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

function unsupportedNodeFileHelper(name) {
  return async function unsupported() {
    throw new Error(`${name}() is only available in the Node.js entrypoints`);
  };
}

Pancake.loadSnapshotFile = unsupportedNodeFileHelper('loadSnapshotFile');
Pancake.loadJsonFile = unsupportedNodeFileHelper('loadJsonFile');

export default Pancake;
