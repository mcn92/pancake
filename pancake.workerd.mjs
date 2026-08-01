import loadEngine from './dist/engine.js';
import loadScalarEngine from './dist/engine.scalar.js';
import simdWasmAsset from './dist/engine.wasm';
import scalarWasmAsset from './dist/engine.scalar.wasm';
import createPancakeApi from './pancake-core.js';
import errorContract from './pancake-errors.js';
import loaderContract from './pancake-loader.js';
import artifactContract from './pancake-artifact.js';
const { PancakeError, PANCAKE_ERROR_CODES, pancakeError } = errorContract;
const { createCachedModuleLoader } = loaderContract;
const { PancakeRangeArtifact, PancakeSketchArtifact } = artifactContract;

let engineVariantPromise = null;

function makeLoadError(message, error) {
  const detail = error && error.message ? error.message : String(error);
  return pancakeError(PANCAKE_ERROR_CODES.WASM_LOAD_FAILED, `${message}: ${detail}`, undefined, error);
}

const moduleLoader = createCachedModuleLoader((variant) =>
  variant === 'simd' ? simdWasmAsset : scalarWasmAsset
);

function selectEngineVariant() {
  engineVariantPromise ??= moduleLoader.supports('simd')
    .then((supported) => supported ? 'simd' : 'scalar');
  return engineVariantPromise;
}

async function loadWorkerdEngine() {
  const engineVariant = await selectEngineVariant();
  if (engineVariant === 'scalar') {
    try {
      return await moduleLoader.instantiate(loadScalarEngine, 'scalar');
    } catch (error) {
      throw makeLoadError('Pancake failed to load the scalar WASM engine', error);
    }
  }

  try {
    return await moduleLoader.instantiate(loadEngine, 'simd');
  } catch (simdError) {
    try {
      const engine = await moduleLoader.instantiate(loadScalarEngine, 'scalar');
      engineVariantPromise = Promise.resolve('scalar');
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

Pancake.RangeArtifact = PancakeRangeArtifact;
Pancake.SketchArtifact = PancakeSketchArtifact;

function unsupportedNodeFileHelper(name) {
  return async function unsupported() {
    throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, `${name}() is only available in the Node.js entrypoints`);
  };
}

Pancake.loadSnapshotFile = unsupportedNodeFileHelper('loadSnapshotFile');
Pancake.loadJsonFile = unsupportedNodeFileHelper('loadJsonFile');

export default Pancake;
