import loadEngine from './dist/engine.js';
import wasmModule from './dist/engine.wasm';
import createPancakeApi from './pancake-core.js';

function loadWebEngine() {
  return loadEngine({
    instantiateWasm(imports, successCallback) {
      WebAssembly.instantiate(wasmModule, imports)
        .then(instance => successCallback(instance))
        .catch(err => { throw err; });
      return {};
    }
  });
}

const Pancake = createPancakeApi(loadWebEngine);

export default Pancake;
