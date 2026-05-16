import loadEngine from './dist/engine.js';
import createPancakeApi from './pancake-core.js';

function loadWebEngine() {
  const wasmUrl = new URL('./dist/engine.wasm', import.meta.url);

  return loadEngine({
    locateFile(path) {
      if (path === 'engine.wasm') {
        return wasmUrl.href;
      }
      return path;
    }
  });
}

const Pancake = createPancakeApi(loadWebEngine);

export default Pancake;
