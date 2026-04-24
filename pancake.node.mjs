import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import loadEngine from './dist/engine.js';
import createPancakeApi from './pancake-core.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function loadNodeEngine() {
  const wasmBinary = readFileSync(path.join(__dirname, 'dist', 'engine.wasm'));
  return loadEngine({
    wasmBinary: wasmBinary.buffer.slice(
      wasmBinary.byteOffset,
      wasmBinary.byteOffset + wasmBinary.byteLength
    )
  });
}

const Pancake = createPancakeApi(loadNodeEngine);

export default Pancake;
