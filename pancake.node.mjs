import { readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import loadEngine from './dist/engine.js';
import loadScalarEngine from './dist/engine.scalar.js';
import createPancakeApi from './pancake-core.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
let engineVariant = null;

const DEFAULT_JSON_FILE_BYTES = 64 * 1024 * 1024;
const DEFAULT_SNAPSHOT_FILE_BYTES = 512 * 1024 * 1024;
const SUPPORTED_SNAPSHOT_MAGICS = new Set([
  0x504E434B, // PNCK envelope
  0x464C4857, // FLHW raw float v0
  0x464C4831, // FLH1 raw float v1
  0x49384857, // I8HW raw int8 v0
  0x49384831, // I8H1 raw int8 v1
]);

function readWasmBinary(fileName) {
  try {
    const wasmBinary = readFileSync(path.join(__dirname, 'dist', fileName));
    return wasmBinary.buffer.slice(
      wasmBinary.byteOffset,
      wasmBinary.byteOffset + wasmBinary.byteLength
    );
  } catch (error) {
    const message = error && error.message ? error.message : String(error);
    throw new Error(`Failed to load Pancake WASM binary (${fileName}): ${message}`);
  }
}

function makeLoadError(message, error) {
  const detail = error && error.message ? error.message : String(error);
  return new Error(`${message}: ${detail}`);
}

function hasSimdSupport(binary) {
  try {
    return WebAssembly.validate(binary);
  } catch {
    return false;
  }
}

function parseJsonLines(text, filePath) {
  const rows = [];
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.length === 0) continue;
    try {
      rows.push(JSON.parse(line));
    } catch (error) {
      const message = error && error.message ? error.message : String(error);
      throw new Error(`Failed to parse JSONL in ${filePath} at line ${i + 1}: ${message}`);
    }
  }
  return rows;
}

function inferJsonFormat(filePath) {
  if (/\.json$/i.test(filePath)) return 'json';
  if (/\.(jsonl|ndjson)$/i.test(filePath)) return 'jsonl';
  throw new Error(`loadJsonFile() could not infer format from '${filePath}'. Use a .json/.jsonl/.ndjson extension or pass opts.format.`);
}

function remapJsonRows(rows, vectorKey, idKey) {
  if (!Array.isArray(rows)) {
    throw new Error('loadJsonFile() expects a JSON array or JSONL sequence of vectors/records');
  }
  return rows.map((row, i) => {
    if (row instanceof Float32Array || Array.isArray(row)) {
      return row;
    }
    if (!row || typeof row !== 'object') {
      throw new Error(`loadJsonFile() expected an object or vector at index ${i}`);
    }
    if (!(vectorKey in row)) {
      throw new Error(`loadJsonFile() missing vectorKey '${vectorKey}' at index ${i}`);
    }
    const mapped = { vector: row[vectorKey] };
    if (Object.prototype.hasOwnProperty.call(row, idKey)) {
      mapped.id = row[idKey];
    }
    return mapped;
  });
}

function validateFilePath(filePath, helperName) {
  if (typeof filePath !== 'string' || filePath.length === 0) {
    throw new Error(`${helperName}() requires a non-empty file path`);
  }
}

function validateMaxFileBytes(maxFileBytes, helperName) {
  if (!Number.isInteger(maxFileBytes) || maxFileBytes <= 0) {
    throw new Error(`${helperName}() maxFileBytes must be a positive integer`);
  }
}

function statRegularFile(filePath, helperName) {
  const stat = statSync(filePath);
  if (!stat.isFile()) {
    throw new Error(`${helperName}() expected a regular file: ${filePath}`);
  }
  return stat;
}

function enforceFileSize(stat, maxFileBytes, helperName, filePath) {
  if (stat.size > maxFileBytes) {
    throw new Error(`${helperName}() file exceeds maxFileBytes (${stat.size} > ${maxFileBytes}): ${filePath}`);
  }
}

function readUtf8FileWithLimit(filePath, helperName, maxFileBytes) {
  validateFilePath(filePath, helperName);
  validateMaxFileBytes(maxFileBytes, helperName);
  const stat = statRegularFile(filePath, helperName);
  enforceFileSize(stat, maxFileBytes, helperName, filePath);
  return readFileSync(filePath, 'utf8');
}

function readBinaryFileWithLimit(filePath, helperName, maxFileBytes) {
  validateFilePath(filePath, helperName);
  validateMaxFileBytes(maxFileBytes, helperName);
  const stat = statRegularFile(filePath, helperName);
  enforceFileSize(stat, maxFileBytes, helperName, filePath);
  return readFileSync(filePath);
}

function validateSnapshotBytes(snapshot, filePath) {
  if (!snapshot || snapshot.byteLength < 4) {
    throw new Error(`loadSnapshotFile() snapshot is too small to be valid: ${filePath}`);
  }
  const view = new DataView(snapshot.buffer, snapshot.byteOffset, snapshot.byteLength);
  const magic = view.getUint32(0, true);
  if (!SUPPORTED_SNAPSHOT_MAGICS.has(magic)) {
    throw new Error(`loadSnapshotFile() unsupported snapshot file type: ${filePath}`);
  }
}

async function loadNodeEngine() {
  if (engineVariant === null) {
    engineVariant = hasSimdSupport(readWasmBinary('engine.wasm')) ? 'simd' : 'scalar';
  }

  if (engineVariant === 'scalar') {
    try {
      return await loadScalarEngine({
        wasmBinary: readWasmBinary('engine.scalar.wasm')
      });
    } catch (error) {
      throw makeLoadError('Pancake failed to load the scalar WASM engine', error);
    }
  }

  try {
    return await loadEngine({
      wasmBinary: readWasmBinary('engine.wasm')
    });
  } catch (simdError) {
    try {
      const engine = await loadScalarEngine({
        wasmBinary: readWasmBinary('engine.scalar.wasm')
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

const Pancake = createPancakeApi(loadNodeEngine);

Pancake.loadSnapshotFile = async function loadSnapshotFile(filePath, opts) {
  const {
    maxFileBytes = DEFAULT_SNAPSHOT_FILE_BYTES,
    ...createOpts
  } = opts || {};

  const snapshot = readBinaryFileWithLimit(filePath, 'loadSnapshotFile', maxFileBytes);
  validateSnapshotBytes(snapshot, filePath);
  const index = await Pancake.create(createOpts);
  try {
    index.import(snapshot);
    return index;
  } catch (error) {
    try {
      index.dispose();
    } catch {}
    throw error;
  }
};

Pancake.loadJsonFile = async function loadJsonFile(filePath, opts = {}) {
  const {
    format = inferJsonFormat(filePath),
    vectorKey = 'vector',
    idKey = 'id',
    maxFileBytes = DEFAULT_JSON_FILE_BYTES,
    ...createOpts
  } = opts;

  if (format !== 'json' && format !== 'jsonl') {
    throw new Error(`loadJsonFile() format must be 'json' or 'jsonl', got '${format}'`);
  }

  const text = readUtf8FileWithLimit(filePath, 'loadJsonFile', maxFileBytes);
  let rows;
  if (format === 'jsonl') {
    rows = parseJsonLines(text, filePath);
  } else {
    try {
      rows = JSON.parse(text);
    } catch (error) {
      const message = error && error.message ? error.message : String(error);
      throw new Error(`Failed to parse JSON in ${filePath}: ${message}`);
    }
  }
  return Pancake.fromVectors(remapJsonRows(rows, vectorKey, idKey), createOpts);
};

export default Pancake;
