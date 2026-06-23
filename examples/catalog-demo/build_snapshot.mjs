import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import Pancake from '../../pancake.node.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const WORKER_EXPORT_MAGIC = 0x57524b31;
const WORKER_EXPORT_VERSION = 1;

function readCorpus(filePath) {
  return fs.readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function extractRawEngineBytes(exported) {
  const bytes = exported instanceof Uint8Array ? exported : new Uint8Array(exported);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = view.getUint32(0, true);
  if (magic !== 0x504E434B) return bytes;
  const version = view.getUint32(4, true);
  if (version === 3) {
    const mappingCount = view.getUint32(24, true);
    const wasmSize = view.getUint32(28, true);
    const wasmOffset = 32 + mappingCount * 8;
    return bytes.slice(wasmOffset, wasmOffset + wasmSize);
  }
  return bytes.slice(20);
}

function encodeWorkerExportEnvelope(rawBytes, metadata) {
  const metaBytes = new TextEncoder().encode(JSON.stringify(metadata));
  const raw = rawBytes instanceof Uint8Array ? rawBytes : new Uint8Array(rawBytes);
  const result = new Uint8Array(16 + metaBytes.length + raw.length);
  const view = new DataView(result.buffer);
  view.setUint32(0, WORKER_EXPORT_MAGIC, true);
  view.setUint32(4, WORKER_EXPORT_VERSION, true);
  view.setUint32(8, metaBytes.length, true);
  view.setUint32(12, raw.length, true);
  result.set(metaBytes, 16);
  result.set(raw, 16 + metaBytes.length);
  return result;
}

async function main() {
  const corpusPath = path.join(__dirname, 'search_corpus.jsonl');
  const packageSnapshotPath = path.join(__dirname, 'catalog_index.pnck');
  const workerSnapshotPath = path.join(__dirname, 'catalog_worker_export.bin');

  const rows = readCorpus(corpusPath).map((row) => ({ id: row.id, vector: row.vector }));
  const opts = {
    dim: 4,
    maxElements: rows.length,
    metric: 'cosine',
    quantized: true,
    M: 16,
    efConstruction: 50,
    efSearch: 100,
  };

  const { index, ids, idMap } = await Pancake.fromVectors(rows, opts);
  try {
    const snapshot = index.export();
    fs.writeFileSync(packageSnapshotPath, snapshot);

    const rawEngineBytes = extractRawEngineBytes(snapshot);
    const workerBlob = encodeWorkerExportEnvelope(rawEngineBytes, {
      dims: opts.dim,
      maxElements: opts.maxElements,
      nextExtId: ids.length,
      initParams: { M: opts.M, efC: opts.efConstruction, efS: opts.efSearch },
      mapping: ids.map((extId, intId) => [intId, extId]),
    });
    fs.writeFileSync(workerSnapshotPath, workerBlob);

    const preview = index.search(new Float32Array([1, 0, 0, 0]), 3).map((hit) => ({
      id: hit.id,
      sourceId: idMap.get(hit.id),
      distance: hit.distance
    }));

    console.log(`Wrote ${path.relative(process.cwd(), packageSnapshotPath)}`);
    console.log(`Wrote ${path.relative(process.cwd(), workerSnapshotPath)}`);
    console.log('Preview query [1,0,0,0]:');
    console.log(JSON.stringify(preview, null, 2));
  } finally {
    index.dispose();
  }
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
