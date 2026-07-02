import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import Pancake from '../../pancake.node.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function readCorpus(filePath) {
  return fs.readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

async function main() {
  const corpusPath = path.join(__dirname, 'search_corpus.jsonl');
  const packageSnapshotPath = path.join(__dirname, 'catalog_index.pnck');

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

    const preview = index.search(new Float32Array([1, 0, 0, 0]), 3).map((hit) => ({
      id: hit.id,
      sourceId: idMap.get(hit.id),
      distance: hit.distance
    }));

    console.log(`Wrote ${path.relative(process.cwd(), packageSnapshotPath)}`);
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
