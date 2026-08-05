import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const WORKER_URL = process.env.PANCAKE_WORKER_URL || 'http://127.0.0.1:8787';
const CATALOG_URL = process.env.CATALOG_DEMO_URL || 'http://127.0.0.1:9090';
const CORPUS_PATH = path.join(__dirname, 'search_corpus.jsonl');
const RUNTIME_MAPPINGS_PATH = path.join(__dirname, 'runtime_mappings.json');

const corpusRows = fs.readFileSync(CORPUS_PATH, 'utf8')
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => JSON.parse(line));

const workerIdToProductId = new Map(corpusRows.map((row, idx) => [idx, row.id]));

if (fs.existsSync(RUNTIME_MAPPINGS_PATH)) {
  const mappings = JSON.parse(fs.readFileSync(RUNTIME_MAPPINGS_PATH, 'utf8'));
  for (const [workerId, productId] of Object.entries(mappings.workerIdToProductId || {})) {
    workerIdToProductId.set(Number(workerId), productId);
  }
}

function queryToDemoVector(queryText) {
  const text = queryText.toLowerCase();
  const vector = [0, 0, 0, 0];

  if (/(jacket|shell|parka|rain|waterproof)/.test(text)) vector[0] += 1;
  if (/(boot|shoe|runner|trail)/.test(text)) vector[1] += 1;
  if (/(pack|backpack|daypack)/.test(text)) vector[2] += 1;
  if (/(tent|shelter|camp)/.test(text)) vector[3] += 1;

  if (/(lightweight|ultralight)/.test(text)) {
    vector[0] += 0.2;
    vector[1] += 0.15;
    vector[3] += 0.1;
  }
  if (/(hiking|backpacking)/.test(text)) {
    vector[0] += 0.1;
    vector[1] += 0.15;
    vector[2] += 0.15;
    vector[3] += 0.15;
  }
  if (vector.every((value) => value === 0)) vector[0] = 1;

  const norm = Math.hypot(...vector);
  return vector.map((value) => value / norm);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) {
    throw new Error(`${options.method || 'GET'} ${url} -> ${response.status}: ${text}`);
  }
  return data;
}

function formatPrice(priceCents) {
  return `$${(priceCents / 100).toFixed(2)}`;
}

function formatLatencyMs(ms) {
  if (!Number.isFinite(ms)) return 'unknown';
  if (ms > 0 && ms < 0.01) return `${(ms * 1000).toFixed(1)} us`;
  return `${ms.toFixed(2)} ms`;
}

async function main() {
  const queryText = process.argv.slice(2).join(' ').trim() || 'lightweight waterproof hiking jacket';
  const query = queryToDemoVector(queryText);

  const search = await fetchJson(`${WORKER_URL}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, k: 3, ef: 100 }),
  });

  const workerIds = search.neighbors.map((neighbor) => neighbor.id);
  const productIds = workerIds.map((workerId) => workerIdToProductId.get(workerId)).filter(Boolean);
  const hydrated = await fetchJson(`${CATALOG_URL}/products?ids=${productIds.join(',')}`);
  const byId = new Map(hydrated.items.map((item) => [item.id, item]));

  console.log(`Query: ${queryText}`);
  console.log(`Demo vector: [${query.map((value) => value.toFixed(3)).join(', ')}]`);
  console.log(`Worker search time: ${formatLatencyMs(search.search_ms)}`);
  console.log('');

  workerIds.forEach((workerId, index) => {
    const productId = workerIdToProductId.get(workerId);
    const item = productId ? byId.get(productId) : null;
    const distance = search.neighbors[index].distance;
    if (!item) {
      console.log(`${index + 1}. workerId=${workerId} productId=${productId ?? 'unknown'} (catalog hydration missing)`);
      return;
    }
    console.log(`${index + 1}. ${item.title}`);
    console.log(`   workerId=${workerId} id=${item.id} distance=${distance.toFixed(4)} inventory=${item.inventory} price=${formatPrice(item.priceCents)}`);
    console.log(`   ${item.description}`);
  });
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
