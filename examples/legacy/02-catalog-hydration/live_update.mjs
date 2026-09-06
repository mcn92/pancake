import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const WORKER_URL = process.env.PIKELET_WORKER_URL || 'http://127.0.0.1:8787';
const CATALOG_URL = process.env.CATALOG_DEMO_URL || 'http://127.0.0.1:9090';
const API_KEY = process.env.PIKELET_API_KEY || '';
const RUNTIME_MAPPINGS_PATH = path.join(__dirname, 'runtime_mappings.json');

const leatherBoots = {
  id: 'boot-leather-hiker',
  title: 'Ridgeline Leather Hiking Boot',
  category: 'boots',
  description: 'Full-grain leather hiking boot with durable trail support.',
  priceCents: 18800,
  inventory: 20,
  url: '/products/boot-leather-hiker',
};

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) {
    throw new Error(`${options.method || 'GET'} ${url} -> ${response.status}: ${text}`);
  }
  return data;
}

function workerHeaders() {
  return {
    'Content-Type': 'application/json',
    ...(API_KEY ? { Authorization: `Bearer ${API_KEY}` } : {}),
  };
}

function readMappings() {
  if (!fs.existsSync(RUNTIME_MAPPINGS_PATH)) return { workerIdToProductId: {} };
  return JSON.parse(fs.readFileSync(RUNTIME_MAPPINGS_PATH, 'utf8'));
}

function writeMappings(mappings) {
  fs.writeFileSync(RUNTIME_MAPPINGS_PATH, `${JSON.stringify(mappings, null, 2)}\n`);
}

async function createProduct() {
  try {
    return await fetchJson(`${CATALOG_URL}/admin/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(leatherBoots),
    });
  } catch (error) {
    if (String(error.message).includes('409')) {
      return { created: leatherBoots, alreadyExisted: true };
    }
    throw error;
  }
}

async function main() {
  await fetchJson(`${WORKER_URL}/delete`, {
    method: 'POST',
    headers: workerHeaders(),
    body: JSON.stringify({ id: 0 }),
  });

  const created = await createProduct();
  const added = await fetchJson(`${WORKER_URL}/add`, {
    method: 'POST',
    headers: workerHeaders(),
    body: JSON.stringify({ vector: [0.05, 1, 0, 0] }),
  });

  const mappings = readMappings();
  mappings.workerIdToProductId[String(added.id)] = leatherBoots.id;
  writeMappings(mappings);

  console.log(JSON.stringify({
    removedFromIndex: 'jacket-rain-shell',
    createdCatalogProduct: created.created.id,
    createdCatalogProductAlreadyExisted: Boolean(created.alreadyExisted),
    addedWorkerId: added.id,
    runtimeMapping: RUNTIME_MAPPINGS_PATH,
  }, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
