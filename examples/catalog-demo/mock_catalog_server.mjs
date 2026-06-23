import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORT = parseInt(process.env.CATALOG_DEMO_PORT || '9090', 10);

const productsPath = path.join(__dirname, 'products.json');
const products = JSON.parse(fs.readFileSync(productsPath, 'utf8'));
const byId = new Map(products.map((product) => [product.id, { ...product }]));

function sendJson(res, status, body) {
  const text = JSON.stringify(body, null, 2);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(text),
  });
  res.end(text);
}

function parseBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => {
      try {
        const text = Buffer.concat(chunks).toString('utf8');
        resolve(text ? JSON.parse(text) : {});
      } catch (error) {
        reject(error);
      }
    });
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://127.0.0.1:${PORT}`);

  if (req.method === 'GET' && url.pathname === '/health') {
    return sendJson(res, 200, { ok: true, count: byId.size });
  }

  if (req.method === 'GET' && url.pathname === '/products') {
    const ids = (url.searchParams.get('ids') || '')
      .split(',')
      .map((id) => id.trim())
      .filter(Boolean);
    const items = ids.map((id) => byId.get(id)).filter(Boolean);
    return sendJson(res, 200, { items });
  }

  if (req.method === 'POST' && url.pathname === '/admin/update') {
    let body;
    try {
      body = await parseBody(req);
    } catch {
      return sendJson(res, 400, { error: 'Invalid JSON' });
    }

    const product = byId.get(body.id);
    if (!product) return sendJson(res, 404, { error: 'Unknown product id' });

    if (body.priceCents !== undefined) {
      if (!Number.isInteger(body.priceCents) || body.priceCents < 0) {
        return sendJson(res, 400, { error: 'priceCents must be a non-negative integer' });
      }
      product.priceCents = body.priceCents;
    }
    if (body.inventory !== undefined) {
      if (!Number.isInteger(body.inventory) || body.inventory < 0) {
        return sendJson(res, 400, { error: 'inventory must be a non-negative integer' });
      }
      product.inventory = body.inventory;
    }

    return sendJson(res, 200, { updated: product });
  }

  return sendJson(res, 404, { error: 'Not found' });
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`Catalog demo server listening on http://127.0.0.1:${PORT}`);
});
