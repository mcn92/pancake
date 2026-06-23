import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const WORKER_URL = process.env.PANCAKE_WORKER_URL || 'http://127.0.0.1:8787';
const SNAPSHOT_PATH = path.join(__dirname, 'catalog_worker_export.bin');

async function main() {
  const body = fs.readFileSync(SNAPSHOT_PATH);
  const response = await fetch(`${WORKER_URL}/import?dims=4`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/octet-stream',
      'Content-Length': String(body.byteLength),
    },
    body,
  });

  const text = await response.text();
  console.log(text);
  if (!response.ok) process.exit(1);
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
