import { copyFile, mkdir, rm, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, 'dist');

await rm(dist, { recursive: true, force: true });
await mkdir(dist, { recursive: true });

await copyFile(join(here, 'index.html'), join(dist, 'index.html'));
await copyFile(join(here, 'styles.css'), join(dist, 'styles.css'));
await copyFile(join(here, 'app.js'), join(dist, 'app.js'));

const apiBase = process.env.PIKELET_WORKER_API_URL || '';
await mkdir(join(dist, 'config'), { recursive: true });
await copyFile(join(here, '_headers'), join(dist, '_headers'));
await copyFile(join(here, '_redirects'), join(dist, '_redirects'));
await writeRuntimeConfig(join(dist, 'config', 'runtime.js'), apiBase);

async function writeRuntimeConfig(path, apiBase) {
  await writeFile(
    path,
    `window.PIKELET_DEMO_CONFIG = ${JSON.stringify({ apiBase }, null, 2)};\n`,
    'utf8'
  );
}
