import path from 'node:path';
import { spawnSync } from 'node:child_process';

const cacheDir = path.join(process.cwd(), '.npm-cache');

const result = spawnSync('npm', ['pack', '--dry-run', '--cache', cacheDir], {
  stdio: 'inherit',
  env: process.env,
});

if (result.error) {
  throw result.error;
}

if (result.status !== 0) {
  process.exit(result.status ?? 1);
}
