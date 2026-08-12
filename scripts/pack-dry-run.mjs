import path from 'node:path';
import { spawnSync } from 'node:child_process';

const cacheDir = path.join(process.cwd(), '.npm-cache');

function npmCliPath() {
  if (process.env.npm_execpath && process.env.npm_execpath.endsWith('.js')) {
    return process.env.npm_execpath;
  }
  return path.join(path.dirname(process.execPath), 'node_modules', 'npm', 'bin', 'npm-cli.js');
}

const result = spawnSync(process.execPath, [npmCliPath(), 'pack', '--dry-run', '--cache', cacheDir], {
  stdio: 'inherit',
  env: process.env,
});

if (result.error) {
  throw result.error;
}

if (result.status !== 0) {
  process.exit(result.status ?? 1);
}
