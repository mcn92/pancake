import path from 'node:path';
import { spawnSync } from 'node:child_process';

const cacheDir = path.join(process.cwd(), '.npm-cache');

function npmCliPath() {
  if (process.env.npm_execpath && process.env.npm_execpath.endsWith('.js')) {
    return { command: process.execPath, args: [process.env.npm_execpath], shell: false };
  }
  if (process.platform === 'win32') return { command: 'cmd.exe', args: ['/d', '/s', '/c', 'npm'], shell: false };
  return { command: 'npm', args: [], shell: false };
}

const npm = npmCliPath();
const result = spawnSync(npm.command, [...npm.args, 'pack', '--dry-run', '--cache', cacheDir], {
  stdio: 'inherit',
  env: process.env,
  shell: npm.shell,
});

if (result.error) {
  throw result.error;
}

if (result.status !== 0) {
  process.exit(result.status ?? 1);
}
