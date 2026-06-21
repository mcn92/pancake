import { spawnSync } from 'node:child_process';

const scalarOnly = process.argv.includes('--scalar-only');

function run(command, args, extraEnv = {}) {
  const result = spawnSync(command, args, {
    stdio: 'inherit',
    env: {
      ...process.env,
      ...extraEnv,
    },
  });

  if (result.error) {
    throw result.error;
  }

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

if (!scalarOnly) {
  run('node', ['scripts/build-engine.mjs']);
}

run('node', ['scripts/build-engine.mjs'], {
  OUT_BASENAME: 'engine.scalar',
  PATCH_ENGINE_JS: '0',
  WASM_SIMD: '0',
});
