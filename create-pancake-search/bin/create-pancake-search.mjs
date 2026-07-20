#!/usr/bin/env node
import { main } from '../src/cli.mjs';

main(process.argv).catch((error) => {
  const code = Number.isInteger(error?.exitCode) ? error.exitCode : 2;
  console.error(error?.message || String(error));
  process.exit(code);
});
