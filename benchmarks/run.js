#!/usr/bin/env node
'use strict';

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const BENCHMARK_DIR = __dirname;
const EXCLUDED = new Set(['bench_args.js', 'run.js']);

function listBenchmarkScripts() {
  return fs.readdirSync(BENCHMARK_DIR)
    .filter(name => name.endsWith('.js') && !EXCLUDED.has(name))
    .sort();
}

function resolveBenchmarkName(input) {
  if (!input) return null;
  const direct = input.endsWith('.js') ? input : `${input}.js`;
  if (fs.existsSync(path.join(BENCHMARK_DIR, direct))) return direct;

  const scripts = listBenchmarkScripts();
  const byBasename = scripts.find(name => path.basename(name, '.js') === input);
  if (byBasename) return byBasename;

  const byAlias = scripts.find(name => {
    const basename = path.basename(name, '.js');
    return basename === `benchmark_${input}`
      || basename === `qps-recall_sweep_${input}`;
  });
  return byAlias || null;
}

function printUsage() {
  console.log('Usage: node benchmarks/run.js <benchmark> [benchmark args]');
  console.log('');
  console.log('Examples:');
  console.log('  node benchmarks/run.js benchmark_nytimes --m 12 --ef-construction 100 --ef-search 200');
  console.log('  node benchmarks/run.js benchmark_sift1m --m 24 --ef-construction 300 --ef-search 150 /path/to/sift');
  console.log('  node benchmarks/run.js --list');
  console.log('');
  console.log('Supported benchmark args forwarded to benchmark scripts:');
  console.log('  --m <int>');
  console.log('  --ef-construction <int>');
  console.log('  --ef-search <int>');
  console.log('  --ef-search-values <comma-separated list>');
}

function main() {
  const argv = process.argv.slice(2);
  if (argv.length === 0 || argv.includes('--help') || argv.includes('-h')) {
    printUsage();
    process.exit(argv.length === 0 ? 1 : 0);
  }

  if (argv.includes('--list')) {
    for (const script of listBenchmarkScripts()) {
      console.log(path.basename(script, '.js'));
    }
    return;
  }

  const benchmarkArg = argv[0];
  const scriptName = resolveBenchmarkName(benchmarkArg);
  if (!scriptName) {
    console.error(`Unknown benchmark: ${benchmarkArg}`);
    console.error('');
    printUsage();
    process.exit(1);
  }

  const child = spawn(process.execPath, [path.join(BENCHMARK_DIR, scriptName), ...argv.slice(1)], {
    stdio: 'inherit',
    cwd: path.resolve(BENCHMARK_DIR, '..'),
    env: process.env,
  });

  child.on('exit', code => {
    process.exit(code === null ? 1 : code);
  });
}

main();
