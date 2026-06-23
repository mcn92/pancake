#!/usr/bin/env node
'use strict';

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const BENCHMARK_DIR = __dirname;
const EXCLUDED = new Set(['bench_args.js', 'run.js']);
const ROOT_DIR = path.resolve(BENCHMARK_DIR, '..');
const RESULTS_DIR = path.join(ROOT_DIR, 'benchmark_results');
const RELEASE_RESULTS_DIR = path.join(RESULTS_DIR, 'release');

function listBenchmarkScripts() {
  return fs.readdirSync(BENCHMARK_DIR)
    .filter(name => name.endsWith('.js') && !EXCLUDED.has(name))
    .sort();
}

function ensureReleaseResultsDir() {
  fs.mkdirSync(RELEASE_RESULTS_DIR, { recursive: true });
}

function listTopLevelResultFiles() {
  if (!fs.existsSync(RESULTS_DIR)) return new Map();

  const entries = fs.readdirSync(RESULTS_DIR, { withFileTypes: true });
  const files = new Map();
  for (const entry of entries) {
    if (!entry.isFile()) continue;
    const filePath = path.join(RESULTS_DIR, entry.name);
    const stat = fs.statSync(filePath);
    files.set(entry.name, { path: filePath, mtimeMs: stat.mtimeMs, size: stat.size });
  }
  return files;
}

function copyReleaseArtifacts(beforeFiles, runStartedAtMs) {
  ensureReleaseResultsDir();
  const afterFiles = listTopLevelResultFiles();
  const copied = [];

  for (const [name, info] of afterFiles) {
    const before = beforeFiles.get(name);
    const isNew = !before;
    const changed = before && (before.mtimeMs !== info.mtimeMs || before.size !== info.size);
    const touchedThisRun = info.mtimeMs >= runStartedAtMs - 1000;
    if (!(isNew || changed || touchedThisRun)) continue;

    const destPath = path.join(RELEASE_RESULTS_DIR, name);
    fs.copyFileSync(info.path, destPath);
    copied.push(destPath);
  }

  return copied.sort();
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

  const beforeFiles = listTopLevelResultFiles();
  const runStartedAtMs = Date.now();
  const child = spawn(process.execPath, [path.join(BENCHMARK_DIR, scriptName), ...argv.slice(1)], {
    stdio: 'inherit',
    cwd: ROOT_DIR,
    env: process.env,
  });

  child.on('exit', code => {
    if (code === 0) {
      const copied = copyReleaseArtifacts(beforeFiles, runStartedAtMs);
      if (copied.length > 0) {
        console.log('');
        console.log('Copied release result artifacts to:');
        for (const filePath of copied) {
          console.log(`  ${path.relative(ROOT_DIR, filePath)}`);
        }
      }
    }
    process.exit(code === null ? 1 : code);
  });
}

main();
