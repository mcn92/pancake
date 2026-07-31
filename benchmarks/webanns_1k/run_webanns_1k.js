'use strict';

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { chromium } = require('playwright');

const ROOT_DIR = path.resolve(__dirname, '..', '..');
const FIXTURE_DIR = path.join(ROOT_DIR, 'examples', 'browser-vite');
const HOST = '127.0.0.1';
const PORT = 4181;
const URL = `http://${HOST}:${PORT}/webanns_1k/index.html`;
const OUT_DIR = path.join(ROOT_DIR, 'benchmark_reports');

function waitForServer(url, timeoutMs = 20000) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const response = await fetch(url);
        if (response.ok) {
          resolve();
          return;
        }
      } catch {}

      if (Date.now() - started >= timeoutMs) {
        reject(new Error(`server failed to start within ${timeoutMs}ms`));
        return;
      }
      setTimeout(tick, 250);
    };
    tick();
  });
}

function startVite() {
  const viteCli = path.join(FIXTURE_DIR, 'node_modules', 'vite', 'bin', 'vite.js');
  const proc = spawn(process.execPath, [viteCli, '--host', HOST, '--port', String(PORT)], {
    cwd: FIXTURE_DIR,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  let output = '';
  proc.stdout.on('data', (chunk) => { output += chunk.toString(); });
  proc.stderr.on('data', (chunk) => { output += chunk.toString(); });
  return { proc, getOutput: () => output };
}

function stopProcess(proc) {
  return new Promise((resolve) => {
    if (!proc || proc.exitCode !== null) {
      resolve();
      return;
    }
    proc.once('exit', () => resolve());
    proc.kill('SIGTERM');
    setTimeout(() => {
      try { proc.kill('SIGKILL'); } catch {}
      resolve();
    }, 3000);
  });
}

function formatMs(value) {
  return `${value.toFixed(4)} ms`;
}

function printSummary(result) {
  console.log(`dataset: ${result.results[0].dataset}`);
  console.log(`browser: ${result.user_agent}`);
  for (const item of result.results) {
    console.log('');
    console.log(item.name);
    console.log(`  build: ${formatMs(item.build_ms)}`);
    console.log(`  snapshot: ${(item.snapshot_bytes / 1024).toFixed(1)} KiB in ${formatMs(item.snapshot_ms)}`);
    console.log(`  random p50/p95/p99: ${formatMs(item.random_query_latency.p50_ms)} / ${formatMs(item.random_query_latency.p95_ms)} / ${formatMs(item.random_query_latency.p99_ms)}`);
    console.log(`  self p50/p95/p99:   ${formatMs(item.self_query_latency.p50_ms)} / ${formatMs(item.self_query_latency.p95_ms)} / ${formatMs(item.self_query_latency.p99_ms)}`);
    console.log(`  self recall@10: ${item.self_query_recall_at_10.toFixed(4)}`);
  }
}

async function main() {
  const { proc, getOutput } = startVite();
  try {
    await waitForServer(URL);
    const browser = await chromium.launch({ headless: true });
    try {
      const page = await browser.newPage();
      await page.goto(URL, { waitUntil: 'networkidle' });
      await page.waitForFunction(() => window.__PANCAKE_WEBANNS_1K__, null, { timeout: 120000 });
      const result = await page.evaluate(() => window.__PANCAKE_WEBANNS_1K__);
      if (!result || !result.ok) {
        throw new Error(result && result.error ? result.error : 'benchmark failed');
      }

      fs.mkdirSync(OUT_DIR, { recursive: true });
      const outPath = path.join(OUT_DIR, `webanns_arxiv_1k_pancake_${Date.now()}.json`);
      fs.writeFileSync(outPath, JSON.stringify(result, null, 2));
      printSummary(result);
      console.log('');
      console.log(`wrote ${outPath}`);
    } finally {
      await browser.close();
    }
  } catch (error) {
    const serverOutput = getOutput().trim();
    if (serverOutput) console.error(serverOutput);
    throw error;
  } finally {
    await stopProcess(proc);
  }
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
