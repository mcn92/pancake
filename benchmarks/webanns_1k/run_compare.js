'use strict';

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { chromium } = require('playwright');

const ROOT_DIR = path.resolve(__dirname, '..', '..');
const FIXTURE_DIR = path.join(ROOT_DIR, 'examples', 'browser-vite');
const HOST = '127.0.0.1';
const PORT = 4182;
const targetSize = Number.parseInt(process.argv[2] || process.env.WEBANNS_SIZE || '1000', 10);
const webannsMode = process.argv[3] || process.env.WEBANNS_MODE || (targetSize === 1000 ? 'cache' : 'build');
const URL = `http://${HOST}:${PORT}/webanns_compare/index.html?size=${targetSize}&webanns=${encodeURIComponent(webannsMode)}`;
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

function fmtMs(value) {
  return `${value.toFixed(4)} ms`;
}

function printSummary(result) {
  console.log(`dataset: ${result.dataset}, vectors=${result.vectors}, dim=${result.dim}, k=${result.top_k}`);
  console.log(`browser: ${result.user_agent}`);
  console.log('');
  console.log('system                         setup/build      random p99    self p99      recall@10      size');
  console.log('-----------------------------  --------------   ----------    ----------    ----------     ----------');
  for (const item of result.results) {
    const setup = item.system === 'webanns'
      ? `${fmtMs(item.init_ms + item.import_with_cache_ms)}`
      : `${fmtMs(item.build_ms)}`;
    const size = item.system === 'webanns'
      ? `${((item.data_bytes + item.graph_bytes) / 1024).toFixed(1)} KiB`
      : `${(item.snapshot_bytes / 1024).toFixed(1)} KiB`;
    console.log(
      `${item.name.padEnd(29)}  ${setup.padStart(14)}   ` +
      `${fmtMs(item.random_query_latency.p99_ms).padStart(10)}    ` +
      `${fmtMs(item.self_query_latency.p99_ms).padStart(10)}    ` +
      `${item.self_query_recall_at_10.toFixed(4).padStart(10)}     ` +
      `${size.padStart(10)}`
    );
  }
}

async function main() {
  const { proc, getOutput } = startVite();
  try {
    await waitForServer(URL);
    const browser = await chromium.launch({ headless: true });
    try {
      const page = await browser.newPage();
      page.on('console', (message) => {
        const text = message.text();
        if (/^(WRAG|Loading|warning!)/.test(text)) return;
        if (message.type() === 'error') console.error(`browser console: ${text}`);
      });
      await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 120000 });
      await page.waitForFunction(() => window.__PANCAKE_WEBANNS_COMPARE__, null, { timeout: 900000 });
      const result = await page.evaluate(() => window.__PANCAKE_WEBANNS_COMPARE__);
      if (!result || !result.ok) {
        throw new Error(result && result.error ? result.error : 'comparison failed');
      }
      fs.mkdirSync(OUT_DIR, { recursive: true });
      const outPath = path.join(OUT_DIR, `webanns_arxiv_${targetSize}_compare_${Date.now()}.json`);
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
