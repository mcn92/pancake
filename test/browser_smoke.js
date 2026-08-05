'use strict';

const fs = require('fs');
const { spawn } = require('child_process');
const { execFileSync } = require('child_process');
const os = require('os');
const path = require('path');
const { chromium, webkit, firefox } = require('playwright');

const ROOT_DIR = process.cwd();
const FIXTURE_DIR = path.join(process.cwd(), 'examples', '01-hello-pack');
const HOST = '127.0.0.1';
const PORT = 4173;
const URL = `http://${HOST}:${PORT}`;

function prepareFixtureDependency() {
  const npmCacheDir = path.join(os.tmpdir(), '.npm-pack-cache');
  const packJson = execFileSync(
    'npm',
    ['pack', '--json', '--ignore-scripts', '--dry-run=false', '--cache', npmCacheDir],
    { cwd: ROOT_DIR, encoding: 'utf8' }
  );

  const [{ filename }] = JSON.parse(packJson);
  const tarballPath = path.join(ROOT_DIR, filename);

  try {
    execFileSync(
      'npm',
      ['install', '--no-save', tarballPath],
      { cwd: FIXTURE_DIR, stdio: 'pipe', encoding: 'utf8' }
    );
  } finally {
    if (fs.existsSync(tarballPath)) {
      fs.unlinkSync(tarballPath);
    }
  }
}

function waitForServer(url, timeoutMs = 20000) {
  const started = Date.now();

  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const res = await fetch(url);
        if (res.ok) {
          resolve();
          return;
        }
      } catch {}

      if (Date.now() - started >= timeoutMs) {
        reject(new Error(`Vite fixture failed to start within ${timeoutMs}ms`));
        return;
      }

      setTimeout(tick, 250);
    };

    tick();
  });
}

function startVite() {
  const viteCli = path.join(ROOT_DIR, 'node_modules', 'vite', 'bin', 'vite.js');
  const proc = spawn(
    process.execPath,
    [viteCli, '--host', HOST, '--port', String(PORT)],
    {
      cwd: FIXTURE_DIR,
      stdio: ['ignore', 'pipe', 'pipe']
    }
  );

  let output = '';
  proc.stdout.on('data', chunk => { output += chunk.toString(); });
  proc.stderr.on('data', chunk => { output += chunk.toString(); });

  return { proc, getOutput: () => output };
}

function stopProcess(proc) {
  return new Promise(resolve => {
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

async function runBrowserTest(browserType, url) {
  const browser = await browserType.launch({ headless: true });

  try {
    const page = await browser.newPage();
    await page.goto(url, { waitUntil: 'networkidle' });

    await page.waitForFunction(() => {
      return !!window.__PANCAKE_BROWSER_SMOKE__;
    }, null, { timeout: 20000 });

    const result = await page.evaluate(() => window.__PANCAKE_BROWSER_SMOKE__);

    if (!result || !result.ok) {
      throw new Error(`Test failed: ${result && result.error ? result.error : 'unknown error'}`);
    }

    const expectedBefore = [result.ids.idA, result.ids.idB, result.ids.idC];
    const expectedAfter = [result.ids.idA, result.ids.idB];

    if (JSON.stringify(result.beforeTopIds) !== JSON.stringify(expectedBefore)) {
      throw new Error(`Unexpected search results before export: ${JSON.stringify(result.beforeTopIds)}`);
    }

    if (JSON.stringify(result.afterTopIds) !== JSON.stringify(expectedAfter)) {
      throw new Error(`Unexpected search results after import: ${JSON.stringify(result.afterTopIds)}`);
    }

    if (result.importedCount !== 2) {
      throw new Error(`Unexpected imported count: ${result.importedCount}`);
    }

    if (result.finalCount !== 3) {
      throw new Error(`Unexpected final count after post-import add: ${result.finalCount}`);
    }

    if (result.nextId !== result.ids.idC + 1) {
      throw new Error(`Unexpected nextId after import: ${result.nextId}`);
    }
  } finally {
    await browser.close();
  }
}

const BROWSERS = [
  { name: 'chromium', type: chromium },
  { name: 'firefox',  type: firefox },
  { name: 'webkit',   type: webkit },
];

async function main() {
  prepareFixtureDependency();

  const { proc, getOutput } = startVite();

  try {
    await waitForServer(URL);

    const failures = [];
    const skipped = [];
    let passed = 0;

    for (const { name, type } of BROWSERS) {
      try {
        await runBrowserTest(type, URL);
        console.log(`  ${name}: passed`);
        passed++;
      } catch (error) {
        if (error.message.includes('missing dependencies') || error.message.includes('Missing libraries')) {
          console.log(`  ${name}: skipped (missing system dependencies)`);
          skipped.push(name);
        } else {
          console.error(`  ${name}: FAILED — ${error.message}`);
          failures.push(name);
        }
      }
    }

    if (passed === 0) {
      throw new Error('Browser smoke test failed: no browser engines available');
    }

    if (failures.length > 0) {
      throw new Error(`Browser smoke test failed on: ${failures.join(', ')}`);
    }

    const summary = skipped.length > 0
      ? `Browser smoke test passed (${passed}/${BROWSERS.length} engines; skipped ${skipped.join(', ')})`
      : `Browser smoke test passed (all ${BROWSERS.length} engines)`;
    console.log(summary);
  } catch (error) {
    const detail = getOutput().trim();
    if (detail) {
      console.error(detail);
    }
    throw error;
  } finally {
    await stopProcess(proc);
  }
}

main().catch(error => {
  console.error(error.stack || error.message);
  process.exit(1);
});
