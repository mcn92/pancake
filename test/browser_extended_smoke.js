'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium, webkit, firefox } = require('playwright');

const HTML_PATH = path.join(__dirname, 'ios-smoke.html');
const HOST = '127.0.0.1';
const PORT = 4174;

function startServer() {
  const html = fs.readFileSync(HTML_PATH);
  const server = http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(html);
  });
  return new Promise(resolve => {
    server.listen(PORT, HOST, () => resolve(server));
  });
}

async function runBrowserTest(browserType, url) {
  const browser = await browserType.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.goto(url, { waitUntil: 'domcontentloaded' });

    await page.waitForFunction(() => !!window.__PANCAKE_SMOKE__, null, {
      timeout: 60000
    });

    const result = await page.evaluate(() => window.__PANCAKE_SMOKE__);
    if (!result || !result.ok) {
      throw new Error(result && result.error ? result.error : 'unknown error');
    }
    return result;
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
  const server = await startServer();
  const url = `http://${HOST}:${PORT}`;

  try {
    const failures = [];
    const skipped = [];
    let passed = 0;

    for (const { name, type } of BROWSERS) {
      try {
        const result = await runBrowserTest(type, url);
        if (result.perf) {
          const p = result.perf;
          console.log(`  ${name}: passed (insert ${p.vectors}x${p.dims}D in ${p.insertMs}ms, search p50=${p.searchP50Ms}ms p99=${p.searchP99Ms}ms, compact ${p.compactMs}ms)`);
        } else {
          console.log(`  ${name}: passed`);
        }
        passed++;
      } catch (error) {
        if (error.message.includes('missing dependencies') ||
            error.message.includes('Missing libraries')) {
          console.log(`  ${name}: skipped (missing system dependencies)`);
          skipped.push(name);
        } else {
          console.error(`  ${name}: FAILED — ${error.message}`);
          failures.push(name);
        }
      }
    }

    if (passed === 0) {
      throw new Error('Extended browser smoke test failed: no browser engines available');
    }

    if (failures.length > 0) {
      throw new Error(`Extended browser smoke test failed on: ${failures.join(', ')}`);
    }

    const summary = skipped.length > 0
      ? `Extended browser smoke test passed (${passed}/${BROWSERS.length} engines; skipped ${skipped.join(', ')})`
      : `Extended browser smoke test passed (all ${BROWSERS.length} engines)`;
    console.log(summary);
  } finally {
    server.close();
  }
}

main().catch(error => {
  console.error(error.stack || error.message);
  process.exit(1);
});
