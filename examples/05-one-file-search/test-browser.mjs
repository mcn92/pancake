#!/usr/bin/env node
// Browser acceptance: serve web/dist + the .pikelet over the range-capable
// server, drive Chromium through a real query and an abstention query, and
// assert range requests actually happened (laziness, not a full download).

import { chromium } from 'playwright';
import { createServer } from './serve.mjs';

const server = createServer();
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const port = server.address().port;

let passed = 0;
let failed = 0;
const check = (name, ok, detail) => {
    if (ok) { passed++; console.log(`  ok: ${name}`); }
    else { failed++; console.log(`  FAIL: ${name}${detail ? ` — ${detail}` : ''}`); }
};

const browser = await chromium.launch();
try {
    const page = await browser.newPage();
    const rangeRequests = [];
    page.on('request', (request) => {
        const range = request.headers().range;
        if (range) rangeRequests.push(range);
    });
    await page.goto(`http://127.0.0.1:${port}/`);
    await page.waitForSelector('#q:not([disabled])', { timeout: 30000 });

    const status = await page.textContent('#status');
    check('page opens the .pikelet and verifies it', /hash verified: true/.test(status), status);
    check('open used HTTP range requests', rangeRequests.length > 0, `ranges=${rangeRequests.length}`);

    const openRanges = rangeRequests.length;
    await page.fill('#q', 'how do cloudflare workers restore snapshots');
    await page.press('#q', 'Enter');
    await page.waitForSelector('.hit', { timeout: 30000 });
    const verdict = await page.textContent('#verdict');
    const hitCount = await page.locator('.hit').count();
    check('real query returns hydrated results', hitCount >= 3, `hits=${hitCount}`);
    check('query verdict is strong with confidence', /strong/.test(verdict) && /confidence/.test(verdict), verdict);
    check('query issued additional range requests', rangeRequests.length > openRanges,
        `open=${openRanges} total=${rangeRequests.length}`);

    await page.fill('#q', 'banana pikelet recipe');
    await page.press('#q', 'Enter');
    await page.waitForFunction(() => /none/.test(document.querySelector('#verdict').textContent), null, { timeout: 30000 });
    const abstained = await page.locator('.hit').count();
    check('out-of-domain query abstains with zero results', abstained === 0, `hits=${abstained}`);
} finally {
    await browser.close();
    server.close();
}

console.log(`\nBrowser one-file search: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
