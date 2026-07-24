#!/usr/bin/env node

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { Miniflare } from 'miniflare';

function argument(name, fallback = null) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function assertScoredSearch(body, label) {
  assert(['strong', 'weak'].includes(body.match_quality), `${label} should report a scored match quality`);
  assert(typeof body.confidence === 'number', `${label} should report confidence`);
}

const bundleDir = path.resolve(argument('bundle-dir', '.tmp-test-work/student-worker-distilled'));
const goldenFixtures = JSON.parse(
  readFileSync(new URL('./fixtures/abstention-golden.json', import.meta.url), 'utf8')
);
const miniflare = new Miniflare({
  compatibilityDate: '2025-04-09',
  compatibilityFlags: ['nodejs_compat'],
  modules: true,
  scriptPath: path.join(bundleDir, 'worker.js'),
  modulesRules: [
    { type: 'CompiledWasm', include: ['**/*.wasm'], fallthrough: true },
    { type: 'Data', include: ['**/*.bin'], fallthrough: true },
  ],
  bindings: { READ_ONLY: '1' },
});

try {
  const page = await miniflare.dispatchFetch('http://demo.test/');
  const html = await page.text();
  assert(page.status === 200, 'GET / should succeed');
  assert(html.includes('Zero runtime dependencies'), 'page should expose the zero-dependency claim');
  assert(html.includes('Distilled docs search'), 'page should identify the distilled demo');

  const cold = await miniflare.dispatchFetch(
    'http://demo.test/search?q=How%20do%20Workers%20restore%20snapshots%20from%20R2%3F&k=3&ef=120'
  );
  const coldBody = await cold.json();
  assert(cold.status === 200, `cold search should succeed: ${JSON.stringify(coldBody)}`);
  assert(coldBody.cache_state === 'cold-restored', 'first search should report cold restoration');
  assert(coldBody.results.length === 3, 'cold search should return the requested top three');
  assertScoredSearch(coldBody, 'cold search');
  assert(coldBody.embedding_ms > 0, 'response should report local embedding time');
  assert(coldBody.search_ms > 0, 'response should report Pancake search time');
  assert(coldBody.encoder?.runtimeDependencies === 0, 'encoder should report zero runtime dependencies');
  assert(coldBody.encoder?.outboundRequests === 0, 'encoder should report zero outbound requests');

  const warm = await miniflare.dispatchFetch(
    'http://demo.test/search?q=How%20does%20filtered%20search%20work%3F&k=3&source=README.md'
  );
  const warmBody = await warm.json();
  assert(warm.status === 200, `warm search should succeed: ${JSON.stringify(warmBody)}`);
  assert(warmBody.cache_state === 'warm-cache', 'second search should report warm cache');
  assertScoredSearch(warmBody, 'warm search');
  assert(warmBody.filter_label === 'README', 'source filter should be applied inside Pancake');
  assert(warmBody.results.every((row) => row.source_path === 'README.md'), 'filtered results should stay in README');

  const noise = await miniflare.dispatchFetch('http://demo.test/search?q=%F0%9F%98%80%F0%9F%98%80%F0%9F%98%80&k=3');
  const noiseBody = await noise.json();
  assert(noise.status === 200, `noise search should abstain cleanly: ${JSON.stringify(noiseBody)}`);
  assert(noiseBody.match_quality === 'none', 'noise query should report no reliable match');
  assert(noiseBody.result_count === 0, 'noise query should suppress results');
  assert(noiseBody.results.length === 0, 'noise query should return an empty results array');
  assert(typeof noiseBody.confidence === 'number', 'noise query should still expose confidence');

  for (const fixture of goldenFixtures) {
    const response = await miniflare.dispatchFetch(
      `http://demo.test/search?q=${encodeURIComponent(fixture.text)}&k=3`
    );
    const body = await response.json();
    assert(response.status === 200, `golden query should succeed: ${fixture.text}`);
    assert(
      body.match_quality === fixture.expected,
      `golden query ${JSON.stringify(fixture.text)} expected ${fixture.expected}, got ${body.match_quality}`
    );
    if (fixture.expected === 'none') {
      assert(body.result_count === 0, `golden none query should suppress results: ${fixture.text}`);
    } else {
      assert(body.result_count > 0, `golden scored query should return results: ${fixture.text}`);
    }
  }

  const health = await miniflare.dispatchFetch('http://demo.test/health');
  const healthBody = await health.json();
  assert(health.status === 200, `health should succeed: ${JSON.stringify(healthBody)}`);
  assert(healthBody.abstention?.present === true, 'health should report the abstention asset as present');

  const readiness = await miniflare.dispatchFetch('http://demo.test/readiness');
  assert(readiness.status === 403, 'admin readiness should reject unauthenticated public requests');

  console.log('Distilled Worker demo checks passed.');
  console.log(`Cold restore ${coldBody.restore_ms.toFixed(2)} ms; embed ${coldBody.embedding_ms.toFixed(2)} ms; search ${coldBody.search_ms.toFixed(2)} ms.`);
} finally {
  await miniflare.dispose();
}
