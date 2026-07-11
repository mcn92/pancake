'use strict';

/**
 * Worker feature tests — auth, rate limiting, CORS, persistence.
 *
 * Starts wrangler dev, sends HTTP requests, checks responses.
 * Run: node test/test_worker_features.js
 */

const { execSync, spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');
const net = require('net');

let PORT = 18787;
let BASE = `http://localhost:${PORT}`;

let passed = 0;
let failed = 0;
const failures = [];

function assert(cond, msg) {
  if (cond) { passed++; }
  else { failed++; failures.push(msg); console.error(`  ✗ FAIL: ${msg}`); }
}

function section(name) {
  console.log(`\n  ${name}`);
  console.log(`  ${'─'.repeat(name.length)}`);
}

async function fetchJSON(path, opts = {}) {
  const url = `${BASE}${path}`;
  const method = opts.method || 'GET';
  const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
  const body = opts.body ? JSON.stringify(opts.body) : undefined;

  const res = await fetch(url, { method, headers, body });
  const text = await res.text();
  let json;
  try { json = JSON.parse(text); } catch { json = null; }
  return { status: res.status, headers: res.headers, json, text };
}

async function fetchRaw(path, opts = {}) {
  const url = `${BASE}${path}`;
  const method = opts.method || 'GET';
  const headers = opts.headers || {};
  const res = await fetch(url, { method, headers });
  return { status: res.status, headers: res.headers };
}

// ---------------------------------------------------------------------------
// Test suites
// ---------------------------------------------------------------------------

async function testAuthNoKey(env) {
  section('Auth — no API_KEY configured (admin routes fail closed)');

  const r = await fetchJSON('/health');
  assert(r.status === 200, '/health returns 200 without auth');

  const r2 = await fetchJSON('/stats');
  // 503 because no index, but neither 401 nor 403 — read routes stay open
  assert(r2.status !== 401 && r2.status !== 403, '/stats stays open when no API_KEY set');

  const r3 = await fetchJSON('/init', {
    method: 'POST',
    body: { dims: 4, maxElements: 10, vectors: [[0, 0, 0, 1]] }
  });
  assert(r3.status === 403, '/init fails closed without API_KEY');
  assert(r3.json?.error?.includes('ALLOW_INSECURE_ADMIN'), '403 explains the ALLOW_INSECURE_ADMIN opt-in');

  const r4 = await fetchJSON('/reset_cache', { method: 'POST' });
  assert(r4.status === 403, '/reset_cache fails closed without API_KEY');
}

async function testInsecureAdminOptIn(env) {
  section('Auth — ALLOW_INSECURE_ADMIN=1 (explicit opt-in)');

  const r = await fetchJSON('/init', {
    method: 'POST',
    body: { dims: 4, maxElements: 10, vectors: [[0, 0, 0, 1]] }
  });
  assert(r.status === 200, '/init succeeds with ALLOW_INSECURE_ADMIN=1 and no API_KEY');

  const r2 = await fetchJSON('/reset_cache', { method: 'POST' });
  assert(r2.status === 200, '/reset_cache succeeds with ALLOW_INSECURE_ADMIN=1 and no API_KEY');
}

async function testAuthWithKey(env) {
  section('Auth — API_KEY configured');

  // No token → 401
  const r1 = await fetchJSON('/stats');
  assert(r1.status === 401, '/stats without token returns 401');
  assert(r1.json?.error === 'Unauthorized', 'error message says Unauthorized');

  // Wrong token → 401
  const r2 = await fetchJSON('/stats', { headers: { Authorization: 'Bearer wrong-key' } });
  assert(r2.status === 401, '/stats with wrong token returns 401');

  // Correct token → not 401 (503 because no index, but auth passed)
  const r3 = await fetchJSON('/stats', { headers: { Authorization: `Bearer ${env.API_KEY}` } });
  assert(r3.status !== 401, '/stats with correct token passes auth');

  // /health skips auth
  const r4 = await fetchJSON('/health');
  assert(r4.status === 200, '/health returns 200 without token even when API_KEY set');
}

async function testRateLimiting(env) {
  section('Rate limiting');

  const headers = env.API_KEY ? { Authorization: `Bearer ${env.API_KEY}` } : {};
  const limit = parseInt(env.RATE_LIMIT_RPM);

  // Send limit+1 requests rapidly (use /stats, not /health which is exempt)
  let hitLimit = false;
  for (let i = 0; i <= limit; i++) {
    const r = await fetchJSON('/stats', { headers });
    if (r.status === 429) {
      hitLimit = true;
      assert(i >= limit, `rate limit hit at request ${i + 1} (limit=${limit})`);
      assert(r.json?.error?.includes('Rate limit'), '429 response has rate limit message');
      break;
    }
  }
  assert(hitLimit, `rate limit was enforced after ${limit} requests`);
}

async function testCorsDefaultOmitted() {
  section('CORS — no ALLOWED_ORIGIN (header omitted)');

  const r = await fetchRaw('/health', { method: 'OPTIONS' });
  assert(r.status === 204, 'OPTIONS returns 204');
  assert(r.headers.get('Access-Control-Allow-Origin') === null,
    'Access-Control-Allow-Origin is omitted when ALLOWED_ORIGIN is unset');
  assert(r.headers.get('Access-Control-Allow-Headers')?.includes('Authorization'),
    'CORS allows Authorization header');
}

async function testCorsRestricted(env) {
  section('CORS — restricted origin');

  const r = await fetchRaw('/health', { method: 'OPTIONS' });
  assert(r.status === 204, 'OPTIONS returns 204');
  assert(r.headers.get('Access-Control-Allow-Origin') === env.ALLOWED_ORIGIN,
    `CORS origin is ${env.ALLOWED_ORIGIN}`);
}

async function testPersistence(env) {
  section('Persistence — R2 save and restore');

  const auth = env.API_KEY ? { Authorization: `Bearer ${env.API_KEY}` } : {};

  // Init an index with a vector
  const dims = 4;
  const vec = [0.1, 0.2, 0.3, 0.4];
  const r1 = await fetchJSON('/init', {
    method: 'POST',
    headers: auth,
    body: { dims, maxElements: 100, vectors: [vec] }
  });
  assert(r1.status === 200, '/init succeeds');
  assert(r1.json?.inserted === 1, 'inserted 1 vector');

  const readiness = await fetchJSON('/readiness', { headers: auth });
  assert(readiness.status === 200, '/readiness inspects the local snapshot');
  assert(readiness.json?.snapshot?.format === 'pancake', '/readiness reports the Pancake snapshot format');
  assert(readiness.json?.snapshot?.dim === dims, '/readiness reports the snapshot dimension');

  const stats = await fetchJSON('/stats', { headers: auth });
  assert(stats.status === 200, '/stats succeeds after init');
  assert(typeof stats.json?.memory_usage?.wasmHeapBytes === 'number',
    '/stats exposes the isolated WASM heap size');

  // Verify index works
  const r2 = await fetchJSON('/search', {
    method: 'POST',
    headers: auth,
    body: { query: vec, k: 1 }
  });
  assert(r2.status === 200, '/search returns 200');
  assert(r2.json?.neighbors?.length === 1, 'search returns 1 neighbor');
}

async function test1536CapabilityContract(env) {
  section('1536D capability contract');

  // The unified handle-based backend supports delete, compact, and ghosts at
  // every dimension, including 1536D. This contract guards against a per-
  // dimension capability gate being reintroduced.
  const auth = env.API_KEY ? { Authorization: `Bearer ${env.API_KEY}` } : {};
  const dims = 1536;
  const vec = new Array(dims).fill(0);
  vec[0] = 1;

  const init = await fetchJSON('/init', {
    method: 'POST',
    headers: auth,
    body: { dims, maxElements: 10, vectors: [vec] }
  });
  assert(init.status === 200, '1536D /init succeeds');

  const health = await fetchJSON('/health', { headers: auth });
  assert(health.status === 200, '/health returns 200 for 1536D index');
  assert(health.json?.dims === 1536, '/health reports dims=1536');

  const del = await fetchJSON('/delete', {
    method: 'POST',
    headers: auth,
    body: { id: 0 }
  });
  assert(del.status === 200, '/delete succeeds for 1536D');

  const compact = await fetchJSON('/compact', {
    method: 'POST',
    headers: auth,
    body: {}
  });
  assert(compact.status === 200, '/compact succeeds for 1536D');

  const stats = await fetchJSON('/stats', { headers: auth });
  assert(stats.status === 200, '/stats returns 200 for 1536D index');
  assert(typeof stats.json?.ghost_count === 'number', '/stats reports numeric ghost_count for 1536D');
}

async function testAddBatchValidationIsAtomic(env) {
  section('/add_batch validates before mutating');

  const auth = env.API_KEY ? { Authorization: `Bearer ${env.API_KEY}` } : {};
  const dims = 4;
  const vec = [0.1, 0.2, 0.3, 0.4];

  const init = await fetchJSON('/init', {
    method: 'POST',
    headers: auth,
    body: { dims, maxElements: 10, vectors: [vec] }
  });
  assert(init.status === 200, '/init succeeds for add_batch atomicity test');

  const badBatch = await fetchJSON('/add_batch', {
    method: 'POST',
    headers: auth,
    body: {
      vectors: [
        [0.4, 0.3, 0.2, 0.1],
        [1, 2, 3]
      ]
    }
  });
  assert(badBatch.status === 400, '/add_batch rejects malformed batch');
  assert(badBatch.json?.code === 'DIMENSION_MISMATCH',
    '/add_batch exposes the stable Pancake error code');

  const stats = await fetchJSON('/stats', { headers: auth });
  assert(stats.status === 200, '/stats succeeds after rejected /add_batch');
  assert(stats.json?.count === 1, 'rejected /add_batch does not partially insert vectors');
}

async function testRestoreAddCompactPreservesRestoredVectors(env) {
  section('Restore -> add -> compact keeps restored vectors');

  const auth = env.API_KEY ? { Authorization: `Bearer ${env.API_KEY}` } : {};
  const dims = 4;
  const v1 = [1, 0, 0, 0];
  const v2 = [0, 1, 0, 0];
  const v3 = [0, 0, 1, 0];

  const init = await fetchJSON('/init', {
    method: 'POST',
    headers: auth,
    body: { dims, maxElements: 10, vectors: [v1, v2] }
  });
  assert(init.status === 200, '/init succeeds for restore/compact test');

  const cleared = await fetchJSON('/reset_cache', {
    method: 'POST',
    headers: auth
  });
  assert(cleared.status === 200, '/reset_cache succeeds');

  const restoredSearches = await Promise.all(Array.from({ length: 5 }, () => fetchJSON('/search', {
    method: 'POST',
    headers: auth,
    body: { query: v1, k: 1, ef: 20 }
  })));
  assert(restoredSearches.every(result => result.status === 200),
    'concurrent /search requests restore the snapshot successfully after reset');

  const healthAfterRestore = await fetchJSON('/health');
  assert(healthAfterRestore.json?.restore?.restoreCount === 1,
    'concurrent cold-start requests share one snapshot restore');

  const add = await fetchJSON('/add', {
    method: 'POST',
    headers: auth,
    body: { vector: v3 }
  });
  assert(add.status === 200, '/add succeeds after restore');

  const compact = await fetchJSON('/compact', {
    method: 'POST',
    headers: auth,
    body: {}
  });
  assert(compact.status === 200, '/compact succeeds after restore and add');

  const stats = await fetchJSON('/stats', { headers: auth });
  assert(stats.status === 200, '/stats succeeds after compact');
  assert(stats.json?.count === 3, 'compact keeps restored vectors after a later add');
}

async function testReadOnlyMode(env) {
  section('Read-only mode');

  const auth = env.API_KEY ? { Authorization: `Bearer ${env.API_KEY}` } : {};

  const health = await fetchJSON('/health');
  assert(health.status === 200, '/health stays public in read-only mode');
  assert(health.json?.read_only === true, '/health reports read_only=true');

  const readiness = await fetchJSON('/readiness', { headers: auth });
  assert(readiness.status === 200, '/readiness succeeds in read-only mode');
  assert(readiness.json?.read_only === true, '/readiness reports read_only=true');

  const init = await fetchJSON('/init', {
    method: 'POST',
    headers: auth,
    body: { dims: 4, maxElements: 10, vectors: [[0, 0, 0, 1]] }
  });
  assert(init.status === 403, '/init is blocked in read-only mode');

  const reset = await fetchJSON('/reset_cache', {
    method: 'POST',
    headers: auth
  });
  assert(reset.status === 403, '/reset_cache is blocked in read-only mode');
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

const WORKER_DIR = path.resolve(__dirname, '..', 'examples', 'worker');
const DEV_VARS_PATH = path.join(WORKER_DIR, '.dev.vars');

function writeDevVars(envVars = {}) {
  const entries = Object.entries(envVars);
  if (entries.length === 0) {
    if (fs.existsSync(DEV_VARS_PATH)) fs.rmSync(DEV_VARS_PATH, { force: true });
    return;
  }

  const lines = entries.map(([k, v]) => `${k}=${v}`);
  fs.writeFileSync(DEV_VARS_PATH, `${lines.join('\n')}\n`, 'utf8');
}

function cleanupDevVars() {
  if (fs.existsSync(DEV_VARS_PATH)) {
    fs.rmSync(DEV_VARS_PATH, { force: true });
  }
}

function getFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      server.close(err => err ? reject(err) : resolve(port));
    });
    server.on('error', reject);
  });
}

function startWorker(envVars = {}) {
  writeDevVars(envVars);
  const args = ['wrangler', 'dev', '--port', String(PORT), '--log-level', 'error'];
  const proc = spawn('npx', args, {
    cwd: WORKER_DIR,
    stdio: ['ignore', 'pipe', 'pipe'],
    shell: true,
  });

  return new Promise((resolve, reject) => {
    let stderr = '';
    let settled = false;
    proc.stderr.on('data', d => { stderr += d.toString(); });
    proc.stdout.on('data', d => { stderr += d.toString(); });

    let check;
    const timeout = setTimeout(() => {
      if (settled) return;
      settled = true;
      clearInterval(check);
      try { proc.kill('SIGTERM'); } catch {}
      reject(new Error(`Worker failed to start within 20s\n${stderr}`));
    }, 20000);

    check = setInterval(async () => {
      try {
        const res = await fetch(`${BASE}/health`);
        if (res.ok) {
          if (settled) return;
          settled = true;
          clearInterval(check);
          clearTimeout(timeout);
          resolve(proc);
        }
      } catch {}
    }, 500);
  });
}

function stopWorker(proc) {
  return new Promise(resolve => {
    proc.on('exit', () => resolve());
    proc.kill('SIGTERM');
    setTimeout(() => {
      try { proc.kill('SIGKILL'); } catch {}
      resolve();
    }, 3000);
  });
}

function waitMs(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function runSuite(name, envVars, tests) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`  Suite: ${name}`);
  console.log(`${'='.repeat(60)}`);

  // Use a fresh port per suite to avoid state leaks
  PORT = await getFreePort();
  BASE = `http://localhost:${PORT}`;

  let proc;
  try {
    proc = await startWorker(envVars);
  } catch (e) {
    console.error(`  ✗ Failed to start worker: ${e.message}`);
    failed++;
    failures.push(`${name}: worker start failed`);
    return;
  }

  try {
    for (const test of tests) {
      await test(envVars);
    }
  } finally {
    await stopWorker(proc);
    cleanupDevVars();
    await waitMs(1000); // let port release
  }
}

async function main() {
  console.log('Worker Feature Tests');
  console.log('====================');

  // Clean local R2 state from prior runs
  const r2Dir = path.join(process.cwd(), '.wrangler', 'state', 'r2');
  if (fs.existsSync(r2Dir)) {
    fs.rmSync(r2Dir, { recursive: true, force: true });
  }
  cleanupDevVars();

  const TEST_KEY = 'test-secret-key-12345';
  const TEST_ORIGIN = 'https://example.com';

  // Suite 1: No auth configured — admin routes fail closed, CORS header omitted
  await runSuite('Fail-closed defaults', {}, [
    testAuthNoKey,
    testCorsDefaultOmitted,
  ]);

  // Suite 2: Auth enabled
  await runSuite('Auth enabled', { API_KEY: TEST_KEY }, [
    testAuthWithKey,
  ]);

  // Suite 3: Explicit insecure opt-in restores open admin access
  await runSuite('Insecure admin opt-in', { ALLOW_INSECURE_ADMIN: '1' }, [
    testInsecureAdminOptIn,
  ]);

  // Suite 4: Rate limiting (set low for fast testing)
  await runSuite('Rate limiting', { RATE_LIMIT_RPM: '5' }, [
    testRateLimiting,
  ]);

  // Suite 5: CORS restricted
  await runSuite('CORS restricted', { ALLOWED_ORIGIN: TEST_ORIGIN }, [
    testCorsRestricted,
  ]);

  // Suite 6: Persistence (admin routes need the local opt-in without API_KEY)
  await runSuite('Persistence', { ALLOW_INSECURE_ADMIN: '1' }, [
    testPersistence,
  ]);

  await runSuite('1536D capability contract', { ALLOW_INSECURE_ADMIN: '1' }, [
    test1536CapabilityContract,
  ]);

  await runSuite('/add_batch atomic validation', { ALLOW_INSECURE_ADMIN: '1' }, [
    testAddBatchValidationIsAtomic,
  ]);

  await runSuite('Restore/add/compact regression', { ALLOW_INSECURE_ADMIN: '1' }, [
    testRestoreAddCompactPreservesRestoredVectors,
  ]);

  await runSuite('Read-only mode', { API_KEY: TEST_KEY, READ_ONLY: '1' }, [
    testReadOnlyMode,
  ]);

  // Summary
  console.log(`\n${'─'.repeat(40)}`);
  console.log(`  Passed: ${passed}`);
  console.log(`  Failed: ${failed}`);
  if (failures.length > 0) {
    console.log('\nFailures:');
    for (const f of failures) console.log(`  • ${f}`);
  }
  process.exit(failed > 0 ? 1 : 0);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
