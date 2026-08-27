#!/usr/bin/env node
// End-to-end check of create-pancake-search as a user experiences it:
// scaffold a project (stub embeddings, no Cloudflare account), install it,
// start its Worker with `wrangler dev`, and query /search — once per runtime.
// Also exercises `compile` (complete kind-3 .pancake from a fixture corpus,
// opened and queried with the in-repo reader).
//
// Two things this catches that the in-repo smoke does not:
//   1. The generated project's pancake-wasm dependency range drifting from
//      the scaffolder's own (0.3.0 shipped ^0.2.0 and every artifact-runtime
//      project answered SNAPSHOT_INVALID: the 0.2.x reader cannot open the
//      format-v3 artifacts the 0.3.0 builder writes). Asserted directly, and
//      the in-repo pancake-wasm version must satisfy the generated range.
//   2. The generated Worker not actually serving queries against the
//      artifacts this tree builds. The project is installed with the
//      in-repo pancake-wasm tarball (npm pack), so the reader under test is
//      the one about to be published, not whatever the registry has.
//
// Usage: node scripts/test-scaffold-e2e.mjs [--runtime artifact|snapshot]...
//        (default: both). Needs network for the generated project's other
//        dependencies (wrangler) and a free local port; no Cloudflare login.

import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const CPS_DIR = path.join(ROOT, 'create-pancake-search');
const CPS_BIN = path.join(CPS_DIR, 'bin', 'create-pancake-search.mjs');
const requestedRuntimes = process.argv.slice(2).flatMap((arg, i, all) => (arg === '--runtime' ? [all[i + 1]] : []));
const RUNTIMES = requestedRuntimes.length ? requestedRuntimes : ['artifact', 'snapshot'];

let passed = 0;
let failed = 0;
function ok(condition, label, detail = '') {
  if (condition) { passed++; console.log(`  ok: ${label}`); }
  else { failed++; console.log(`  FAIL: ${label}${detail ? ` — ${detail}` : ''}`); }
}
function run(cmd, args, opts = {}) {
  const res = spawnSync(cmd, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], ...opts });
  if (res.status !== 0) {
    throw new Error(`${cmd} ${args.join(' ')} failed (${res.status}):\n${res.stdout}\n${res.stderr}`);
  }
  return res.stdout;
}
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const npx = process.platform === 'win32' ? 'npx.cmd' : 'npx';

// Minimal caret-range check, enough for the ranges this repo writes; anything
// else is reported as a failure so a reviewer looks at it.
function caretSatisfies(range, version) {
  const m = /^\^(\d+)\.(\d+)\.(\d+)$/.exec(range);
  const v = /^(\d+)\.(\d+)\.(\d+)$/.exec(version);
  if (!m || !v) return null;
  const [rMaj, rMin, rPat] = m.slice(1).map(Number);
  const [vMaj, vMin, vPat] = v.slice(1).map(Number);
  if (rMaj > 0) return vMaj === rMaj && (vMin > rMin || (vMin === rMin && vPat >= rPat));
  if (rMin > 0) return vMaj === 0 && vMin === rMin && vPat >= rPat;
  return vMaj === 0 && vMin === 0 && vPat === rPat;
}

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, '127.0.0.1', () => { const { port } = srv.address(); srv.close((e) => (e ? reject(e) : resolve(port))); });
    srv.on('error', reject);
  });
}
async function waitFor(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try { const r = await fetch(url); if (r.ok) return true; } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 1000));
  }
  return false;
}
function killTree(proc) {
  if (!proc || proc.exitCode !== null) return;
  try { process.kill(-proc.pid, 'SIGTERM'); } catch { try { proc.kill('SIGTERM'); } catch { /* gone */ } }
}

const work = fs.mkdtempSync(path.join(os.tmpdir(), 'pancake-scaffold-e2e-'));
const rootPkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));
const cpsPkg = JSON.parse(fs.readFileSync(path.join(CPS_DIR, 'package.json'), 'utf8'));
const cpsRange = cpsPkg.dependencies?.['pancake-wasm'];

console.log(`Scaffold e2e: pancake-wasm ${rootPkg.version} (in-repo), create-pancake-search ${cpsPkg.version}, runtimes ${RUNTIMES.join(', ')}`);
console.log(`work dir: ${work}`);

// In-repo tarballs for BOTH packages, installed into each generated project:
// pancake-wasm so the reader under test is the one about to be published,
// and create-pancake-search because the generated devDependency pins the
// in-repo version, which does not exist on the registry until it is
// published — before a release, resolving it from npm fails with ETARGET.
const packOut = run(npm, ['pack', '--pack-destination', work, '--ignore-scripts'], { cwd: ROOT });
const tarball = path.join(work, packOut.trim().split('\n').pop());
ok(fs.existsSync(tarball), `packed in-repo pancake-wasm: ${path.basename(tarball)}`);
const cpsPackOut = run(npm, ['pack', '--pack-destination', work, '--ignore-scripts'], { cwd: CPS_DIR });
const cpsTarball = path.join(work, cpsPackOut.trim().split('\n').pop());
ok(fs.existsSync(cpsTarball), `packed in-repo create-pancake-search: ${path.basename(cpsTarball)}`);

// The compile command: a complete kind-3 .pancake from a small fixture
// corpus, opened and queried with the in-repo complete reader. A fixture
// rather than ROOT/docs because the inline transformer embeds for real
// (no stub path), and the guard rail: --runtime complete must point at
// compile instead of silently building a snapshot project (the pre-0.6
// behavior).
console.log('\n--- compile ---');
{
  const fixtureDir = path.join(work, 'compile-fixture');
  fs.mkdirSync(fixtureDir, { recursive: true });
  const fixtures = {
    'volcanoes.md': 'Volcanoes form where magma rises through cracks in the crust and erupts onto the surface. Repeated eruptions of lava and ash build the cone over thousands of years, usually along tectonic plate boundaries where one plate subducts beneath another and melts into fresh magma below.',
    'routers.md': 'A network router forwards packets between different networks by reading the destination address in each packet header and consulting its routing table. Routes are configured statically or learned through routing protocols, and home routers also perform network address translation with a built-in firewall.',
    'sourdough.md': 'Sourdough bread rises without commercial yeast by relying on a starter, a live culture of wild yeast and lactic acid bacteria kept alive with regular feedings of flour and water. The long fermentation develops flavor and structure, and the acidity gives the crumb its characteristic tang.',
  };
  for (const [name, text] of Object.entries(fixtures)) fs.writeFileSync(path.join(fixtureDir, name), `# ${name}\n\n${(text + ' ').repeat(3)}`);
  const outFile = path.join(work, 'compiled', 'search.pancake');
  run(process.execPath, [CPS_BIN, 'compile', '--source', fixtureDir, '--out', outFile], { cwd: CPS_DIR });
  ok(fs.existsSync(outFile), 'compile wrote the .pancake artifact');

  const { openPancakeFile } = await import(path.join(ROOT, 'complete', 'index.mjs'));
  const search = await openPancakeFile(outFile);
  try {
    const info = search.info();
    ok(info.corpusIntegrity === 'per-record-sha256' && info.indexRowIntegrity === 'per-row-sha256',
      'compiled artifact is format 2 with per-record and per-row integrity', JSON.stringify({ corpus: info.corpusIntegrity, index: info.indexRowIntegrity }));
    const out = await search.query('how does bread rise without yeast', { k: 2 });
    ok(out.results[0]?.sourcePath?.includes('sourdough') || out.results[0]?.title?.includes('sourdough'),
      'compiled artifact answers a natural-language query with no host encoder', JSON.stringify(out.results[0] || {}).slice(0, 200));
  } finally {
    await search.close();
  }

  const rejected = spawnSync(process.execPath,
    [CPS_BIN, '--name', path.join(work, 'proj-complete'), '--source', fixtureDir, '--runtime', 'complete', '--no-deploy', '--yes'],
    { encoding: 'utf8', cwd: CPS_DIR });
  ok(rejected.status !== 0 && `${rejected.stderr}${rejected.stdout}`.includes('compile --source'),
    'scaffold --runtime complete is rejected with a pointer to compile', `${rejected.status} ${rejected.stderr.slice(0, 200)}`);
}

let worker = null;
try {
  for (const runtime of RUNTIMES) {
    console.log(`\n--- runtime: ${runtime} ---`);
    const projectDir = path.join(work, `proj-${runtime}`);
    run(process.execPath, [CPS_BIN, '--name', projectDir, '--source', path.join(ROOT, 'docs'), '--runtime', runtime, '--no-deploy', '--yes', '--force'], {
      cwd: CPS_DIR,
      env: { ...process.env, PANCAKE_SEARCH_STUB_EMBEDDINGS: '1' },
    });
    const genPkgPath = path.join(projectDir, 'package.json');
    ok(fs.existsSync(genPkgPath), 'scaffold wrote package.json');
    const genPkg = JSON.parse(fs.readFileSync(genPkgPath, 'utf8'));
    const genRange = genPkg.dependencies?.['pancake-wasm'];
    ok(genRange === cpsRange, `generated pancake-wasm range equals the scaffolder's own (${cpsRange})`, `generated ${genRange}`);
    const sat = caretSatisfies(genRange, rootPkg.version);
    ok(sat === true, `in-repo pancake-wasm ${rootPkg.version} satisfies generated range ${genRange}`, sat === null ? 'unrecognized range shape' : 'not satisfied');

    // Install as generated, with both in-repo tarballs on top.
    run(npm, ['install', '--no-audit', '--no-fund', '--no-progress', tarball, cpsTarball], { cwd: projectDir });
    const installed = JSON.parse(fs.readFileSync(path.join(projectDir, 'node_modules', 'pancake-wasm', 'package.json'), 'utf8')).version;
    ok(installed === rootPkg.version, `generated project runs pancake-wasm ${installed}`);

    const localToml = fs.existsSync(path.join(projectDir, 'wrangler.local.toml')) ? 'wrangler.local.toml' : 'wrangler.toml';
    const port = await freePort();
    const args = ['wrangler', 'dev', '--config', localToml, '--port', String(port), '--log-level', 'error'];
    if (localToml === 'wrangler.toml') args.push('--var', 'LOCAL_STUB_AI:1');
    let workerLog = '';
    worker = spawn(npx, args, { cwd: projectDir, detached: true, stdio: ['ignore', 'pipe', 'pipe'] });
    worker.stdout.on('data', (d) => { workerLog += d; });
    worker.stderr.on('data', (d) => { workerLog += d; });

    const base = `http://127.0.0.1:${port}`;
    const up = await waitFor(`${base}/health`, 120_000);
    ok(up, `wrangler dev answers /health on :${port}`, workerLog.slice(-800));
    if (up) {
      const health = await (await fetch(`${base}/health`)).json();
      ok(health.runtime_mode === runtime, `/health reports runtime_mode ${runtime}`, JSON.stringify(health).slice(0, 300));
      const res = await fetch(`${base}/search?q=${encodeURIComponent('pancake search artifact')}&k=3`);
      const body = await res.json().catch(() => ({}));
      ok(res.status === 200 && !body.error, `/search returns 200 without error`, `${res.status} ${JSON.stringify(body).slice(0, 300)}`);
      ok(Array.isArray(body.results) && body.results.length > 0 && typeof body.results[0].title === 'string',
        `/search returns hydrated results (${body.result_count ?? '?'} for k=3)`, JSON.stringify(body).slice(0, 300));
      if (runtime === 'artifact') {
        ok(body.artifact && body.artifact.query_range_requests > 0, 'artifact runtime served the query via range reads', JSON.stringify(body.artifact || {}).slice(0, 200));
      }
      const second = await (await fetch(`${base}/search?q=${encodeURIComponent('worker snapshot restore')}&k=2`)).json().catch(() => ({}));
      ok(Array.isArray(second.results) && second.results.length > 0, 'second query (warm) returns results');
    }
    killTree(worker);
    await new Promise((r) => setTimeout(r, 1500));
    worker = null;
  }
} finally {
  killTree(worker);
}

console.log(`\nScaffold e2e: ${passed} passed, ${failed} failed`);
if (failed === 0) fs.rmSync(work, { recursive: true, force: true });
else console.log(`kept ${work} for inspection`);
process.exit(failed === 0 ? 0 : 1);
