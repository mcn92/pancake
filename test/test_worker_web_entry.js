'use strict';

const { spawn } = require('child_process');
const path = require('path');
const net = require('net');

let passed = 0;
let failed = 0;

function assert(cond, msg) {
  if (cond) {
    passed++;
  } else {
    failed++;
    console.error(`  ✗ FAIL: ${msg}`);
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

function startWorker(port) {
  const cwd = path.join(process.cwd(), 'test', 'fixtures', 'worker_web_entry');
  const proc = spawn('npx', ['wrangler', 'dev', '--port', String(port), '--log-level', 'error'], {
    cwd,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  return new Promise((resolve, reject) => {
    let output = '';
    let settled = false;

    const finishError = (err) => {
      if (settled) return;
      settled = true;
      try { proc.kill('SIGTERM'); } catch {}
      reject(new Error(`${err}\n${output}`));
    };

    proc.stdout.on('data', d => { output += d.toString(); });
    proc.stderr.on('data', d => { output += d.toString(); });
    proc.on('exit', code => {
      if (!settled) finishError(`worker exited early with code ${code}`);
    });

    const timeout = setTimeout(() => finishError('worker failed to start within 20s'), 20000);
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`http://127.0.0.1:${port}/health`);
        if (res.ok) {
          if (settled) return;
          settled = true;
          clearTimeout(timeout);
          clearInterval(interval);
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

async function main() {
  console.log('Worker Web Entry Smoke Test');
  console.log('===========================');

  const port = await getFreePort();
  let proc;
  try {
    proc = await startWorker(port);

    const res = await fetch(`http://127.0.0.1:${port}/smoke`);
    const body = await res.json();

    assert(res.status === 200, '/smoke returns 200');
    assert(body.ok === true, 'response marks ok=true');
    assert(body.count === 3, 'index count is 3');
    assert(body.topId === 1, 'nearest neighbor id is 1');
    assert(body.resultCount === 2, 'requested result count is returned');
    assert(typeof body.topDistance === 'number', 'distance is numeric');
  } finally {
    if (proc) await stopWorker(proc);
  }

  console.log(`\nPassed: ${passed}`);
  console.log(`Failed: ${failed}`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch(err => {
  console.error(err.stack || err.message);
  process.exit(1);
});
