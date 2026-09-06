#!/usr/bin/env node
// Real-HTTP companion to measure-row-commitments.mjs: replay the captured
// per-query read runs against local servers with injected per-request
// latency, to turn the byte/run model into wall-clock — the request-count
// overhead is the one cost the static model cannot price.
//
// Protocols: HTTP/1.1 with a browser-like 6-connection keep-alive pool, and
// HTTP/2 (plaintext session) multiplexing up to 64 streams. Each response is
// delayed by the injected latency, so requests cost RTT the way they do
// against a real host; loopback bandwidth is effectively free, so byte costs
// are underweighted — read them from the static model instead.
//
// Scenarios per query (from row-commitment-replay.json, geometry P=16/D=16):
//   rows         rerank row runs only (today's reader)
//   +digests@0   rows plus separate-region digest pages, digest gap 0
//   +digests@4k  rows plus separate-region digest pages, digest gap 4 KiB
//   interleaved  digest blocks stored beside their rows, one shared run set
//
//   node measure-row-commitments-http.mjs [queriesPerCell] [longLatQueries]
//
// Output: table + row-commitment-http-results.json.

import fs from 'node:fs';
import http from 'node:http';
import http2 from 'node:http2';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const replay = JSON.parse(fs.readFileSync(path.join(here, 'row-commitment-replay.json'), 'utf8'));

const QUERIES_SHORT = Number(process.argv[2] || 20); // queries per cell at the short latency
const QUERIES_LONG = Number(process.argv[3] || 8);   // queries per cell at the long latency
const LATENCIES_MS = [20, 80];
const H1_SOCKETS = 6;   // browser per-origin connection limit
const H2_STREAMS = 64;  // concurrent multiplexed streams
const SCENARIOS = ['rows', 'digests0', 'digests4k', 'interleaved'];

const chunk = Buffer.alloc(64 * 1024);

function startServers(latencyMs) {
    const respond = (len, write, end) => {
        setTimeout(() => {
            let left = len;
            while (left > 0) { const n = Math.min(left, chunk.length); write(chunk.subarray(0, n)); left -= n; }
            end();
        }, latencyMs);
    };
    const h1 = http.createServer((req, res) => {
        const len = Number(new URL(req.url, 'http://x').searchParams.get('l')) || 1;
        res.setHeader('content-length', len);
        respond(len, (b) => res.write(b), () => res.end());
    });
    const h2 = http2.createServer();
    h2.on('stream', (stream, headers) => {
        const len = Number(new URL(headers[':path'], 'http://x').searchParams.get('l')) || 1;
        stream.respond({ ':status': 200, 'content-length': len });
        respond(len, (b) => stream.write(b), () => stream.end());
    });
    return Promise.all([
        new Promise((r) => h1.listen(0, '127.0.0.1', () => r(h1))),
        new Promise((r) => h2.listen(0, '127.0.0.1', () => r(h2))),
    ]);
}

function runPool(tasks, limit) {
    let next = 0;
    const workers = Array.from({ length: Math.min(limit, tasks.length) }, async () => {
        while (next < tasks.length) { const i = next++; await tasks[i](); }
    });
    return Promise.all(workers);
}

function h1Fetch(agent, port, [offset, length]) {
    return new Promise((resolve, reject) => {
        const req = http.get({ host: '127.0.0.1', port, path: `/b?o=${offset}&l=${length}`, agent }, (res) => {
            res.on('data', () => {});
            res.on('end', resolve);
        });
        req.on('error', reject);
    });
}

function h2Fetch(session, [offset, length]) {
    return new Promise((resolve, reject) => {
        const stream = session.request({ ':path': `/b?o=${offset}&l=${length}` });
        stream.on('response', () => {});
        stream.on('data', () => {});
        stream.on('end', resolve);
        stream.on('error', reject);
        stream.end();
    });
}

function runsFor(query, scenario) {
    if (scenario === 'rows') return query.rows;
    if (scenario === 'digests0') return [...query.rows, ...query.digests0];
    if (scenario === 'digests4k') return [...query.rows, ...query.digests4k];
    return query.interleaved;
}

async function measure(protocol, port, latencyMs, queries, scenario) {
    const times = [];
    let agent = null;
    let session = null;
    if (protocol === 'h1') agent = new http.Agent({ keepAlive: true, maxSockets: H1_SOCKETS });
    else {
        session = http2.connect(`http://127.0.0.1:${port}`);
        await new Promise((r) => session.once('connect', r));
    }
    const limit = protocol === 'h1' ? H1_SOCKETS * 4 : H2_STREAMS;
    for (const query of queries) {
        const runs = runsFor(query, scenario);
        const t0 = process.hrtime.bigint();
        await runPool(runs.map((run) => () => (protocol === 'h1' ? h1Fetch(agent, port, run) : h2Fetch(session, run))), limit);
        times.push(Number(process.hrtime.bigint() - t0) / 1e6);
    }
    if (agent) agent.destroy();
    if (session) session.close();
    const mean = times.reduce((s, v) => s + v, 0) / times.length;
    const meanRuns = queries.reduce((s, q) => s + runsFor(q, scenario).length, 0) / queries.length;
    return { meanMs: mean, meanRuns };
}

async function main() {
    const out = { latenciesMs: LATENCIES_MS, h1Sockets: H1_SOCKETS, h2Streams: H2_STREAMS, cells: [] };
    for (const [wl, queriesAll] of Object.entries(replay.scenarios)) {
        console.log(`\n=== workload ${wl} (geometry ${replay.geometry.P} rows/page, ${replay.geometry.D} B digests) ===`);
        for (const latencyMs of LATENCIES_MS) {
            const [h1srv, h2srv] = await startServers(latencyMs);
            const nQueries = latencyMs === LATENCIES_MS[0] ? QUERIES_SHORT : QUERIES_LONG;
            const queries = queriesAll.slice(0, nQueries);
            console.log(`latency ${latencyMs} ms, ${queries.length} queries:`);
            console.log('scenario      |  h1 (6 conn) ms/query (runs) |  h2 (64 str) ms/query | h1 overhead vs rows | h2 overhead');
            let base = { h1: 0, h2: 0 };
            for (const scenario of SCENARIOS) {
                const h1 = await measure('h1', h1srv.address().port, latencyMs, queries, scenario);
                const h2 = await measure('h2', h2srv.address().port, latencyMs, queries, scenario);
                if (scenario === 'rows') base = { h1: h1.meanMs, h2: h2.meanMs };
                const cell = {
                    workload: wl, latencyMs, scenario,
                    meanRuns: Number(h1.meanRuns.toFixed(1)),
                    h1Ms: Number(h1.meanMs.toFixed(0)), h2Ms: Number(h2.meanMs.toFixed(0)),
                    h1OverheadPct: Number((100 * (h1.meanMs / base.h1 - 1)).toFixed(1)),
                    h2OverheadPct: Number((100 * (h2.meanMs / base.h2 - 1)).toFixed(1)),
                };
                out.cells.push(cell);
                console.log(`${scenario.padEnd(13)} | ${String(cell.h1Ms).padStart(12)} (${String(cell.meanRuns).padStart(6)}) | ${String(cell.h2Ms).padStart(9)} | `
                    + `${scenario === 'rows' ? '     —' : String(cell.h1OverheadPct).padStart(5) + '%'} | ${scenario === 'rows' ? '  —' : String(cell.h2OverheadPct).padStart(5) + '%'}`);
            }
            h1srv.close();
            h2srv.close();
        }
    }
    fs.writeFileSync(path.join(here, 'row-commitment-http-results.json'), `${JSON.stringify(out, null, 2)}\n`);
    console.log('\nresults written to row-commitment-http-results.json');
}

main().catch((err) => { console.error(err); process.exit(1); });
