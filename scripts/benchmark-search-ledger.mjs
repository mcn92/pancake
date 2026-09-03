#!/usr/bin/env node
/**
 * The search ledger: artifact performance on the axes a general audience
 * can price, not the ANN-leaderboard axes. Composes benchmark-range-storage
 * (real loopback HTTP 206 measurement) across network-distance presets and
 * renders one Markdown report per invocation:
 *
 *   - time to first answer, from nothing (open + first query, no prefetch)
 *   - what one query costs: requests, KiB, and $ per million queries on
 *     object-store pricing (defaults: Cloudflare R2 and AWS S3, overridable)
 *   - queries per GB of egress; artifact storage cost per month
 *   - warm-session answer time (caches populated)
 *   - infrastructure required (measured column: a static file host)
 *
 * Method is deliberately boring and reproducible: every number comes from
 * benchmark-range-storage --json runs (fresh child process per artifact,
 * byte accounting cross-checked against the server), with network distance
 * modeled as a fixed injected per-response delay — labeled as a model, not
 * a WAN claim. The report embeds the exact commands to reproduce it.
 *
 * Usage:
 *   node scripts/benchmark-search-ledger.mjs \
 *     [--out report.md] [--query-limit 12] [--presets city=5,continent=25,world=90] \
 *     [--r2-... | --s3-... pricing overrides, see --help] \
 *     <artifact.pikelet>:<queries.json> [...]
 */

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(here, '..');
const BENCH = path.join(here, 'benchmark-range-storage.mjs');

const DEFAULT_PRESETS = [['city', 5], ['continent', 25], ['world', 90]];
// Pricing defaults (USD), overridable; sources belong in the rendered
// report so readers can check them against current price pages.
const PRICING = {
  r2: { name: 'Cloudflare R2', egressPerGb: 0, readsPerMillion: 0.36, storagePerGbMonth: 0.015 },
  s3: { name: 'AWS S3', egressPerGb: 0.09, readsPerMillion: 0.40, storagePerGbMonth: 0.023 },
};

function usage(code = 0) {
  const out = code ? console.error : console.log;
  out(`usage: node scripts/benchmark-search-ledger.mjs [options] <artifact.pikelet>:<queries.json> [...]

Options:
  --out <file>            Markdown report path (default search-ledger.md)
  --query-limit <n>       Queries per pass (default 12)
  --presets a=ms,b=ms     Distance presets (default city=5,continent=25,world=90)
  --encoder-module <f>    Passed through for kind-2 artifacts
  --r2-egress-per-gb, --r2-reads-per-million, --r2-storage-per-gb-month
  --s3-egress-per-gb, --s3-reads-per-million, --s3-storage-per-gb-month
  -h, --help`);
  process.exit(code);
}

const args = process.argv.slice(2);
const options = { out: 'search-ledger.md', queryLimit: 12, presets: DEFAULT_PRESETS, encoderModule: null, sets: [] };
for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (a === '-h' || a === '--help') usage(0);
  else if (a === '--out') options.out = args[++i];
  else if (a === '--query-limit') options.queryLimit = Number(args[++i]);
  else if (a === '--encoder-module') options.encoderModule = path.resolve(args[++i]);
  else if (a === '--presets') {
    options.presets = args[++i].split(',').map((p) => {
      const [name, ms] = p.split('=');
      if (!name || !Number.isFinite(Number(ms))) throw new Error(`bad preset ${p}`);
      return [name, Number(ms)];
    });
  } else if (a.startsWith('--r2-') || a.startsWith('--s3-')) {
    const tier = a.slice(2, 4);
    const key = a.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    if (!(key in PRICING[tier])) throw new Error(`unknown pricing option ${a}`);
    PRICING[tier][key] = Number(args[++i]);
  } else if (a.startsWith('-')) throw new Error(`unknown option ${a}`);
  else {
    const sep = a.lastIndexOf(':');
    if (sep < 1) throw new Error(`expected <artifact>:<queries>, got ${a}`);
    options.sets.push({ artifact: path.resolve(a.slice(0, sep)), queries: path.resolve(a.slice(sep + 1)) });
  }
}
if (!options.sets.length) usage(1);
for (const s of options.sets) {
  if (!fs.existsSync(s.artifact)) throw new Error(`artifact not found: ${s.artifact}`);
  if (!fs.existsSync(s.queries)) throw new Error(`queries not found: ${s.queries}`);
}

function runBench(set, delayMs) {
  const jsonPath = path.join(fs.mkdtempSync(path.join(ROOT, '.tmp-ledger-')), 'r.json');
  const cmd = [
    BENCH, '--server-delay-ms', String(delayMs), '--no-prefetch-encoder',
    '--query-limit', String(options.queryLimit), '--queries', set.queries,
    ...(options.encoderModule ? ['--encoder-module', options.encoderModule, '--no-verify-encoder'] : []),
    '--json', jsonPath, set.artifact,
  ];
  const child = spawnSync(process.execPath, cmd, { cwd: ROOT, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  if (child.status !== 0) {
    throw new Error(`bench failed for ${set.artifact} @ ${delayMs}ms:\n${(child.stderr || child.stdout || '').slice(-2000)}`);
  }
  const parsed = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
  fs.rmSync(path.dirname(jsonPath), { recursive: true, force: true });
  return parsed.results[0];
}

const mib = (b) => b / 1048576;
const kib = (b) => b / 1024;
const money = (v) => (v < 0.01 && v > 0 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`);

function perMillionQueries(bytesPerQuery, requestsPerQuery, tier) {
  const egress = (bytesPerQuery * 1e6 / 1e9) * tier.egressPerGb;
  const reads = requestsPerQuery * tier.readsPerMillion;
  return egress + reads;
}

const runs = [];
for (const set of options.sets) {
  const name = path.basename(set.artifact);
  const perPreset = {};
  for (const [presetName, ms] of options.presets) {
    console.log(`measuring ${name} @ ${presetName} (${ms}ms)...`);
    perPreset[presetName] = { ms, result: runBench(set, ms) };
  }
  runs.push({ set, name, perPreset });
}

const lines = [];
const push = (s = '') => lines.push(s);
const presetNames = options.presets.map(([n]) => n);
const mid = presetNames[Math.min(1, presetNames.length - 1)];

push('# The search ledger');
push();
push('What it costs to answer a search query from a static file on ordinary');
push('object storage — no server, no search service, no index API. Every');
push('number below is measured over real HTTP range requests against a');
push('byte-range server; network distance is modeled as a fixed injected');
push('delay per response (a model, not a WAN measurement) at: '
  + options.presets.map(([n, ms]) => `${n} = ${ms}ms`).join(', ') + '.');
push();
for (const run of runs) {
  const any = run.perPreset[presetNames[0]].result;
  const midRun = run.perPreset[mid].result;
  push(`## ${run.name}`);
  push();
  push(`${any.records.toLocaleString()} records, ${mib(any.fileBytes).toFixed(1)} MiB artifact`
    + (any.lexicalTerms ? ', hybrid retrieval (BM25 + vector)' : ''));
  push();
  push('| | ' + presetNames.join(' | ') + ' |');
  push('|---|' + presetNames.map(() => '---:').join('|') + '|');
  const row = (label, fn) => push(`| ${label} | ` + presetNames.map((p) => fn(run.perPreset[p].result)).join(' | ') + ' |');
  row('ready to search (open)', (r) => `${(r.open.ms / 1000).toFixed(2)} s`);
  row('first answer, from nothing', (r) => `${((r.open.ms + r.firstQuery.ms) / 1000).toFixed(2)} s`);
  row('answer, session warm (p50)', (r) => `${r.repeatPass.latencyP50Ms.toFixed(0)} ms`);
  row('answer, session warm (p95)', (r) => `${r.repeatPass.latencyP95Ms.toFixed(0)} ms`);
  push();
  // Steady-state per-query transfer: the cold session minus its first
  // query, which carries one-time costs (the encoder download when
  // prefetch is off). Charging those to every query would overstate cost
  // by an order of magnitude.
  const n = midRun.firstPass.n;
  const steadyBytes = n > 1 ? (midRun.firstPass.bytesTotal - midRun.firstQuery.bytes) / (n - 1) : midRun.firstPass.bytesMean;
  const steadyReqs = n > 1 ? (midRun.firstPass.requestsTotal - midRun.firstQuery.requests) / (n - 1) : midRun.firstPass.requestsMean;
  push('| per query (fresh caches, encoder loaded) | value |');
  push('|---|---:|');
  push(`| HTTP range requests | ${steadyReqs.toFixed(1)} |`);
  push(`| bytes transferred | ${kib(steadyBytes).toFixed(0)} KiB |`);
  push(`| queries per GB of egress | ${Math.round(1e9 / Math.max(1, steadyBytes)).toLocaleString()} |`);
  for (const tier of Object.values(PRICING)) {
    push(`| ${tier.name}: 1M queries | ${money(perMillionQueries(steadyBytes, steadyReqs, tier))} |`);
    push(`| ${tier.name}: store the artifact | ${money(mib(midRun.fileBytes) / 1024 * tier.storagePerGbMonth)}/month |`);
  }
  push(`| servers to operate | 0 (any static host with byte ranges) |`);
  push();
  push(`Open transfers ${mib(midRun.open.bytes).toFixed(2)} MiB of the ${mib(midRun.fileBytes).toFixed(1)} MiB`
    + ` file (${(100 * midRun.open.bytes / midRun.fileBytes).toFixed(1)}%) in ${midRun.open.requests} range`
    + ` requests; warm repeats transfer ${kib(midRun.repeatPass.bytesMean).toFixed(1)} KiB/query.`
    + (midRun.encoderKind === 'inline-transformer-v1'
      ? ' "First answer" includes downloading the self-contained query encoder serially; by default the reader prefetches it in the background during open, so a user who types a few seconds after page load pays little or none of it.'
      : ''));
  push();
}
push('## Method and reproduction');
push();
push('Measured by `scripts/benchmark-range-storage.mjs`: each artifact is served');
push('by a loopback HTTP server implementing HEAD + 206 byte ranges (the server');
push('cross-checks that bytes requested equal bytes served), opened through the');
push('same `httpRangeSource()` a browser uses against a CDN, in a fresh process');
push('per artifact. Encoder prefetch is disabled so "first answer" is the honest');
push('serial worst case. Pricing defaults: R2 zero egress, $0.36/M class-B reads,');
push('$0.015/GB-month; S3 $0.09/GB egress, $0.40/M GETs, $0.023/GB-month — check');
push('current price pages; override with the --r2-*/--s3-* flags.');
push();
push('```');
for (const run of runs) {
  push(`node scripts/benchmark-search-ledger.mjs ${path.relative(ROOT, run.set.artifact)}:${path.relative(ROOT, run.set.queries)}`);
}
push('```');
push();
push(`Generated ${new Date().toISOString()} with query-limit ${options.queryLimit}.`);

fs.writeFileSync(options.out, lines.join('\n') + '\n');
console.log(`\nwrote ${options.out}`);
