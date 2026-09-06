#!/usr/bin/env node
// POC cost model: turns the harness's measured bytes/query into a monthly
// hosting estimate and compares it against current managed-search spend.
//
//   node cost.mjs --results results.json --queries-per-month 1000000 \
//       [--current-monthly 500] [--integration-cost 8000]
//
// Pricing defaults are the plan's examples — override with your CDN's rates:
//   --egress-per-gb 0.085       $/GB egress
//   --storage-per-gb 0.023      $/GB-month storage
//   --per-million-requests 0    $/1M requests (CDNs that bill per request)
//   --build-monthly 0           $/month for artifact rebuilds (CI compute, amortized)
//   --ops-monthly 0             $/month ops overhead you attribute to this
//
// Without --results, pass the measurements directly:
//   --bytes-per-query 51200 --requests-per-query 4 --artifact-bytes 25000000
//
// The bytes/query figure used is the mean over the first (cold-reader) pass —
// the conservative choice, since a CDN edge cache only makes real traffic
// cheaper than this.

import fs from 'node:fs';

function parseArgs(argv) {
    const args = {
        egressPerGb: 0.085,
        storagePerGb: 0.023,
        perMillionRequests: 0,
        buildMonthly: 0,
        opsMonthly: 0,
    };
    const num = (v, flag) => {
        const n = Number(v);
        if (!Number.isFinite(n)) throw new Error(`${flag} needs a number, got ${v}`);
        return n;
    };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--results') args.results = argv[++i];
        else if (a === '--queries-per-month') args.queriesPerMonth = num(argv[++i], a);
        else if (a === '--bytes-per-query') args.bytesPerQuery = num(argv[++i], a);
        else if (a === '--requests-per-query') args.requestsPerQuery = num(argv[++i], a);
        else if (a === '--artifact-bytes') args.artifactBytes = num(argv[++i], a);
        else if (a === '--egress-per-gb') args.egressPerGb = num(argv[++i], a);
        else if (a === '--storage-per-gb') args.storagePerGb = num(argv[++i], a);
        else if (a === '--per-million-requests') args.perMillionRequests = num(argv[++i], a);
        else if (a === '--build-monthly') args.buildMonthly = num(argv[++i], a);
        else if (a === '--ops-monthly') args.opsMonthly = num(argv[++i], a);
        else if (a === '--current-monthly') args.currentMonthly = num(argv[++i], a);
        else if (a === '--integration-cost') args.integrationCost = num(argv[++i], a);
        else throw new Error(`unknown flag ${a}`);
    }
    if (args.queriesPerMonth === undefined) throw new Error('--queries-per-month is required');
    return args;
}

const args = parseArgs(process.argv.slice(2));

if (args.results) {
    const results = JSON.parse(fs.readFileSync(args.results, 'utf8'));
    const cold = results.passes[0];
    args.bytesPerQuery ??= cold.summary.bytesPerQuery.mean;
    args.requestsPerQuery ??= cold.summary.rangeReadsPerQuery.median;
    args.artifactBytes ??= results.artifact.fileBytes;
    console.log(`from ${args.results}: ${cold.label}, mean ${(args.bytesPerQuery / 1024).toFixed(1)} KiB/query, `
        + `${args.requestsPerQuery} reads/query, artifact ${(args.artifactBytes / 1048576).toFixed(1)} MiB`);
}
if (args.bytesPerQuery === undefined) throw new Error('need --results or --bytes-per-query');

const GB = 1024 ** 3;
const egressGb = (args.queriesPerMonth * args.bytesPerQuery) / GB;
const egressCost = egressGb * args.egressPerGb;
const storageCost = ((args.artifactBytes ?? 0) / GB) * args.storagePerGb;
const requestCost = args.requestsPerQuery !== undefined
    ? (args.queriesPerMonth * args.requestsPerQuery / 1e6) * args.perMillionRequests
    : 0;
const total = egressCost + storageCost + requestCost + args.buildMonthly + args.opsMonthly;

const usd = (n) => `$${n.toFixed(2)}`;
const rate = (n) => `$${n}`;
console.log(`\nmonthly estimate at ${args.queriesPerMonth.toLocaleString()} queries/month:`);
console.log(`  egress   ${egressGb.toFixed(2)} GB x ${rate(args.egressPerGb)}/GB = ${usd(egressCost)}`);
if (args.artifactBytes !== undefined) {
    console.log(`  storage  ${(args.artifactBytes / GB).toFixed(3)} GB x ${rate(args.storagePerGb)}/GB-mo = ${usd(storageCost)}`);
}
if (requestCost > 0) console.log(`  requests ${usd(requestCost)}`);
if (args.buildMonthly > 0) console.log(`  builds   ${usd(args.buildMonthly)}`);
if (args.opsMonthly > 0) console.log(`  ops      ${usd(args.opsMonthly)}`);
console.log(`  total    ${usd(total)}/month`);

if (args.currentMonthly !== undefined) {
    const savings = args.currentMonthly - total;
    console.log(`\nvs current spend ${usd(args.currentMonthly)}/month: `
        + (savings > 0 ? `saves ${usd(savings)}/month (${((savings / args.currentMonthly) * 100).toFixed(0)}%)` : `costs ${usd(-savings)}/month MORE`));
    if (args.integrationCost !== undefined && savings > 0) {
        console.log(`breakeven on ${usd(args.integrationCost)} integration cost: ${(args.integrationCost / savings).toFixed(1)} months`);
    }
}
