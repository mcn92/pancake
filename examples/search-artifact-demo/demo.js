#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const Pancake = require('../../pancake.js');

const DEFAULT_ARTIFACT = path.join(
    __dirname,
    '..',
    '..',
    'benchmark_results',
    'layout',
    'pancake-sift1m-u8-metis-split.pancake-range'
);
const DEFAULT_QUERY_FILE = path.join(__dirname, '..', '..', 'sift', 'sift_query.fvecs');

function getArg(name, fallback) {
    const index = process.argv.indexOf(`--${name}`);
    return index >= 0 && index + 1 < process.argv.length ? process.argv[index + 1] : fallback;
}

function hasArg(name) {
    return process.argv.includes(`--${name}`);
}

function parsePositiveInt(name, fallback) {
    const raw = getArg(name, String(fallback));
    const value = Number.parseInt(raw, 10);
    if (!Number.isInteger(value) || value <= 0) {
        throw new Error(`--${name} must be a positive integer`);
    }
    return value;
}

function readFvecs(filePath, limit) {
    const buffer = fs.readFileSync(filePath);
    const vectors = [];
    let offset = 0;
    while (offset + 4 <= buffer.byteLength && vectors.length < limit) {
        const dim = buffer.readInt32LE(offset);
        offset += 4;
        const bytes = dim * 4;
        if (dim <= 0 || offset + bytes > buffer.byteLength) {
            throw new Error(`Invalid fvecs record in ${filePath}`);
        }
        const vector = new Float32Array(dim);
        for (let d = 0; d < dim; d++) {
            vector[d] = buffer.readFloatLE(offset);
            offset += 4;
        }
        vectors.push(vector);
    }
    return vectors;
}

function summarize(values) {
    if (values.length === 0) return { min: 0, mean: 0, p50: 0, p95: 0, max: 0 };
    const sorted = [...values].sort((a, b) => a - b);
    const pick = (q) => sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))];
    return {
        min: sorted[0],
        mean: values.reduce((sum, value) => sum + value, 0) / values.length,
        p50: pick(0.5),
        p95: pick(0.95),
        max: sorted[sorted.length - 1],
    };
}

async function main() {
    if (hasArg('help')) {
        console.log('Usage: node examples/search-artifact-demo/demo.js [--artifact file] [--query-file sift_query.fvecs] [--queries 10] [--k 10] [--ef-search 10] [--compact]');
        process.exit(0);
    }

    const artifactPath = getArg('artifact', DEFAULT_ARTIFACT);
    const queryFile = getArg('query-file', DEFAULT_QUERY_FILE);
    const queryCount = parsePositiveInt('queries', 10);
    const k = parsePositiveInt('k', 10);
    const efSearch = parsePositiveInt('ef-search', 10);

    if (!fs.existsSync(artifactPath)) {
        throw new Error([
            `Artifact not found: ${artifactPath}`,
            '',
            'Generate or copy the v2 split artifact before running this demo.',
            'Expected default:',
            `  ${path.relative(process.cwd(), DEFAULT_ARTIFACT)}`,
            '',
            'Current SIFT1M artifact inputs used during development:',
            '  snapshot: /tmp/pancake-sift1m-u8.pnck',
            '  partition: benchmark_results/layout/pancake-sift1m-base.metis.part.5036',
        ].join('\n'));
    }
    if (!fs.existsSync(queryFile)) {
        throw new Error(`Query file not found: ${queryFile}`);
    }

    const queries = readFvecs(queryFile, queryCount);
    if (queries.length === 0) throw new Error(`No queries found in ${queryFile}`);

    const artifact = await Pancake.openRangeArtifactFile(artifactPath);
    try {
        const perQueryRequests = [];
        const perQueryBytes = [];
        const perQueryRounds = [];
        const sampleResults = [];

        for (let i = 0; i < queries.length; i++) {
            const before = artifact.stats();
            const result = await artifact.search(queries[i], k, { efSearch });
            const after = artifact.stats();
            perQueryRequests.push(after.rangeRequests - before.rangeRequests);
            perQueryBytes.push(after.rangeBytes - before.rangeBytes);
            perQueryRounds.push(result.rounds.length);
            if (i < 3) {
                sampleResults.push({
                    query: i,
                    topK: result.results.map((row) => ({
                        id: row.id,
                        distance: Number(row.distance.toFixed(4)),
                    })),
                });
            }
        }

        const stats = artifact.stats();
        const output = {
            artifact: path.resolve(artifactPath),
            queries: queries.length,
            k,
            efSearch,
            graph: {
                count: artifact.count,
                dim: artifact.dim,
                maxLevel: artifact.maxLevel,
                recordBytes: artifact.recordBytes,
            },
            residentRouter: {
                records: artifact.routerResident.records,
                mib: artifact.routerResident.bytes / 1048576,
            },
            lazyBaseReads: {
                requests: summarize(perQueryRequests),
                bytes: summarize(perQueryBytes),
                meanMiB: summarize(perQueryBytes).mean / 1048576,
                rounds: summarize(perQueryRounds),
                cumulativeRequests: stats.rangeRequests,
                cumulativeFetchedMiB: stats.rangeBytes / 1048576,
                cachedNodes: stats.cachedNodes,
            },
            sampleResults,
        };
        if (hasArg('compact')) {
            console.log([
                `Search Artifact demo: ${output.graph.count.toLocaleString()} vectors, ${output.graph.dim}D`,
                `resident router: ${output.residentRouter.records.toLocaleString()} records (${output.residentRouter.mib.toFixed(1)} MiB)`,
                `lazy reads: ${output.lazyBaseReads.cumulativeRequests.toLocaleString()} requests, ${output.lazyBaseReads.cumulativeFetchedMiB.toFixed(2)} MiB fetched across ${output.queries} queries`,
                `per query: ${output.lazyBaseReads.requests.mean.toFixed(1)} requests, ${output.lazyBaseReads.meanMiB.toFixed(3)} MiB, ${output.lazyBaseReads.rounds.mean.toFixed(1)} rounds mean`,
                `sample top-3 ids: ${output.sampleResults[0].topK.slice(0, 3).map((row) => row.id).join(', ')}`,
            ].join('\n'));
        } else {
            console.log(JSON.stringify(output, null, 2));
        }
    } finally {
        await artifact.close();
    }
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : error);
    process.exit(1);
});
