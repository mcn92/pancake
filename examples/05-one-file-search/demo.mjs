#!/usr/bin/env node
// Query the composed reader from the command line:
//   node examples/05-one-file-search/demo.mjs "how do workers restore snapshots"
// With no arguments, runs the manifest's own sample queries.

import { openDocsSearch, docsAssetPaths } from './search-reader.mjs';

const search = await openDocsSearch(docsAssetPaths());
const info = search.info();
console.log(`one-file-search spike: ${info.chunks} chunks, ${info.dim}D, `
    + `resident ${(info.residentBytes / 1024).toFixed(1)} KiB (hash verified: ${info.residentVerified})`);

const queries = process.argv.slice(2).length > 0
    ? [process.argv.slice(2).join(' ')]
    : info.sampleQueries;

for (const q of queries) {
    const out = await search.query(q, { k: 3 });
    console.log(`\n> ${q}`);
    console.log(`  match: ${out.matchQuality}${out.confidence !== undefined ? ` (confidence ${out.confidence.toFixed(3)})` : ''}`);
    for (const r of out.results) {
        console.log(`  ${r.distance.toFixed(4)}  ${r.title} — ${r.sourcePath}${r.anchor ? `#${r.anchor}` : ''}`);
        console.log(`          ${(r.preview || r.text).slice(0, 100).replace(/\s+/g, ' ')}...`);
    }
    if (out.results.length === 0) console.log('  (abstained — no results returned)');
}

await search.close();
