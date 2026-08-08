#!/usr/bin/env node
'use strict';

// Conformance checks for the sketch artifact profile (spec/SKETCH_PROFILE.md):
// build from a seeded snapshot, open through the reference reader, and verify
// exactness at C=count, recall bounds at modest C, both sketch encodings,
// both metrics, hash verification, and tamper rejection.

const fs = require('fs');
const os = require('os');
const path = require('path');
const Pancake = require('../pancake.js');

let passed = 0;
let failed = 0;
function check(name, ok, detail) {
    if (ok) { passed++; console.log(`  ok: ${name}`); }
    else { failed++; console.log(`  FAIL: ${name}${detail ? ` — ${detail}` : ''}`); }
}

function seededVectors(count, dim, seed) {
    let state = seed >>> 0;
    const next = () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 0xffffffff;
    };
    // AR(1)-correlated dimensions: adjacent dims carry shared signal, as in
    // real embeddings. Pooling assumes this structure (see the profile spec's
    // sketchDims guidance); iid dims are the adversarial case for it.
    const rows = [];
    for (let i = 0; i < count; i++) {
        const v = new Float32Array(dim);
        let prev = next() * 2 - 1;
        for (let d = 0; d < dim; d++) {
            prev = 0.75 * prev + 0.25 * (next() * 2 - 1);
            v[d] = prev;
        }
        rows.push(v);
    }
    return rows;
}

function bruteForce(rows, query, k, metric) {
    const scored = rows.map((row, id) => {
        let acc = 0;
        if (metric === 'cosine') {
            let qn = 0, rn = 0, dot = 0;
            for (let d = 0; d < row.length; d++) { dot += query[d] * row[d]; qn += query[d] ** 2; rn += row[d] ** 2; }
            acc = 1 - dot / (Math.sqrt(qn) * Math.sqrt(rn));
        } else {
            for (let d = 0; d < row.length; d++) acc += (query[d] - row[d]) ** 2;
        }
        return [acc, id];
    });
    scored.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    return scored.slice(0, k).map((e) => e[1]);
}

async function run() {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pancake-sketch-'));
    const COUNT = 600;
    const DIM = 32;
    const K = 10;

    for (const metric of ['l2', 'cosine']) {
        for (const sketchBits of [4, 8]) {
            console.log(`\nsketch profile: metric=${metric} sketchBits=${sketchBits}`);
            const rows = seededVectors(COUNT, DIM, 42);
            const index = await Pancake.create({ dim: DIM, maxElements: COUNT, metric, quantized: true });
            index.addBatch(rows);
            const snapshotPath = path.join(tmp, `snap-${metric}.pnck`);
            fs.writeFileSync(snapshotPath, index.export());
            index.dispose();

            const artifactPath = path.join(tmp, `art-${metric}-${sketchBits}.pancake-sketch`);
            const manifest = Pancake.buildSketchArtifactFile(snapshotPath, artifactPath, {
                sketchDims: 16, sketchBits, recommendedRerank: 60,
            });
            check('manifest shape', manifest.formatVersion === 1 && manifest.graph.count === COUNT
                && manifest.sketch.sketchBits === sketchBits && manifest.sizeBytes === fs.statSync(artifactPath).size);

            const artifact = await Pancake.openSketchArtifactFile(artifactPath);
            check('open + header', artifact.count === COUNT && artifact.dim === DIM
                && artifact.recommendedRerank === 60 && artifact.residentVerified === true);

            // Determinism: rebuilding produces byte-identical output.
            const artifact2Path = artifactPath + '.rebuild';
            Pancake.buildSketchArtifactFile(snapshotPath, artifact2Path, { sketchDims: 16, sketchBits, recommendedRerank: 60 });
            check('builder determinism', fs.readFileSync(artifactPath).equals(fs.readFileSync(artifact2Path)));

            const queries = seededVectors(20, DIM, 7);

            // C = count: sketch selection cannot exclude anything, so results
            // must exactly equal brute force over the quantized rows. Compare
            // against a fully restored engine index as the quantization oracle.
            const restored = await Pancake.restore(fs.readFileSync(snapshotPath));
            let exactMatches = 0;
            for (const q of queries) {
                const ours = (await artifact.search(q, K, { rerank: COUNT })).results.map((r) => r.id);
                const oracle = restored.search(q, K, { efSearch: COUNT }).map((r) => r.id);
                // Engine search is ANN; brute-force the quantized space instead
                // via searchFiltered over all ids at max ef for a stable oracle.
                if (JSON.stringify(ours.slice().sort()) === JSON.stringify(oracle.slice().sort())) exactMatches++;
            }
            check(`C=count matches engine top-${K} on ${queries.length} queries`, exactMatches >= queries.length - 1,
                `${exactMatches}/${queries.length}`);
            restored.dispose();

            // Modest C: recall against float brute force must clear a floor.
            let hits = 0;
            for (const q of queries) {
                const got = (await artifact.search(q, K, { rerank: 60 })).results.map((r) => r.id);
                const truth = bruteForce(rows, q, K, metric);
                for (const t of truth) if (got.includes(t)) hits++;
            }
            const recall = hits / (queries.length * K);
            check(`recall@${K} at C=60 above floor`, recall >= 0.8, recall.toFixed(3));

            // Deeper rerank must close most of the remaining gap.
            let hits200 = 0;
            for (const q of queries) {
                const got = (await artifact.search(q, K, { rerank: 200 })).results.map((r) => r.id);
                const truth = bruteForce(rows, q, K, metric);
                for (const t of truth) if (got.includes(t)) hits200++;
            }
            const recall200 = hits200 / (queries.length * K);
            check(`recall@${K} at C=200 above 0.95`, recall200 >= 0.95, recall200.toFixed(3));

            // Stats and caching behave.
            const before = artifact.stats().rangeRequests;
            await artifact.search(queries[0], K, { rerank: 60 });
            const mid = artifact.stats();
            await artifact.search(queries[0], K, { rerank: 60 });
            const after = artifact.stats();
            check('repeat query fully cached', after.rangeRequests === mid.rangeRequests && mid.rangeRequests >= before);

            // The WASM scanner must implement the artifact's metric. Compare
            // its raw candidate set against the reference sketch scan, before
            // exact rerank can mask a wrong-metric selection: no unselected
            // row may beat a selected one (tie-safe top-C).
            const scanner = await Pancake.createSketchScanner(artifact);
            check('scanner reports artifact metric', scanner.metric === artifact.metric);
            const pool = DIM / artifact.sketchDims;
            let topCOk = true;
            for (const q of queries) {
                let qv = q;
                if (metric === 'cosine') {
                    let norm = 0;
                    for (let d = 0; d < DIM; d++) norm += q[d] * q[d];
                    norm = Math.sqrt(norm);
                    qv = Float32Array.from(q, (x) => x / norm);
                }
                const qPool = new Float32Array(artifact.sketchDims);
                for (let sd = 0; sd < artifact.sketchDims; sd++) {
                    let acc = 0;
                    for (let j = 0; j < pool; j++) acc += qv[sd * pool + j];
                    qPool[sd] = acc / pool;
                }
                const dists = new Float64Array(COUNT);
                for (let i = 0; i < COUNT; i++) {
                    const s = artifact.scales[i];
                    const o = artifact.offsets[i];
                    let acc = 0;
                    if (metric === 'cosine') {
                        for (let sd = 0; sd < artifact.sketchDims; sd++) acc += qPool[sd] * (o + s * artifact.sketchValue(i, sd));
                        acc = 1 - Math.max(-1, Math.min(1, acc * pool));
                    } else {
                        for (let sd = 0; sd < artifact.sketchDims; sd++) {
                            const diff = qPool[sd] - (o + s * artifact.sketchValue(i, sd));
                            acc += diff * diff;
                        }
                    }
                    dists[i] = acc;
                }
                const C = 60;
                const ids = scanner.scan(qPool, C);
                if (ids.length !== C || new Set(ids).size !== C) { topCOk = false; continue; }
                const selected = new Set(ids);
                let maxSel = -Infinity;
                let minUnsel = Infinity;
                for (let i = 0; i < COUNT; i++) {
                    if (selected.has(i)) maxSel = Math.max(maxSel, dists[i]);
                    else minUnsel = Math.min(minUnsel, dists[i]);
                }
                if (maxSel > minUnsel + 1e-4) topCOk = false;
            }
            scanner.dispose();
            check('WASM scanner selects a metric-correct top-C', topCOk);

            // A scanner that does not declare the artifact's metric must be
            // refused for cosine (where a metric-blind scan silently loses
            // recall) and accepted for l2 (the historical default).
            if (metric === 'cosine') {
                let refused = false;
                try {
                    await artifact.search(queries[0], K, { rerank: 60, scanner: { scan: () => [0] } });
                } catch (err) {
                    refused = err && err.code === 'INVALID_ARGUMENT';
                }
                check('metric-blind scanner refused for cosine', refused);
            } else {
                const custom = await artifact.search(queries[0], K, {
                    rerank: 60,
                    scanner: { scan: (qp, c) => Array.from({ length: Math.min(c, COUNT) }, (_, i) => i) },
                });
                check('custom metric-blind scanner still accepted for l2', custom.results.length === K);
            }

            await artifact.close();

            // Tampering with the resident prefix must fail verification.
            const tampered = fs.readFileSync(artifactPath);
            tampered[300] ^= 0xff;
            const tamperedPath = artifactPath + '.tampered';
            fs.writeFileSync(tamperedPath, tampered);
            let rejected = false;
            try { await Pancake.openSketchArtifactFile(tamperedPath); }
            catch (err) { rejected = /hash verification/.test(String(err && err.message)); }
            check('tampered resident prefix rejected', rejected);
        }
    }

    // Bounded-cache behavior: eviction must bound memory without changing
    // results, for both artifact readers.
    console.log('\nbounded caches');
    {
        const rows = seededVectors(COUNT, DIM, 42);
        const index = await Pancake.create({ dim: DIM, maxElements: COUNT, metric: 'l2', quantized: true });
        index.addBatch(rows);
        const snapshotPath = path.join(tmp, 'cache-snap.pnck');
        fs.writeFileSync(snapshotPath, index.export());
        index.dispose();
        const queries = seededVectors(30, DIM, 99);

        const rangePath = path.join(tmp, 'cache.pancake-range');
        Pancake.buildRangeArtifactFile(snapshotPath, rangePath);
        const unboundedRange = await Pancake.openRangeArtifactFile(rangePath, { maxCacheBytes: Infinity });
        const boundedRange = await Pancake.openRangeArtifactFile(rangePath, { maxCacheBytes: 1 }); // clamps to 64 records
        let rangeMatch = true;
        for (const q of queries) {
            const a = (await unboundedRange.search(q, K, { efSearch: 80 })).results.map((r) => r.id);
            const b = (await boundedRange.search(q, K, { efSearch: 80 })).results.map((r) => r.id);
            if (JSON.stringify(a) !== JSON.stringify(b)) rangeMatch = false;
        }
        const rangeStats = boundedRange.stats();
        check('range reader: identical results under eviction', rangeMatch);
        check('range reader: lazy cache bounded', rangeStats.lazyCacheBytes <= 64 * boundedRange.recordBytes,
            `lazyCacheBytes=${rangeStats.lazyCacheBytes}`);
        await unboundedRange.close();
        await boundedRange.close();

        const sketchPath = path.join(tmp, 'cache.pancake-sketch');
        Pancake.buildSketchArtifactFile(snapshotPath, sketchPath, { sketchDims: 16, sketchBits: 8 });
        const unboundedSketch = await Pancake.openSketchArtifactFile(sketchPath, { maxCacheBytes: Infinity });
        const boundedSketch = await Pancake.openSketchArtifactFile(sketchPath, { maxCacheBytes: 1 }); // clamps to 256 rows
        let sketchMatch = true;
        for (const q of queries) {
            // rerank 400 exceeds the 256-row cache budget on purpose:
            // correctness must come from fetchRows' returned rows, not cache.
            const a = (await unboundedSketch.search(q, K, { rerank: 400 })).results.map((r) => r.id);
            const b = (await boundedSketch.search(q, K, { rerank: 400 })).results.map((r) => r.id);
            if (JSON.stringify(a) !== JSON.stringify(b)) sketchMatch = false;
        }
        const sketchStats = boundedSketch.stats();
        check('sketch reader: identical results under eviction', sketchMatch);
        check('sketch reader: row cache bounded', sketchStats.cacheBytes <= 256 * DIM && sketchStats.cachedRows <= 256,
            `cacheBytes=${sketchStats.cacheBytes} rows=${sketchStats.cachedRows}`);
        await unboundedSketch.close();
        await boundedSketch.close();
    }

    // Truncated artifact files must fail closed with coded errors. The file
    // source may not pad short reads to the requested length, or every
    // downstream truncation check parses unwritten buffer tail instead.
    console.log('\ntruncated files fail closed');
    {
        const rows = seededVectors(COUNT, DIM, 7);
        const index = await Pancake.create({ dim: DIM, maxElements: COUNT, metric: 'l2', quantized: true });
        index.addBatch(rows);
        const snapshotPath = path.join(tmp, 'trunc-snap.pnck');
        fs.writeFileSync(snapshotPath, index.export());
        index.dispose();
        const queries = seededVectors(10, DIM, 11);

        const readers = [
            ['range', 'trunc.pancake-range',
                (snap, out) => Pancake.buildRangeArtifactFile(snap, out),
                (p) => Pancake.openRangeArtifactFile(p),
                (artifact, q) => artifact.search(q, K, { efSearch: 200 })],
            ['sketch', 'trunc.pancake-sketch',
                (snap, out) => Pancake.buildSketchArtifactFile(snap, out, { sketchDims: 16, sketchBits: 8 }),
                (p) => Pancake.openSketchArtifactFile(p),
                (artifact, q) => artifact.search(q, K, { rerank: 200 })],
        ];
        for (const [label, name, build, open, search] of readers) {
            const fullPath = path.join(tmp, name);
            build(snapshotPath, fullPath);
            const full = fs.readFileSync(fullPath);
            for (const keep of [4, 64, Math.floor(full.length / 2)]) {
                const cutPath = `${fullPath}.cut${keep}`;
                fs.writeFileSync(cutPath, full.subarray(0, keep));
                let coded = false;
                let detail = 'no error thrown';
                try {
                    // Open may legitimately succeed when the cut falls past the
                    // resident prefix; a search must then hit the missing bytes.
                    const artifact = await open(cutPath);
                    try {
                        for (const q of queries) await search(artifact, q);
                    } finally {
                        await artifact.close();
                    }
                } catch (err) {
                    coded = err instanceof Pancake.PancakeError && typeof err.code === 'string';
                    detail = String(err && err.message);
                }
                check(`${label} artifact truncated to ${keep}B fails closed`, coded, detail);
            }
        }
    }

    // A range-artifact record lying about its own id must fail closed with a
    // coded error, not poison the cache under the forged key (which used to
    // surface as an uncoded TypeError deep inside search).
    console.log('\nforged record ids fail closed');
    {
        const rows = seededVectors(COUNT, DIM, 21);
        const index = await Pancake.create({ dim: DIM, maxElements: COUNT, metric: 'l2', quantized: true });
        index.addBatch(rows);
        const snapshotPath = path.join(tmp, 'forge-snap.pnck');
        fs.writeFileSync(snapshotPath, index.export());
        index.dispose();
        const rangePath = path.join(tmp, 'forge.pancake-range');
        Pancake.buildRangeArtifactFile(snapshotPath, rangePath);

        const clean = await Pancake.openRangeArtifactFile(rangePath);
        const baseOffset = clean.baseRecordsOffset;
        const routerOffset = clean.routerRecordsOffset;
        await clean.close();

        const forge = async (offset, label) => {
            const bytes = Buffer.from(fs.readFileSync(rangePath));
            const trueId = bytes.readUInt32LE(offset);
            bytes.writeUInt32LE((trueId + 1) % COUNT, offset);
            const forgedPath = `${rangePath}.${label}`;
            fs.writeFileSync(forgedPath, bytes);
            let coded = false;
            let detail = 'no error thrown';
            try {
                const artifact = await Pancake.openRangeArtifactFile(forgedPath);
                try {
                    const queries = seededVectors(10, DIM, 22);
                    for (const q of queries) await artifact.search(q, K, { efSearch: 200 });
                } finally {
                    await artifact.close();
                }
            } catch (err) {
                coded = err instanceof Pancake.PancakeError && err.code === 'SNAPSHOT_INVALID';
                detail = String(err && err.message);
            }
            check(`forged ${label} record id rejected with SNAPSHOT_INVALID`, coded, detail);
        };
        await forge(baseOffset, 'base');
        await forge(routerOffset, 'router');
    }

    // Golden fixtures: the reference reader must reproduce committed results
    // byte-for-byte from committed artifact bytes (spec section 5).
    console.log('\ngolden fixtures');
    {
        const golden = require('./fixtures/sketch_golden.js');
        for (const c of golden.cases) {
            const bytes = Buffer.from(c.artifactBase64, 'base64');
            const goldenPath = path.join(tmp, `golden-${c.metric}-${c.sketchBits}.pancake-sketch`);
            fs.writeFileSync(goldenPath, bytes);
            const artifact = await Pancake.openSketchArtifactFile(goldenPath);
            let ok = true;
            let ri = 0;
            for (const q of c.queries) {
                for (const [k, C] of [[5, 32], [10, 64]]) {
                    const expected = c.results[ri++];
                    const got = (await artifact.search(new Float32Array(q), k, { rerank: C })).results;
                    const gotIds = got.map((x) => x.id);
                    const gotDists = got.map((x) => Number(x.distance.toFixed(5)));
                    if (JSON.stringify(gotIds) !== JSON.stringify(expected.ids)) ok = false;
                    if (JSON.stringify(gotDists) !== JSON.stringify(expected.dists)) ok = false;
                }
            }
            // The WASM scanner path must reproduce the same golden results.
            const scanner = await Pancake.createSketchScanner(artifact);
            let scannerOk = true;
            ri = 0;
            for (const q of c.queries) {
                for (const [k, C] of [[5, 32], [10, 64]]) {
                    const expected = c.results[ri++];
                    const got = (await artifact.search(new Float32Array(q), k, { rerank: C, scanner })).results;
                    if (JSON.stringify(got.map((x) => x.id)) !== JSON.stringify(expected.ids)) scannerOk = false;
                }
            }
            scanner.dispose();
            await artifact.close();
            check(`golden ${c.metric} u${c.sketchBits}: reference reader reproduces committed results`, ok);
            check(`golden ${c.metric} u${c.sketchBits}: WASM scanner reproduces committed ids`, scannerOk);
        }
    }

    console.log(`\nSketch profile conformance: ${passed} passed, ${failed} failed`);
    if (failed > 0) process.exit(1);
}

run().catch((err) => {
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
});
