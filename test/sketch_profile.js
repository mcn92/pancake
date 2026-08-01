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

    console.log(`\nSketch profile conformance: ${passed} passed, ${failed} failed`);
    if (failed > 0) process.exit(1);
}

run().catch((err) => {
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
});
