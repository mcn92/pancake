// Artifact-layer hardening conformance (range + sketch readers, snapshot
// parser): the lifecycle and hostile-input rules that an external review on
// 2026-08-21 found missing. Hermetic — fixtures are built in-process from a
// tiny quantized index — and fast.
//
//   1. a rejected openFile() releases its file descriptor (range and sketch);
//   2. query vectors follow the core engine's contract: numeric elements,
//      finite components for every metric, right dimension;
//   3. k / rerank / efSearch are validated and bounded by the row count, so a
//      k of 1e9 over a small artifact returns count results without trying
//      to allocate 1e9 slots;
//   4. the uint8 snapshot parser refuses adjacency counts above M/M0, out of
//      range neighbor ids, and implausible graph parameters before any
//      allocation.
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const Pancake = require('../pancake.js');
const {
    PancakeRangeArtifact, PancakeSketchArtifact, buildRangeArtifact, buildSketchArtifact, parseUint8Snapshot,
} = require('../pancake-artifact.js');

let passed = 0, failed = 0;
function check(label, cond, detail = '') {
    if (cond) { passed++; console.log('  ok:', label); }
    else { failed++; console.log('  FAIL:', label, detail ? `— ${detail}` : ''); }
}
async function rejects(label, fn, code, pattern) {
    try {
        await fn();
        check(label, false, 'resolved instead of throwing');
    } catch (err) {
        const ok = (!code || err.code === code) && (!pattern || pattern.test(String(err.message)));
        check(label, ok, ok ? '' : `threw ${err.code}: ${String(err.message).slice(0, 140)}`);
    }
}
function openFds() {
    try { return fs.readdirSync('/proc/self/fd').length; } catch { return null; }
}

async function buildFixtures(tmp, metric) {
    const dim = 8, count = 40;
    const index = await Pancake.create({ dim, maxElements: 100, metric, quantized: true });
    const vectors = [];
    for (let n = 0; n < count; n++) {
        const v = new Float32Array(dim);
        for (let d = 0; d < dim; d++) v[d] = Math.sin(n * 7 + d * 1.3);
        index.add(v);
        vectors.push(v);
    }
    const snapshot = index.export();
    index.dispose();
    const rangePath = path.join(tmp, `${metric}.pancake-range`);
    const sketchPath = path.join(tmp, `${metric}.pancake-sketch`);
    buildRangeArtifact(snapshot, rangePath, { layout: 'rcm' });
    buildSketchArtifact(snapshot, sketchPath, { recommendedRerank: 20 });
    return { dim, count, snapshot, vectors, rangePath, sketchPath };
}

async function main() {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pancake-hardening-'));

    // 1. Failed opens do not leak descriptors.
    console.log('1. openFile() releases its descriptor when the open is rejected');
    const garbage = path.join(tmp, 'garbage.bin');
    fs.writeFileSync(garbage, Buffer.alloc(4096, 0xab));
    const before = openFds();
    let rangeRejected = 0, sketchRejected = 0;
    for (let i = 0; i < 50; i++) {
        try { await PancakeRangeArtifact.openFile(garbage); } catch { rangeRejected++; }
        try { await PancakeSketchArtifact.openFile(garbage); } catch { sketchRejected++; }
    }
    const after = openFds();
    check('50 garbage range opens and 50 garbage sketch opens are all rejected', rangeRejected === 50 && sketchRejected === 50);
    if (before === null) console.log('  (skip: /proc/self/fd unavailable on this platform; descriptor count not measured)');
    else check('open descriptor count is unchanged after 100 rejected opens', after - before <= 0, `before ${before}, after ${after}`);
    await rejects('openFile() on a missing path rejects with ENOENT and leaves no descriptor',
        () => PancakeRangeArtifact.openFile(path.join(tmp, 'missing.pancake-range')), undefined, /ENOENT/);
    if (before !== null) check('descriptor count still unchanged after the missing-file open', openFds() - before <= 0);

    for (const metric of ['l2', 'cosine']) {
        console.log(`\n2-3. query contract and bounded k on ${metric} artifacts`);
        const fx = await buildFixtures(tmp, metric);
        const range = await PancakeRangeArtifact.openFile(fx.rangePath);
        const sketch = await PancakeSketchArtifact.openFile(fx.sketchPath);
        const q = fx.vectors[3];
        const strings = Array.from(q, (v) => String(v));
        const withNaN = Float32Array.from(q); withNaN[2] = NaN;
        const withInf = Float32Array.from(q); withInf[5] = Infinity;

        const baseline = (await range.search(q, 5)).results;
        check('range search returns results for a valid query', baseline.length === 5 && Number.isFinite(baseline[0].distance));
        await rejects('range: numeric-string coordinates are rejected', () => range.search(strings, 5), 'INVALID_VECTOR', /only numbers/);
        await rejects('range: NaN component is rejected', () => range.search(withNaN, 5), 'INVALID_VECTOR', /non-finite/);
        await rejects('range: Infinity component is rejected', () => range.search(withInf, 5), 'INVALID_VECTOR', /non-finite/);
        await rejects('range: wrong dimension is rejected', () => range.search(new Float32Array(fx.dim + 1), 5), 'DIMENSION_MISMATCH');
        await rejects('range: k=0 is rejected', () => range.search(q, 0), 'INVALID_ARGUMENT', /positive integer/);
        await rejects('range: fractional k is rejected', () => range.search(q, 2.5), 'INVALID_ARGUMENT');
        await rejects('range: efSearch must be a positive integer', () => range.search(q, 5, { efSearch: -1 }), 'INVALID_ARGUMENT', /efSearch/);
        const hugeRange = (await range.search(q, 1e9)).results;
        check('range: k=1e9 returns at most count results', hugeRange.length <= fx.count && hugeRange.length > 0, `${hugeRange.length}`);
        check('range: k=1e9 top result equals the k=5 top result', hugeRange[0].id === baseline[0].id);

        const sBaseline = (await sketch.search(q, 5)).results;
        check('sketch search returns results for a valid query', sBaseline.length === 5 && Number.isFinite(sBaseline[0].distance));
        await rejects('sketch: numeric-string coordinates are rejected', () => sketch.search(strings, 5), 'INVALID_VECTOR', /only numbers/);
        await rejects('sketch: NaN component is rejected', () => sketch.search(withNaN, 5), 'INVALID_VECTOR', /non-finite/);
        await rejects('sketch: wrong dimension is rejected', () => sketch.search(new Float32Array(fx.dim + 1), 5), 'DIMENSION_MISMATCH');
        await rejects('sketch: k=0 is rejected', () => sketch.search(q, 0), 'INVALID_ARGUMENT', /positive integer/);
        await rejects('sketch: rerank must be a positive integer', () => sketch.search(q, 5, { rerank: 1.5 }), 'INVALID_ARGUMENT', /rerank/);
        const t0 = Date.now();
        const hugeSketch = (await sketch.search(q, 1e9, { rerank: 1e9 })).results;
        check('sketch: k=1e9 / rerank=1e9 returns at most count results without a giant allocation',
            hugeSketch.length <= fx.count && hugeSketch.length > 0 && Date.now() - t0 < 5000, `${hugeSketch.length} in ${Date.now() - t0} ms`);
        check('sketch: k=1e9 top result equals the k=5 top result', hugeSketch[0].id === sBaseline[0].id);
        await range.close();
        await sketch.close();
        await sketch.close();
        check('sketch close() is idempotent', true);
        await rejects('reads through a closed NodeFileRangeSource are refused', () => range.source.read(0, 16), 'INVALID_ARGUMENT', /closed/);
    }

    // 4. Snapshot parser bounds.
    console.log('\n4. uint8 snapshot parser refuses hostile adjacency before allocating');
    const fx = await buildFixtures(tmp, 'l2');
    // Locate the raw uint8 payload inside the export envelope by its magic
    // (parseUint8Snapshot accepts a bare payload), then tamper node 0's first
    // adjacency count.
    const magic = Buffer.from([0x31, 0x48, 0x38, 0x49]); // UINT8_HNSW_MAGIC_V1 LE
    const snap = Buffer.from(fx.snapshot);
    const rawAt = snap.indexOf(magic);
    check('raw uint8 payload located in the export', rawAt >= 0);
    const raw = Buffer.from(snap.subarray(rawAt));
    const graph = parseUint8Snapshot(raw);
    check('untampered payload parses', graph.count === fx.count && graph.M0 >= graph.M);
    const levelsAt = 40 + fx.count * 8 + fx.count * fx.dim;
    const sizeAt = levelsAt + 4; // node 0: u32 level, then u32 size for level 0
    const hugeEdges = Buffer.from(raw); hugeEdges.writeUInt32LE(0xfffffff0, sizeAt);
    const t0 = Date.now();
    await rejects('adjacency count above M0 is rejected (no allocation)', async () => parseUint8Snapshot(hugeEdges), 'SNAPSHOT_INVALID', /adjacency exceeds/);
    check('...and quickly', Date.now() - t0 < 1000);
    const overCap = Buffer.from(raw); overCap.writeUInt32LE(graph.M0 + 1, sizeAt);
    await rejects('adjacency count of M0+1 at layer 0 is rejected', async () => parseUint8Snapshot(overCap), 'SNAPSHOT_INVALID', /adjacency exceeds/);
    const badNeighbor = Buffer.from(raw); badNeighbor.writeUInt32LE(fx.count + 7, sizeAt + 4); // first neighbor id of node 0
    await rejects('neighbor id >= count is rejected', async () => parseUint8Snapshot(badNeighbor), 'SNAPSHOT_INVALID', /neighbor id/);
    const badParams = Buffer.from(raw); badParams.writeUInt32LE(0, 24); // M = 0
    await rejects('implausible graph parameters (M=0) are rejected', async () => parseUint8Snapshot(badParams), 'SNAPSHOT_INVALID', /graph parameters/);
    const truncatedEdges = Buffer.from(raw.subarray(0, sizeAt + 8)); // claims edges, bytes end
    await rejects('adjacency that runs past the buffer is rejected as truncated', async () => parseUint8Snapshot(truncatedEdges), 'SNAPSHOT_INVALID', /truncated|adjacency/);

    fs.rmSync(tmp, { recursive: true, force: true });
    console.log(`\nArtifact hardening: ${passed} passed, ${failed} failed`);
    process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => { console.error(err); process.exit(1); });
