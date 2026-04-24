/**
 * Pancake Test Suite
 *
 * Tests core invariants of the Pancake index.
 * Run with: npm test
 *
 * No test framework required — plain Node.js.
 */

'use strict';

const Pancake = require('./pancake.js');

// ─── Minimal test harness ────────────────────────────────────────────────────

let passed = 0;
let failed = 0;
const failures = [];

function assert(condition, message) {
    if (condition) {
        passed++;
    } else {
        failed++;
        failures.push(message);
        console.error(`  ✗ FAIL: ${message}`);
    }
}

function assertNear(a, b, tolerance = 1e-4, message) {
    assert(Math.abs(a - b) <= tolerance, `${message} (got ${a}, expected ~${b})`);
}

function assertThrows(fn, message) {
    try {
        fn();
        failed++;
        failures.push(`${message} — expected throw but none occurred`);
        console.error(`  ✗ FAIL: ${message} — expected throw but none occurred`);
    } catch (e) {
        passed++;
    }
}

async function assertThrowsAsync(fn, message) {
    try {
        await fn();
        failed++;
        failures.push(`${message} — expected throw but none occurred`);
        console.error(`  ✗ FAIL: ${message} — expected throw but none occurred`);
    } catch (e) {
        passed++;
    }
}

function section(name) {
    console.log(`\n  ${name}`);
    console.log(`  ${'─'.repeat(name.length)}`);
}

// ─── Vector utilities ─────────────────────────────────────────────────────────

function randomVec(dim) {
    const v = new Float32Array(dim);
    for (let i = 0; i < dim; i++) v[i] = Math.random() * 2 - 1;
    return v;
}

function normalizedVec(dim) {
    const v = randomVec(dim);
    let norm = 0;
    for (let i = 0; i < dim; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    for (let i = 0; i < dim; i++) v[i] /= norm;
    return v;
}

function zeroVec(dim) {
    return new Float32Array(dim);
}

function unitVec(dim, axis) {
    const v = new Float32Array(dim);
    v[axis] = 1.0;
    return v;
}

function cosineDist(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return 1 - dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function l2Dist(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        sum += d * d;
    }
    return Math.sqrt(sum);
}

// ─── Default config ───────────────────────────────────────────────────────────

const DIM = 128;

const DEFAULT_CONFIG = {
    dim:            DIM,
    metric:         'cosine',
    maxElements:    1000,
    M:              16,
    efConstruction: 200,
    efSearch:       100,
};

// ─── Test suites ──────────────────────────────────────────────────────────────

async function testCreation() {
    section('Creation');

    // Basic creation
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        assert(idx !== null && idx !== undefined, 'create() returns an index');
        assert(idx.count === 0, 'fresh index has count 0');
        assert(idx.memory > 0, 'fresh index has positive memory');
        idx.dispose();
    }

    // L2 metric
    {
        const idx = await Pancake.create({ ...DEFAULT_CONFIG, metric: 'l2' });
        assert(idx !== null, 'create() with l2 metric succeeds');
        idx.dispose();
    }

    // Different dimensions
    for (const dim of [32, 64, 256, 384]) {
        const idx = await Pancake.create({ ...DEFAULT_CONFIG, dim });
        assert(idx !== null, `create() with dim=${dim} succeeds`);
        idx.dispose();
    }

    // Missing required field
    await assertThrowsAsync(
        () => Pancake.create({ metric: 'cosine', maxElements: 100 }),
        'create() without dim throws'
    );

    await assertThrowsAsync(
        () => Pancake.create({ ...DEFAULT_CONFIG, compressed: DIM }),
        'create() with removed compressed option throws'
    );

    await assertThrowsAsync(
        () => Pancake.create({ ...DEFAULT_CONFIG, varianceSample: [normalizedVec(DIM)] }),
        'create() with removed varianceSample option throws'
    );
}


async function testAdd() {
    section('Add');

    const idx = await Pancake.create(DEFAULT_CONFIG);

    // Add Float32Array
    const vec = normalizedVec(DIM);
    const id0 = idx.add(vec);
    assert(typeof id0 === 'number', 'add() returns a numeric id');
    assert(idx.count === 1, 'count increments after add');

    // Add plain number[]
    const vec2 = Array.from(normalizedVec(DIM));
    const id1 = idx.add(vec2);
    assert(typeof id1 === 'number', 'add() accepts plain number[]');
    assert(idx.count === 2, 'count increments after second add');
    assert(id1 !== id0, 'ids are unique');

    // Wrong dimension throws
    assertThrows(
        () => idx.add(new Float32Array(DIM + 1)),
        'add() with wrong dimension throws'
    );
    assertThrows(
        () => idx.add(new Float32Array(DIM - 1)),
        'add() with dim-1 throws'
    );

    // Count unchanged after failed adds
    assert(idx.count === 2, 'count unchanged after failed adds');

    idx.dispose();
}


async function testAddBatch() {
    section('AddBatch');

    const idx = await Pancake.create(DEFAULT_CONFIG);

    const vecs = Array.from({ length: 10 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);

    assert(Array.isArray(ids), 'addBatch() returns an array');
    assert(ids.length === 10, 'addBatch() returns correct number of ids');
    assert(idx.count === 10, 'count reflects batch add');

    const unique = new Set(ids);
    assert(unique.size === 10, 'addBatch() ids are all unique');

    // Empty batch
    const emptyIds = idx.addBatch([]);
    assert(Array.isArray(emptyIds) && emptyIds.length === 0, 'addBatch([]) returns empty array');
    assert(idx.count === 10, 'count unchanged after empty batch');

    idx.dispose();
}


async function testSearch() {
    section('Search');

    const idx = await Pancake.create(DEFAULT_CONFIG);

    // Insert known vectors and verify self-recall
    const vecs = Array.from({ length: 100 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);

    // Search for each inserted vector — should find itself at rank 1
    let selfRecall = 0;
    for (let i = 0; i < vecs.length; i++) {
        const results = idx.search(vecs[i], 1);
        if (results.length > 0 && results[0].id === ids[i]) selfRecall++;
    }
    assert(selfRecall === vecs.length, `self-recall 100/100 (got ${selfRecall})`);

    // Result structure
    const results = idx.search(vecs[0], 5);
    assert(Array.isArray(results), 'search() returns array');
    assert(results.length === 5, 'search() returns requested k results');
    assert('id' in results[0], 'results have id field');
    assert('distance' in results[0], 'results have distance field');
    assert(typeof results[0].distance === 'number', 'distance is a number');

    // Results are sorted by distance ascending
    for (let i = 1; i < results.length; i++) {
        assert(
            results[i].distance >= results[i - 1].distance,
            `results sorted by distance (pos ${i})`
        );
    }

    // Distance to self is near zero
    assertNear(results[0].distance, 0, 0.01, 'distance to self is near zero');

    // k > count returns count results
    const bigK = idx.search(vecs[0], 1000);
    assert(bigK.length === idx.count, 'search with k>count returns count results');

    const overHundred = idx.search(vecs[0], 150);
    assert(overHundred.length === idx.count, 'search with k>100 is not truncated');

    // k=1
    const one = idx.search(vecs[0], 1);
    assert(one.length === 1, 'search with k=1 returns 1 result');

    idx.dispose();
}


async function testSearchMetrics() {
    section('Search — metric correctness');

    const DIM2 = 4;  // Small dim for exact verification

    // Cosine: unit vectors, known angles
    {
        const idx = await Pancake.create({
            ...DEFAULT_CONFIG,
            dim: DIM2,
            metric: 'cosine',
            maxElements: 10,
        });

        const a = new Float32Array([1, 0, 0, 0]);
        const b = new Float32Array([0, 1, 0, 0]);  // orthogonal → dist 1.0
        const c = new Float32Array([-1, 0, 0, 0]); // opposite  → dist 2.0

        idx.add(a);
        idx.add(b);
        idx.add(c);

        const results = idx.search(a, 3);
        assertNear(results[0].distance, 0.0, 0.01, 'cosine: self distance ≈ 0');
        assertNear(results[1].distance, 1.0, 0.01, 'cosine: orthogonal distance ≈ 1');
        assertNear(results[2].distance, 2.0, 0.01, 'cosine: opposite distance ≈ 2');

        idx.dispose();
    }

    // L2: known distances
    {
        const idx = await Pancake.create({
            ...DEFAULT_CONFIG,
            dim: DIM2,
            metric: 'l2',
            maxElements: 10,
        });

        const origin = new Float32Array([0, 0, 0, 0]);
        const unit   = new Float32Array([1, 0, 0, 0]);  // l2 dist = 1
        const far    = new Float32Array([3, 4, 0, 0]);  // l2 dist = 5

        idx.add(origin);
        idx.add(unit);
        idx.add(far);

        const results = idx.search(origin, 3);
        assertNear(results[0].distance, 0.0, 0.01, 'l2: self distance ≈ 0');
        assertNear(results[1].distance, 1.0, 0.01, 'l2: unit distance ≈ 1');
        assertNear(results[2].distance, 5.0, 0.01, 'l2: (3,4) distance ≈ 5');

        idx.dispose();
    }
}


async function testDelete() {
    section('Delete');

    const idx = await Pancake.create(DEFAULT_CONFIG);

    const vecs = Array.from({ length: 10 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);

    // Delete one vector
    idx.delete(ids[0]);
    assert(idx.count === 10, 'count unchanged after delete (soft delete)');
    assert(idx.ghostCount > 0, 'ghostCount > 0 after delete');

    // Deleted vector does not appear in search results
    const results = idx.search(vecs[0], 10);
    const deletedAppears = results.some(r => r.id === ids[0]);
    assert(!deletedAppears, 'deleted vector excluded from search results');

    // Delete remaining vectors
    for (let i = 1; i < ids.length; i++) idx.delete(ids[i]);
    assert(idx.ghostCount === 10, 'ghostCount equals total deleted');

    idx.dispose();
}


async function testGhostRatio() {
    section('Ghost ratio');

    const idx = await Pancake.create(DEFAULT_CONFIG);

    const vecs = Array.from({ length: 100 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);

    assert(idx.ghostRatio === 0, 'ghostRatio is 0 with no deletes');

    // Delete half
    for (let i = 0; i < 50; i++) idx.delete(ids[i]);

    assertNear(idx.ghostRatio, 0.5, 0.01, 'ghostRatio ≈ 0.5 after deleting half');

    idx.dispose();
}


async function testCompact() {
    section('Compact');

    const idx = await Pancake.create(DEFAULT_CONFIG);

    const vecs = Array.from({ length: 100 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);

    const memBefore = idx.memory;

    // Delete half and compact
    for (let i = 0; i < 50; i++) idx.delete(ids[i]);
    idx.compact();

    assert(idx.ghostCount === 0, 'ghostCount is 0 after compact');
    assert(idx.ghostRatio === 0, 'ghostRatio is 0 after compact');

    const memAfter = idx.memory;
    assert(memAfter < memBefore, `memory decreased after compact (${memBefore} → ${memAfter})`);

    // Remaining vectors still searchable after compact
    let recallAfterCompact = 0;
    for (let i = 50; i < 100; i++) {
        const results = idx.search(vecs[i], 1);
        if (results.length > 0 && results[0].id === ids[i]) recallAfterCompact++;
    }
    assert(recallAfterCompact === 50, `recall intact after compact (${recallAfterCompact}/50)`);

    idx.dispose();
}


async function testMemory() {
    section('Memory');

    const idx = await Pancake.create(DEFAULT_CONFIG);
    const memEmpty = idx.memory;

    // Memory grows with insertions
    const vecs = Array.from({ length: 100 }, () => normalizedVec(DIM));
    idx.addBatch(vecs);

    assert(idx.memory > memEmpty, 'memory grows after insertions');

    idx.dispose();
}


async function testDispose() {
    section('Dispose');

    const idx = await Pancake.create(DEFAULT_CONFIG);
    idx.add(normalizedVec(DIM));
    idx.dispose();

    // All operations after dispose should throw
    assertThrows(() => idx.add(normalizedVec(DIM)),   'add() after dispose throws');
    assertThrows(() => idx.search(normalizedVec(DIM), 1), 'search() after dispose throws');
    assertThrows(() => idx.delete(0),                 'delete() after dispose throws');
    assertThrows(() => idx.compact(),                 'compact() after dispose throws');
    assertThrows(() => idx.count,                     'count after dispose throws');
    assertThrows(() => idx.memory,                    'memory after dispose throws');

    // Double dispose should not throw
    try {
        idx.dispose();
        passed++;
    } catch (e) {
        failed++;
        failures.push('double dispose threw unexpectedly');
    }
}


async function testDeterminism() {
    section('Determinism');

    // Same vectors, same order → same search results
    const vecs = Array.from({ length: 50 }, () => normalizedVec(DIM));
    const query = normalizedVec(DIM);

    const runSearch = async () => {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        idx.addBatch(vecs);
        const results = idx.search(query, 5);
        idx.dispose();
        return results.map(r => r.id);
    };

    const run1 = await runSearch();
    const run2 = await runSearch();

    const deterministic = run1.every((id, i) => id === run2[i]);
    assert(deterministic, 'search results are deterministic across identical builds');
}


async function testSearchDuringMutation() {
    section('Search during mutation');

    const idx = await Pancake.create(DEFAULT_CONFIG);

    // Build initial index
    const vecs = Array.from({ length: 100 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);

    // Interleave deletes and searches — search should never return deleted ids
    for (let i = 0; i < 50; i++) {
        idx.delete(ids[i]);
        const results = idx.search(vecs[i], 10);
        const contaminated = results.some(r => ids.slice(0, i + 1).includes(r.id));
        assert(!contaminated, `no deleted ids in search results after ${i + 1} deletes`);
    }

    idx.dispose();
}


async function testExportImport() {
    section('Export / Import');

    const idx = await Pancake.create(DEFAULT_CONFIG);

    const vecs = Array.from({ length: 50 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);

    const query   = vecs[0];
    const before  = idx.search(query, 5).map(r => r.id);
    const exported = idx.export();

    assert(exported instanceof Uint8Array || exported instanceof ArrayBuffer,
        'export() returns binary data');
    assert(exported.byteLength > 0, 'exported data is non-empty');

    idx.dispose();

    // Import into fresh index
    const idx2 = await Pancake.create(DEFAULT_CONFIG);
    idx2.import(exported);

    assert(idx2.count === 50, 'imported index has correct count');

    const after = idx2.search(query, 5).map(r => r.id);
    const roundTrip = before.every((id, i) => id === after[i]);
    assert(roundTrip, 'search results identical after export/import round-trip');

    idx2.dispose();
}


async function testLargeIndex() {
    section('Scale — 1000 vectors');

    const idx = await Pancake.create({
        ...DEFAULT_CONFIG,
        maxElements: 1000,
    });

    const vecs = Array.from({ length: 1000 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);

    assert(idx.count === 1000, 'count correct at 1000 vectors');

    // Self-recall on a sample
    let recall = 0;
    const sample = 50;
    for (let i = 0; i < sample; i++) {
        const results = idx.search(vecs[i], 1);
        if (results.length > 0 && results[0].id === ids[i]) recall++;
    }
    // HNSW is approximate — expect high but not necessarily perfect recall
    assert(recall >= 45, `recall@1 ≥ 90% at 1000 vectors (got ${recall}/${sample})`);

    idx.dispose();
}


async function testQuantized() {
    section('Quantized mode');

    const QDIM = 128;
    const QCONFIG = {
        dim:            QDIM,
        metric:         'cosine',
        maxElements:    500,
        M:              16,
        efConstruction: 200,
        efSearch:       100,
        quantized:      true,
    };

    // Creation
    const idx = await Pancake.create(QCONFIG);
    assert(idx !== null, 'create() with quantized: true succeeds');
    assert(idx.count === 0, 'fresh quantized index has count 0');

    // Add vectors
    const vecs = Array.from({ length: 100 }, () => normalizedVec(QDIM));
    const ids  = idx.addBatch(vecs);
    assert(idx.count === 100, 'quantized: count correct after addBatch');

    // Self-recall (allow slightly lower due to quantization)
    let selfRecall = 0;
    for (let i = 0; i < vecs.length; i++) {
        const results = idx.search(vecs[i], 1);
        if (results.length > 0 && results[0].id === ids[i]) selfRecall++;
    }
    assert(selfRecall >= 90, `quantized: self-recall ≥ 90% (got ${selfRecall}/100)`);

    // Delete + compact
    for (let i = 0; i < 20; i++) idx.delete(ids[i]);
    assert(idx.ghostCount > 0, 'quantized: ghostCount > 0 after deletes');

    idx.compact();
    assert(idx.ghostCount === 0, 'quantized: ghostCount 0 after compact');

    // Surviving vectors still searchable
    let recallAfterCompact = 0;
    for (let i = 20; i < 100; i++) {
        const results = idx.search(vecs[i], 1);
        if (results.length > 0 && results[0].id === ids[i]) recallAfterCompact++;
    }
    assert(recallAfterCompact >= 70, `quantized: recall after compact ≥ 87% (got ${recallAfterCompact}/80)`);

    // Export / Import
    const exported = idx.export();
    assert(exported.byteLength > 0, 'quantized: export produces data');

    const idx2 = await Pancake.create(QCONFIG);
    idx2.import(exported);
    assert(idx2.count === 80, 'quantized: imported count correct');

    idx2.dispose();
    idx.dispose();
}


async function testErrorPaths() {
    section('Error paths');

    // Import corrupted data
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        const garbage = new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7]);
        let threw = false;
        try {
            idx.import(garbage);
        } catch (e) {
            threw = true;
        }
        assert(threw, 'import with corrupted data throws');
        idx.dispose();
    }

    // Import empty data
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        let threw = false;
        try {
            idx.import(new Uint8Array(0));
        } catch (e) {
            threw = true;
        }
        assert(threw, 'import with empty data throws');
        idx.dispose();
    }

    // Import from ArrayBuffer (not Uint8Array)
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        const vecs = Array.from({ length: 10 }, () => normalizedVec(DIM));
        idx.addBatch(vecs);
        const exported = idx.export();
        idx.dispose();

        const idx2 = await Pancake.create(DEFAULT_CONFIG);
        idx2.import(exported.buffer); // ArrayBuffer, not Uint8Array
        assert(idx2.count === 10, 'import from ArrayBuffer works');
        idx2.dispose();
    }

    // Import into non-empty index (overwrites)
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        const vecs1 = Array.from({ length: 20 }, () => normalizedVec(DIM));
        idx.addBatch(vecs1);

        const idx2 = await Pancake.create(DEFAULT_CONFIG);
        const vecs2 = Array.from({ length: 10 }, () => normalizedVec(DIM));
        idx2.addBatch(vecs2);
        const snapshot = idx2.export();
        idx2.dispose();

        idx.import(snapshot);
        assert(idx.count === 10, 'import into non-empty index overwrites (count matches source)');
        idx.dispose();
    }

    // add() with wrong dimension after successful adds
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        idx.add(normalizedVec(DIM));
        assertThrows(
            () => idx.add(new Float32Array(DIM * 2)),
            'add() wrong dim after successful add still throws'
        );
        assert(idx.count === 1, 'count unchanged after failed add');
        idx.dispose();
    }

    // Import config mismatch: dim 64 → 128
    {
        const src = await Pancake.create({ ...DEFAULT_CONFIG, dim: 64 });
        src.addBatch(Array.from({ length: 10 }, () => normalizedVec(64)));
        const exported = src.export();
        src.dispose();

        const dst = await Pancake.create({ ...DEFAULT_CONFIG, dim: 128 });
        let msg = '';
        try { dst.import(exported); } catch (e) { msg = e.message; }
        assert(msg.includes('dim mismatch'), 'import dim 64→128 throws with dim mismatch message');
        dst.dispose();
    }

    // Import config mismatch: dim 128 → 64
    {
        const src = await Pancake.create({ ...DEFAULT_CONFIG, dim: 128 });
        src.addBatch(Array.from({ length: 10 }, () => normalizedVec(128)));
        const exported = src.export();
        src.dispose();

        const dst = await Pancake.create({ ...DEFAULT_CONFIG, dim: 64 });
        let msg = '';
        try { dst.import(exported); } catch (e) { msg = e.message; }
        assert(msg.includes('dim mismatch'), 'import dim 128→64 throws with dim mismatch message');
        dst.dispose();
    }

    // Import config mismatch: cosine → l2
    {
        const src = await Pancake.create({ ...DEFAULT_CONFIG, metric: 'cosine' });
        src.addBatch(Array.from({ length: 10 }, () => normalizedVec(DIM)));
        const exported = src.export();
        src.dispose();

        const dst = await Pancake.create({ ...DEFAULT_CONFIG, metric: 'l2' });
        let msg = '';
        try { dst.import(exported); } catch (e) { msg = e.message; }
        assert(msg.includes('metric mismatch'), 'import cosine→l2 throws with metric mismatch message');
        dst.dispose();
    }

    // Import of deprecated DCT/PCA envelope is rejected
    {
        const src = await Pancake.create(DEFAULT_CONFIG);
        src.addBatch(Array.from({ length: 10 }, () => normalizedVec(DIM)));
        const exported = src.export();
        src.dispose();

        const deprecated = new Uint8Array(exported);
        const view = new DataView(deprecated.buffer);
        view.setUint32(4, 1, true);
        view.setUint32(12, 64, true);

        const dst = await Pancake.create(DEFAULT_CONFIG);
        let msg = '';
        try { dst.import(deprecated); } catch (e) { msg = e.message; }
        assert(msg.includes('no longer supported'), 'import rejects deprecated DCT/PCA envelopes');
        dst.dispose();
    }
}


async function testEdgeCases() {
    section('Edge cases');

    // Search on empty index
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        const results = idx.search(normalizedVec(DIM), 5);
        assert(Array.isArray(results), 'search on empty index returns array');
        assert(results.length === 0, 'search on empty index returns 0 results');
        idx.dispose();
    }

    // k=0 search
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        idx.addBatch(Array.from({ length: 10 }, () => normalizedVec(DIM)));
        const results = idx.search(normalizedVec(DIM), 0);
        assert(results.length === 0, 'search with k=0 returns 0 results');
        idx.dispose();
    }

    // Delete non-existent ID (should not throw)
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        idx.addBatch(Array.from({ length: 5 }, () => normalizedVec(DIM)));
        let threw = false;
        try {
            idx.delete(9999);
        } catch (e) {
            threw = true;
        }
        assert(!threw, 'delete non-existent ID does not throw');
        assert(idx.count === 5, 'count unchanged after deleting non-existent ID');
        assert(idx.ghostCount === 0, 'ghostCount unchanged after deleting non-existent ID');
        idx.dispose();
    }

    // Double-delete same ID
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        const vecs = Array.from({ length: 5 }, () => normalizedVec(DIM));
        const ids = idx.addBatch(vecs);
        idx.delete(ids[0]);
        let threw = false;
        try {
            idx.delete(ids[0]);
        } catch (e) {
            threw = true;
        }
        assert(!threw, 'double-delete same ID does not throw');
        idx.dispose();
    }

    // Compact with no deletions (no ghosts)
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        const vecs = Array.from({ length: 20 }, () => normalizedVec(DIM));
        const ids = idx.addBatch(vecs);
        idx.compact(); // no-op compact
        assert(idx.count === 20, 'compact with no ghosts preserves count');
        assert(idx.ghostCount === 0, 'ghostCount still 0 after no-op compact');

        // Vectors still searchable
        let recall = 0;
        for (let i = 0; i < vecs.length; i++) {
            const results = idx.search(vecs[i], 1);
            if (results.length > 0 && results[0].id === ids[i]) recall++;
        }
        assert(recall === 20, `recall intact after no-op compact (${recall}/20)`);
        idx.dispose();
    }

    // Compact after deleting ALL vectors
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        const vecs = Array.from({ length: 10 }, () => normalizedVec(DIM));
        const ids = idx.addBatch(vecs);
        for (const id of ids) idx.delete(id);
        idx.compact();
        assert(idx.ghostCount === 0, 'ghostCount 0 after compact of fully-deleted index');
        const results = idx.search(normalizedVec(DIM), 5);
        assert(results.length === 0, 'search returns nothing after full delete + compact');
        idx.dispose();
    }

    // Export on empty index
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        const exported = idx.export();
        assert(exported.byteLength > 0, 'export on empty index produces data');

        const idx2 = await Pancake.create(DEFAULT_CONFIG);
        idx2.import(exported);
        assert(idx2.count === 0, 'import of empty export gives count 0');
        idx2.dispose();
        idx.dispose();
    }

    // All getters after dispose
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        idx.add(normalizedVec(DIM));
        idx.dispose();

        assertThrows(() => idx.export(),             'export() after dispose throws');
        assertThrows(() => idx.import(new Uint8Array(1)), 'import() after dispose throws');
        assertThrows(() => idx.ghostCount,           'ghostCount after dispose throws');
        assertThrows(() => idx.ghostRatio,           'ghostRatio after dispose throws');
        // dim is a plain property, not guarded — it doesn't access WASM
        assert(idx.dim === DIM, 'dim still readable after dispose (no WASM call)');
    }

    // search() with plain number[] query (not Float32Array)
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        const vec = normalizedVec(DIM);
        idx.add(vec);
        const query = Array.from(vec);
        const results = idx.search(query, 1);
        assert(results.length === 1, 'search with plain number[] query works');
        assertNear(results[0].distance, 0, 0.01, 'search with number[] query finds self');
        idx.dispose();
    }

    // addBatch with single element
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        const ids = idx.addBatch([normalizedVec(DIM)]);
        assert(ids.length === 1, 'addBatch with single vector returns 1 id');
        assert(idx.count === 1, 'count is 1 after single-element batch');
        idx.dispose();
    }
}


// ─── New tests ───────────────────────────────────────────────────────────────

async function testMaxElementsOverflow() {
    section('maxElements overflow');

    // maxElements is a HARD limit in the C++ layer: insert() returns UINT32_MAX
    // when count_ >= max_elements_, and pancake.js translates that to a throw.
    //
    // IMPORTANT: The WASM backend uses a single global index. Each Pancake.create()
    // reinitializes that global, clobbering any previously created index. All
    // overflow assertions must therefore use a single index instance.

    const SMALL = 5;
    const idx = await Pancake.create({
        ...DEFAULT_CONFIG,
        maxElements: SMALL,
    });

    // Fill to declared capacity
    const ids = [];
    for (let i = 0; i < SMALL; i++) {
        ids.push(idx.add(normalizedVec(DIM)));
    }
    assert(idx.count === SMALL, `count is ${SMALL} at capacity`);

    // One more must throw
    let threw = false;
    let errMsg = '';
    try {
        idx.add(normalizedVec(DIM));
    } catch (e) {
        threw = true;
        errMsg = e.message;
    }
    assert(threw, 'add() beyond maxElements throws');
    assert(
        errMsg.toLowerCase().includes('full') || errMsg.toLowerCase().includes('failed'),
        `overflow error message is meaningful (got: "${errMsg}")`
    );

    // Count and search still correct after the failed add
    assert(idx.count === SMALL, 'count unchanged after overflow attempt');
    const results = idx.search(normalizedVec(DIM), 3);
    assert(results.length === 3, 'search still works after overflow attempt');

    // addBatch hitting the limit also throws — free one slot first so we
    // can verify the batch fails when the *batch itself* causes overflow,
    // without needing a second create() that would clobber the global.
    idx.delete(ids[0]);
    idx.compact(); // live count_ drops to 4, max_elements_ still 5
    idx.add(normalizedVec(DIM)); // back to 5 (at capacity again)
    let batchThrew = false;
    try {
        idx.addBatch([normalizedVec(DIM), normalizedVec(DIM)]);
    } catch (e) {
        batchThrew = true;
    }
    assert(batchThrew, 'addBatch() that exceeds maxElements throws');

    idx.dispose();
}


async function testExportImportWithGhosts() {
    section('Export / Import — with ghosts');

    // KNOWN LIMITATION: The serialize() format does not persist the deleted[]
    // bitmap. deserialize() resets all vectors to alive (deleted_.assign(count_,0)).
    // This means ghost vectors are silently resurrected on import. The test
    // documents this behaviour so callers know to compact() before export if
    // they need a clean snapshot.

    const idx = await Pancake.create(DEFAULT_CONFIG);
    const vecs = Array.from({ length: 20 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);

    // Delete some vectors before exporting
    const deletedIndices = [0, 5, 10];
    for (const i of deletedIndices) idx.delete(ids[i]);
    assert(idx.ghostCount === 3, 'ghostCount is 3 before export');

    const exported = idx.export();

    // Re-import into a fresh index
    const idx2 = await Pancake.create(DEFAULT_CONFIG);
    idx2.import(exported);

    // Because deleted[] is not serialized, all 20 vectors come back alive.
    // This is a known limitation — compact() before export to avoid it.
    assert(idx2.count === 20, 'import restores all 20 vectors (deleted[] not serialized — known limitation)');
    assert(idx2.ghostCount === 0, 'imported index reports 0 ghosts (deleted state not persisted)');

    // All vectors (including the 3 that were deleted pre-export) are now findable
    let recall = 0;
    for (const vec of vecs) {
        const r = idx2.search(vec, 1);
        if (r.length > 0 && r[0].distance < 0.01) recall++;
    }
    assert(recall >= 18, `all vectors findable after import (${recall}/20)`);

    // Contrast: compact() before export strips ghosts cleanly
    idx.compact();
    assert(idx.count === 17, 'compact() reduces count to 17 live vectors');
    const compactExported = idx.export();

    const idx3 = await Pancake.create(DEFAULT_CONFIG);
    idx3.import(compactExported);
    assert(idx3.count === 17, 'import after compact() gives 17 vectors');

    // Deleted vectors are no longer present after compact+export+import
    let ghostLeak = 0;
    for (const i of deletedIndices) {
        const r = idx3.search(vecs[i], 1);
        if (r.length > 0 && r[0].distance < 0.001) ghostLeak++;
    }
    assert(ghostLeak === 0, 'compact()+export()+import() produces clean snapshot with no ghost data');

    idx3.dispose();
    idx2.dispose();
    idx.dispose();
}


async function testAddBatchPartialFailure() {
    section('AddBatch — partial failure');

    // addBatch with a wrong-dimension vector mid-batch
    {
        const idx = await Pancake.create({ ...DEFAULT_CONFIG, maxElements: 20 });
        const goodVec  = normalizedVec(DIM);
        const badVec   = new Float32Array(DIM + 10); // wrong dim
        const goodVec2 = normalizedVec(DIM);

        let threw = false;
        try {
            idx.addBatch([goodVec, badVec, goodVec2]);
        } catch (e) {
            threw = true;
        }
        assert(threw, 'addBatch with wrong-dim vector throws');

        // Index must not be corrupted — search should still work
        const countAfter = idx.count;
        assert(countAfter >= 0, 'count is non-negative after partial batch failure');

        if (countAfter > 0) {
            const results = idx.search(goodVec, 1);
            assert(Array.isArray(results), 'search still returns array after partial batch failure');
        }

        idx.dispose();
    }

    // addBatch where ALL vectors have wrong dim
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        let threw = false;
        try {
            idx.addBatch([new Float32Array(DIM + 1), new Float32Array(DIM + 1)]);
        } catch (e) {
            threw = true;
        }
        assert(threw, 'addBatch with all wrong-dim vectors throws');
        assert(idx.count === 0, 'count still 0 after all-wrong-dim batch');
        idx.dispose();
    }
}


async function testZeroAndDegenerateVectors() {
    section('Degenerate vectors');

    // Zero vector — cosine is undefined; engine should not crash or corrupt
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        const zero = zeroVec(DIM);
        try {
            idx.add(zero);
        } catch (e) {
            // Throwing is acceptable
        }
        // Regardless of what happened above, index must still function
        idx.add(normalizedVec(DIM));
        const results = idx.search(normalizedVec(DIM), 1);
        assert(Array.isArray(results), 'index still functional after zero-vector add attempt');
        idx.dispose();
    }

    // NaN vector — must not silently corrupt
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        const nanVec = new Float32Array(DIM).fill(NaN);
        try {
            idx.add(nanVec);
        } catch (e) {
            // Throwing is acceptable
        }
        idx.add(normalizedVec(DIM));
        const results = idx.search(normalizedVec(DIM), 1);
        assert(Array.isArray(results), 'index functional after NaN vector add attempt');
        idx.dispose();
    }

    // Infinity vector — same contract as NaN
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        const infVec = new Float32Array(DIM).fill(Infinity);
        try {
            idx.add(infVec);
        } catch (e) {
            // Throwing is acceptable
        }
        idx.add(normalizedVec(DIM));
        const results = idx.search(normalizedVec(DIM), 1);
        assert(Array.isArray(results), 'index functional after Infinity vector add attempt');
        idx.dispose();
    }

    // Zero query vector — search must not crash or hang
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        idx.addBatch(Array.from({ length: 10 }, () => normalizedVec(DIM)));
        try {
            const results = idx.search(zeroVec(DIM), 5);
            assert(Array.isArray(results), 'search with zero query returns array');
        } catch (e) {
            // Throwing is also acceptable
        }
        idx.dispose();
    }
}


async function testDeterminismIds() {
    section('Determinism — ID assignment');

    // Same vectors, same order → same IDs
    const vecs = Array.from({ length: 20 }, () => normalizedVec(DIM));

    const build = async () => {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        const ids = idx.addBatch(vecs);
        idx.dispose();
        return ids;
    };

    const ids1 = await build();
    const ids2 = await build();

    const sameIds = ids1.every((id, i) => id === ids2[i]);
    assert(sameIds, 'IDs assigned are deterministic across identical builds');

    // IDs are sequential starting from 0
    assert(ids1[0] === 0, 'first assigned ID is 0');
    for (let i = 1; i < ids1.length; i++) {
        assert(ids1[i] === ids1[i - 1] + 1, `IDs are sequential (pos ${i})`);
    }

    // After compact(), surviving external IDs are stable
    {
        const idx = await Pancake.create(DEFAULT_CONFIG);
        const ids = idx.addBatch(vecs);
        const survivorId = ids[10];

        idx.delete(ids[0]);
        idx.delete(ids[1]);
        idx.compact();

        const results = idx.search(vecs[10], 1);
        assert(
            results.length > 0 && results[0].id === survivorId,
            'external IDs stable after compact (survivor found by original id)'
        );

        idx.dispose();
    }
}


async function testMetricCorrectnessQuantized() {
    section('Metric correctness — quantized INT8');

    const DIM2 = 32;
    const QCONFIG = {
        dim:            DIM2,
        metric:         'cosine',
        maxElements:    20,
        M:              8,
        efConstruction: 100,
        efSearch:       50,
        quantized:      true,
    };

    const idx = await Pancake.create(QCONFIG);

    // a = unit vector along axis 0
    // b = close to a (normalized)
    // c = opposite of a → max cosine distance
    const a = new Float32Array(DIM2); a[0] = 1.0;
    const b = new Float32Array(DIM2); b[0] = 0.99; b[1] = 0.14;
    let normB = 0; for (let i = 0; i < DIM2; i++) normB += b[i] * b[i]; normB = Math.sqrt(normB);
    for (let i = 0; i < DIM2; i++) b[i] /= normB;
    const c = new Float32Array(DIM2); c[0] = -1.0;

    const idA = idx.add(a);
    const idB = idx.add(b);
    const idC = idx.add(c);

    const results = idx.search(a, 3);
    assert(results.length === 3, 'quantized: search returns 3 results');
    assert(results[0].id === idA, 'quantized: self is nearest');
    assert(results[1].id === idB, 'quantized: near vector is rank 2');
    assert(results[2].id === idC, 'quantized: opposite vector is rank 3');

    assert(results[0].distance <= results[1].distance, 'quantized: dist[0] ≤ dist[1]');
    assert(results[1].distance <= results[2].distance, 'quantized: dist[1] ≤ dist[2]');
    assertNear(results[0].distance, 0.0, 0.05, 'quantized: self-distance ≈ 0');
    assert(results[2].distance > 1.5, `quantized: opposite distance > 1.5 (got ${results[2].distance.toFixed(3)})`);

    idx.dispose();
}


// ─── Runner ───────────────────────────────────────────────────────────────────

async function main() {
    console.log('');
    console.log('Pancake Test Suite');
    console.log('==================');

    const suites = [
        testCreation,
        testAdd,
        testAddBatch,
        testSearch,
        testSearchMetrics,
        testDelete,
        testGhostRatio,
        testCompact,
        testMemory,
        testDispose,
        testDeterminism,
        testSearchDuringMutation,
        testExportImport,
        testLargeIndex,
        testQuantized,
        testErrorPaths,
        testEdgeCases,
        testMaxElementsOverflow,
        testExportImportWithGhosts,
        testAddBatchPartialFailure,
        testZeroAndDegenerateVectors,
        testDeterminismIds,
        testMetricCorrectnessQuantized,
    ];

    const supplementalSuites = require('./test/supplemental_suite.js')({
        assert,
        assertNear,
        assertThrows,
        assertThrowsAsync,
        section,
    });
    suites.push(...supplementalSuites);

    for (const suite of suites) {
        try {
            await suite();
        } catch (e) {
            failed++;
            failures.push(`${suite.name} threw unexpectedly: ${e.message}`);
            console.error(`  ✗ ${suite.name} threw: ${e.message}`);
        }
    }

    console.log('');
    console.log('─'.repeat(40));
    console.log(`  Passed: ${passed}`);
    console.log(`  Failed: ${failed}`);

    if (failures.length > 0) {
        console.log('\nFailures:');
        failures.forEach(f => console.log(`  • ${f}`));
    }

    console.log('');
    process.exit(failed > 0 ? 1 : 0);
}

main().catch(e => {
    console.error('Test runner error:', e);
    process.exit(1);
});
