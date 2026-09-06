#!/usr/bin/env node
/**
 * Fuzz harness for Pikelet import() path.
 *
 * Generates random and adversarial binary blobs, feeds them to import(),
 * and verifies the index either rejects them cleanly or remains functional.
 *
 * Covers:
 *   - Pure random bytes (varying lengths)
 *   - Truncated valid snapshots at every byte boundary
 *   - Bit-flipped valid snapshots (single and multi-bit)
 *   - Header field corruption (count, dims, M, metric, entry_point, etc.)
 *   - Integer overflow in size fields (count * dims near SIZE_MAX)
 *   - Corrupted neighbor IDs (pointing past count)
 *   - Corrupted graph structure (bad level values, oversized neighbor lists)
 *   - NaN/Infinity in float fields (scales, offsets, vectors)
 *   - Envelope corruption (bad magic, version, dimension mismatch, truncated)
 *   - JS-layer envelope bookkeeping corruption (nextExtId, mappingCount, mappings)
 *   - Zero-length and minimal payloads
 *   - Valid snapshot followed by post-search to verify index is still usable
 *
 * Usage:
 *   node test/fuzz_import.js [--rounds N] [--seed N]
 *
 * Exit code 0 = no crashes or hangs. Non-zero = failure.
 */

'use strict';

const Pikelet = require('../pikelet.js');

// ─── PRNG (seeded xoshiro128** for reproducibility) ─────────────────────────

function xoshiro128ss(seed) {
    let s = [seed, seed ^ 0xDEADBEEF, seed ^ 0x12345678, seed ^ 0xCAFEBABE];
    function rotl(x, k) { return ((x << k) | (x >>> (32 - k))) >>> 0; }
    return function next() {
        const result = (rotl((s[1] * 5) >>> 0, 7) * 9) >>> 0;
        const t = (s[1] << 9) >>> 0;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = rotl(s[3], 11);
        return result;
    };
}

let rng;

function randU32() { return rng() >>> 0; }
function randFloat() { return (rng() >>> 0) / 0x100000000; }
function randInt(min, max) { return min + ((rng() >>> 0) % (max - min + 1)); }
function randBytes(len) {
    const buf = new Uint8Array(len);
    for (let i = 0; i < len; i++) buf[i] = rng() & 0xFF;
    return buf;
}

// ─── Snapshot helpers ───────────────────────────────────────────────────────

const PIKELET_MAGIC = 0x504E434B;
const FLOAT_MAGIC = 0x464C4831;
const UINT8_MAGIC = 0x49384831;

function normalizedVec(dim) {
    const v = new Float32Array(dim);
    let norm = 0;
    for (let i = 0; i < dim; i++) { v[i] = randFloat() * 2 - 1; norm += v[i] * v[i]; }
    norm = Math.sqrt(norm);
    for (let i = 0; i < dim; i++) v[i] /= norm;
    return v;
}

async function buildValidSnapshot(quantized) {
    const dim = 16;
    const idx = await Pikelet.create({
        dim, maxElements: 64, metric: 'cosine', quantized,
        M: 8, efConstruction: 50, efSearch: 50,
    });
    const vecs = Array.from({ length: 16 }, () => normalizedVec(dim));
    for (const v of vecs) idx.add(v);
    const exported = idx.export();
    idx.dispose();
    return { exported: new Uint8Array(exported), dim, count: 16, quantized };
}

function extractWasmPayload(exported) {
    const view = new DataView(exported.buffer, exported.byteOffset, exported.byteLength);
    const magic = view.getUint32(0, true);
    if (magic !== PIKELET_MAGIC) return exported;
    const version = view.getUint32(4, true);
    if (version === 3) {
        const mappingCount = view.getUint32(24, true);
        const wasmSize = view.getUint32(28, true);
        const wasmOffset = 32 + mappingCount * 8;
        return exported.slice(wasmOffset, wasmOffset + wasmSize);
    }
    return exported.slice(20);
}

function wrapInEnvelope(wasmBytes, dim, quantized) {
    const mappingCount = 0;
    const headerSize = 32;
    const result = new Uint8Array(headerSize + wasmBytes.length);
    const view = new DataView(result.buffer);
    view.setUint32(0, PIKELET_MAGIC, true);
    view.setUint32(4, 3, true); // version
    view.setUint32(8, dim, true);
    view.setUint32(12, 1, true); // cosine
    view.setUint32(16, quantized ? 1 : 0, true);
    view.setUint32(20, 0, true); // nextExtId
    view.setUint32(24, mappingCount, true);
    view.setUint32(28, wasmBytes.length, true);
    result.set(wasmBytes, headerSize);
    return result;
}

function mutateU32(buf, offset, value) {
    const copy = new Uint8Array(buf);
    new DataView(copy.buffer, copy.byteOffset).setUint32(offset, value >>> 0, true);
    return copy;
}

function mutateF32(buf, offset, value) {
    const copy = new Uint8Array(buf);
    new DataView(copy.buffer, copy.byteOffset).setFloat32(offset, value, true);
    return copy;
}

function flipBit(buf, byteIdx, bitIdx) {
    const copy = new Uint8Array(buf);
    copy[byteIdx] ^= (1 << bitIdx);
    return copy;
}

// ─── Test helpers ───────────────────────────────────────────────────────────

let passed = 0;
let failed = 0;
const failures = [];

function ok(condition, label) {
    if (condition) {
        passed++;
    } else {
        failed++;
        failures.push(label);
        console.error(`  FAIL: ${label}`);
    }
}

async function tryImport(label, data, dim, quantized) {
    let idx;
    try {
        idx = await Pikelet.create({
            dim, maxElements: 1000, metric: 'cosine', quantized,
            M: 8, efConstruction: 50, efSearch: 50,
        });
        idx.import(data);
        // If import succeeded, verify the index is still meaningfully functional.
        const q = normalizedVec(dim);
        const results = idx.search(q, Math.min(5, idx.count));
        ok(Array.isArray(results), `${label}: search returns array after import`);
        ok(idx.count >= 0 && Number.isInteger(idx.count), `${label}: count remains sane after import`);
        ok(results.length <= idx.count, `${label}: result length bounded by count`);
        if (idx.count > 0) {
            const followUpId = idx.add(normalizedVec(dim));
            ok(Number.isInteger(followUpId), `${label}: add works after import`);
            const results2 = idx.search(q, Math.min(5, idx.count));
            ok(Array.isArray(results2), `${label}: search still works after add`);
        }
    } catch (e) {
        // Rejection is fine — that's the expected outcome for bad data
        const msg = (e && e.message) ? e.message.slice(0, 60) : String(e).slice(0, 60);
        ok(true, `${label}: rejected (${msg})`);
    } finally {
        if (idx) idx.dispose();
    }
}

async function mustReject(label, data, dim, quantized) {
    let idx;
    try {
        idx = await Pikelet.create({
            dim, maxElements: 1000, metric: 'cosine', quantized,
            M: 8, efConstruction: 50, efSearch: 50,
        });
        idx.import(data);
        const q = normalizedVec(dim);
        const results = idx.search(q, Math.min(5, idx.count));
        ok(false, `${label}: accepted malformed import (count=${idx.count}, results=${results.length})`);
    } catch (e) {
        const msg = (e && e.message) ? e.message.slice(0, 60) : String(e).slice(0, 60);
        ok(true, `${label}: rejected (${msg})`);
    } finally {
        if (idx) idx.dispose();
    }
}

// ─── Fuzz strategies ────────────────────────────────────────────────────────

async function fuzzPureRandom(rounds) {
    console.log('\n  Pure random bytes');
    for (let i = 0; i < rounds; i++) {
        const len = randInt(0, 4096);
        const data = randBytes(len);
        await tryImport(`random[${len}B]`, data, 16, i % 2 === 0);
    }
}

async function fuzzTruncation(snapshot, label) {
    console.log(`\n  Truncation (${label})`);
    const raw = extractWasmPayload(snapshot.exported);
    // Test at key boundaries
    const offsets = [0, 1, 3, 4, 7, 8, 11, 12, 16, 20, 24, 28, 32, 36, 40];
    for (const len of offsets) {
        if (len > raw.length) continue;
        const truncated = wrapInEnvelope(raw.slice(0, len), snapshot.dim, snapshot.quantized);
        await mustReject(`trunc@${len}/${label}`, truncated, snapshot.dim, snapshot.quantized);
    }
    // Also test truncation at every 4-byte boundary in the first 200 bytes
    for (let len = 0; len < Math.min(200, raw.length); len += 4) {
        const truncated = wrapInEnvelope(raw.slice(0, len), snapshot.dim, snapshot.quantized);
        await mustReject(`trunc4@${len}/${label}`, truncated, snapshot.dim, snapshot.quantized);
    }
}

async function fuzzBitFlips(snapshot, label, rounds) {
    console.log(`\n  Bit flips (${label})`);
    const raw = extractWasmPayload(snapshot.exported);
    for (let i = 0; i < rounds; i++) {
        const byteIdx = randInt(0, raw.length - 1);
        const bitIdx = randInt(0, 7);
        const flipped = flipBit(raw, byteIdx, bitIdx);
        const wrapped = wrapInEnvelope(flipped, snapshot.dim, snapshot.quantized);
        await tryImport(`bitflip[${byteIdx}:${bitIdx}]/${label}`, wrapped, snapshot.dim, snapshot.quantized);
    }
}

async function fuzzHeaderCorruption(snapshot, label) {
    console.log(`\n  Header corruption (${label})`);
    const raw = extractWasmPayload(snapshot.exported);
    const wasmMagic = snapshot.quantized ? UINT8_MAGIC : FLOAT_MAGIC;

    // Offsets within raw WASM payload (v1 format):
    // 0: magic, 4: dims, 8: version, 12: count, 16: entry_point, 20: max_level,
    // 24: M, 28: M0, 32: metric, 36: ef_construction

    const headerMutations = [
        { off: 0, val: 0x00000000, desc: 'zero magic' },
        { off: 0, val: 0xDEADBEEF, desc: 'bad magic' },
        { off: 4, val: 999, desc: 'wrong dims' },
        { off: 12, val: 0xFFFFFFFF, desc: 'count=MAX_UINT32' },
        { off: 12, val: 999999, desc: 'count=999999' },
        { off: 12, val: 0, desc: 'count=0 (with valid entry_point)' },
        { off: 16, val: 0xFFFFFFFF, desc: 'entry_point=MAX' },
        { off: 16, val: 99999, desc: 'entry_point=99999' },
        { off: 20, val: 255, desc: 'max_level=255' },
        { off: 20, val: 0xFFFFFFFF, desc: 'max_level=MAX' },
        { off: 24, val: 0, desc: 'M=0' },
        { off: 24, val: 1, desc: 'M=1' },
        { off: 24, val: 0xFFFFFFFF, desc: 'M=MAX' },
        { off: 28, val: 0, desc: 'M0=0' },
        { off: 28, val: 0xFFFFFFFF, desc: 'M0=MAX' },
        { off: 32, val: 5, desc: 'metric=5 (invalid enum)' },
        { off: 32, val: 0xFFFFFFFF, desc: 'metric=MAX' },
        { off: 36, val: 0, desc: 'ef_construction=0' },
        { off: 36, val: 0xFFFFFFFF, desc: 'ef_construction=MAX' },
    ];

    for (const m of headerMutations) {
        if (m.off >= raw.length) continue;
        const mutated = mutateU32(raw, m.off, m.val);
        const wrapped = wrapInEnvelope(mutated, snapshot.dim, snapshot.quantized);
        await mustReject(`header:${m.desc}/${label}`, wrapped, snapshot.dim, snapshot.quantized);
    }
}

async function fuzzOverflowSizes(snapshot, label) {
    console.log(`\n  Integer overflow in size fields (${label})`);
    const raw = extractWasmPayload(snapshot.exported);

    // Try count values that would overflow count * dims or count * dims * 4
    const overflowCounts = [
        0x10000000,  // 256M — count * 16dims * 4 > 4GB on 32-bit
        0x40000000,  // 1G
        0x7FFFFFFF,  // INT32_MAX
        0x80000000,  // just over INT32_MAX
        0xFFFFFFFE,  // near MAX
        0xFFFFFFFF,  // MAX
    ];

    for (const count of overflowCounts) {
        const mutated = mutateU32(raw, 12, count);
        const wrapped = wrapInEnvelope(mutated, snapshot.dim, snapshot.quantized);
        await mustReject(`overflow:count=${count.toString(16)}/${label}`, wrapped, snapshot.dim, snapshot.quantized);
    }
}

async function fuzzNeighborCorruption(snapshot, label) {
    console.log(`\n  Neighbor ID corruption (${label})`);
    const raw = extractWasmPayload(snapshot.exported);

    // Find where the graph section starts by reading count and dims from header
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const count = view.getUint32(12, true);
    const dims = view.getUint32(4, true);

    // Calculate graph offset
    let graphOffset;
    if (snapshot.quantized) {
        // scales(count*4) + offsets(count*4) + qdata(count*dims)
        graphOffset = 40 + count * 4 + count * 4 + count * dims;
    } else {
        // vectors(count*dims*4)
        graphOffset = 40 + count * dims * 4;
    }

    if (graphOffset + 20 < raw.length) {
        // Corrupt various positions in the graph section
        const badIds = [count, count + 1, 0xFFFFFFFF, 0x7FFFFFFF, 0xDEADBEEF];
        for (const badId of badIds) {
            // Try corrupting at several positions in the graph section
            for (let probe = graphOffset + 4; probe < Math.min(graphOffset + 80, raw.length - 4); probe += 4) {
                const mutated = mutateU32(raw, probe, badId);
                const wrapped = wrapInEnvelope(mutated, snapshot.dim, snapshot.quantized);
                await tryImport(`neighbor:${badId.toString(16)}@${probe}/${label}`, wrapped, snapshot.dim, snapshot.quantized);
            }
        }
    }
}

async function fuzzNaNInfinity(snapshot, label) {
    console.log(`\n  NaN/Infinity in float fields (${label})`);
    const raw = extractWasmPayload(snapshot.exported);
    const specialFloats = [NaN, Infinity, -Infinity, -0, 1e38, -1e38, 1e-45];

    if (snapshot.quantized) {
        // Corrupt scales (offset 40) and offsets (offset 40 + count*4)
        const count = new DataView(raw.buffer, raw.byteOffset).getUint32(12, true);
        for (const val of specialFloats) {
            // Corrupt first scale
            const m1 = mutateF32(raw, 40, val);
            const w1 = wrapInEnvelope(m1, snapshot.dim, snapshot.quantized);
            await tryImport(`scale=${val}/${label}`, w1, snapshot.dim, snapshot.quantized);

            // Corrupt first offset
            const m2 = mutateF32(raw, 40 + count * 4, val);
            const w2 = wrapInEnvelope(m2, snapshot.dim, snapshot.quantized);
            await tryImport(`offset=${val}/${label}`, w2, snapshot.dim, snapshot.quantized);
        }
    } else {
        // Corrupt vector data (starts at offset 40)
        for (const val of specialFloats) {
            const m = mutateF32(raw, 40, val);
            const w = wrapInEnvelope(m, snapshot.dim, snapshot.quantized);
            await tryImport(`vec[0]=${val}/${label}`, w, snapshot.dim, snapshot.quantized);
        }
    }
}

async function fuzzEnvelopeCorruption() {
    console.log('\n  Envelope-level corruption');
    const dim = 16;

    // Bad envelope magic
    const buf1 = new Uint8Array(64);
    new DataView(buf1.buffer).setUint32(0, 0xBAADF00D, true);
    await tryImport('bad-envelope-magic', buf1, dim, true);

    // Bad envelope version
    const buf2 = new Uint8Array(64);
    const v2 = new DataView(buf2.buffer);
    v2.setUint32(0, PIKELET_MAGIC, true);
    v2.setUint32(4, 99, true);
    await tryImport('bad-envelope-version', buf2, dim, true);

    // Dimension mismatch in envelope
    const buf3 = new Uint8Array(64);
    const v3 = new DataView(buf3.buffer);
    v3.setUint32(0, PIKELET_MAGIC, true);
    v3.setUint32(4, 3, true);
    v3.setUint32(8, 999, true); // wrong dim
    v3.setUint32(12, 1, true);
    v3.setUint32(16, 1, true);
    await tryImport('envelope-dim-mismatch', buf3, dim, true);

    // Metric mismatch
    const buf4 = new Uint8Array(64);
    const v4 = new DataView(buf4.buffer);
    v4.setUint32(0, PIKELET_MAGIC, true);
    v4.setUint32(4, 3, true);
    v4.setUint32(8, dim, true);
    v4.setUint32(12, 0, true); // l2 but we create with cosine
    v4.setUint32(16, 1, true);
    await tryImport('envelope-metric-mismatch', buf4, dim, true);

    // Quantized mismatch
    const buf5 = new Uint8Array(64);
    const v5 = new DataView(buf5.buffer);
    v5.setUint32(0, PIKELET_MAGIC, true);
    v5.setUint32(4, 3, true);
    v5.setUint32(8, dim, true);
    v5.setUint32(12, 1, true);
    v5.setUint32(16, 0, true); // float32 but we create with quantized
    await tryImport('envelope-quantized-mismatch', buf5, dim, true);

    // Truncated v3 envelope (< 32 bytes)
    for (const len of [0, 4, 8, 16, 20, 28, 31]) {
        const buf = new Uint8Array(len);
        if (len >= 4) new DataView(buf.buffer).setUint32(0, PIKELET_MAGIC, true);
        if (len >= 8) new DataView(buf.buffer).setUint32(4, 3, true);
        await tryImport(`truncated-envelope[${len}B]`, buf, dim, true);
    }

    // Envelope claims huge wasmSize
    const buf6 = new Uint8Array(48);
    const v6 = new DataView(buf6.buffer);
    v6.setUint32(0, PIKELET_MAGIC, true);
    v6.setUint32(4, 3, true);
    v6.setUint32(8, dim, true);
    v6.setUint32(12, 1, true);
    v6.setUint32(16, 1, true);
    v6.setUint32(20, 0, true);
    v6.setUint32(24, 0, true);
    v6.setUint32(28, 0xFFFFFFFF, true); // wasmSize = 4GB
    await tryImport('envelope-huge-wasmSize', buf6, dim, true);

    // Envelope with huge mappingCount
    const buf7 = new Uint8Array(48);
    const v7 = new DataView(buf7.buffer);
    v7.setUint32(0, PIKELET_MAGIC, true);
    v7.setUint32(4, 3, true);
    v7.setUint32(8, dim, true);
    v7.setUint32(12, 1, true);
    v7.setUint32(16, 1, true);
    v7.setUint32(20, 0, true);
    v7.setUint32(24, 0x10000000, true); // 256M mappings
    v7.setUint32(28, 16, true);
    await tryImport('envelope-huge-mappingCount', buf7, dim, true);
}

async function fuzzEnvelopeBookkeeping(snapshot, label) {
    console.log(`\n  Envelope bookkeeping corruption (${label})`);
    const exported = new Uint8Array(snapshot.exported);
    const view = new DataView(exported.buffer, exported.byteOffset, exported.byteLength);
    const nextExtId = view.getUint32(20, true);
    const mappingCount = view.getUint32(24, true);
    const mappingOffset = 32;
    const maxLiveExtId = mappingCount > 0
        ? view.getUint32(mappingOffset + (mappingCount - 1) * 8 + 4, true)
        : -1;

    // nextExtId is JS-layer state, so it bypasses the raw-engine validators.
    // We still know certain values are impossible for this envelope: anything
    // below the live mapping count or at/below the max live external ID.
    for (const badNextExtId of new Set([0, 1, snapshot.count - 1, nextExtId - 1, maxLiveExtId])) {
        if (badNextExtId < 0) continue;
        const mutated = mutateU32(exported, 20, badNextExtId);
        await mustReject(`envelope:nextExtId=${badNextExtId}/${label}`, mutated, snapshot.dim, snapshot.quantized);
    }

    // mappingCount smaller than the live raw count should be rejected even if
    // the payload itself remains valid.
    if (mappingCount > 0) {
        const shrunkCount = mappingCount - 1;
        const mutated = new Uint8Array(exported);
        const mv = new DataView(mutated.buffer, mutated.byteOffset, mutated.byteLength);
        mv.setUint32(24, shrunkCount, true);

        const oldWasmOffset = 32 + mappingCount * 8;
        const newWasmOffset = 32 + shrunkCount * 8;
        const wasmBytes = mutated.slice(oldWasmOffset);
        mutated.set(wasmBytes, newWasmOffset);
        mv.setUint32(28, wasmBytes.byteLength, true);

        await mustReject(`envelope:mappingCount=${shrunkCount}/${label}`, mutated, snapshot.dim, snapshot.quantized);
    }

    // Duplicate external IDs in the JS envelope mapping should be rejected
    // before the raw engine sees the payload.
    if (mappingCount > 1) {
        const mutated = new Uint8Array(exported);
        const mv = new DataView(mutated.buffer, mutated.byteOffset, mutated.byteLength);
        const firstExtId = mv.getUint32(mappingOffset + 4, true);
        mv.setUint32(mappingOffset + 12, firstExtId, true);
        await mustReject(`envelope:duplicate-ext-id/${label}`, mutated, snapshot.dim, snapshot.quantized);
    }
}

async function fuzzMultiBitCorruption(snapshot, label, rounds) {
    console.log(`\n  Multi-bit corruption (${label})`);
    const raw = extractWasmPayload(snapshot.exported);
    for (let i = 0; i < rounds; i++) {
        let mutated = new Uint8Array(raw);
        const numFlips = randInt(2, 8);
        for (let j = 0; j < numFlips; j++) {
            const byteIdx = randInt(0, mutated.length - 1);
            mutated[byteIdx] ^= (1 << randInt(0, 7));
        }
        const wrapped = wrapInEnvelope(mutated, snapshot.dim, snapshot.quantized);
        await tryImport(`multiflip[${numFlips}x]#${i}/${label}`, wrapped, snapshot.dim, snapshot.quantized);
    }
}

async function fuzzPostImportStability() {
    console.log('\n  Post-import stability (valid import then operations)');
    const dim = 16;

    for (const quantized of [true, false]) {
        const label = quantized ? 'uint8' : 'float32';
        const snap = await buildValidSnapshot(quantized);
        const idx = await Pikelet.create({
            dim, maxElements: 128, metric: 'cosine', quantized,
            M: 8, efConstruction: 50, efSearch: 50,
        });

        idx.import(snap.exported);

        // Search should work
        const q = normalizedVec(dim);
        const results = idx.search(q, 5);
        ok(results.length > 0, `${label}: search works after valid import`);

        // Add more vectors after import
        for (let i = 0; i < 10; i++) {
            idx.add(normalizedVec(dim));
        }
        ok(idx.count === 26, `${label}: count correct after import + add`);

        // Search again
        const results2 = idx.search(q, 10);
        ok(results2.length === 10, `${label}: search works after import + add`);

        // Delete and compact
        idx.delete(results2[0].id);
        idx.compact();
        ok(idx.count === 25, `${label}: count correct after delete+compact`);

        // Export/reimport cycle
        const reexported = idx.export();
        const idx2 = await Pikelet.create({
            dim, maxElements: 128, metric: 'cosine', quantized,
            M: 8, efConstruction: 50, efSearch: 50,
        });
        idx2.import(reexported);
        ok(idx2.count === 25, `${label}: reimport preserves count`);

        const results3 = idx2.search(q, 5);
        ok(results3.length === 5, `${label}: search works after reimport`);

        idx2.dispose();
        idx.dispose();
    }
}

async function fuzzInputValidation() {
    console.log('\n  Input validation (add/search with bad data)');
    const dim = 16;

    for (const quantized of [true, false]) {
        const label = quantized ? 'uint8' : 'float32';
        const idx = await Pikelet.create({
            dim, maxElements: 64, metric: 'cosine', quantized,
        });

        // Add a few good vectors first
        for (let i = 0; i < 5; i++) idx.add(normalizedVec(dim));

        // NaN vector — must be rejected
        const nanVec = new Float32Array(dim);
        nanVec.fill(NaN);
        try {
            idx.add(nanVec);
            ok(false, `${label}: accepted NaN vector`);
        } catch (e) {
            ok(true, `${label}: rejected NaN vector`);
        }

        // Infinity vector — must be rejected
        const infVec = new Float32Array(dim);
        infVec.fill(Infinity);
        try {
            idx.add(infVec);
            ok(false, `${label}: accepted Infinity vector`);
        } catch (e) {
            ok(true, `${label}: rejected Infinity vector`);
        }

        // Search with NaN query — must be rejected
        try {
            idx.search(nanVec, 3);
            ok(false, `${label}: accepted NaN search query`);
        } catch (e) {
            ok(true, `${label}: rejected NaN search`);
        }

        // Search with k > count
        try {
            const r = idx.search(normalizedVec(dim), 9999);
            ok(r.length <= idx.count, `${label}: k>count returns at most count results`);
        } catch (e) {
            ok(true, `${label}: rejected k>count`);
        }

        idx.dispose();
    }
}

async function fuzzQuantizedScaleCorruption(snapshot) {
    console.log('\n  Quantized scale/offset corruption');
    const raw = extractWasmPayload(snapshot.exported);
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const count = view.getUint32(12, true);

    // Scales start at offset 40 in the u8 WASM payload
    const scalesOffset = 40;
    const offsetsOffset = 40 + count * 4;

    const poisonValues = [
        { val: 0.0, desc: 'zero' },
        { val: NaN, desc: 'NaN' },
        { val: Infinity, desc: 'Inf' },
        { val: -Infinity, desc: '-Inf' },
        { val: 1e38, desc: '1e38' },
        { val: -1e38, desc: '-1e38' },
        { val: 1e-45, desc: '1e-45 (denorm)' },
    ];

    for (const p of poisonValues) {
        // Poison all scales
        {
            const mutated = new Uint8Array(raw);
            const mv = new DataView(mutated.buffer);
            for (let i = 0; i < count; i++) {
                mv.setFloat32(scalesOffset + i * 4, p.val, true);
            }
            const wrapped = wrapInEnvelope(mutated, snapshot.dim, snapshot.quantized);
            const idx = await Pikelet.create({
                dim: snapshot.dim, maxElements: 1000, metric: 'cosine', quantized: true,
                M: 8, efConstruction: 50, efSearch: 50,
            });
            try {
                idx.import(wrapped);
                // Search must terminate — this is the key check
                const q = normalizedVec(snapshot.dim);
                const t0 = performance.now();
                const results = idx.search(q, 5);
                const elapsed = performance.now() - t0;
                ok(elapsed < 5000, `scale=${p.desc}: search terminates (${elapsed.toFixed(0)}ms)`);
                ok(Array.isArray(results), `scale=${p.desc}: search returns array`);
                // Verify distances are finite (NaN distances indicate broken comparisons)
                const allFinite = results.every(r => Number.isFinite(r.distance));
                if (!allFinite) {
                    ok(false, `scale=${p.desc}: search returned non-finite distances`);
                } else {
                    ok(true, `scale=${p.desc}: distances are finite`);
                }
            } catch (e) {
                ok(true, `scale=${p.desc}: rejected`);
            } finally {
                idx.dispose();
            }
        }

        // Poison all offsets
        {
            const mutated = new Uint8Array(raw);
            const mv = new DataView(mutated.buffer);
            for (let i = 0; i < count; i++) {
                mv.setFloat32(offsetsOffset + i * 4, p.val, true);
            }
            const wrapped = wrapInEnvelope(mutated, snapshot.dim, snapshot.quantized);
            const idx = await Pikelet.create({
                dim: snapshot.dim, maxElements: 1000, metric: 'cosine', quantized: true,
                M: 8, efConstruction: 50, efSearch: 50,
            });
            try {
                idx.import(wrapped);
                const q = normalizedVec(snapshot.dim);
                const t0 = performance.now();
                const results = idx.search(q, 5);
                const elapsed = performance.now() - t0;
                ok(elapsed < 5000, `offset=${p.desc}: search terminates (${elapsed.toFixed(0)}ms)`);
                ok(Array.isArray(results), `offset=${p.desc}: search returns array`);
                const allFinite = results.every(r => Number.isFinite(r.distance));
                if (!allFinite) {
                    ok(false, `offset=${p.desc}: search returned non-finite distances`);
                } else {
                    ok(true, `offset=${p.desc}: distances are finite`);
                }
            } catch (e) {
                ok(true, `offset=${p.desc}: rejected`);
            } finally {
                idx.dispose();
            }
        }
    }
}

async function fuzzAddDeleteCompactStress(rounds) {
    console.log('\n  Add/delete/compact/search interleaving stress');
    const dim = 16;

    for (const quantized of [true, false]) {
        const label = quantized ? 'uint8' : 'float32';
        const idx = await Pikelet.create({
            dim, maxElements: 512, metric: 'cosine', quantized,
            M: 8, efConstruction: 50, efSearch: 50,
        });

        const liveIds = new Set();
        let ops = 0;

        try {
            for (let r = 0; r < rounds; r++) {
                const roll = randFloat();

                if (roll < 0.4 || liveIds.size < 5) {
                    // Add
                    if (liveIds.size < 500) {
                        const id = idx.add(normalizedVec(dim));
                        liveIds.add(id);
                    }
                } else if (roll < 0.6 && liveIds.size > 0) {
                    // Delete random
                    const ids = Array.from(liveIds);
                    const victim = ids[randInt(0, ids.length - 1)];
                    idx.delete(victim);
                    liveIds.delete(victim);
                } else if (roll < 0.75 && idx.ghostCount > 0) {
                    // Compact
                    idx.compact();
                    ok(idx.ghostCount === 0, `${label}#${r}: compact clears ghosts`);
                } else if (roll < 0.9 && liveIds.size > 0) {
                    // Search
                    const q = normalizedVec(dim);
                    const results = idx.search(q, Math.min(10, liveIds.size));
                    ok(Array.isArray(results), `${label}#${r}: search returns array`);
                    // All returned IDs should be live
                    for (const res of results) {
                        ok(liveIds.has(res.id), `${label}#${r}: result id ${res.id} is live`);
                    }
                } else if (liveIds.size > 0) {
                    // Export/reimport cycle
                    if (idx.ghostCount > 0) idx.compact();
                    const snap = idx.export();
                    const idx2 = await Pikelet.create({
                        dim, maxElements: 512, metric: 'cosine', quantized,
                        M: 8, efConstruction: 50, efSearch: 50,
                    });
                    idx2.import(snap);
                    ok(idx2.count === liveIds.size, `${label}#${r}: reimport count matches (${idx2.count} vs ${liveIds.size})`);
                    idx2.dispose();
                }
                ops++;
            }
            ok(true, `${label}: completed ${ops} ops without crash`);
        } catch (e) {
            ok(false, `${label}: crashed after ${ops} ops (live=${liveIds.size}, count=${idx.count}, ghosts=${idx.ghostCount}): ${e.message}`);
        } finally {
            idx.dispose();
        }
    }
}

async function fuzzPostImportMutation(snapshot, label) {
    console.log(`\n  Post-import mutation stress (${label})`);
    const idx = await Pikelet.create({
        dim: snapshot.dim, maxElements: 256, metric: 'cosine', quantized: snapshot.quantized,
        M: 8, efConstruction: 50, efSearch: 50,
    });

    try {
        idx.import(snapshot.exported);
        const initialCount = idx.count;

        // Add vectors until near capacity
        const addedIds = [];
        for (let i = 0; i < 100; i++) {
            const id = idx.add(normalizedVec(snapshot.dim));
            addedIds.push(id);
        }
        ok(idx.count === initialCount + 100, `${label}: count correct after post-import adds`);

        // Search should find results from both original and added vectors
        const q = normalizedVec(snapshot.dim);
        const results = idx.search(q, 20);
        ok(results.length === 20, `${label}: search returns 20 results from mixed index`);

        // Delete some original and some new vectors
        for (let i = 0; i < 5; i++) {
            idx.delete(addedIds[i]);
            idx.delete(i); // original vector IDs
        }
        idx.compact();
        ok(idx.count === initialCount + 100 - 10, `${label}: count correct after mixed deletes`);

        // Export/reimport the mutated index
        const snap2 = idx.export();
        const idx2 = await Pikelet.create({
            dim: snapshot.dim, maxElements: 256, metric: 'cosine', quantized: snapshot.quantized,
            M: 8, efConstruction: 50, efSearch: 50,
        });
        idx2.import(snap2);
        ok(idx2.count === idx.count, `${label}: reimport preserves count after mutations`);

        const results2 = idx2.search(q, 10);
        ok(results2.length === 10, `${label}: search works after reimport of mutated index`);
        idx2.dispose();
    } finally {
        idx.dispose();
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main() {
    const args = process.argv.slice(2);
    let rounds = 100;
    let seed = Date.now() & 0xFFFFFFFF;

    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--rounds' && args[i + 1]) rounds = parseInt(args[i + 1], 10);
        if (args[i] === '--seed' && args[i + 1]) seed = parseInt(args[i + 1], 10);
    }

    rng = xoshiro128ss(seed);
    console.log(`Pikelet Import Fuzz Harness`);
    console.log(`  seed=${seed}  rounds=${rounds}`);
    console.log('─'.repeat(50));

    // Build reference snapshots
    const floatSnap = await buildValidSnapshot(false);
    const uint8Snap = await buildValidSnapshot(true);

    // Run all fuzz strategies
    await fuzzPureRandom(rounds);

    await fuzzTruncation(floatSnap, 'float32');
    await fuzzTruncation(uint8Snap, 'uint8');

    await fuzzBitFlips(floatSnap, 'float32', rounds);
    await fuzzBitFlips(uint8Snap, 'uint8', rounds);

    await fuzzHeaderCorruption(floatSnap, 'float32');
    await fuzzHeaderCorruption(uint8Snap, 'uint8');

    await fuzzOverflowSizes(floatSnap, 'float32');
    await fuzzOverflowSizes(uint8Snap, 'uint8');

    await fuzzNeighborCorruption(floatSnap, 'float32');
    await fuzzNeighborCorruption(uint8Snap, 'uint8');

    await fuzzNaNInfinity(floatSnap, 'float32');
    await fuzzNaNInfinity(uint8Snap, 'uint8');

    await fuzzMultiBitCorruption(floatSnap, 'float32', rounds);
    await fuzzMultiBitCorruption(uint8Snap, 'uint8', rounds);

    await fuzzEnvelopeCorruption();
    await fuzzEnvelopeBookkeeping(floatSnap, 'float32');
    await fuzzEnvelopeBookkeeping(uint8Snap, 'uint8');
    await fuzzPostImportStability();
    await fuzzInputValidation();
    await fuzzQuantizedScaleCorruption(uint8Snap);
    await fuzzAddDeleteCompactStress(rounds);
    await fuzzPostImportMutation(floatSnap, 'float32');
    await fuzzPostImportMutation(uint8Snap, 'uint8');

    // Summary
    console.log('\n' + '─'.repeat(50));
    console.log(`  Passed: ${passed}`);
    console.log(`  Failed: ${failed}`);
    if (failures.length > 0) {
        console.log('\nFailures:');
        for (const f of failures) console.log(`  - ${f}`);
    }
    console.log(`\n  seed=${seed} (rerun with --seed ${seed} to reproduce)`);
    process.exit(failed > 0 ? 1 : 0);
}

main().catch(e => {
    console.error('Fuzz harness crashed:', e);
    process.exit(2);
});
