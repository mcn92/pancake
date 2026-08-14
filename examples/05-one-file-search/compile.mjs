#!/usr/bin/env node
// Compile the five Search Artifact components into one .pancake file per
// spec/COMPLETE_PROFILE.md (Draft 1), and inspect/verify the result:
//
//   node compile.mjs                 # compile 03's assets -> pancake-docs.pancake
//   node compile.mjs --inspect pancake-docs.pancake
//
// Layout: 64 B header | canonical-JSON manifest | segment table (48 B/entry)
// | segments (index = embedded .pancake-sketch, corpus = offsets + JSON
// records, query-interp = encoder + calibration, evaluation = JSON), all
// 16-byte aligned. Identity = SHA-256 of the manifest bytes.

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { assertIdentityMapping, docsAssetPaths } from './search-reader.mjs';

const require = createRequire(import.meta.url);
const Pancake = require('../../pancake.js');

const MAGIC = 0x31465350; // "PSF1"
const HEADER_BYTES = 64;
const TABLE_ENTRY_BYTES = 48;
const KINDS = { index: 1, corpus: 2, 'query-interp': 3, evaluation: 4 };
const KIND_NAMES = Object.fromEntries(Object.entries(KINDS).map(([name, id]) => [id, name]));

const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest();
const align16 = (n) => Math.ceil(n / 16) * 16;

// Canonical JSON: recursively sorted keys, no insignificant whitespace. The
// serialized bytes in the file are the bytes the identity digest covers.
function canonicalJson(value) {
    if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
    if (value && typeof value === 'object') {
        const keys = Object.keys(value).sort();
        return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalJson(value[k])}`).join(',')}}`;
    }
    return JSON.stringify(value);
}

function buildCorpusSegment(corpusById) {
    const count = corpusById.size;
    const records = [];
    for (let id = 0; id < count; id++) {
        const r = corpusById.get(id);
        if (!r) throw new Error(`corpus record ${id} missing — ids must be dense [0, count)`);
        records.push(Buffer.from(JSON.stringify({
            title: r.title, text: r.text, preview: r.preview,
            sourcePath: r.sourcePath, anchor: r.anchor,
        }), 'utf8'));
    }
    const prefix = 4 + 8 * (count + 1);
    const total = prefix + records.reduce((sum, b) => sum + b.length, 0);
    const out = Buffer.alloc(total);
    out.writeUInt32LE(count, 0);
    let cursor = prefix;
    for (let id = 0; id < count; id++) {
        out.writeBigUInt64LE(BigInt(cursor), 4 + 8 * id);
        records[id].copy(out, cursor);
        cursor += records[id].length;
    }
    out.writeBigUInt64LE(BigInt(cursor), 4 + 8 * count);
    return out;
}

function buildQueryInterpSegment(encoderBytes, calibrationBytes) {
    const out = Buffer.alloc(12 + encoderBytes.length + calibrationBytes.length);
    out.writeUInt32LE(1, 0); // shared encoder+calibration version
    out.writeUInt32LE(encoderBytes.length, 4);
    out.writeUInt32LE(calibrationBytes.length, 8);
    encoderBytes.copy(out, 12);
    calibrationBytes.copy(out, 12 + encoderBytes.length);
    return out;
}

// The full student evaluation is per-row and large; the evaluation segment
// carries the golden queries plus the evaluation's scalar/summary fields
// (arrays over 100 entries dropped, noted by key).
function buildEvaluationSegment(goldenQueries, studentEvaluation) {
    const summary = {};
    const dropped = [];
    for (const [key, value] of Object.entries(studentEvaluation)) {
        if (Array.isArray(value) && value.length > 100) dropped.push(key);
        else summary[key] = value;
    }
    return Buffer.from(canonicalJson({
        goldenQueries,
        student: summary,
        studentFieldsOmitted: dropped,
    }), 'utf8');
}

function compile(paths, outPath) {
    const sourceManifest = JSON.parse(fs.readFileSync(paths.manifestPath, 'utf8'));
    const corpusRaw = JSON.parse(fs.readFileSync(paths.corpusPath, 'utf8'));
    const corpusById = new Map(Object.entries(corpusRaw).map(([id, r]) => [Number(id), r]));
    const snapshotBytes = fs.readFileSync(paths.indexPath);
    assertIdentityMapping(snapshotBytes);

    const { bytes: sketchBytes } = Pancake.buildSketchArtifactBytes(snapshotBytes, {
        recommendedRerank: sourceManifest.efSearch || 120,
    });
    const goldenQueries = JSON.parse(fs.readFileSync(path.join(
        path.dirname(paths.manifestPath), '..', 'fixtures', 'abstention-golden.json'), 'utf8'));

    const segments = [
        { kind: 'index', bytes: Buffer.from(sketchBytes) },
        { kind: 'corpus', bytes: buildCorpusSegment(corpusById) },
        {
            kind: 'query-interp',
            bytes: buildQueryInterpSegment(
                fs.readFileSync(paths.encoderPath),
                fs.readFileSync(paths.calibrationPath)
            ),
        },
        {
            kind: 'evaluation',
            bytes: buildEvaluationSegment(goldenQueries,
                JSON.parse(fs.readFileSync(paths.manifestPath.replace('docs-manifest', 'docs-student-evaluation'), 'utf8'))),
        },
    ];

    const manifest = {
        profile: 'pancake-complete-v1',
        corpus: { records: corpusById.size, provenance: null },
        dim: sourceManifest.dim,
        metric: sourceManifest.metric,
        encoder: sourceManifest.encoder,
        segments: segments.map((s) => ({
            kind: s.kind,
            sha256: sha256(s.bytes).toString('hex'),
            bytes: s.bytes.length,
        })),
        sampleQueries: sourceManifest.sampleQueries || [],
    };
    const manifestBytes = Buffer.from(canonicalJson(manifest), 'utf8');

    const tableOffset = HEADER_BYTES + manifestBytes.length;
    let cursor = align16(tableOffset + segments.length * TABLE_ENTRY_BYTES);
    const table = Buffer.alloc(segments.length * TABLE_ENTRY_BYTES);
    for (let i = 0; i < segments.length; i++) {
        const entry = i * TABLE_ENTRY_BYTES;
        table.writeUInt32LE(KINDS[segments[i].kind], entry);
        table.writeBigUInt64LE(BigInt(cursor), entry + 8);
        table.writeBigUInt64LE(BigInt(segments[i].bytes.length), entry + 16);
        segments[i].offset = cursor;
        cursor = align16(cursor + segments[i].bytes.length);
    }
    const fileBytes = segments.length
        ? segments[segments.length - 1].offset + segments[segments.length - 1].bytes.length
        : tableOffset;

    const out = Buffer.alloc(fileBytes);
    out.writeUInt32LE(MAGIC, 0);
    out.writeUInt32LE(1, 4);
    out.writeUInt32LE(manifestBytes.length, 8);
    out.writeUInt32LE(segments.length, 12);
    out.writeBigUInt64LE(BigInt(fileBytes), 16);
    sha256(manifestBytes).copy(out, 24);
    manifestBytes.copy(out, HEADER_BYTES);
    table.copy(out, tableOffset);
    for (const segment of segments) segment.bytes.copy(out, segment.offset);

    fs.writeFileSync(outPath, out);
    return { outPath, fileBytes, identity: sha256(manifestBytes).toString('hex'), manifest };
}

function inspect(filePath) {
    const bytes = fs.readFileSync(filePath);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (view.getUint32(0, true) !== MAGIC) throw new Error('not a .pancake file (bad magic)');
    if (view.getUint32(4, true) !== 1) throw new Error(`unsupported format version ${view.getUint32(4, true)}`);
    const manifestBytes = view.getUint32(8, true);
    const segmentCount = view.getUint32(12, true);
    const fileBytes = Number(view.getBigUint64(16, true));
    if (fileBytes !== bytes.length) throw new Error(`fileBytes ${fileBytes} != actual ${bytes.length}`);

    const manifestBuf = bytes.subarray(HEADER_BYTES, HEADER_BYTES + manifestBytes);
    const manifestOk = sha256(manifestBuf).equals(bytes.subarray(24, 56));
    const manifest = JSON.parse(manifestBuf.toString('utf8'));

    console.log(`${path.basename(filePath)}: ${(fileBytes / 1048576).toFixed(2)} MiB, profile ${manifest.profile}`);
    console.log(`identity (manifest sha256): ${Buffer.from(bytes.subarray(24, 56)).toString('hex')}`);
    console.log(`manifest digest verifies: ${manifestOk}`);
    console.log(`corpus: ${manifest.corpus.records} records, ${manifest.dim}D ${manifest.metric}`);
    console.log('segments:');
    const tableOffset = HEADER_BYTES + manifestBytes;
    for (let i = 0; i < segmentCount; i++) {
        const entry = tableOffset + i * TABLE_ENTRY_BYTES;
        const kind = view.getUint32(entry, true);
        const offset = Number(view.getBigUint64(entry + 8, true));
        const length = Number(view.getBigUint64(entry + 16, true));
        const declared = manifest.segments[i];
        const digestOk = sha256(bytes.subarray(offset, offset + length)).toString('hex') === declared.sha256;
        const aligned = offset % 16 === 0;
        console.log(`  ${String(KIND_NAMES[kind] || kind).padEnd(13)} offset ${String(offset).padStart(9)}  `
            + `${(length / 1024).toFixed(1).padStart(9)} KiB  digest ${digestOk ? 'ok' : 'MISMATCH'}  aligned ${aligned}`);
        if (!digestOk || !aligned || declared.kind !== KIND_NAMES[kind]) process.exitCode = 1;
    }
    if (!manifestOk) process.exitCode = 1;
}

const args = process.argv.slice(2);
if (args[0] === '--inspect') {
    inspect(args[1] || path.join(path.dirname(new URL(import.meta.url).pathname), 'pancake-docs.pancake'));
} else {
    const outPath = args[0] || path.join(path.dirname(new URL(import.meta.url).pathname), 'pancake-docs.pancake');
    const result = compile(docsAssetPaths(), outPath);
    console.log(`compiled ${result.outPath}`);
    console.log(`  ${(result.fileBytes / 1048576).toFixed(2)} MiB, identity ${result.identity.slice(0, 16)}...`);
    for (const s of result.manifest.segments) {
        console.log(`  ${s.kind.padEnd(13)} ${(s.bytes / 1024).toFixed(1).padStart(9)} KiB  ${s.sha256.slice(0, 12)}...`);
    }
}
