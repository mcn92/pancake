// Shared .pancake container assembly (spec/COMPLETE_PROFILE.md): canonical
// JSON, header + manifest + segment table layout, and streaming file
// writes so multi-hundred-MB segments never need one contiguous output
// buffer. Used by compile.mjs (docs) and compile-wiki.mjs (wiki pack).

import fs from 'node:fs';
import crypto from 'node:crypto';

export const MAGIC = 0x31465350; // "PSF1"
export const HEADER_BYTES = 64;
export const TABLE_ENTRY_BYTES = 48;
export const KINDS = { index: 1, corpus: 2, 'query-interp': 3, evaluation: 4 };
export const KIND_NAMES = Object.fromEntries(Object.entries(KINDS).map(([name, id]) => [id, name]));

export const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest();
export const align16 = (n) => Math.ceil(n / 16) * 16;

// Canonical JSON: recursively sorted keys, no insignificant whitespace. The
// serialized bytes in the file are the bytes the identity digest covers.
export function canonicalJson(value) {
    if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
    if (value && typeof value === 'object') {
        const keys = Object.keys(value).sort();
        return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalJson(value[k])}`).join(',')}}`;
    }
    return JSON.stringify(value);
}

// Build the query-interp segment (spec 3.6): shared version, kind, then
// encoder and calibration blocks.
export function buildQueryInterpSegment(kind, encoderBytes, calibrationBytes) {
    const out = Buffer.alloc(16 + encoderBytes.length + calibrationBytes.length);
    out.writeUInt32LE(1, 0);
    out.writeUInt32LE(kind, 4);
    out.writeUInt32LE(encoderBytes.length, 8);
    out.writeUInt32LE(calibrationBytes.length, 12);
    encoderBytes.copy(out, 16);
    calibrationBytes.copy(out, 16 + encoderBytes.length);
    return out;
}

// Build the corpus segment (spec 3.5) from an array of record buffers.
export function buildCorpusSegmentFromBuffers(records) {
    const count = records.length;
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

/**
 * Assemble and write a .pancake. `segments` is an ordered array of
 * { kind, bytes } with kind from KINDS; `manifestFields` is everything the
 * manifest carries except `segments`, which is derived here.
 * Returns { outPath, fileBytes, identity, manifest }.
 */
export function assemblePancakeFile(manifestFields, segments, outPath) {
    const manifest = {
        ...manifestFields,
        segments: segments.map((s) => ({
            kind: s.kind,
            sha256: sha256(s.bytes).toString('hex'),
            bytes: s.bytes.length,
        })),
    };
    const manifestBytes = Buffer.from(canonicalJson(manifest), 'utf8');
    const identity = sha256(manifestBytes);

    const tableOffset = HEADER_BYTES + manifestBytes.length;
    let cursor = align16(tableOffset + segments.length * TABLE_ENTRY_BYTES);
    const table = Buffer.alloc(segments.length * TABLE_ENTRY_BYTES);
    const placed = [];
    for (let i = 0; i < segments.length; i++) {
        const entry = i * TABLE_ENTRY_BYTES;
        table.writeUInt32LE(KINDS[segments[i].kind], entry);
        table.writeBigUInt64LE(BigInt(cursor), entry + 8);
        table.writeBigUInt64LE(BigInt(segments[i].bytes.length), entry + 16);
        placed.push({ offset: cursor, bytes: segments[i].bytes });
        cursor = align16(cursor + segments[i].bytes.length);
    }
    const last = placed[placed.length - 1];
    const fileBytes = last.offset + last.bytes.length;

    const header = Buffer.alloc(HEADER_BYTES);
    header.writeUInt32LE(MAGIC, 0);
    header.writeUInt32LE(1, 4);
    header.writeUInt32LE(manifestBytes.length, 8);
    header.writeUInt32LE(segments.length, 12);
    header.writeBigUInt64LE(BigInt(fileBytes), 16);
    identity.copy(header, 24);

    // Streamed writes: segments can be hundreds of MB, so never concat.
    const fd = fs.openSync(outPath, 'w');
    try {
        fs.writeSync(fd, header, 0, header.length, 0);
        fs.writeSync(fd, manifestBytes, 0, manifestBytes.length, HEADER_BYTES);
        fs.writeSync(fd, table, 0, table.length, tableOffset);
        for (const { offset, bytes } of placed) {
            fs.writeSync(fd, bytes, 0, bytes.length, offset);
        }
        fs.ftruncateSync(fd, fileBytes);
    } finally {
        fs.closeSync(fd);
    }
    return { outPath, fileBytes, identity: identity.toString('hex'), manifest };
}
