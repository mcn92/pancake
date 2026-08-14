// The one-file reader: open a .pancake (spec/COMPLETE_PROFILE.md) and query
// it. Environment-neutral — the same module runs in Node (file path or
// range source) and the browser (HTTP range source, via a bundler for the
// CJS sketch-reader dependency). The query path is pure JS: encoder, sketch
// reference scan, hydration, and calibration all run without the WASM
// engine, which is a compile-time dependency only.
//
//   const search = await openPancakeFile('pancake-docs.pancake');   // Node
//   const search = await openPancakeFile(httpRangeSource(url));     // browser
//   const out = await search.query('how do workers restore snapshots');

import { loadStudentModel, embedTextWithStudent } from '../03-edge-docs-search/student-embedder.mjs';
import { computeMatchQuality, computePreSearchAbstention } from './abstention.mjs';
import { PancakeSketchArtifact } from '../../pancake-artifact.js';

const MAGIC = 0x31465350; // "PSF1"
const HEADER_BYTES = 64;
const TABLE_ENTRY_BYTES = 48;
const KIND_NAMES = { 1: 'index', 2: 'corpus', 3: 'query-interp', 4: 'evaluation' };

const decoder = new TextDecoder();

function toHex(bytes) {
    let out = '';
    for (const b of bytes) out += b.toString(16).padStart(2, '0');
    return out;
}

// SHA-256 via WebCrypto (browser, workerd, Node 18+); node:crypto fallback
// for older Node. Fails closed: verification is not optional in this reader.
async function sha256hex(bytes) {
    if (globalThis.crypto?.subtle) {
        return toHex(new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', bytes)));
    }
    const { createHash } = await import('node:crypto');
    return createHash('sha256').update(bytes).digest('hex');
}

async function fileSource(filePath) {
    const fs = await import('node:fs');
    const fd = fs.openSync(filePath, 'r');
    const size = fs.fstatSync(fd).size;
    return {
        size,
        async read(offset, length) {
            const buffer = new Uint8Array(length);
            const bytesRead = fs.readSync(fd, buffer, 0, length, offset);
            return buffer.subarray(0, bytesRead);
        },
        async close() { fs.closeSync(fd); },
    };
}

// A source restricted to one segment's window, for handing a container
// region to a reader that expects offset 0 = its own start (the embedded
// sketch artifact).
function windowSource(source, offset, length) {
    return {
        size: length,
        async read(off, len) { return source.read(offset + off, len); },
        async close() {},
    };
}

const viewOf = (bytes) => new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

/**
 * Open a .pancake complete artifact from a file path (Node) or a
 * { read(offset, length), size? } range source (any runtime).
 * Returns { query(text, {k}), info(), evaluation(), close() }.
 */
export async function openPancakeFile(input) {
    const source = typeof input === 'string' ? await fileSource(input) : input;
    const owned = typeof input === 'string';
    try {
        const header = new Uint8Array(await source.read(0, HEADER_BYTES));
        const headerView = viewOf(header);
        if (header.length < HEADER_BYTES || headerView.getUint32(0, true) !== MAGIC) {
            throw new Error('not a .pancake file (bad magic)');
        }
        const version = headerView.getUint32(4, true);
        if (version !== 1) throw new Error(`unsupported .pancake format version ${version}`);
        const manifestBytes = headerView.getUint32(8, true);
        const segmentCount = headerView.getUint32(12, true);
        const fileBytes = Number(headerView.getBigUint64(16, true));
        if (manifestBytes < 2 || manifestBytes > 16 * 1024 * 1024 || segmentCount < 1 || segmentCount > 64) {
            throw new Error('.pancake header is implausible');
        }
        const identity = toHex(header.subarray(24, 56));

        const manifestBuf = new Uint8Array(await source.read(HEADER_BYTES, manifestBytes));
        if (await sha256hex(manifestBuf) !== identity) {
            throw new Error('.pancake manifest failed identity verification');
        }
        const manifest = JSON.parse(decoder.decode(manifestBuf));
        if (manifest.profile !== 'pancake-complete-v1') {
            throw new Error(`unsupported profile ${manifest.profile}`);
        }

        const table = new Uint8Array(await source.read(HEADER_BYTES + manifestBytes, segmentCount * TABLE_ENTRY_BYTES));
        const tableView = viewOf(table);
        const segments = new Map();
        for (let i = 0; i < segmentCount; i++) {
            const entry = i * TABLE_ENTRY_BYTES;
            const kind = KIND_NAMES[tableView.getUint32(entry, true)];
            const offset = Number(tableView.getBigUint64(entry + 8, true));
            const length = Number(tableView.getBigUint64(entry + 16, true));
            const declared = manifest.segments[i];
            if (!declared || declared.kind !== kind || declared.bytes !== length
                || offset % 16 !== 0 || offset + length > fileBytes) {
                throw new Error(`.pancake segment table disagrees with manifest at entry ${i}`);
            }
            segments.set(kind, { offset, length, sha256: declared.sha256 });
        }
        for (const required of ['index', 'corpus', 'query-interp']) {
            if (!segments.has(required)) throw new Error(`.pancake is missing the ${required} segment`);
        }

        // Query interpretation: one eager read, digest-verified, then split
        // into encoder + calibration (one unit, shared version).
        const qi = segments.get('query-interp');
        const qiBytes = new Uint8Array(await source.read(qi.offset, qi.length));
        if (await sha256hex(qiBytes) !== qi.sha256) {
            throw new Error('.pancake query-interp segment failed hash verification');
        }
        const qiView = viewOf(qiBytes);
        const encoderLen = qiView.getUint32(4, true);
        const calibrationLen = qiView.getUint32(8, true);
        if (12 + encoderLen + calibrationLen !== qi.length) {
            throw new Error('.pancake query-interp segment layout is inconsistent');
        }
        const student = loadStudentModel(qiBytes.subarray(12, 12 + encoderLen));
        const abstention = JSON.parse(decoder.decode(qiBytes.subarray(12 + encoderLen)));

        // Index: the embedded sketch artifact, opened by the existing reader
        // against a segment-windowed source (resident hash verified there).
        const idx = segments.get('index');
        const sketch = await PancakeSketchArtifact.open(windowSource(source, idx.offset, idx.length));
        if (sketch.count !== manifest.corpus.records) {
            throw new Error(`.pancake index count ${sketch.count} != corpus records ${manifest.corpus.records}`);
        }

        // Corpus: resident offsets table; records hydrate by range read.
        const corpus = segments.get('corpus');
        const countBuf = new Uint8Array(await source.read(corpus.offset, 4));
        const recordCount = viewOf(countBuf).getUint32(0, true);
        if (recordCount !== manifest.corpus.records) {
            throw new Error('.pancake corpus count disagrees with manifest');
        }
        const offsetsBuf = new Uint8Array(await source.read(corpus.offset + 4, 8 * (recordCount + 1)));
        const offsetsView = viewOf(offsetsBuf);
        const recordOffsets = new Array(recordCount + 1);
        for (let i = 0; i <= recordCount; i++) {
            recordOffsets[i] = Number(offsetsView.getBigUint64(8 * i, true));
            if (recordOffsets[i] > corpus.length || (i > 0 && recordOffsets[i] < recordOffsets[i - 1])) {
                throw new Error('.pancake corpus offsets are inconsistent');
            }
        }

        const hydrate = async (id) => {
            const start = recordOffsets[id];
            const end = recordOffsets[id + 1];
            const bytes = new Uint8Array(await source.read(corpus.offset + start, end - start));
            return JSON.parse(decoder.decode(bytes));
        };

        return {
            info() {
                return {
                    identity,
                    records: recordCount,
                    dim: manifest.dim,
                    metric: manifest.metric,
                    fileBytes,
                    residentBytes: sketch.residentBytes + 8 * (recordCount + 1),
                    residentVerified: sketch.stats().residentVerified,
                    sampleQueries: manifest.sampleQueries || [],
                };
            },

            async query(text, options = {}) {
                const k = Number.isInteger(options.k) && options.k > 0 ? options.k : 5;
                const trimmed = String(text || '').trim();
                if (!trimmed) throw new Error('query text is required');
                const embedded = embedTextWithStudent(trimmed, student);
                const pre = computePreSearchAbstention(embedded, abstention);
                let hits = [];
                if (!pre) hits = (await sketch.search(embedded.vector, k)).results;
                const quality = pre || computeMatchQuality(hits, embedded, abstention);
                const returned = quality.match_quality === 'none' ? [] : hits;
                const results = [];
                for (const hit of returned) {
                    const record = await hydrate(hit.id);
                    results.push({ id: hit.id, distance: hit.distance, ...record });
                }
                return { matchQuality: quality.match_quality, confidence: quality.confidence, results };
            },

            async evaluation() {
                const ev = segments.get('evaluation');
                if (!ev) return null;
                const bytes = new Uint8Array(await source.read(ev.offset, ev.length));
                if (await sha256hex(bytes) !== ev.sha256) {
                    throw new Error('.pancake evaluation segment failed hash verification');
                }
                return JSON.parse(decoder.decode(bytes));
            },

            async close() {
                await sketch.close();
                if (owned) await source.close();
            },
        };
    } catch (err) {
        if (owned && source.close) await source.close().catch(() => {});
        throw err;
    }
}
