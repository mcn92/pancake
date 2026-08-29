// The one-file reader: open a .pancake (spec/COMPLETE_PROFILE.md) and query
// it. Environment-neutral — the same module runs in Node (file path or
// range source) and the browser (HTTP range source, via a bundler for the
// CJS sketch-reader dependency). The query path is pure JS: encoder, sketch
// reference scan, hydration, and calibration all run without the core HNSW
// engine; kind-3 artifacts load the reader-owned transformer WASM kernels.
//
//   const search = await openPancakeFile('pancake-docs.pancake');   // Node
//   const search = await openPancakeFile(httpRangeSource(url));     // browser
//   const out = await search.query('how do workers restore snapshots');
//
// Every byte of the artifact is untrusted input (contract section 7): each
// offset, length, and count is validated — safe integer, within the header's
// fileBytes and the source's size, under the per-open read budget — before a
// read is issued or a buffer allocated, and every read must return exactly
// the bytes asked for. Format 2 files (profile pancake-complete-v2) carry
// per-record corpus digests, so each hydrated record is verified on its own
// range read; format 1 files verify eager segments whole and hydrate records
// under the segment digest only (reported via info().corpusIntegrity).

import { loadStudentModel, embedTextWithStudent } from './student-embedder.mjs';
import { computeMatchQuality, computePreSearchAbstention } from './student-abstention.mjs';
import { createAbstentionScorer } from './retrieval-abstention.mjs';
import { PancakeSketchArtifact } from '../pancake-artifact.js';
import { MAGIC, HEADER_BYTES, TABLE_ENTRY_BYTES, KINDS, KIND_NAMES } from './format.mjs';

import {
    KERNEL_LAYOUT, expectedBlobBytes, parseInlineTransformerEncoder,
    createInlineTransformerEmbedder, INLINE_TEST_VECTOR_TEXTS,
    buildInlineTestVectors, verifyInlineTestVectors,
    validateExpectedEmbedding, validateTestVectorTolerance,
} from './inline-transformer.mjs';

export { httpRangeSource } from './sources.mjs';
export {
    KERNEL_LAYOUT, expectedBlobBytes, parseInlineTransformerEncoder,
    createInlineTransformerEmbedder, INLINE_TEST_VECTOR_TEXTS,
    buildInlineTestVectors, verifyInlineTestVectors,
    validateExpectedEmbedding, validateTestVectorTolerance,
};

// Supported container formats: header formatVersion -> manifest profile.
export const SUPPORTED_PROFILES = Object.freeze({ 1: 'pancake-complete-v1', 2: 'pancake-complete-v2' });
export const CORPUS_LAYOUT_V2 = 'records-v2';

// Read budgets. Open-path reads (manifest, segment table, query-interp,
// corpus tables, evaluation) default to DEFAULT_OPEN_READ_BYTES per read and
// are configurable per open via options.maxReadBytes; Infinity defers to
// the absolute backstop. Records hydrate under options.maxRecordBytes.
const DEFAULT_OPEN_READ_BYTES = 256 * 1024 * 1024;
const MAX_ARTIFACT_READ_BYTES = 2 * 1024 * 1024 * 1024;
const DEFAULT_RECORD_BYTES = 16 * 1024 * 1024;
const MAX_MANIFEST_BYTES = 16 * 1024 * 1024;
const MAX_SEGMENTS = 64;
const MAX_RECORDS = 2 ** 31 - 1;
const MAX_PAGE_RECORDS = 65536;
const DIGEST_PAGE_CACHE = 64;
const RECORD_CACHE = 256;
// The embedded index is a .pancake-sketch artifact (SKETCH_PROFILE.md);
// its fixed-size header is what format 2 commits to in the manifest.
const SKETCH_HEADER_BYTES = 256;
const METRIC_NAMES = { 0: 'l2', 1: 'cosine' };

const decoder = new TextDecoder();
const MAX_SAFE_BIG = BigInt(Number.MAX_SAFE_INTEGER);

function toHex(bytes) {
    let out = '';
    for (const b of bytes) out += b.toString(16).padStart(2, '0');
    return out;
}

const isSha256Hex = (value) => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);

// SHA-256 via WebCrypto (browser, workerd, Node 18+); node:crypto fallback
// for older Node. Fails closed: verification is not optional in this reader.
async function sha256hex(bytes) {
    if (globalThis.crypto?.subtle) {
        return toHex(new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', bytes)));
    }
    const { createHash } = await import(/* webpackIgnore: true */ /* @vite-ignore */ 'node:crypto');
    return createHash('sha256').update(bytes).digest('hex');
}

function asUint8Array(bytes) {
    if (bytes instanceof Uint8Array) return bytes;
    if (bytes instanceof ArrayBuffer) return new Uint8Array(bytes);
    if (ArrayBuffer.isView(bytes)) return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    throw new Error('.pancake range source returned a non-binary value');
}

function resolveBudget(value, label, fallback) {
    if (value === undefined) return fallback;
    if (value === Infinity) return MAX_ARTIFACT_READ_BYTES;
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
        throw new Error(`${label} must be a positive number or Infinity`);
    }
    return Math.min(value, MAX_ARTIFACT_READ_BYTES);
}

// u64 fields become JS numbers only when they are exactly representable.
function u64(view, offset, label) {
    const big = view.getBigUint64(offset, true);
    if (big > MAX_SAFE_BIG) throw new Error(`.pancake ${label} exceeds the safe integer range`);
    return Number(big);
}

// Validate an artifact-derived (offset, length) before touching the source:
// safe non-negative integers, no overflow, within the read budget, within
// the header's fileBytes, and within the source's size when it reports one.
function checkRange(source, offset, length, label, limit, fileBytes) {
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length)
        || offset < 0 || length < 0 || !Number.isSafeInteger(offset + length)) {
        throw new Error(`.pancake ${label} range is out of bounds`);
    }
    if (length > limit) {
        throw new Error(`.pancake ${label} read of ${length} bytes exceeds the ${limit}-byte read budget`);
    }
    if (fileBytes !== undefined && offset + length > fileBytes) {
        throw new Error(`.pancake ${label} range extends past the file (${offset}+${length} > ${fileBytes})`);
    }
    if (Number.isSafeInteger(source.size) && offset + length > source.size) {
        throw new Error(`.pancake ${label} range extends past the source (${offset}+${length} > ${source.size})`);
    }
}

async function readChecked(source, offset, length, label, limit, fileBytes) {
    checkRange(source, offset, length, label, limit, fileBytes);
    const bytes = asUint8Array(await source.read(offset, length));
    if (bytes.length !== length) {
        throw new Error(`.pancake ${label} read returned ${bytes.length} of ${length} bytes (truncated or misbehaving source)`);
    }
    return bytes;
}

async function fileSource(filePath) {
    const fs = await import(/* webpackIgnore: true */ /* @vite-ignore */ 'node:fs');
    let fd = fs.openSync(filePath, 'r');
    let size;
    try {
        size = fs.fstatSync(fd).size;
    } catch (err) {
        fs.closeSync(fd);
        throw err;
    }
    return {
        size,
        preferredParallelism: Infinity,
        preferredGapBytes: 2048,
        async read(offset, length) {
            if (fd === null) throw new Error('.pancake file source is closed');
            const buffer = new Uint8Array(length);
            let bytesRead = 0;
            while (bytesRead < length) {
                const chunk = fs.readSync(fd, buffer, bytesRead, length - bytesRead, offset + bytesRead);
                if (chunk === 0) break;
                bytesRead += chunk;
            }
            return buffer.subarray(0, bytesRead);
        },
        // Idempotent: a second close() is a no-op, not EBADF.
        async close() {
            if (fd !== null) {
                const handle = fd;
                fd = null;
                fs.closeSync(handle);
            }
        },
    };
}

// A source restricted to one segment's window, for handing a container
// region to a reader that expects offset 0 = its own start (the embedded
// sketch artifact). Reads outside the window are refused here, before the
// underlying source sees them.
function windowSource(source, offset, length) {
    return {
        size: length,
        preferredParallelism: source.preferredParallelism,
        preferredGapBytes: source.preferredGapBytes,
        async read(off, len) {
            if (!Number.isSafeInteger(off) || !Number.isSafeInteger(len) || off < 0 || len < 0 || off + len > length) {
                throw new Error('.pancake index segment read is outside the segment window');
            }
            return source.read(offset + off, len);
        },
        async close() {},
    };
}

function align16(n) {
    return Math.ceil(n / 16) * 16;
}

const viewOf = (bytes) => new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

function toFloat32(vector, dim, what) {
    const v = vector instanceof Float32Array ? vector
        : (Array.isArray(vector) || ArrayBuffer.isView(vector)) ? Float32Array.from(vector) : null;
    if (!v || v.length !== dim) {
        throw new Error(`${what} must return a ${dim}-dimensional vector`);
    }
    for (let d = 0; d < dim; d++) {
        if (!Number.isFinite(v[d])) throw new Error(`${what} returned a non-finite component`);
    }
    return v;
}

function base64Bytes(text) {
    if (typeof text !== 'string') return new Uint8Array(0);
    return typeof Buffer !== 'undefined'
        ? Buffer.from(text, 'base64')
        : Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
}

// Contract 4.4 mode 2: a host-supplied encoder for a pinned external model
// is checked against the declaration's verification vectors before the
// artifact serves a single query. Dimension and every component within the
// declared tolerance; any disagreement refuses the open.
export async function verifyHostEncoder(declaration, encodeQuery, dim) {
    const vectors = declaration.testVectors;
    if (!Array.isArray(vectors) || vectors.length === 0) {
        throw new Error('.pancake kind-2 declaration carries no verification vectors');
    }
    if (declaration.dim !== undefined && declaration.dim !== dim) {
        throw new Error(`.pancake kind-2 declaration dim ${declaration.dim} disagrees with manifest dim ${dim}`);
    }
    for (const tv of vectors) {
        if (!tv || typeof tv.text !== 'string') {
            throw new Error('.pancake kind-2 verification vector is malformed');
        }
        // The expected embedding must be dim finite numbers: a malformed
        // entry would turn every comparison into NaN, and NaN > maxDiff is
        // false — the vector would "pass" without comparing anything.
        const label = `.pancake kind-2 verification vector "${tv.text.slice(0, 40)}…"`;
        validateExpectedEmbedding(tv.embedding, dim, label);
        const tolerance = validateTestVectorTolerance(tv.tolerance, label);
        const vector = toFloat32(await encodeQuery(tv.text), dim, 'options.encodeQuery');
        let maxDiff = 0;
        for (let d = 0; d < dim; d++) {
            const diff = Math.abs(vector[d] - tv.embedding[d]);
            if (diff > maxDiff) maxDiff = diff;
        }
        if (!(maxDiff <= tolerance)) {
            throw new Error(`host encoder failed the artifact's verification vector "${tv.text.slice(0, 40)}…": `
                + `max component diff ${maxDiff.toExponential(2)} exceeds tolerance ${tolerance} `
                + `(declared model ${declaration.model})`);
        }
    }
    return { checked: vectors.length };
}

/**
 * Open a .pancake complete artifact from a file path (Node) or a
 * { read(offset, length), size? } range source (any runtime).
 * Returns { query(text, {k}), info(), evaluation(), close() }.
 */
export async function openPancakeFile(input, options = {}) {
    const source = typeof input === 'string' ? await fileSource(input) : input;
    const owned = typeof input === 'string';
    if (!source || typeof source.read !== 'function') {
        throw new Error('openPancakeFile() requires a file path or a range source with read(offset, length)');
    }
    const maxReadBytes = resolveBudget(options.maxReadBytes, 'maxReadBytes', DEFAULT_OPEN_READ_BYTES);
    const maxRecordBytes = resolveBudget(options.maxRecordBytes, 'maxRecordBytes', DEFAULT_RECORD_BYTES);
    const verifyRecords = options.verifyRecords !== false;
    // Opt-in full authentication of the lazy rerank rows at open: one
    // streamed pass over the vectors segment against the (identity-anchored)
    // vectorsSha256. Without it, rows that feed reranking are covered by the
    // whole-segment commitment but not verified per read — the documented
    // transitional stance (spec section 6) — and verifyVectors() can run the
    // same pass at any later point.
    const verifyIndexVectors = options.verifyIndexVectors === true;
    const verifyEncoder = options.verifyEncoder !== false;
    // Declared outside the try so a failure after the encoder is created
    // (count mismatch, bad corpus tables, ...) still releases its buffers.
    let disposeEncoder = () => {};
    try {
        // Header: the only read whose bounds come from nowhere but the spec.
        const header = await readChecked(source, 0, HEADER_BYTES, 'header', maxReadBytes);
        const headerView = viewOf(header);
        if (headerView.getUint32(0, true) !== MAGIC) {
            throw new Error('not a .pancake file (bad magic)');
        }
        const formatVersion = headerView.getUint32(4, true);
        const profile = SUPPORTED_PROFILES[formatVersion];
        if (!profile) throw new Error(`unsupported .pancake format version ${formatVersion}`);
        const manifestBytes = headerView.getUint32(8, true);
        const segmentCount = headerView.getUint32(12, true);
        const fileBytes = u64(headerView, 16, 'header fileBytes');
        if (manifestBytes < 2 || manifestBytes > MAX_MANIFEST_BYTES || segmentCount < 1 || segmentCount > MAX_SEGMENTS) {
            throw new Error('.pancake header is implausible');
        }
        if (Number.isSafeInteger(source.size) && source.size !== fileBytes) {
            throw new Error(`.pancake is truncated or padded: header says ${fileBytes} bytes, source has ${source.size}`);
        }
        const tableBytes = segmentCount * TABLE_ENTRY_BYTES;
        if (HEADER_BYTES + manifestBytes + tableBytes > fileBytes) {
            throw new Error('.pancake header places the manifest or segment table past the file');
        }
        const identity = toHex(header.subarray(24, 56));

        const manifestBuf = await readChecked(source, HEADER_BYTES, manifestBytes, 'manifest', maxReadBytes, fileBytes);
        if (await sha256hex(manifestBuf) !== identity) {
            throw new Error('.pancake manifest failed identity verification');
        }
        let manifest;
        try { manifest = JSON.parse(decoder.decode(manifestBuf)); } catch (err) {
            throw new Error('.pancake manifest is not valid JSON', { cause: err });
        }
        if (!manifest || typeof manifest !== 'object' || manifest.profile !== profile) {
            throw new Error(`unsupported profile ${manifest?.profile} for format version ${formatVersion}`);
        }
        if (!Array.isArray(manifest.segments) || manifest.segments.length !== segmentCount) {
            throw new Error('.pancake manifest segment list disagrees with the header segment count');
        }
        const dim = manifest.dim;
        if (!Number.isSafeInteger(dim) || dim < 1 || dim > 65536) {
            throw new Error('.pancake manifest dim is implausible');
        }
        if (!manifest.corpus || typeof manifest.corpus !== 'object') {
            throw new Error('.pancake manifest has no corpus block');
        }

        const table = await readChecked(source, HEADER_BYTES + manifestBytes, tableBytes, 'segment table', maxReadBytes, fileBytes);
        const tableView = viewOf(table);
        const segments = new Map();
        let expectedOffset = align16(HEADER_BYTES + manifestBytes + tableBytes);
        for (let i = 0; i < segmentCount; i++) {
            const entry = i * TABLE_ENTRY_BYTES;
            const kindNumber = tableView.getUint32(entry, true);
            const kind = KIND_NAMES[kindNumber];
            const offset = u64(tableView, entry + 8, `segment ${i} offset`);
            const length = u64(tableView, entry + 16, `segment ${i} length`);
            const declared = manifest.segments[i];
            if (!declared || typeof declared !== 'object' || declared.bytes !== length || !isSha256Hex(declared.sha256)
                || offset !== expectedOffset || !Number.isSafeInteger(offset + length) || offset + length > fileBytes) {
                throw new Error(`.pancake segment table disagrees with manifest at entry ${i}`);
            }
            if (kind === undefined) {
                // Unknown kinds are skipped, not failed (spec 3.3); they are
                // still committed through the manifest, so the declared name
                // must at least not claim to be a kind this reader knows.
                if (typeof declared.kind !== 'string' || declared.kind in KINDS) {
                    throw new Error(`.pancake segment table entry ${i} has unknown kind ${kindNumber} but the manifest names it ${declared.kind}`);
                }
            } else {
                if (declared.kind !== kind) {
                    throw new Error(`.pancake segment table disagrees with manifest at entry ${i}`);
                }
                if (segments.has(kind)) throw new Error(`.pancake carries more than one ${kind} segment`);
                segments.set(kind, { offset, length, sha256: declared.sha256 });
            }
            expectedOffset = align16(offset + length);
        }
        for (const required of ['index', 'corpus', 'query-interp']) {
            if (!segments.has(required)) throw new Error(`.pancake is missing the ${required} segment`);
        }

        // Corpus tables: count + offsets (+ page table for layout v2),
        // one contiguous region at the segment start, sized from the
        // identity-verified manifest and cross-checked against the segment's
        // own words after the read.
        const qi = segments.get('query-interp');
        const idx = segments.get('index');
        const corpus = segments.get('corpus');
        const declaredRecords = manifest.corpus.records;
        if (!Number.isSafeInteger(declaredRecords) || declaredRecords < 0 || declaredRecords > MAX_RECORDS) {
            throw new Error('.pancake manifest corpus.records is implausible');
        }
        const layout = manifest.corpus.layout;
        const perRecord = formatVersion >= 2;
        if (perRecord && layout !== CORPUS_LAYOUT_V2) {
            throw new Error(`.pancake format ${formatVersion} requires corpus layout ${CORPUS_LAYOUT_V2}, manifest says ${layout}`);
        }
        if (!perRecord && layout !== undefined) {
            throw new Error(`.pancake format 1 carries corpus layout v1, manifest says ${layout}`);
        }
        let pageRecords = 0;
        let pages = 0;
        let offsetsAt;
        let tablesBytes;
        let recordDigestsAt = 0;
        let recordsAt;
        if (perRecord) {
            pageRecords = manifest.corpus.pageRecords;
            if (!Number.isSafeInteger(pageRecords) || pageRecords < 1 || pageRecords > MAX_PAGE_RECORDS) {
                throw new Error('.pancake manifest corpus.pageRecords is implausible');
            }
            pages = Math.ceil(declaredRecords / pageRecords);
            if (manifest.corpus.pages !== pages || !isSha256Hex(manifest.corpus.pageTableSha256)
                || manifest.corpus.recordDigest !== 'sha256') {
                throw new Error('.pancake manifest corpus integrity block is inconsistent');
            }
            offsetsAt = 8;
            tablesBytes = offsetsAt + 8 * (declaredRecords + 1) + 32 * pages;
            recordDigestsAt = tablesBytes;
            recordsAt = recordDigestsAt + 32 * declaredRecords;
        } else {
            offsetsAt = 4;
            tablesBytes = offsetsAt + 8 * (declaredRecords + 1);
            recordsAt = tablesBytes;
        }
        if (!Number.isSafeInteger(recordsAt) || recordsAt > corpus.length) {
            throw new Error('.pancake manifest corpus.records is implausible for the corpus segment');
        }
        // Format 2 commits to the embedded sketch's 256-byte header
        // (metric/dim/count and the sketch's residentSha256/vectorsSha256),
        // anchoring the sketch's own integrity chain to the identity; the
        // sketch's self-checks alone verify against fields an attacker who
        // rewrites the segment also controls.
        if (perRecord && (!isSha256Hex(manifest.index?.headerSha256) || idx.length < SKETCH_HEADER_BYTES)) {
            throw new Error('.pancake manifest carries no index header commitment (index.headerSha256)');
        }

        // After the segment table, every remaining open-path extent is known,
        // so the query-interp segment, the embedded sketch (two dependent
        // reads of its own), and the corpus tables fetch as one concurrent
        // wave instead of dependent rounds — at 100 ms/request that is most
        // of the open time.
        const [qiBytes, sketch, tables] = await Promise.all([
            (async () => {
                const bytes = await readChecked(source, qi.offset, qi.length, 'query-interp segment', maxReadBytes, fileBytes);
                if (await sha256hex(bytes) !== qi.sha256) {
                    throw new Error('.pancake query-interp segment failed hash verification');
                }
                return bytes;
            })(),
            PancakeSketchArtifact.open(windowSource(source, idx.offset, idx.length), { maxReadBytes }),
            readChecked(source, corpus.offset, tablesBytes, 'corpus tables', maxReadBytes, fileBytes),
            perRecord ? (async () => {
                const headerBytes = await readChecked(source, idx.offset, SKETCH_HEADER_BYTES, 'index header', maxReadBytes, fileBytes);
                if (await sha256hex(headerBytes) !== manifest.index.headerSha256) {
                    throw new Error('.pancake index header failed hash verification against the manifest');
                }
            })() : null,
        ]);

        // Query interpretation: version word, kind, two length-prefixed
        // regions that must tile the segment exactly.
        if (qi.length < 16) throw new Error('.pancake query-interp segment is too short');
        const qiView = viewOf(qiBytes);
        const qiVersion = qiView.getUint32(0, true);
        if (qiVersion !== 1) throw new Error(`unsupported query-interpretation version ${qiVersion}`);
        const qiKind = qiView.getUint32(4, true);
        const encoderLen = qiView.getUint32(8, true);
        const calibrationLen = qiView.getUint32(12, true);
        if (16 + encoderLen + calibrationLen !== qi.length) {
            throw new Error('.pancake query-interp segment layout is inconsistent');
        }
        const encoderBytes = qiBytes.subarray(16, 16 + encoderLen);
        let calibrationJson;
        try { calibrationJson = JSON.parse(decoder.decode(qiBytes.subarray(16 + encoderLen))); } catch (err) {
            throw new Error('.pancake calibration is not valid JSON', { cause: err });
        }
        if (!calibrationJson || typeof calibrationJson !== 'object') {
            throw new Error('.pancake calibration must be a JSON object');
        }

        // kind 1 (student-inline-v1): pure-JS inline encoder + feature-stream
        // abstention. kind 2 (external-transformers-v1): the encoder is a
        // pinned declaration the HOST must satisfy via options.encodeQuery,
        // verified against the declaration's test vectors before serving;
        // calibration scores retrieval signals + query text. kind 3
        // (inline-transformer-v1): the pinned teacher as data, executed by
        // reader-owned kernels, verified against its own test vectors.
        let embed;
        let preScore = () => null;
        let scoreQuality;
        let encoderInfo;
        let encoderVerified = null;
        const retrievalScorer = () => {
            const scorer = createAbstentionScorer(calibrationJson.asset, base64Bytes(calibrationJson.vocabBloomBase64));
            const VERDICTS = { answer: 'strong', weak: 'weak', abstain: 'none' };
            return async (hits, context) => {
                if (!scorer) return { match_quality: 'unscored' };
                // The coverage term grounds the verdict in the top passages'
                // text, so those records hydrate before scoring. When the
                // query is answered the pages are already hot for result
                // hydration; an abstained query costs these extra reads.
                // Hydration failures propagate — on format 2 a record that
                // fails its digest must fail the query, not skew its verdict.
                const topTexts = scorer.usesPassage && hits.length
                    ? (await Promise.all(hits.slice(0, scorer.passagesNeeded || 1)
                        .map((hit) => hydrate(hit.id)))).map((record) => record?.text)
                    : [];
                const scored = scorer.score(context.text, hits, topTexts);
                return { match_quality: VERDICTS[scored.verdict] || scored.verdict, confidence: scored.p };
            };
        };
        if (qiKind === 1) {
            const student = loadStudentModel(encoderBytes);
            encoderInfo = { kind: 'student-inline-v1' };
            embed = (text) => {
                const embedded = embedTextWithStudent(text, student);
                return { vector: toFloat32(embedded.vector, dim, 'inline student encoder'), embedded };
            };
            // Pre-search abstention skips the index entirely when it fires,
            // exactly as the Worker does.
            preScore = (context) => computePreSearchAbstention(context.embedded, calibrationJson);
            scoreQuality = (hits, context) => computeMatchQuality(hits, context.embedded, calibrationJson);
        } else if (qiKind === 2) {
            let declaration;
            try { declaration = JSON.parse(decoder.decode(encoderBytes)); } catch (err) {
                throw new Error('.pancake kind-2 encoder declaration is not valid JSON', { cause: err });
            }
            if (!declaration || typeof declaration !== 'object') throw new Error('.pancake kind-2 encoder declaration must be a JSON object');
            encoderInfo = declaration;
            const encodeQuery = options.encodeQuery;
            if (encodeQuery !== undefined && typeof encodeQuery !== 'function') {
                throw new Error('options.encodeQuery must be a function');
            }
            const hasVectors = Array.isArray(declaration.testVectors) && declaration.testVectors.length > 0;
            if (encodeQuery && verifyEncoder) {
                if (hasVectors) {
                    await verifyHostEncoder(declaration, encodeQuery, dim);
                    encoderVerified = true;
                } else if (options.allowUnverifiedEncoder === true) {
                    encoderVerified = false;
                } else {
                    throw new Error(`.pancake declares an external encoder (${declaration.model}) without verification vectors, `
                        + 'so the host encoder cannot be checked against it (contract section 4.4); '
                        + 'pass options.allowUnverifiedEncoder: true to serve it anyway, marked unverified');
                }
            } else if (encodeQuery) {
                encoderVerified = false;
            }
            embed = async (text) => {
                if (!encodeQuery) {
                    throw new Error(`.pancake declares an external encoder (${declaration.model}); `
                        + 'pass options.encodeQuery to openPancakeFile');
                }
                return { vector: toFloat32(await encodeQuery(text), dim, 'options.encodeQuery'), text };
            };
            scoreQuality = retrievalScorer();
        } else if (qiKind === 3) {
            // Kernels are reader-owned (spec 3.6) and load dynamically so
            // kind-1/2 files never touch them; the node-only glue is hidden
            // from bundlers, the web glue stays analyzable so browser builds
            // carry it. options.createEncoder overrides the loader.
            let createEncoder = options.createEncoder;
            if (!createEncoder) {
                try {
                    createEncoder = globalThis.process?.versions?.node
                        ? (await import(/* webpackIgnore: true */ /* @vite-ignore */ './encoder-kernels/encoder.node.mjs')).default
                        : (await import('./encoder-kernels/encoder.mjs')).default;
                } catch (err) {
                    throw new Error('kind-3 artifact requires the inline-transformer kernels at '
                        + 'complete/encoder-kernels/ (pancake-wasm/complete); run '
                        + 'examples/05-one-file-search/encoder-spike/build-encoder.sh to rebuild them',
                    { cause: err });
                }
            }
            const { declaration, vocabText, blob } = parseInlineTransformerEncoder(encoderBytes);
            encoderInfo = declaration;
            if (declaration.dim !== undefined && declaration.dim !== dim) {
                throw new Error(`.pancake kind-3 declaration dim ${declaration.dim} disagrees with manifest dim ${dim}`);
            }
            const embedder = await createInlineTransformerEmbedder({ declaration, vocabText, blob, createEncoder, verify: verifyEncoder });
            encoderVerified = verifyEncoder && Array.isArray(declaration.testVectors) && declaration.testVectors.length > 0;
            disposeEncoder = () => embedder.dispose();
            embed = async (text) => {
                const { vector } = await embedder.embed(`${declaration.prefixPolicy?.query || ''}${text}`);
                return { vector: toFloat32(vector, dim, 'inline transformer encoder'), text };
            };
            scoreQuality = retrievalScorer();
        } else {
            throw new Error(`unsupported query-interpretation kind ${qiKind}`);
        }

        // Index: the embedded sketch artifact, opened in the wave above
        // against a segment-windowed source (resident hash verified there).
        if (sketch.count !== declaredRecords) {
            throw new Error(`.pancake index count ${sketch.count} != corpus records ${declaredRecords}`);
        }
        if (sketch.dim !== undefined && sketch.dim !== dim) {
            throw new Error(`.pancake index dim ${sketch.dim} != manifest dim ${dim}`);
        }
        // The metric decides the search semantics; the sketch header's copy
        // must agree with the identity-verified manifest (on format 2 the
        // header itself is hash-anchored above; on format 1 this cross-check
        // is the only binding).
        if (METRIC_NAMES[sketch.metric] !== undefined && typeof manifest.metric === 'string'
            && METRIC_NAMES[sketch.metric] !== manifest.metric) {
            throw new Error(`.pancake index metric ${METRIC_NAMES[sketch.metric]} != manifest metric ${manifest.metric}`);
        }
        if (verifyIndexVectors) {
            await sketch.verifyVectors();
        }

        // Corpus tables: cross-check the segment's own words, then the
        // offsets — monotonic, starting exactly where the tables end, ending
        // inside the segment — and for layout v2 the page table's digest.
        const tablesView = viewOf(tables);
        const recordCount = tablesView.getUint32(0, true);
        if (recordCount !== declaredRecords) {
            throw new Error('.pancake corpus count disagrees with manifest');
        }
        if (perRecord && tablesView.getUint32(4, true) !== pageRecords) {
            throw new Error('.pancake corpus page size disagrees with manifest');
        }
        const recordOffsets = new Float64Array(recordCount + 1);
        for (let i = 0; i <= recordCount; i++) {
            const value = u64(tablesView, offsetsAt + 8 * i, `corpus offset ${i}`);
            if (value > corpus.length || (i > 0 && value < recordOffsets[i - 1]) || (i === 0 && value !== recordsAt)) {
                throw new Error('.pancake corpus offsets are inconsistent');
            }
            recordOffsets[i] = value;
        }
        let pageTable = null;
        if (perRecord) {
            pageTable = tables.slice(offsetsAt + 8 * (recordCount + 1), tablesBytes);
            if (await sha256hex(pageTable) !== manifest.corpus.pageTableSha256) {
                throw new Error('.pancake corpus page table failed hash verification');
            }
        }
        const corpusIntegrity = perRecord ? (verifyRecords ? 'per-record-sha256' : 'per-record-sha256 (unverified by option)') : 'segment-sha256';

        // Per-page record-digest cache: a page is fetched (one read of
        // 32 * pageRecords bytes), verified against the page table, and kept
        // for the records that share it. Pending fetches are shared so
        // concurrent hydrations of one page cost one read.
        const digestPages = new Map();
        const pageDigests = (page) => {
            if (digestPages.has(page)) {
                const cached = digestPages.get(page);
                digestPages.delete(page);
                digestPages.set(page, cached);
                return cached;
            }
            const pending = (async () => {
                const from = Math.min(recordCount, page * pageRecords);
                const to = Math.min(recordCount, (page + 1) * pageRecords);
                const bytes = await readChecked(source, corpus.offset + recordDigestsAt + 32 * from, 32 * (to - from),
                    `corpus digest page ${page}`, maxReadBytes, fileBytes);
                if (await sha256hex(bytes) !== toHex(pageTable.subarray(32 * page, 32 * page + 32))) {
                    digestPages.delete(page);
                    throw new Error(`.pancake corpus digest page ${page} failed hash verification`);
                }
                return bytes;
            })();
            digestPages.set(page, pending);
            // A failed fetch (transport error, short read, hash mismatch) must
            // not stay cached as a rejected promise that every later record in
            // the page re-throws; evict it so the next hydration retries.
            pending.catch(() => { if (digestPages.get(page) === pending) digestPages.delete(page); });
            if (digestPages.size > DIGEST_PAGE_CACHE) digestPages.delete(digestPages.keys().next().value);
            return pending;
        };

        const recordCache = new Map();
        const cacheRecord = (id, record) => {
            if (recordCache.has(id)) recordCache.delete(id);
            recordCache.set(id, record);
            if (recordCache.size > RECORD_CACHE) recordCache.delete(recordCache.keys().next().value);
            return record;
        };

        const hydrate = async (id) => {
            if (!Number.isSafeInteger(id) || id < 0 || id >= recordCount) {
                throw new Error(`.pancake result id ${id} is outside the corpus`);
            }
            if (recordCache.has(id)) {
                const cached = recordCache.get(id);
                recordCache.delete(id);
                recordCache.set(id, cached);
                return cached;
            }
            const start = recordOffsets[id];
            const end = recordOffsets[id + 1];
            const bytes = await readChecked(source, corpus.offset + start, end - start, `corpus record ${id}`, maxRecordBytes, fileBytes);
            if (perRecord && verifyRecords) {
                const page = Math.floor(id / pageRecords);
                const digests = await pageDigests(page);
                const at = 32 * (id - page * pageRecords);
                if (await sha256hex(bytes) !== toHex(digests.subarray(at, at + 32))) {
                    throw new Error(`.pancake corpus record ${id} failed integrity verification`);
                }
            }
            let record;
            try { record = JSON.parse(decoder.decode(bytes)); } catch (err) {
                throw new Error(`.pancake corpus record ${id} is not valid JSON`, { cause: err });
            }
            if (!record || typeof record !== 'object' || Array.isArray(record)) {
                throw new Error(`.pancake corpus record ${id} is not a JSON object`);
            }
            return cacheRecord(id, record);
        };

        // Lifecycle: close() is idempotent (the encoder's WASM buffers and the
        // file descriptor are released once), and the query surface refuses
        // use after close instead of reading through a freed encoder.
        let closed = false;
        const assertOpen = () => {
            if (closed) throw new Error('.pancake reader is closed');
        };

        return {
            info() {
                return {
                    identity,
                    formatVersion,
                    profile,
                    records: recordCount,
                    dim,
                    metric: manifest.metric,
                    encoder: encoderInfo,
                    encoderVerified,
                    corpusIntegrity,
                    fileBytes,
                    residentBytes: sketch.residentBytes + 8 * (recordCount + 1) + (pageTable ? pageTable.length : 0),
                    residentVerified: sketch.stats().residentVerified,
                    // True only after a full vectors pass (verifyIndexVectors
                    // at open, or verifyVectors() later).
                    vectorsVerified: sketch.vectorsVerified === true,
                    // 'per-row-sha256' when the embedded sketch is format 2
                    // (every rerank row verified on its own read against the
                    // identity-anchored page-hash table); 'segment-sha256'
                    // for format-1 sketches, where rows are committed but
                    // unverified until a full vectors pass.
                    indexRowIntegrity: sketch.formatVersion >= 2 && sketch.verifyRows !== false
                        ? 'per-row-sha256' : 'segment-sha256',
                    sampleQueries: Array.isArray(manifest.sampleQueries) ? manifest.sampleQueries : [],
                };
            },

            async query(text, queryOptions = {}) {
                assertOpen();
                // k: absent means the default; anything supplied must be a
                // positive integer (an explicit 0, -1, or 2.5 is an error,
                // not a silent fallback) and is capped to the corpus size
                // before it sizes the search.
                let k = 5;
                if (queryOptions.k !== undefined) {
                    if (!Number.isSafeInteger(queryOptions.k) || queryOptions.k < 1) {
                        throw new Error('query() k must be a positive integer');
                    }
                    k = queryOptions.k;
                }
                k = Math.min(k, Math.max(recordCount, 1));
                const trimmed = String(text || '').trim();
                if (!trimmed) throw new Error('query text is required');
                const context = await embed(trimmed);
                const pre = preScore(context);
                let hits = [];
                if (!pre) {
                    hits = (await sketch.search(context.vector, k, {
                        rerank: queryOptions.rerank,
                        parallelism: queryOptions.parallelism ?? queryOptions.rerankParallelism ?? options.rerankParallelism,
                        gap: queryOptions.gap ?? queryOptions.rerankGap ?? options.rerankGap,
                        maxRangeBytes: queryOptions.maxRangeBytes ?? options.rerankMaxRangeBytes,
                    })).results;
                }
                const quality = pre || await scoreQuality(hits, context);
                const returned = quality.match_quality === 'none' ? [] : hits;
                // The search's id and distance are authoritative: they are
                // written last so a corpus record carrying its own `id` or
                // `distance` field cannot overwrite them (reserved names).
                const results = await Promise.all(returned.map(async (hit) => {
                    const record = await hydrate(hit.id);
                    return { ...record, id: hit.id, distance: hit.distance };
                }));
                return { matchQuality: quality.match_quality, confidence: quality.confidence, results };
            },

            /** Hydrate one corpus record by id (verified per record on format 2). */
            async record(id) {
                assertOpen();
                return hydrate(id);
            },

            /**
             * Full pass over the index's lazy vector rows against the
             * identity-anchored vectorsSha256 (streamed where possible).
             * Until this resolves (or verifyIndexVectors ran at open), rows
             * that feed reranking are committed but not verified per read.
             */
            async verifyVectors(verifyOptions) {
                assertOpen();
                return sketch.verifyVectors(verifyOptions);
            },

            async evaluation() {
                assertOpen();
                const ev = segments.get('evaluation');
                if (!ev) return null;
                const bytes = await readChecked(source, ev.offset, ev.length, 'evaluation segment', maxReadBytes, fileBytes);
                if (await sha256hex(bytes) !== ev.sha256) {
                    throw new Error('.pancake evaluation segment failed hash verification');
                }
                let parsed;
                try { parsed = JSON.parse(decoder.decode(bytes)); } catch (err) {
                    throw new Error('.pancake evaluation segment is not valid JSON', { cause: err });
                }
                if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
                    throw new Error('.pancake evaluation segment must be a JSON object');
                }
                return parsed;
            },

            async close() {
                if (closed) return;
                closed = true;
                disposeEncoder();
                await sketch.close();
                if (owned) await source.close();
            },
        };
    } catch (err) {
        try { disposeEncoder(); } catch { /* already released */ }
        if (owned && source.close) await source.close().catch(() => {});
        throw err;
    }
}
