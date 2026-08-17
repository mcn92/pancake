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

import { loadStudentModel, embedTextWithStudent } from '../03-edge-docs-search/student-embedder.mjs';
import { computeMatchQuality, computePreSearchAbstention } from './abstention.mjs';
import { createAbstentionScorer } from '../04-static-wiki-pack/web/src/abstention.js';
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
        preferredParallelism: Infinity,
        preferredGapBytes: 2048,
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
        preferredParallelism: source.preferredParallelism,
        preferredGapBytes: source.preferredGapBytes,
        async read(off, len) { return source.read(offset + off, len); },
        async close() {},
    };
}

function align16(n) {
    return Math.ceil(n / 16) * 16;
}

const viewOf = (bytes) => new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

/**
 * Open a .pancake complete artifact from a file path (Node) or a
 * { read(offset, length), size? } range source (any runtime).
 * Returns { query(text, {k}), info(), evaluation(), close() }.
 */
export async function openPancakeFile(input, options = {}) {
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
        if (source.size !== undefined && source.size !== fileBytes) {
            throw new Error(`.pancake is truncated or padded: header says ${fileBytes} bytes, source has ${source.size}`);
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
        let expectedOffset = align16(HEADER_BYTES + manifestBytes + segmentCount * TABLE_ENTRY_BYTES);
        for (let i = 0; i < segmentCount; i++) {
            const entry = i * TABLE_ENTRY_BYTES;
            const kind = KIND_NAMES[tableView.getUint32(entry, true)];
            const offset = Number(tableView.getBigUint64(entry + 8, true));
            const length = Number(tableView.getBigUint64(entry + 16, true));
            const declared = manifest.segments[i];
            if (!declared || declared.kind !== kind || declared.bytes !== length
                || offset !== expectedOffset || offset + length > fileBytes) {
                throw new Error(`.pancake segment table disagrees with manifest at entry ${i}`);
            }
            segments.set(kind, { offset, length, sha256: declared.sha256 });
            expectedOffset = align16(offset + length);
        }
        for (const required of ['index', 'corpus', 'query-interp']) {
            if (!segments.has(required)) throw new Error(`.pancake is missing the ${required} segment`);
        }

        // Query interpretation: one eager read, digest-verified, then split
        // into encoder + calibration (one unit, shared version + kind).
        const qi = segments.get('query-interp');
        const qiBytes = new Uint8Array(await source.read(qi.offset, qi.length));
        if (await sha256hex(qiBytes) !== qi.sha256) {
            throw new Error('.pancake query-interp segment failed hash verification');
        }
        const qiView = viewOf(qiBytes);
        const qiKind = qiView.getUint32(4, true);
        const encoderLen = qiView.getUint32(8, true);
        const calibrationLen = qiView.getUint32(12, true);
        if (16 + encoderLen + calibrationLen !== qi.length) {
            throw new Error('.pancake query-interp segment layout is inconsistent');
        }
        const encoderBytes = qiBytes.subarray(16, 16 + encoderLen);
        const calibrationJson = JSON.parse(decoder.decode(qiBytes.subarray(16 + encoderLen)));

        // kind 1 (student-inline-v1): pure-JS inline encoder + feature-stream
        // abstention. kind 2 (external-transformers-v1): the encoder is a
        // pinned declaration the HOST must satisfy via options.encodeQuery;
        // calibration scores retrieval signals + query text.
        let embed;
        let scoreQuality;
        let encoderInfo;
        let disposeEncoder = () => {};
        if (qiKind === 1) {
            const student = loadStudentModel(encoderBytes);
            encoderInfo = { kind: 'student-inline-v1' };
            embed = (text) => {
                const embedded = embedTextWithStudent(text, student);
                return { vector: embedded.vector, embedded };
            };
            // Pre-search abstention skips the index entirely when it fires,
            // exactly as the Worker does.
            var preScore = (context) => computePreSearchAbstention(context.embedded, calibrationJson);
            scoreQuality = (hits, context) => computeMatchQuality(hits, context.embedded, calibrationJson);
        } else if (qiKind === 2) {
            const declaration = JSON.parse(decoder.decode(encoderBytes));
            encoderInfo = declaration;
            const encodeQuery = options.encodeQuery;
            embed = async (text) => {
                if (!encodeQuery) {
                    throw new Error(`.pancake declares an external encoder (${declaration.model}); `
                        + 'pass options.encodeQuery to openPancakeFile');
                }
                return { vector: await encodeQuery(text), text };
            };
            const bloomBytes = typeof Buffer !== 'undefined'
                ? Buffer.from(calibrationJson.vocabBloomBase64, 'base64')
                : Uint8Array.from(atob(calibrationJson.vocabBloomBase64), (c) => c.charCodeAt(0));
            const scorer = createAbstentionScorer(calibrationJson.asset, bloomBytes);
            const VERDICTS = { answer: 'strong', weak: 'weak', abstain: 'none' };
            preScore = () => null;
            scoreQuality = (hits, context) => {
                if (!scorer) return { match_quality: 'unscored' };
                const scored = scorer.score(context.text, hits);
                return { match_quality: VERDICTS[scored.verdict] || scored.verdict, confidence: scored.p };
            };
        } else if (qiKind === 3) {
            // inline-transformer-v1: declaration + vocab + weight blob as
            // segment data; parsing and the embed loop live in the shared
            // create-pancake-search module so this reader cannot drift from
            // the packaged one (dynamic imports so kind-1/2 files never load
            // the kernels).
            let parseInlineTransformerEncoder;
            let createInlineTransformerEmbedder;
            let createEncoder;
            try {
                ({ parseInlineTransformerEncoder, createInlineTransformerEmbedder } =
                    await import('../../create-pancake-search/src/inline-transformer.mjs'));
                createEncoder = globalThis.process?.versions?.node
                    ? (await import('../../create-pancake-search/src/encoder-kernels/encoder.node.mjs')).default
                    : (await import('../../create-pancake-search/src/encoder-kernels/encoder.mjs')).default;
            } catch (err) {
                throw new Error('kind-3 artifact requires the shared inline-transformer reader at '
                    + 'create-pancake-search/src/inline-transformer.mjs and its kernels at '
                    + 'create-pancake-search/src/encoder-kernels/; run '
                    + 'examples/05-one-file-search/encoder-spike/build-encoder.sh to rebuild the kernels',
                { cause: err });
            }
            const { declaration, vocabText, blob } = parseInlineTransformerEncoder(encoderBytes);
            encoderInfo = declaration;
            const embedder = await createInlineTransformerEmbedder({ declaration, vocabText, blob, createEncoder });
            disposeEncoder = () => embedder.dispose();
            embed = async (text) => {
                const { vector } = await embedder.embed(`${declaration.prefixPolicy?.query || ''}${text}`);
                return { vector, text };
            };
            const bloomBytes = typeof Buffer !== 'undefined'
                ? Buffer.from(calibrationJson.vocabBloomBase64, 'base64')
                : Uint8Array.from(atob(calibrationJson.vocabBloomBase64), (c) => c.charCodeAt(0));
            const { createAbstentionScorer: makeScorer } = await import('../04-static-wiki-pack/web/src/abstention.js');
            const scorer3 = makeScorer(calibrationJson.asset, bloomBytes);
            const VERDICTS3 = { answer: 'strong', weak: 'weak', abstain: 'none' };
            preScore = () => null;
            scoreQuality = (hits, context) => {
                if (!scorer3) return { match_quality: 'unscored' };
                const scored = scorer3.score(context.text, hits);
                return { match_quality: VERDICTS3[scored.verdict] || scored.verdict, confidence: scored.p };
            };
        } else {
            throw new Error(`unsupported query-interpretation kind ${qiKind}`);
        }

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

        const recordCache = new Map();
        const cacheRecord = (id, record) => {
            if (recordCache.has(id)) recordCache.delete(id);
            recordCache.set(id, record);
            if (recordCache.size > 256) recordCache.delete(recordCache.keys().next().value);
            return record;
        };

        const hydrate = async (id) => {
            if (recordCache.has(id)) {
                const cached = recordCache.get(id);
                recordCache.delete(id);
                recordCache.set(id, cached);
                return cached;
            }
            const start = recordOffsets[id];
            const end = recordOffsets[id + 1];
            const bytes = new Uint8Array(await source.read(corpus.offset + start, end - start));
            return cacheRecord(id, JSON.parse(decoder.decode(bytes)));
        };

        return {
            info() {
                return {
                    identity,
                    records: recordCount,
                    dim: manifest.dim,
                    metric: manifest.metric,
                    encoder: encoderInfo,
                    fileBytes,
                    residentBytes: sketch.residentBytes + 8 * (recordCount + 1),
                    residentVerified: sketch.stats().residentVerified,
                    sampleQueries: manifest.sampleQueries || [],
                };
            },

            async query(text, queryOptions = {}) {
                const k = Number.isInteger(queryOptions.k) && queryOptions.k > 0 ? queryOptions.k : 5;
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
                const quality = pre || scoreQuality(hits, context);
                const returned = quality.match_quality === 'none' ? [] : hits;
                const results = await Promise.all(returned.map(async (hit) => {
                    const record = await hydrate(hit.id);
                    return { id: hit.id, distance: hit.distance, ...record };
                }));
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
                disposeEncoder();
                await sketch.close();
                if (owned) await source.close();
            },
        };
    } catch (err) {
        if (owned && source.close) await source.close().catch(() => {});
        throw err;
    }
}
