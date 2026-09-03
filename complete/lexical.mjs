// Reader for the lexical index segment (kind 5, layout bm25-v1) — the
// static inverted index a complete artifact MAY carry for hybrid retrieval.
// Byte layout and hashing mirror buildLexicalSegment in builder.mjs; this
// module must stay runtime-neutral (browsers, Workers, Node).
//
// The phase-1 reader receives the whole hash-verified segment; the format
// itself is lazy-friendly (fixed-width hash-sorted term table, absolute
// offsets) so a range-reading opener can replace this one without a format
// revision.

const BM25_K1 = 1.2;
const BM25_B = 0.75;

function fnv1a32(str, seed) {
    let h = 0x811c9dc5 ^ seed;
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
}

// Query tokens follow the builder's shape (lowercase [a-z0-9']+, 2..32
// chars). No stopword list ships or is needed: a token the builder skipped
// simply finds no postings.
const tokenize = (text) => (String(text).toLowerCase().match(/[a-z0-9']+/g) || [])
    .filter((w) => w.length >= 2 && w.length <= 32);

// Lazy opener for wiki-scale lexical segments: only the 64-byte header and
// the doclen array (4 bytes per record — 4 MiB at a million chunks) become
// resident; term lookups and postings stay remote. Term lookup exploits the
// format's own structure: fnv hashes are uniformly distributed and the term
// table is sorted by them, so the table is an implicit interpolation index —
// predict the entry position from the hash value, read a ~512-entry window
// (12 KiB), and the term is almost always inside on the first read, with
// bounded re-aims for the tail. A query term then costs ~1 window read plus
// 1 postings read, issued in parallel across the query's terms.
//
// Integrity stance is transitional, like format-1 sketch rows: the segment
// is committed via the manifest digest but lazy reads are not individually
// verified (the eager opener verifies the whole segment; per-page digests
// are the format-v2 path if this gap needs closing).
export async function openLexicalIndexLazy(read, segLength) {
    const headerBytes = await read(0, 64);
    if (!(headerBytes instanceof Uint8Array) || headerBytes.length < 64 || segLength < 64) {
        throw new Error('.pikelet lexical segment is too short');
    }
    const hv = new DataView(headerBytes.buffer, headerBytes.byteOffset, headerBytes.byteLength);
    const version = hv.getUint32(0, true);
    if (version !== 1) throw new Error(`unsupported lexical segment version ${version}`);
    const docCount = hv.getUint32(4, true);
    const termCount = hv.getUint32(8, true);
    const totalTokens = Number(hv.getBigUint64(12, true));
    const doclenOffset = hv.getUint32(20, true);
    const termTableOffset = hv.getUint32(24, true);
    const postingsOffset = hv.getUint32(28, true);
    const postingsBytes = Number(hv.getBigUint64(32, true));
    if (doclenOffset !== 64
        || termTableOffset !== doclenOffset + 4 * docCount
        || postingsOffset !== termTableOffset + 24 * termCount
        || postingsOffset + postingsBytes !== segLength) {
        throw new Error('.pikelet lexical segment regions do not tile the segment');
    }
    const avgdl = docCount > 0 ? Math.max(1, totalTokens / docCount) : 1;
    const doclenBytes = docCount > 0 ? await read(doclenOffset, 4 * docCount) : new Uint8Array(0);
    const doclenView = new DataView(doclenBytes.buffer, doclenBytes.byteOffset, doclenBytes.byteLength);
    const doclenOf = (id) => doclenView.getUint32(4 * id, true);

    const WINDOW = 512;
    const windowCache = new Map(); // startIdx -> entry bytes (small LRU)
    async function readWindow(startIdx, count) {
        const key = `${startIdx}:${count}`;
        const hit = windowCache.get(key);
        if (hit) return hit;
        const bytes = await read(termTableOffset + 24 * startIdx, 24 * count);
        if (windowCache.size >= 64) windowCache.delete(windowCache.keys().next().value);
        windowCache.set(key, bytes);
        return bytes;
    }
    const entryAt = (bytes, i) => {
        const view = new DataView(bytes.buffer, bytes.byteOffset + 24 * i, 24);
        return {
            hashLo: view.getUint32(0, true),
            hashHi: view.getUint32(4, true),
            postingsRel: Number(view.getBigUint64(8, true)),
            postingsLen: view.getUint32(16, true),
            df: view.getUint32(20, true),
        };
    };
    const cmp = (aHi, aLo, bHi, bLo) => (aHi !== bHi ? (aHi < bHi ? -1 : 1) : aLo === bLo ? 0 : (aLo < bLo ? -1 : 1));

    async function findTerm(hashLo, hashHi) {
        if (termCount === 0) return null;
        let loIdx = 0;
        let hiIdx = termCount - 1;
        let loHash = 0;
        let hiHash = 0xffffffff;
        for (let iter = 0; iter < 10 && loIdx <= hiIdx; iter++) {
            const span = hiHash > loHash ? hiHash - loHash : 1;
            let guess = loIdx + Math.floor((hiIdx - loIdx) * ((hashHi - loHash) / span));
            guess = Math.min(hiIdx, Math.max(loIdx, guess));
            const start = Math.min(Math.max(loIdx, guess - (WINDOW >> 1)), Math.max(loIdx, hiIdx - WINDOW + 1));
            const count = Math.min(WINDOW, hiIdx - start + 1);
            const bytes = await readWindow(start, count);
            const first = entryAt(bytes, 0);
            const last = entryAt(bytes, count - 1);
            if (cmp(hashHi, hashLo, first.hashHi, first.hashLo) < 0) {
                hiIdx = start - 1;
                hiHash = first.hashHi;
                continue;
            }
            if (cmp(hashHi, hashLo, last.hashHi, last.hashLo) > 0) {
                loIdx = start + count;
                loHash = last.hashHi;
                continue;
            }
            let a = 0;
            let b = count - 1;
            while (a <= b) {
                const mid = (a + b) >> 1;
                const e = entryAt(bytes, mid);
                const c = cmp(hashHi, hashLo, e.hashHi, e.hashLo);
                if (c === 0) return e;
                if (c < 0) b = mid - 1;
                else a = mid + 1;
            }
            return null; // window brackets the hash; absence is definitive
        }
        return null;
    }

    return {
        docCount,
        termCount,
        totalTokens,
        lazy: true,
        async search(text, n) {
            const terms = [...new Set(tokenize(text))];
            if (!terms.length || docCount === 0) return [];
            const found = (await Promise.all(terms.map((t) => findTerm(fnv1a32(t, 0), fnv1a32(t, 0x9e3779b9)))))
                .filter(Boolean);
            if (!found.length) return [];
            const postings = await Promise.all(found.map((e) => read(postingsOffset + e.postingsRel, e.postingsLen)));
            const scores = new Map();
            found.forEach((e, i) => {
                const bytes = postings[i];
                const idf = Math.log(1 + (docCount - e.df + 0.5) / (e.df + 0.5));
                const cursor = { at: 0, end: bytes.length };
                const readVarint = () => {
                    let value = 0;
                    let shift = 0;
                    for (;;) {
                        if (cursor.at >= cursor.end) throw new Error('.pikelet lexical postings truncated');
                        const b = bytes[cursor.at++];
                        value += (b & 0x7f) * 2 ** shift;
                        if (!(b & 0x80)) return value;
                        shift += 7;
                        if (shift > 35) throw new Error('.pikelet lexical postings varint overflow');
                    }
                };
                let docId = 0;
                for (let p = 0; p < e.df; p++) {
                    docId = p === 0 ? readVarint() : docId + readVarint();
                    const tf = readVarint();
                    if (docId >= docCount) throw new Error('.pikelet lexical postings doc id out of range');
                    const dl = doclenOf(docId);
                    const norm = tf + BM25_K1 * (1 - BM25_B + (BM25_B * dl) / avgdl);
                    scores.set(docId, (scores.get(docId) || 0) + (idf * tf * (BM25_K1 + 1)) / norm);
                }
            });
            return [...scores.entries()]
                .map(([id, score]) => ({ id, score }))
                .sort((a, b) => (b.score - a.score) || (a.id - b.id))
                .slice(0, n);
        },
    };
}

export function openLexicalIndex(bytes) {
    if (!(bytes instanceof Uint8Array) || bytes.length < 64) {
        throw new Error('.pikelet lexical segment is too short');
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const version = view.getUint32(0, true);
    if (version !== 1) throw new Error(`unsupported lexical segment version ${version}`);
    const docCount = view.getUint32(4, true);
    const termCount = view.getUint32(8, true);
    const totalTokens = Number(view.getBigUint64(12, true));
    const doclenOffset = view.getUint32(20, true);
    const termTableOffset = view.getUint32(24, true);
    const postingsOffset = view.getUint32(28, true);
    const postingsBytes = Number(view.getBigUint64(32, true));
    if (doclenOffset !== 64
        || termTableOffset !== doclenOffset + 4 * docCount
        || postingsOffset !== termTableOffset + 24 * termCount
        || postingsOffset + postingsBytes !== bytes.length) {
        throw new Error('.pikelet lexical segment regions do not tile the segment');
    }
    const avgdl = docCount > 0 ? Math.max(1, totalTokens / docCount) : 1;

    const entryAt = (i) => termTableOffset + 24 * i;
    function findTerm(hashLo, hashHi) {
        let lo = 0;
        let hi = termCount - 1;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            const eLo = view.getUint32(entryAt(mid), true);
            const eHi = view.getUint32(entryAt(mid) + 4, true);
            if (eHi === hashHi && eLo === hashLo) return mid;
            if (eHi < hashHi || (eHi === hashHi && eLo < hashLo)) lo = mid + 1;
            else hi = mid - 1;
        }
        return -1;
    }

    function readVarint(cursor) {
        let value = 0;
        let shift = 0;
        for (;;) {
            if (cursor.at >= cursor.end) throw new Error('.pikelet lexical postings truncated');
            const b = bytes[cursor.at++];
            value += (b & 0x7f) * 2 ** shift;
            if (!(b & 0x80)) return value;
            shift += 7;
            if (shift > 35) throw new Error('.pikelet lexical postings varint overflow');
        }
    }

    return {
        docCount,
        termCount,
        totalTokens,
        // BM25 over the query's unique tokens; returns [{id, score}] sorted
        // by score descending (ties by id), at most n entries.
        search(text, n) {
            const terms = [...new Set(tokenize(text))];
            if (!terms.length || docCount === 0) return [];
            const scores = new Map();
            for (const term of terms) {
                const i = findTerm(fnv1a32(term, 0), fnv1a32(term, 0x9e3779b9));
                if (i < 0) continue;
                const rel = Number(view.getBigUint64(entryAt(i) + 8, true));
                const len = view.getUint32(entryAt(i) + 16, true);
                const df = view.getUint32(entryAt(i) + 20, true);
                if (rel + len > postingsBytes) throw new Error('.pikelet lexical postings out of bounds');
                const idf = Math.log(1 + (docCount - df + 0.5) / (df + 0.5));
                const cursor = { at: postingsOffset + rel, end: postingsOffset + rel + len };
                let docId = 0;
                for (let p = 0; p < df; p++) {
                    docId = p === 0 ? readVarint(cursor) : docId + readVarint(cursor);
                    const tf = readVarint(cursor);
                    if (docId >= docCount) throw new Error('.pikelet lexical postings doc id out of range');
                    const dl = view.getUint32(doclenOffset + 4 * docId, true);
                    const norm = tf + BM25_K1 * (1 - BM25_B + (BM25_B * dl) / avgdl);
                    scores.set(docId, (scores.get(docId) || 0) + (idf * tf * (BM25_K1 + 1)) / norm);
                }
            }
            return [...scores.entries()]
                .map(([id, score]) => ({ id, score }))
                .sort((a, b) => (b.score - a.score) || (a.id - b.id))
                .slice(0, n);
        },
    };
}
