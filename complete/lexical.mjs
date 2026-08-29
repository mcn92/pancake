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

export function openLexicalIndex(bytes) {
    if (!(bytes instanceof Uint8Array) || bytes.length < 64) {
        throw new Error('.pancake lexical segment is too short');
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
        throw new Error('.pancake lexical segment regions do not tile the segment');
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
            if (cursor.at >= cursor.end) throw new Error('.pancake lexical postings truncated');
            const b = bytes[cursor.at++];
            value += (b & 0x7f) * 2 ** shift;
            if (!(b & 0x80)) return value;
            shift += 7;
            if (shift > 35) throw new Error('.pancake lexical postings varint overflow');
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
                if (rel + len > postingsBytes) throw new Error('.pancake lexical postings out of bounds');
                const idf = Math.log(1 + (docCount - df + 0.5) / (df + 0.5));
                const cursor = { at: postingsOffset + rel, end: postingsOffset + rel + len };
                let docId = 0;
                for (let p = 0; p < df; p++) {
                    docId = p === 0 ? readVarint(cursor) : docId + readVarint(cursor);
                    const tf = readVarint(cursor);
                    if (docId >= docCount) throw new Error('.pancake lexical postings doc id out of range');
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
