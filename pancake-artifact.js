'use strict';

const { pancakeError, PANCAKE_ERROR_CODES } = require('./pancake-errors.js');

const PANCAKE_MAGIC = 0x504E434B;
const V1_ENVELOPE_HEADER_SIZE = 24;
const V2_ENVELOPE_HEADER_SIZE = 20;
const V3_ENVELOPE_HEADER_SIZE = 32;
const MAPPING_ENTRY_SIZE = 8;
const UINT8_HNSW_MAGIC_V1 = 0x49384831;
const RANGE_MAGIC = 0x31415250; // PRA1
const HEADER_BYTES = 128;

// Read limits are layered, because header fields are untrusted and a hostile
// count/offset must not turn into a multi-GB read or allocation before the
// truncation checks run:
// - MAX_ARTIFACT_READ_BYTES is the absolute backstop on any single read,
//   blocking the 32-bit-overflow class of pathological requests. It is also
//   enforced independently inside NodeFileRangeSource.
// - Open-path reads (id map, router, resident sketch tier) default to the
//   much tighter DEFAULT_OPEN_READ_BYTES, configurable per open() via
//   options.maxReadBytes — resident segments are small by design, so a read
//   near this limit means a hostile or misconfigured header.
// - Query-path coalesced ranges are split at MAX_COALESCED_RANGE_BYTES so no
//   gap/parallelism combination can drive one giant fetch.
// - Whole-segment verification streams in VERIFY_CHUNK_BYTES chunks where a
//   streaming hash exists (Node); WebCrypto cannot stream SHA-256, so other
//   runtimes fall back to a single read bounded by the open budget.
const MAX_ARTIFACT_READ_BYTES = 2 * 1024 * 1024 * 1024;
const DEFAULT_OPEN_READ_BYTES = 256 * 1024 * 1024;
const MAX_COALESCED_RANGE_BYTES = 16 * 1024 * 1024;
const VERIFY_CHUNK_BYTES = 64 * 1024 * 1024;

// Validate an artifact-derived (offset, length) before issuing source.read.
// Rejects non-integer/negative/overflowing ranges, anything past the cap, and
// — when the source reports a numeric size — anything past the source end, so
// the read is refused instead of attempted. Returns the length for convenience.
function checkArtifactRange(source, offset, length, label, limit = MAX_ARTIFACT_READ_BYTES) {
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length)
        || offset < 0 || length < 0 || !Number.isSafeInteger(offset + length)) {
        throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID,
            `Artifact ${label} range is out of bounds`, { offset, length });
    }
    const max = Math.min(limit, MAX_ARTIFACT_READ_BYTES);
    if (length > max) {
        throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID,
            `Artifact ${label} range exceeds the maximum read size`, { length, max });
    }
    if (source && Number.isSafeInteger(source.size) && offset + length > source.size) {
        throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID,
            `Artifact ${label} range extends past the source size`, { offset, length, size: source.size });
    }
    return length;
}

// Read a checked artifact region: validate (offset, length) before touching
// the source so a hostile header cannot drive a giant read or allocation.
async function readChecked(source, offset, length, label, limit) {
    checkArtifactRange(source, offset, length, label, limit);
    return asUint8Array(await source.read(offset, length));
}

// Per-open read budget: undefined keeps the default, Infinity defers to the
// absolute backstop, anything else must be a positive finite number and is
// honored strictly — a budget too small for the artifact's resident segments
// fails the open with a coded error rather than being silently raised.
function resolveMaxReadBytes(value) {
    if (value === undefined) return DEFAULT_OPEN_READ_BYTES;
    if (value === Infinity) return MAX_ARTIFACT_READ_BYTES;
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT,
            'maxReadBytes must be a positive number or Infinity', { maxReadBytes: value });
    }
    return value;
}
const HEADER_BYTES_V2 = 256;
const RANGE_KIND_U8 = 1;
// v3 appends three whole-segment SHA-256 digests (id map, router, base) in
// the second half of the 256-byte header, after the v2 field block.
const RANGE_VERSION = 3;
const RANGE_DIGESTS_OFFSET = 128;
const RANGE_DIGEST_BYTES = 32;
const ROUTER_LOCATION_MASK = 0x80000000;
const LOCATION_ORDINAL_MASK = 0x7fffffff;

function asUint8Array(bytes) {
    if (bytes instanceof Uint8Array) return bytes;
    if (bytes instanceof ArrayBuffer) return new Uint8Array(bytes);
    if (ArrayBuffer.isView(bytes)) {
        return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    }
    throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'Range source returned a non-binary value');
}

// Both readers accept a maxCacheBytes option: undefined keeps the default,
// Infinity disables the bound, and any other value must be a positive number.
// Each reader raises small budgets to a floor so the LRU always holds enough
// records for one search round.
function resolveMaxCacheBytes(value, floorBytes, defaultBytes) {
    if (value === undefined) return defaultBytes;
    if (value === Infinity) return Infinity;
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'maxCacheBytes must be a positive number or Infinity', { maxCacheBytes: value });
    }
    return Math.max(value, floorBytes);
}

function readU32(view, state) {
    const value = view.getUint32(state.offset, true);
    state.offset += 4;
    return value;
}

function readU16(view, state) {
    const value = view.getUint16(state.offset, true);
    state.offset += 2;
    return value;
}

function readF32(view, state) {
    const value = view.getFloat32(state.offset, true);
    state.offset += 4;
    return value;
}

function compareDistancesAsc(a, b) {
    return a.distance - b.distance || a.id - b.id;
}

class MinHeap {
    constructor(compare) {
        this.items = [];
        this.compare = compare;
    }

    get size() {
        return this.items.length;
    }

    push(value) {
        const items = this.items;
        items.push(value);
        this.bubbleUp(items.length - 1);
    }

    bubbleUp(index) {
        const items = this.items;
        const value = items[index];
        while (index > 0) {
            const parent = (index - 1) >> 1;
            if (this.compare(items[parent], value) <= 0) break;
            items[index] = items[parent];
            index = parent;
        }
        items[index] = value;
    }

    pop() {
        const items = this.items;
        if (items.length === 0) return undefined;
        const top = items[0];
        const value = items.pop();
        if (items.length > 0) {
            let index = 0;
            while (true) {
                let child = index * 2 + 1;
                if (child >= items.length) break;
                if (child + 1 < items.length && this.compare(items[child + 1], items[child]) < 0) {
                    child++;
                }
                if (this.compare(items[child], value) >= 0) break;
                items[index] = items[child];
                index = child;
            }
            items[index] = value;
        }
        return top;
    }

    peek() {
        return this.items[0];
    }

    sorted() {
        return [...this.items].sort(this.compare);
    }
}

class NodeFileRangeSource {
    constructor(filePath) {
        const fs = require('fs');
        this.fs = fs;
        this.filePath = filePath;
        this.fd = fs.openSync(filePath, 'r');
        this.size = fs.statSync(filePath).size;
    }

    async read(offset, length) {
        // Defense in depth: the callers validate artifact-derived ranges, but
        // a future caller must not be able to drive a giant Buffer.alloc or a
        // read past the file end through this source directly.
        if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length)
            || offset < 0 || length < 0 || length > MAX_ARTIFACT_READ_BYTES
            || offset + length > this.size) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID,
                'NodeFileRangeSource.read() range is out of bounds', { offset, length, size: this.size });
        }
        const buffer = Buffer.alloc(length);
        let bytesRead = 0;
        while (bytesRead < length) {
            const chunk = this.fs.readSync(this.fd, buffer, bytesRead, length - bytesRead, offset + bytesRead);
            if (chunk === 0) break;
            bytesRead += chunk;
        }
        // Return only the bytes actually read so callers' truncation checks
        // fire instead of parsing an unwritten buffer tail.
        return bytesRead === length ? buffer : buffer.subarray(0, bytesRead);
    }

    async close() {
        if (this.fd !== null) {
            this.fs.closeSync(this.fd);
            this.fd = null;
        }
    }
}

class PancakeRangeArtifact {
    constructor(source, header, idMap, options = {}) {
        this.source = source;
        this.version = header.version;
        this.kind = header.kind;
        this.dim = header.dim;
        this.count = header.count;
        this.entryPoint = header.entryPoint;
        this.maxLevel = header.maxLevel;
        this.M = header.M;
        this.M0 = header.M0;
        this.metric = header.metric;
        this.recordBytes = header.recordBytes;
        this.idMapOffset = header.idMapOffset;
        this.recordsOffset = header.recordsOffset;
        this.parts = header.parts;
        this.routerCount = header.routerCount;
        this.baseCount = header.baseCount;
        this.routerRecordsOffset = header.routerRecordsOffset;
        this.baseRecordsOffset = header.baseRecordsOffset;
        this.originalToLocation = idMap;
        this.cache = new Map();
        // Lazily fetched records live in a byte-budgeted LRU so a long-lived
        // reader stays bounded; the router segment (this.cache) is permanent.
        this.lazyCache = new Map();
        this.lazyCacheBytes = 0;
        this.maxCacheBytes = resolveMaxCacheBytes(options.maxCacheBytes, 64 * this.recordBytes, 64 * 1024 * 1024);
        this.currentRanges = [];
        this.routerResident = { records: 0, bytes: 0 };
        this.rangeRequests = 0;
        this.rangeBytes = 0;
        this.rangeNodesDecoded = 0;
        this.loadRouter = options.loadRouter !== false;
        this.maxReadBytes = resolveMaxReadBytes(options.maxReadBytes);
        this.verify = options.verify !== false;
        // v3 artifacts carry whole-segment digests; older versions have none.
        this.digests = null;
        this.segmentVerified = { idMap: false, router: false, base: false };
    }

    static async open(source, options = {}) {
        if (!source || typeof source.read !== 'function') {
            throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'PancakeRangeArtifact.open() requires a range source with read(offset, length)');
        }
        const headerBytes = await readChecked(source, 0, HEADER_BYTES, 'header');
        if (headerBytes.byteLength < HEADER_BYTES) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact header is truncated');
        }
        const header = parseHeader(headerBytes);
        // Resolved before construction: the id-map read below is driven by an
        // untrusted header count and must respect the caller's budget.
        const maxReadBytes = resolveMaxReadBytes(options.maxReadBytes);
        let digests = null;
        if (header.version >= 3) {
            const digestBytes = await readChecked(source, RANGE_DIGESTS_OFFSET, RANGE_DIGEST_BYTES * 3, 'header digests');
            if (digestBytes.byteLength !== RANGE_DIGEST_BYTES * 3) {
                throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact header digests are truncated');
            }
            digests = {
                idMap: new Uint8Array(digestBytes.subarray(0, RANGE_DIGEST_BYTES)),
                router: new Uint8Array(digestBytes.subarray(RANGE_DIGEST_BYTES, RANGE_DIGEST_BYTES * 2)),
                base: new Uint8Array(digestBytes.subarray(RANGE_DIGEST_BYTES * 2, RANGE_DIGEST_BYTES * 3)),
            };
        }
        const idMapBytes = await readChecked(source, header.idMapOffset, header.count * 4, 'id map', maxReadBytes);
        if (idMapBytes.byteLength !== header.count * 4) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact id map is truncated');
        }
        const copied = new Uint8Array(idMapBytes.byteLength);
        copied.set(idMapBytes);
        const idMap = new Uint32Array(copied.buffer);
        const artifact = new PancakeRangeArtifact(source, header, idMap, options);
        artifact.digests = digests;
        if (digests && artifact.verify) {
            await verifySha256(copied, digests.idMap, 'Range artifact id map');
            artifact.segmentVerified.idMap = true;
        }
        if (artifact.loadRouter && artifact.version >= 2) {
            artifact.routerResident = await artifact.loadRouterSegment();
            artifact.resetStats();
        }
        return artifact;
    }

    static async openFile(filePath, options = {}) {
        return PancakeRangeArtifact.open(new NodeFileRangeSource(filePath), options);
    }

    async close() {
        if (this.source && typeof this.source.close === 'function') {
            await this.source.close();
        }
    }

    resetStats() {
        this.rangeRequests = 0;
        this.rangeBytes = 0;
        this.rangeNodesDecoded = 0;
        this.currentRanges = [];
    }

    clearCache(options = {}) {
        this.cache.clear();
        this.lazyCache.clear();
        this.lazyCacheBytes = 0;
        if (options.reloadRouter !== false && this.loadRouter && this.version >= 2) {
            return this.loadRouterSegment().then((resident) => {
                this.routerResident = resident;
                this.resetStats();
                return resident;
            });
        }
        return Promise.resolve({ records: 0, bytes: 0 });
    }

    stats() {
        return {
            rangeRequests: this.rangeRequests,
            rangeBytes: this.rangeBytes,
            rangeNodesDecoded: this.rangeNodesDecoded,
            cachedNodes: this.cache.size + this.lazyCache.size,
            lazyCacheBytes: this.lazyCacheBytes,
            routerResident: { ...this.routerResident },
            segmentVerified: { ...this.segmentVerified },
        };
    }

    // Verify the lazily-read base segment against the header's whole-segment
    // digest (v3+). Reads the segment in bounded chunks with a streaming hash
    // (Node); runtimes without one fall back to a single read bounded by the
    // open budget. Intended for producers, CI, and hosts that materialize
    // artifacts, not the query path. Verifying individual lazy ranges needs
    // per-chunk commitments, which arrive with the contract's
    // complete-profile manifest.
    async verifyBaseSegment(options = {}) {
        if (!this.digests) {
            throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT,
                'Range artifact predates segment digests (format v3); nothing to verify against', { version: this.version });
        }
        const bytes = this.baseCount * this.recordBytes;
        await verifySegmentSha256(this.source, this.baseRecordsOffset, bytes, this.digests.base,
            'Range artifact base segment', { chunkBytes: options.chunkBytes, oneShotLimit: this.maxReadBytes });
        this.segmentVerified.base = true;
        return true;
    }

    markRanges() {
        return this.currentRanges.length;
    }

    rangesSince(mark) {
        return this.currentRanges.slice(mark);
    }

    recordAddressForId(id) {
        if (!Number.isInteger(id) || id < 0 || id >= this.count) {
            throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, `Node id ${id} is outside artifact bounds`, { id, count: this.count });
        }
        const location = this.originalToLocation[id];
        if (this.version >= 2) {
            const ordinal = location & LOCATION_ORDINAL_MASK;
            if ((location & ROUTER_LOCATION_MASK) !== 0) {
                if (ordinal >= this.routerCount) throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, `Router ordinal ${ordinal} is outside artifact`);
                return this.routerRecordsOffset + ordinal * this.recordBytes;
            }
            if (ordinal >= this.baseCount) throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, `Base ordinal ${ordinal} is outside artifact`);
            return this.baseRecordsOffset + ordinal * this.recordBytes;
        }
        if (location >= this.count) throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, `Node id ${id} is not addressable in artifact`);
        return this.recordsOffset + location * this.recordBytes;
    }

    cachedNode(id) {
        const resident = this.cache.get(id);
        if (resident !== undefined) return resident;
        const lazy = this.lazyCache.get(id);
        if (lazy !== undefined) {
            // Touch for LRU: move to the tail of insertion order.
            this.lazyCache.delete(id);
            this.lazyCache.set(id, lazy);
            return lazy;
        }
        return undefined;
    }

    cacheLazyNode(id, node) {
        if (this.lazyCache.has(id)) return;
        this.lazyCache.set(id, node);
        this.lazyCacheBytes += this.recordBytes;
        while (this.lazyCacheBytes > this.maxCacheBytes && this.lazyCache.size > 1) {
            const oldest = this.lazyCache.keys().next().value;
            this.lazyCache.delete(oldest);
            this.lazyCacheBytes -= this.recordBytes;
        }
    }

    async readNode(id) {
        const cached = this.cachedNode(id);
        if (cached !== undefined) return cached;
        await this.prefetch([id]);
        const node = this.cachedNode(id);
        if (node === undefined) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID,
                'Range artifact record for id was not resolved by its address', { id });
        }
        return node;
    }

    async prefetch(ids, options = {}) {
        const addresses = [];
        const seen = new Set();
        for (const id of ids) {
            if (this.cache.has(id) || this.lazyCache.has(id) || seen.has(id)) continue;
            seen.add(id);
            addresses.push(this.recordAddressForId(id));
        }
        if (addresses.length === 0) return 0;
        addresses.sort((a, b) => a - b);
        const gap = Math.max(0, options.gap || 0);
        const gapBytes = this.version >= 2 ? gap : gap * this.recordBytes;
        const rangeParallelism = Math.max(1, Math.trunc(options.parallelism || 1));
        // Coalesced runs are split at maxRangeBytes (record-aligned) so no
        // gap/parallelism combination — or a bulk prefetch of the whole base
        // segment — turns into one unbounded read and allocation.
        const maxRangeBytes = Math.max(this.recordBytes,
            Math.floor(Math.trunc(options.maxRangeBytes || MAX_COALESCED_RANGE_BYTES) / this.recordBytes) * this.recordBytes);
        const ranges = [];
        let runStart = addresses[0];
        let runEnd = addresses[0] + this.recordBytes;
        const flush = () => {
            for (let piece = runStart; piece < runEnd; piece += maxRangeBytes) {
                ranges.push([piece, Math.min(piece + maxRangeBytes, runEnd)]);
            }
        };
        const readRange = async ([start, end]) => {
            const bytes = end - start;
            const buffer = await readChecked(this.source, start, bytes, 'record');
            if (buffer.byteLength !== bytes) {
                throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact record read returned a truncated range', { offset: start, bytes, actual: buffer.byteLength });
            }
            return { start, end, bytes, buffer };
        };
        const decodeRange = ({ start, end, bytes, buffer }) => {
            this.rangeRequests++;
            this.rangeBytes += bytes;
            this.currentRanges.push([start, end]);
            for (let off = 0; off + this.recordBytes <= bytes; off += this.recordBytes) {
                const record = buffer.subarray(off, off + this.recordBytes);
                const originalId = new DataView(record.buffer, record.byteOffset, record.byteLength).getUint32(0, true);
                // The embedded id is untrusted: it must map back to the byte
                // address this record was read from, or a lying record would
                // poison the cache under an attacker-chosen key and searches
                // for the real id would die uncoded.
                if (originalId >= this.count || this.recordAddressForId(originalId) !== start + off) {
                    throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID,
                        'Range artifact record id does not match its address',
                        { originalId, address: start + off });
                }
                if (!this.cache.has(originalId) && !this.lazyCache.has(originalId)) {
                    this.cacheLazyNode(originalId, this.decodeNode(record));
                    this.rangeNodesDecoded++;
                }
            }
        };
        for (let i = 1; i < addresses.length; i++) {
            const address = addresses[i];
            if (address <= runEnd + gapBytes) {
                runEnd = address + this.recordBytes;
            } else {
                flush();
                runStart = address;
                runEnd = address + this.recordBytes;
            }
        }
        flush();
        for (let i = 0; i < ranges.length; i += rangeParallelism) {
            const batch = ranges.slice(i, i + rangeParallelism);
            const results = await Promise.all(batch.map(readRange));
            for (const result of results) decodeRange(result);
        }
        return ranges.length;
    }

    async loadRouterSegment() {
        if (this.version < 2 || this.routerCount === 0) return { records: 0, bytes: 0 };
        const bytes = this.routerCount * this.recordBytes;
        const buffer = await readChecked(this.source, this.routerRecordsOffset, bytes, 'router segment', this.maxReadBytes);
        if (buffer.byteLength !== bytes) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact router segment is truncated');
        }
        if (this.digests && this.verify) {
            await verifySha256(buffer, this.digests.router, 'Range artifact router segment');
            this.segmentVerified.router = true;
        }
        for (let i = 0; i < this.routerCount; i++) {
            const record = buffer.subarray(i * this.recordBytes, (i + 1) * this.recordBytes);
            const originalId = new DataView(record.buffer, record.byteOffset, record.byteLength).getUint32(0, true);
            const address = this.routerRecordsOffset + i * this.recordBytes;
            // Same untrusted-id check as decodeRange: the embedded id must map
            // back to this router slot's address.
            if (originalId >= this.count || this.recordAddressForId(originalId) !== address) {
                throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID,
                    'Range artifact router record id does not match its address',
                    { originalId, address });
            }
            if (!this.cache.has(originalId)) this.cache.set(originalId, this.decodeNode(record));
        }
        return { records: this.routerCount, bytes };
    }

    decodeNode(bytes) {
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const state = { offset: 0 };
        const id = readU32(view, state);
        const level = readU16(view, state);
        const baseCount = readU16(view, state);
        // Records are untrusted bytes. Counts beyond the header geometry would
        // read filler (or zeros) as edges; fail closed instead.
        if (level > this.maxLevel || baseCount > this.M0) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact record structure is inconsistent', { id, level, baseCount });
        }
        const upperCounts = new Uint16Array(this.maxLevel);
        for (let i = 0; i < this.maxLevel; i++) {
            upperCounts[i] = readU16(view, state);
            if (upperCounts[i] > this.M) {
                throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact record structure is inconsistent', { id, level: i + 1, edges: upperCounts[i] });
            }
        }
        const qdata = new Uint8Array(this.dim);
        qdata.set(bytes.subarray(state.offset, state.offset + this.dim));
        state.offset += this.dim;
        const scale = readF32(view, state);
        const offset = readF32(view, state);
        const base = new Uint32Array(baseCount);
        for (let i = 0; i < this.M0; i++) {
            const neighbor = readU32(view, state);
            if (i < baseCount) {
                if (neighbor >= this.count) {
                    throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact record has an out-of-bounds neighbor', { id, neighbor, count: this.count });
                }
                base[i] = neighbor;
            }
        }
        const upper = Array.from({ length: this.maxLevel }, () => new Uint32Array(0));
        for (let levelIndex = 0; levelIndex < this.maxLevel; levelIndex++) {
            const edges = new Uint32Array(upperCounts[levelIndex]);
            for (let i = 0; i < this.M; i++) {
                const neighbor = readU32(view, state);
                if (i < edges.length) {
                    if (neighbor >= this.count) {
                        throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact record has an out-of-bounds neighbor', { id, neighbor, count: this.count });
                    }
                    edges[i] = neighbor;
                }
            }
            upper[levelIndex] = edges;
        }
        return { id, level, base, upper, qdata, scale, offset };
    }

    distance(query, node) {
        if (this.metric === 1) {
            let dot = 0;
            for (let d = 0; d < this.dim; d++) {
                dot += query[d] * (node.offset + node.scale * node.qdata[d]);
            }
            if (dot > 1) dot = 1;
            else if (dot < -1) dot = -1;
            return 1 - dot;
        }
        let sum = 0;
        for (let d = 0; d < this.dim; d++) {
            const decoded = node.offset + node.scale * node.qdata[d];
            const diff = query[d] - decoded;
            sum += diff * diff;
        }
        return sum;
    }

    async search(queryInput, k, options = {}) {
        const query = normalizeQuery(queryInput, this.dim, this.metric);
        if (!Number.isInteger(k) || k <= 0) {
            throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'search() k must be a positive integer', { k });
        }
        const efSearch = Math.max(options.efSearch || 100, k);
        const expansionBatch = Math.max(1, Math.trunc(options.expansionBatch || 1));
        const rounds = [];
        const prefetchRound = async (ids) => {
            if (!ids.length) return;
            const beforeRequests = this.rangeRequests;
            const beforeBytes = this.rangeBytes;
            const beforeRanges = this.markRanges();
            await this.prefetch(ids, { gap: options.gap || 0, parallelism: options.rangeParallelism || options.parallelism || 1, maxRangeBytes: options.maxRangeBytes });
            const ranges = this.rangesSince(beforeRanges);
            rounds.push({
                ids: ids.length,
                requests: this.rangeRequests - beforeRequests,
                bytes: this.rangeBytes - beforeBytes,
                rangeBytes: ranges.map(([start, end]) => end - start),
            });
        };

        let current = this.entryPoint;
        await prefetchRound([current]);
        let currentNode = await this.readNode(current);
        let currentDistance = this.distance(query, currentNode);

        for (let level = this.maxLevel; level > 0; level--) {
            let changed = true;
            while (changed) {
                changed = false;
                const edges = currentNode.upper[level - 1] || [];
                await prefetchRound(edges);
                for (const neighbor of edges) {
                    const node = await this.readNode(neighbor);
                    const distance = this.distance(query, node);
                    if (distance < currentDistance) {
                        current = neighbor;
                        currentNode = node;
                        currentDistance = distance;
                        changed = true;
                    }
                }
            }
        }

        const visited = new Uint8Array(this.count);
        const candidates = new MinHeap(compareDistancesAsc);
        const results = new MinHeap((a, b) => b.distance - a.distance || b.id - a.id);
        currentNode = await this.readNode(current);
        const distance0 = this.distance(query, currentNode);
        candidates.push({ distance: distance0, id: current });
        results.push({ distance: distance0, id: current });
        visited[current] = 1;

        while (candidates.size > 0) {
            const batch = [];
            while (batch.length < expansionBatch && candidates.size > 0) {
                const candidate = candidates.peek();
                const worst = results.peek();
                if (results.size >= efSearch && worst && candidate && candidate.distance > worst.distance) break;
                batch.push(candidates.pop());
            }
            if (!batch.length) break;

            const toVisit = [];
            for (const candidate of batch) {
                const node = await this.readNode(candidate.id);
                for (const neighbor of node.base) {
                    if (visited[neighbor]) continue;
                    visited[neighbor] = 1;
                    toVisit.push(neighbor);
                }
            }
            await prefetchRound(toVisit);
            for (const neighbor of toVisit) {
                const neighborNode = await this.readNode(neighbor);
                const distance = this.distance(query, neighborNode);
                const currentWorst = results.peek();
                if (results.size < efSearch || distance < currentWorst.distance) {
                    candidates.push({ distance, id: neighbor });
                    results.push({ distance, id: neighbor });
                    if (results.size > efSearch) results.pop();
                }
            }
        }

        const top = results.items.sort(compareDistancesAsc).slice(0, k);
        // Traversal orders by squared L2; the API contract (README "Distance
        // values") reports Euclidean, matching PancakeIndex.search.
        if (this.metric !== 1) {
            for (const hit of top) hit.distance = Math.sqrt(hit.distance);
        }
        return {
            results: top,
            rounds,
            stats: this.stats(),
        };
    }
}

function normalizeQuery(query, dim, metric = 0) {
    if (!(query instanceof Float32Array)) {
        if (!Array.isArray(query)) {
            throw pancakeError(PANCAKE_ERROR_CODES.INVALID_VECTOR, 'search() query must be a Float32Array or number[]');
        }
        query = Float32Array.from(query);
    }
    if (query.length !== dim) {
        throw pancakeError(PANCAKE_ERROR_CODES.DIMENSION_MISMATCH, `search() query dimension ${query.length} does not match artifact dimension ${dim}`, { queryDim: query.length, dim });
    }
    if (metric === 1) {
        let norm = 0;
        for (let i = 0; i < query.length; i++) norm += query[i] * query[i];
        norm = Math.sqrt(norm);
        if (!Number.isFinite(norm) || norm <= 1e-30) {
            throw pancakeError(PANCAKE_ERROR_CODES.INVALID_VECTOR, 'search() query has invalid cosine norm');
        }
        const normalized = new Float32Array(query.length);
        for (let i = 0; i < query.length; i++) normalized[i] = query[i] / norm;
        return normalized;
    }
    return query;
}

function parseHeader(headerBytes) {
    const view = new DataView(headerBytes.buffer, headerBytes.byteOffset, headerBytes.byteLength);
    const state = { offset: 0 };
    const magic = readU32(view, state);
    if (magic !== RANGE_MAGIC) {
        throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Not a Pancake range artifact', { magic });
    }
    const version = readU32(view, state);
    const kind = readU32(view, state);
    if (kind !== RANGE_KIND_U8) {
        throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported Pancake range artifact kind', { kind });
    }
    const header = {
        version,
        kind,
        dim: readU32(view, state),
        count: readU32(view, state),
        entryPoint: readU32(view, state),
        maxLevel: readU32(view, state),
        M: readU32(view, state),
        M0: readU32(view, state),
        metric: readU32(view, state),
        recordBytes: readU32(view, state),
        idMapOffset: readU32(view, state),
        recordsOffset: readU32(view, state),
        parts: readU32(view, state),
        routerCount: 0,
        baseCount: 0,
        routerRecordsOffset: 0,
        baseRecordsOffset: 0,
    };
    if (header.metric !== 0) {
        if (header.metric !== 1) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported Pancake range artifact metric', { metric: header.metric });
        }
    }
    if (version >= 2) {
        header.routerCount = readU32(view, state);
        header.baseCount = readU32(view, state);
        header.routerRecordsOffset = readU32(view, state);
        header.baseRecordsOffset = readU32(view, state);
    } else {
        header.baseCount = header.count;
        header.baseRecordsOffset = header.recordsOffset;
    }
    if (version < 1 || version > RANGE_VERSION) {
        throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported Pancake range artifact version', { version });
    }
    // recordBytes must agree with the layout the geometry implies, or
    // decodeNode would read past record boundaries on a corrupt header.
    const expectedRecordBytes = 4 + 2 + 2 + header.maxLevel * 2 + header.dim
        + 8 + header.M0 * 4 + header.maxLevel * header.M * 4;
    if (header.dim < 1 || header.count < 1 || header.M < 1 || header.M0 < 1
        || header.maxLevel > 64
        || header.entryPoint >= header.count
        || header.recordBytes !== expectedRecordBytes
        || (version >= 2 && header.routerCount + header.baseCount !== header.count)) {
        throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact header geometry is inconsistent', {
            dim: header.dim, count: header.count, maxLevel: header.maxLevel,
            M: header.M, M0: header.M0, recordBytes: header.recordBytes, expectedRecordBytes,
        });
    }
    return header;
}

function unwrapSnapshot(bytes) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (bytes.byteLength < 4 || view.getUint32(0, true) !== PANCAKE_MAGIC) return bytes;
    // Envelope fields are untrusted: every size read from them must be
    // checked against the actual buffer before slicing, so a corrupt
    // envelope fails closed with a coded error instead of a raw RangeError.
    if (bytes.byteLength < 8) {
        throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Pancake envelope is truncated', { byteLength: bytes.byteLength });
    }
    const version = view.getUint32(4, true);
    if (version === 1 || version === 2) {
        const headerSize = version === 1 ? V1_ENVELOPE_HEADER_SIZE : V2_ENVELOPE_HEADER_SIZE;
        if (bytes.byteLength < headerSize) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Pancake envelope is truncated', { version, byteLength: bytes.byteLength });
        }
        return bytes.subarray(headerSize);
    }
    if (version === 3) {
        if (bytes.byteLength < V3_ENVELOPE_HEADER_SIZE) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Pancake envelope is truncated', { version, byteLength: bytes.byteLength });
        }
        const mappingCount = view.getUint32(24, true);
        const rawSize = view.getUint32(28, true);
        const rawOffset = V3_ENVELOPE_HEADER_SIZE + mappingCount * MAPPING_ENTRY_SIZE;
        if (rawOffset + rawSize > bytes.byteLength) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Pancake envelope declares more data than the snapshot contains', { mappingCount, rawSize, byteLength: bytes.byteLength });
        }
        return bytes.subarray(rawOffset, rawOffset + rawSize);
    }
    throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, `Unsupported Pancake envelope version ${version}`, { version });
}

function parseUint8Snapshot(bytes) {
    const raw = unwrapSnapshot(asUint8Array(bytes));
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    let offset = 0;
    // Snapshot bytes are untrusted: sizes read from the header drive every
    // subsequent read, so each read checks the remaining buffer and fails
    // closed with a coded error instead of a raw RangeError.
    const truncated = () => pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'uint8 snapshot is truncated', { offset, byteLength: raw.byteLength });
    const u32 = () => {
        if (raw.byteLength - offset < 4) throw truncated();
        const value = view.getUint32(offset, true);
        offset += 4;
        return value;
    };
    const f32 = () => {
        if (raw.byteLength - offset < 4) throw truncated();
        const value = view.getFloat32(offset, true);
        offset += 4;
        return value;
    };

    const magic = u32();
    const dim = u32();
    const version = u32();
    const count = u32();
    const entryPoint = u32();
    const maxLevel = u32();
    const M = u32();
    const M0 = u32();
    const metric = u32();
    const efConstruction = u32();
    if (magic !== UINT8_HNSW_MAGIC_V1) {
        throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Search Artifact export currently supports uint8 Pancake snapshots only');
    }
    // Contract §6: reject unknown future versions instead of parsing them
    // as v2 — a changed layout must fail closed, not misparse.
    if (version > 2) {
        throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported uint8 snapshot format version', { version });
    }
    if (metric !== 0 && metric !== 1) {
        throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported uint8 snapshot metric', { metric });
    }
    // Same structural sanity the engine's own deserialize enforces; the
    // level cap mirrors MAX_DESERIALIZE_LEVEL in src/uint8_float_hnsw.hpp.
    if (dim < 1 || count < 1 || entryPoint >= count || maxLevel > 64
        || !Number.isSafeInteger(count * dim)
        || raw.byteLength - offset < count * 8 + count * dim) {
        throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'uint8 snapshot header is inconsistent', { dim, count, entryPoint, maxLevel, byteLength: raw.byteLength });
    }

    const scales = new Float32Array(count);
    const offsets = new Float32Array(count);
    for (let i = 0; i < count; i++) scales[i] = f32();
    for (let i = 0; i < count; i++) offsets[i] = f32();
    const qdata = raw.subarray(offset, offset + count * dim);
    offset += qdata.byteLength;

    const levels = new Uint16Array(count);
    const base = new Array(count);
    const upper = Array.from({ length: count }, () => []);
    for (let id = 0; id < count; id++) {
        const level = u32();
        if (level > maxLevel) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'uint8 snapshot node level exceeds header maxLevel', { id, level, maxLevel });
        }
        levels[id] = level;
        for (let l = 0; l <= level; l++) {
            const size = u32();
            const edges = new Uint32Array(size);
            for (let e = 0; e < size; e++) {
                edges[e] = u32();
                offset += 4;
            }
            if (l === 0) base[id] = edges;
            else upper[id][l - 1] = edges;
        }
        if (!base[id]) base[id] = new Uint32Array(0);
    }

    return { kind: 'u8', dim, version, count, entryPoint, maxLevel, M, M0, metric, efConstruction, scales, offsets, qdata, levels, base, upper };
}

function artifactRecordBytes(index) {
    return 4 + 2 + 2 + index.maxLevel * 2 + index.dim + 8 + index.M0 * 4 + index.maxLevel * index.M * 4;
}

function writeNodeRecord(index, id, record) {
    record.fill(0);
    let offset = 0;
    record.writeUInt32LE(id, offset); offset += 4;
    record.writeUInt16LE(index.levels[id], offset); offset += 2;
    const baseEdges = index.base[id];
    record.writeUInt16LE(baseEdges.length, offset); offset += 2;
    for (let level = 1; level <= index.maxLevel; level++) {
        record.writeUInt16LE((index.upper[id][level - 1] || []).length, offset);
        offset += 2;
    }
    Buffer.from(index.qdata.buffer, index.qdata.byteOffset + id * index.dim, index.dim).copy(record, offset);
    offset += index.dim;
    record.writeFloatLE(index.scales[id], offset); offset += 4;
    record.writeFloatLE(index.offsets[id], offset); offset += 4;
    for (let i = 0; i < index.M0; i++) {
        record.writeUInt32LE(i < baseEdges.length ? baseEdges[i] : 0xFFFFFFFF, offset);
        offset += 4;
    }
    for (let level = 1; level <= index.maxLevel; level++) {
        const edges = index.upper[id][level - 1] || [];
        for (let i = 0; i < index.M; i++) {
            record.writeUInt32LE(i < edges.length ? edges[i] : 0xFFFFFFFF, offset);
            offset += 4;
        }
    }
}

function reverseCuthillMckeePermutation(index) {
    const count = index.count;
    const degrees = new Uint32Array(count);
    for (let id = 0; id < count; id++) degrees[id] = index.base[id].length;
    const starts = Array.from({ length: count }, (_, id) => id);
    starts.sort((a, b) => degrees[a] - degrees[b] || a - b);
    const visited = new Uint8Array(count);
    const order = new Uint32Array(count);
    const queue = new Uint32Array(count);
    let orderLen = 0;
    for (const start of starts) {
        if (visited[start]) continue;
        let read = 0;
        let write = 0;
        queue[write++] = start;
        visited[start] = 1;
        while (read < write) {
            const id = queue[read++];
            order[orderLen++] = id;
            const neighbors = Array.from(index.base[id]).filter((neighbor) => !visited[neighbor]);
            neighbors.sort((a, b) => degrees[a] - degrees[b] || a - b);
            for (const neighbor of neighbors) {
                if (visited[neighbor]) continue;
                visited[neighbor] = 1;
                queue[write++] = neighbor;
            }
        }
    }
    const ordinalToOriginal = new Uint32Array(count);
    for (let pos = 0; pos < orderLen; pos++) ordinalToOriginal[pos] = order[orderLen - 1 - pos];
    return ordinalToOriginal;
}

function identityPermutation(count) {
    const out = new Uint32Array(count);
    for (let i = 0; i < count; i++) out[i] = i;
    return out;
}

function buildOrdinalToOriginal(index, layout) {
    if (layout === 'identity') return identityPermutation(index.count);
    if (layout === 'rcm') return reverseCuthillMckeePermutation(index);
    throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, `Unsupported Search Artifact layout '${layout}'`, { layout });
}

function exportSplitArtifact(index, outPath, options = {}) {
    const fs = require('fs');
    const layoutName = options.layout || 'rcm';
    const ordinalToOriginal = buildOrdinalToOriginal(index, layoutName);
    const recBytes = artifactRecordBytes(index);
    const idMapOffset = HEADER_BYTES_V2;
    const routerIds = [];
    const baseIds = [];
    const locationMap = new Uint32Array(index.count);
    for (let ordinal = 0; ordinal < index.count; ordinal++) {
        const id = ordinalToOriginal[ordinal];
        if (index.levels[id] > 0) routerIds.push(id);
        else baseIds.push(id);
    }
    for (let i = 0; i < routerIds.length; i++) locationMap[routerIds[i]] = ROUTER_LOCATION_MASK | i;
    for (let i = 0; i < baseIds.length; i++) locationMap[baseIds[i]] = i;

    const routerRecordsOffset = idMapOffset + index.count * 4;
    const baseRecordsOffset = routerRecordsOffset + routerIds.length * recBytes;
    const totalBytes = baseRecordsOffset + baseIds.length * recBytes;
    const fd = fs.openSync(outPath, 'w');
    // Whole-segment digests (v3): streamed while the segments are written so
    // the file is hashed exactly once, then stamped into the header at the
    // end. The id map and router are verified by the reader at open; the base
    // segment is verifiable on demand via verifyBaseSegment().
    const crypto = require('crypto');
    const idMapBuffer = Buffer.from(locationMap.buffer);
    const idMapDigest = crypto.createHash('sha256').update(idMapBuffer).digest();
    const routerHash = crypto.createHash('sha256');
    const baseHash = crypto.createHash('sha256');
    let routerDigest;
    let baseDigest;
    try {
        fs.writeSync(fd, idMapBuffer, 0, index.count * 4, idMapOffset);

        const record = Buffer.alloc(recBytes);
        for (let i = 0; i < routerIds.length; i++) {
            writeNodeRecord(index, routerIds[i], record);
            routerHash.update(record);
            fs.writeSync(fd, record, 0, record.length, routerRecordsOffset + i * recBytes);
        }
        for (let i = 0; i < baseIds.length; i++) {
            writeNodeRecord(index, baseIds[i], record);
            baseHash.update(record);
            fs.writeSync(fd, record, 0, record.length, baseRecordsOffset + i * recBytes);
        }

        const header = Buffer.alloc(HEADER_BYTES_V2);
        let h = 0;
        header.writeUInt32LE(RANGE_MAGIC, h); h += 4;
        header.writeUInt32LE(RANGE_VERSION, h); h += 4;
        header.writeUInt32LE(RANGE_KIND_U8, h); h += 4;
        header.writeUInt32LE(index.dim, h); h += 4;
        header.writeUInt32LE(index.count, h); h += 4;
        header.writeUInt32LE(index.entryPoint, h); h += 4;
        header.writeUInt32LE(index.maxLevel, h); h += 4;
        header.writeUInt32LE(index.M, h); h += 4;
        header.writeUInt32LE(index.M0, h); h += 4;
        header.writeUInt32LE(index.metric, h); h += 4;
        header.writeUInt32LE(recBytes, h); h += 4;
        header.writeUInt32LE(idMapOffset, h); h += 4;
        header.writeUInt32LE(routerRecordsOffset, h); h += 4;
        header.writeUInt32LE(options.parts || 0, h); h += 4;
        header.writeUInt32LE(routerIds.length, h); h += 4;
        header.writeUInt32LE(baseIds.length, h); h += 4;
        header.writeUInt32LE(routerRecordsOffset, h); h += 4;
        header.writeUInt32LE(baseRecordsOffset, h); h += 4;
        routerDigest = routerHash.digest();
        baseDigest = baseHash.digest();
        idMapDigest.copy(header, RANGE_DIGESTS_OFFSET);
        routerDigest.copy(header, RANGE_DIGESTS_OFFSET + RANGE_DIGEST_BYTES);
        baseDigest.copy(header, RANGE_DIGESTS_OFFSET + RANGE_DIGEST_BYTES * 2);
        fs.writeSync(fd, header, 0, header.length, 0);
    } finally {
        fs.closeSync(fd);
    }
    return {
        format: 'pancake-range-artifact',
        formatVersion: RANGE_VERSION,
        integrity: {
            idMapSha256: idMapDigest.toString('hex'),
            routerSha256: routerDigest.toString('hex'),
            baseSha256: baseDigest.toString('hex'),
        },
        file: outPath,
        sizeBytes: totalBytes,
        kind: 'u8-affine',
        metric: index.metric === 0 ? 'l2' : 'cosine',
        layout: { permutation: layoutName, split: 'router_then_base' },
        graph: {
            count: index.count,
            dim: index.dim,
            entryPoint: index.entryPoint,
            maxLevel: index.maxLevel,
            M: index.M,
            M0: index.M0,
        },
        addressing: {
            headerBytes: HEADER_BYTES_V2,
            idMapOffset,
            recordBytes: recBytes,
            routerCount: routerIds.length,
            routerBytes: routerIds.length * recBytes,
            routerRecordsOffset,
            baseCount: baseIds.length,
            baseBytes: baseIds.length * recBytes,
            baseRecordsOffset,
        },
    };
}

function buildRangeArtifact(snapshotBytes, outPath, options = {}) {
    if (typeof outPath !== 'string' || outPath.length === 0) {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'buildRangeArtifact() requires an output path');
    }
    const index = parseUint8Snapshot(snapshotBytes);
    return exportSplitArtifact(index, outPath, options);
}

function buildRangeArtifactFile(snapshotPath, outPath, options = {}) {
    const fs = require('fs');
    const snapshot = fs.readFileSync(snapshotPath);
    return buildRangeArtifact(snapshot, outPath, options);
}

// ============================================================================
// Sketch artifact profile (.pancake-sketch) — spec/SKETCH_PROFILE.md
// ============================================================================
//
// Layout: 256-byte header | scales f32[count] | offsets f32[count] |
// packed pooled sketches | zero padding to 16-byte alignment | raw u8 rows.
// Everything before vectorsOffset is the resident prefix; row addresses are
// computed (vectorsOffset + id*dim), never looked up.

const SKETCH_MAGIC = 0x31415350; // PSA1
const SKETCH_HEADER_BYTES = 256;
const SKETCH_KIND_U8 = 1;

function sha256Bytes(bytes) {
    const crypto = require('crypto');
    return crypto.createHash('sha256').update(bytes).digest();
}

async function sha256BytesAsync(bytes) {
    if (globalThis.crypto && globalThis.crypto.subtle) {
        const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
        return new Uint8Array(digest);
    }
    try {
        return new Uint8Array(sha256Bytes(bytes));
    } catch {
        return null; // no crypto available in this environment
    }
}

// Verify bytes against an expected 32-byte SHA-256 digest. Verification was
// requested by the caller, so a missing crypto backend fails closed rather
// than admitting unverified bytes.
async function verifySha256(bytes, expected, label) {
    const digest = await sha256BytesAsync(bytes);
    if (!digest) {
        throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID,
            'Artifact verification requested but no crypto backend is available; pass verify:false to skip');
    }
    for (let b = 0; b < 32; b++) {
        if (digest[b] !== expected[b]) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, `${label} failed hash verification`);
        }
    }
}

// Verify a whole segment against its digest without buffering it entirely.
// With a streaming hash backend (Node crypto) the segment is read and hashed
// in bounded chunks, so peak memory stays at chunkBytes no matter how large
// the segment. WebCrypto has no incremental SHA-256, so without a streaming
// backend the verify falls back to one bounded read — and refuses segments
// larger than the one-shot limit rather than buffering them.
async function verifySegmentSha256(source, offset, totalBytes, expected, label, options = {}) {
    const chunkBytes = Math.max(4096, Math.trunc(options.chunkBytes || VERIFY_CHUNK_BYTES));
    let hash = null;
    try {
        hash = require('crypto').createHash('sha256');
    } catch {
        hash = null;
    }
    if (hash) {
        let done = 0;
        while (done < totalBytes) {
            const len = Math.min(chunkBytes, totalBytes - done);
            const bytes = await readChecked(source, offset + done, len, label);
            if (bytes.byteLength !== len) {
                throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID,
                    `Artifact ${label} is truncated`, { offset: offset + done, expected: len, actual: bytes.byteLength });
            }
            hash.update(bytes);
            done += len;
        }
        const digest = new Uint8Array(hash.digest());
        for (let b = 0; b < 32; b++) {
            if (digest[b] !== expected[b]) {
                throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, `${label} failed hash verification`);
            }
        }
        return;
    }
    const oneShotLimit = options.oneShotLimit || DEFAULT_OPEN_READ_BYTES;
    if (totalBytes > oneShotLimit) {
        throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID,
            `Artifact ${label} is too large to verify without a streaming crypto backend`,
            { totalBytes, oneShotLimit });
    }
    const bytes = await readChecked(source, offset, totalBytes, label, oneShotLimit);
    if (bytes.byteLength !== totalBytes) {
        throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID,
            `Artifact ${label} is truncated`, { offset, expected: totalBytes, actual: bytes.byteLength });
    }
    await verifySha256(bytes, expected, label);
}

function buildSketchArtifact(snapshotBytes, outPath, options = {}) {
    if (typeof outPath !== 'string' || outPath.length === 0) {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'buildSketchArtifact() requires an output path');
    }
    return exportSketchArtifact(parseUint8Snapshot(snapshotBytes), outPath, options);
}

// Accepts any index-like source of quantized rows ({ dim, count, metric,
// qdata, scales, offsets }) — the profile is derivable from a snapshot, a
// range artifact, or any producer that emits row-affine u8 vectors.
function exportSketchArtifact(index, outPath, options = {}) {
    const dim = index.dim;
    const count = index.count;
    const sketchDims = options.sketchDims || (dim % 2 === 0 ? dim / 2 : dim);
    const sketchBits = options.sketchBits || 4;
    const recommendedRerank = options.recommendedRerank || 0;
    if (![4, 8].includes(sketchBits)) {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'sketchBits must be 4 or 8', { sketchBits });
    }
    if (!Number.isInteger(sketchDims) || sketchDims < 1 || dim % sketchDims !== 0) {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'sketchDims must divide dim', { sketchDims, dim });
    }
    if (sketchBits === 4 && sketchDims % 2 !== 0) {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'sketchDims must be even for 4-bit sketches', { sketchDims });
    }
    // Optional staged-boot micro tier: a second, coarser pooling of the same
    // quantized rows, stored after the full sketches so v1 readers see it
    // only as opaque resident tail bytes (their layout check permits a gap
    // between sketches end and vectorsOffset, and the resident hash already
    // covers it). Pooling commutes with the per-row affine map, so the micro
    // tier reuses scales/offsets unchanged, exactly like the full tier.
    const microDims = options.microDims || 0;
    // 4-bit default: the 2026-08-04 sweep on the 456k wiki corpus measured
    // bit depth as worthless at fixed bytes (48d/4b == 48d/8b within noise)
    // while halved pooling at the same bytes gained +28 recall points
    // (96d/4b 87.8% vs 48d/8b 59.4% capture at C=800). Micro tiers should
    // spend their byte budget on dims, never on bits.
    const microBits = microDims ? (options.microBits || 4) : 0;
    if (microDims) {
        if (!Number.isInteger(microDims) || microDims < 1 || sketchDims % microDims !== 0 || microDims >= sketchDims) {
            throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'microDims must divide sketchDims and be smaller', { microDims, sketchDims });
        }
        if (![4, 8].includes(microBits) || (microBits === 4 && microDims % 2 !== 0)) {
            throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'invalid micro sketch encoding', { microBits, microDims });
        }
    }
    const pool = dim / sketchDims;
    const sketchRowBytes = (sketchDims * sketchBits) / 8;
    const microPool = microDims ? dim / microDims : 0;
    const microRowBytes = microDims ? (microDims * microBits) / 8 : 0;

    const scalesOffset = SKETCH_HEADER_BYTES;
    const offsetsOffset = scalesOffset + count * 4;
    const sketchesOffset = offsetsOffset + count * 4;
    const sketchesEnd = sketchesOffset + count * sketchRowBytes;
    const microOffset = microDims ? sketchesEnd : 0;
    const residentEnd = microDims ? microOffset + count * microRowBytes : sketchesEnd;
    const vectorsOffset = Math.ceil(residentEnd / 16) * 16;
    const fileBytes = vectorsOffset + count * dim;

    const out = Buffer.alloc(fileBytes);
    out.writeUInt32LE(SKETCH_MAGIC, 0);
    out.writeUInt32LE(1, 4);
    out.writeUInt32LE(SKETCH_KIND_U8, 8);
    out.writeUInt32LE(index.metric, 12);
    out.writeUInt32LE(dim, 16);
    out.writeUInt32LE(count, 20);
    out.writeUInt32LE(sketchDims, 24);
    out.writeUInt32LE(sketchBits, 28);
    out.writeUInt32LE(scalesOffset, 32);
    out.writeUInt32LE(offsetsOffset, 36);
    out.writeUInt32LE(sketchesOffset, 40);
    out.writeUInt32LE(vectorsOffset, 44);
    out.writeBigUInt64LE(BigInt(fileBytes), 48);
    out.writeUInt32LE(recommendedRerank, 120);
    if (microDims) {
        out.writeUInt32LE(microDims, 124);
        out.writeUInt32LE(microBits, 128);
        out.writeUInt32LE(microOffset, 132);
    }

    for (let i = 0; i < count; i++) out.writeFloatLE(index.scales[i], scalesOffset + i * 4);
    for (let i = 0; i < count; i++) out.writeFloatLE(index.offsets[i], offsetsOffset + i * 4);

    for (let i = 0; i < count; i++) {
        const rowBase = i * dim;
        for (let sd = 0; sd < sketchDims; sd++) {
            let acc = 0;
            for (let j = 0; j < pool; j++) acc += index.qdata[rowBase + sd * pool + j];
            const pooled = Math.round(acc / pool);
            if (sketchBits === 8) {
                out[sketchesOffset + i * sketchRowBytes + sd] = pooled;
            } else {
                const q = Math.min(15, Math.round(pooled / 17));
                const byteIndex = sketchesOffset + i * sketchRowBytes + (sd >> 1);
                if (sd % 2 === 0) out[byteIndex] = (out[byteIndex] & 0xf0) | q;
                else out[byteIndex] = (out[byteIndex] & 0x0f) | (q << 4);
            }
        }
        if (microDims) {
            for (let sd = 0; sd < microDims; sd++) {
                let acc = 0;
                for (let j = 0; j < microPool; j++) acc += index.qdata[rowBase + sd * microPool + j];
                const pooled = Math.round(acc / microPool);
                if (microBits === 8) {
                    out[microOffset + i * microRowBytes + sd] = pooled;
                } else {
                    const q = Math.min(15, Math.round(pooled / 17));
                    const byteIndex = microOffset + i * microRowBytes + (sd >> 1);
                    if (sd % 2 === 0) out[byteIndex] = (out[byteIndex] & 0xf0) | q;
                    else out[byteIndex] = (out[byteIndex] & 0x0f) | (q << 4);
                }
            }
        }
    }

    Buffer.from(index.qdata.buffer, index.qdata.byteOffset, count * dim).copy(out, vectorsOffset);

    sha256Bytes(out.subarray(SKETCH_HEADER_BYTES, vectorsOffset)).copy(out, 56);
    sha256Bytes(out.subarray(vectorsOffset)).copy(out, 88);
    if (microDims) {
        // Stage-1 hash: scales+offsets plus the micro segment, exactly the
        // bytes a staged open serves from, verifiable before the full
        // sketches arrive. The full resident hash still covers everything.
        const stage1 = Buffer.concat([
            out.subarray(SKETCH_HEADER_BYTES, sketchesOffset),
            out.subarray(microOffset, microOffset + count * microRowBytes),
        ]);
        sha256Bytes(stage1).copy(out, 136);
    }

    const fs = require('fs');
    fs.writeFileSync(outPath, out);
    return {
        format: 'pancake-sketch-artifact',
        formatVersion: 1,
        file: outPath,
        sizeBytes: fileBytes,
        metric: index.metric === 1 ? 'cosine' : 'l2',
        graph: { count, dim },
        sketch: { sketchDims, sketchBits, pool, residentBytes: vectorsOffset },
        micro: microDims ? { microDims, microBits, microPool, stage1Bytes: sketchesOffset + count * microRowBytes } : null,
        addressing: { scalesOffset, offsetsOffset, sketchesOffset, vectorsOffset },
        recommendedRerank,
    };
}

function buildSketchArtifactFile(snapshotPath, outPath, options = {}) {
    const fs = require('fs');
    return buildSketchArtifact(fs.readFileSync(snapshotPath), outPath, options);
}

class PancakeSketchArtifact {
    constructor(source) {
        this.source = source;
        // Fetched rows live in a byte-budgeted LRU; search correctness never
        // depends on retention because fetchRows returns rows directly.
        this.cache = new Map();
        this.cacheBytes = 0;
        this.maxCacheBytes = 64 * 1024 * 1024;
        this.rangeRequests = 0;
        this.rangeBytes = 0;
        this.maxReadBytes = DEFAULT_OPEN_READ_BYTES;
        this.residentVerified = false;
        this.vectorsVerified = false;
    }

    cachedRow(id) {
        const row = this.cache.get(id);
        if (row !== undefined) {
            this.cache.delete(id);
            this.cache.set(id, row);
        }
        return row;
    }

    // Drop all cached rows. Cache state never affects result semantics, so
    // this only changes fetch behavior (every subsequent search starts cold).
    clearCache() {
        this.cache.clear();
        this.cacheBytes = 0;
    }

    cacheRow(id, row) {
        if (this.cache.has(id)) return;
        this.cache.set(id, row);
        this.cacheBytes += this.dim;
        while (this.cacheBytes > this.maxCacheBytes && this.cache.size > 1) {
            const oldest = this.cache.keys().next().value;
            this.cache.delete(oldest);
            this.cacheBytes -= this.dim;
        }
    }

    static async open(source, options = {}) {
        if (!source || typeof source.read !== 'function') {
            throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'SketchArtifact.open() requires a range source with read(offset, length)');
        }
        const artifact = new PancakeSketchArtifact(source);
        artifact.maxReadBytes = resolveMaxReadBytes(options.maxReadBytes);
        const header = await readChecked(source, 0, SKETCH_HEADER_BYTES, 'header');
        if (header.byteLength !== SKETCH_HEADER_BYTES) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact header is truncated');
        }
        const view = new DataView(header.buffer, header.byteOffset, header.byteLength);
        if (view.getUint32(0, true) !== SKETCH_MAGIC) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Not a Pancake sketch artifact (bad magic)');
        }
        const version = view.getUint32(4, true);
        if (version !== 1) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported sketch artifact version', { version });
        }
        if (view.getUint32(8, true) !== SKETCH_KIND_U8) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported sketch artifact kind');
        }
        artifact.metric = view.getUint32(12, true);
        artifact.dim = view.getUint32(16, true);
        artifact.count = view.getUint32(20, true);
        artifact.sketchDims = view.getUint32(24, true);
        artifact.sketchBits = view.getUint32(28, true);
        const scalesOffset = view.getUint32(32, true);
        const offsetsOffset = view.getUint32(36, true);
        const sketchesOffset = view.getUint32(40, true);
        artifact.vectorsOffset = view.getUint32(44, true);
        const fileBytes = Number(view.getBigUint64(48, true));
        artifact.recommendedRerank = view.getUint32(120, true);
        // Retained so the lazy tier stays verifiable after open — see
        // verifyVectors(). Copied out of the header read buffer.
        artifact.vectorsSha256 = new Uint8Array(header.subarray(88, 120));
        artifact.microDims = view.getUint32(124, true);
        artifact.microBits = artifact.microDims ? view.getUint32(128, true) : 0;
        const microOffset = artifact.microDims ? view.getUint32(132, true) : 0;
        artifact.maxCacheBytes = resolveMaxCacheBytes(options.maxCacheBytes, 256 * artifact.dim, artifact.maxCacheBytes);

        const { metric, dim, count, sketchDims, sketchBits, vectorsOffset } = artifact;
        if (metric !== 0 && metric !== 1) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported sketch artifact metric', { metric });
        }
        if (dim < 1 || count < 1 || sketchDims < 1 || dim % sketchDims !== 0) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Invalid sketch artifact geometry', { dim, count, sketchDims });
        }
        if (![4, 8].includes(sketchBits) || (sketchBits === 4 && sketchDims % 2 !== 0)) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Invalid sketch encoding', { sketchBits, sketchDims });
        }
        const sketchRowBytes = (sketchDims * sketchBits) / 8;
        const expectVectors = count * dim;
        if (!Number.isSafeInteger(fileBytes) || !Number.isSafeInteger(expectVectors)
            || scalesOffset !== SKETCH_HEADER_BYTES
            || offsetsOffset !== scalesOffset + count * 4
            || sketchesOffset !== offsetsOffset + count * 4
            || vectorsOffset < sketchesOffset + count * sketchRowBytes
            || vectorsOffset % 16 !== 0
            || fileBytes !== vectorsOffset + expectVectors) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact layout is inconsistent');
        }
        let microRowBytes = 0;
        if (artifact.microDims) {
            microRowBytes = (artifact.microDims * artifact.microBits) / 8;
            if (sketchDims % artifact.microDims !== 0 || artifact.microDims >= sketchDims
                || ![4, 8].includes(artifact.microBits) || (artifact.microBits === 4 && artifact.microDims % 2 !== 0)
                || microOffset !== sketchesOffset + count * sketchRowBytes
                || microOffset + count * microRowBytes > vectorsOffset) {
                throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact micro tier layout is inconsistent', { microDims: artifact.microDims });
            }
        }
        artifact.microOffset = microOffset;

        const verify = options.verify !== false;
        const staged = options.staged === true && artifact.microDims > 0;
        artifact.tier = 'full';
        artifact.fullyResident = Promise.resolve(artifact);

        if (!staged) {
            const resident = await readChecked(source, SKETCH_HEADER_BYTES, vectorsOffset - SKETCH_HEADER_BYTES, 'resident prefix', artifact.maxReadBytes);
            if (resident.byteLength !== vectorsOffset - SKETCH_HEADER_BYTES) {
                throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact resident prefix is truncated');
            }
            // Copy into an aligned buffer so the typed-array views are valid
            // regardless of the source's byteOffset.
            const residentCopy = new Uint8Array(resident.byteLength);
            residentCopy.set(resident);
            artifact._adoptResident(residentCopy, count, sketchRowBytes, microRowBytes, microOffset);
            if (verify) await artifact._verifyFullResident(residentCopy, header);
            return artifact;
        }

        // Staged boot: serve queries from the coarse micro tier after
        // fetching only scales+offsets and the micro segment (one parallel
        // wave after the header), then complete residency in the background.
        // Stage 1 is independently hash-verified; stage 2 re-verifies the
        // full resident hash before the tier swap, so both states honor the
        // integrity contract. Result semantics differ between tiers, so the
        // tier is reported on every search result rather than hidden.
        artifact.tier = 'micro';
        const [affineBytes, microBytes] = await Promise.all([
            readChecked(source, SKETCH_HEADER_BYTES, sketchesOffset - SKETCH_HEADER_BYTES, 'stage-1 affine', artifact.maxReadBytes),
            readChecked(source, microOffset, count * microRowBytes, 'stage-1 micro', artifact.maxReadBytes),
        ]);
        if (affineBytes.byteLength !== sketchesOffset - SKETCH_HEADER_BYTES || microBytes.byteLength !== count * microRowBytes) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact stage-1 read is truncated');
        }
        const affineCopy = new Uint8Array(affineBytes.byteLength);
        affineCopy.set(affineBytes);
        const microCopy = new Uint8Array(microBytes.byteLength);
        microCopy.set(microBytes);
        if (verify) {
            const stage1 = new Uint8Array(affineCopy.byteLength + microCopy.byteLength);
            stage1.set(affineCopy, 0);
            stage1.set(microCopy, affineCopy.byteLength);
            const digest = await sha256BytesAsync(stage1);
            if (!digest) {
                // Verification requested but no crypto backend: fail closed.
                throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID,
                    'Sketch artifact verification requested but no crypto backend is available; pass verify:false to skip');
            }
            const expected = header.subarray(136, 168);
            for (let b = 0; b < 32; b++) {
                if (digest[b] !== expected[b]) {
                    throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact stage-1 prefix failed hash verification');
                }
            }
            artifact.residentVerified = true;
        }
        artifact.scales = new Float32Array(affineCopy.buffer, 0, count);
        artifact.offsets = new Float32Array(affineCopy.buffer, count * 4, count);
        artifact.sketches = null;
        artifact.microSketches = microCopy;
        artifact.residentBytes = SKETCH_HEADER_BYTES + affineCopy.byteLength + microCopy.byteLength;

        const onStage = typeof options.onStage === 'function' ? options.onStage : null;
        if (onStage) onStage({ tier: 'micro', residentBytes: artifact.residentBytes });
        artifact.fullyResident = (async () => {
            const rest = await readChecked(source, sketchesOffset, vectorsOffset - sketchesOffset, 'stage-2 sketches', artifact.maxReadBytes);
            if (rest.byteLength !== vectorsOffset - sketchesOffset) {
                throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact stage-2 read is truncated');
            }
            const residentCopy = new Uint8Array((sketchesOffset - SKETCH_HEADER_BYTES) + rest.byteLength);
            residentCopy.set(affineCopy, 0);
            residentCopy.set(rest, sketchesOffset - SKETCH_HEADER_BYTES);
            if (verify) await artifact._verifyFullResident(residentCopy, header);
            artifact._adoptResident(residentCopy, count, sketchRowBytes, microRowBytes, microOffset);
            artifact.tier = 'full';
            if (onStage) onStage({ tier: 'full', residentBytes: artifact.residentBytes });
            return artifact;
        })();
        // Surfacing failures is the caller's job via the promise; an
        // unobserved rejection must not crash the process mid-boot.
        artifact.fullyResident.catch(() => {});
        return artifact;
    }

    _adoptResident(residentCopy, count, sketchRowBytes, microRowBytes, microOffset) {
        this.scales = new Float32Array(residentCopy.buffer, 0, count);
        this.offsets = new Float32Array(residentCopy.buffer, count * 4, count);
        this.sketches = new Uint8Array(residentCopy.buffer, count * 8, count * sketchRowBytes);
        if (this.microDims) {
            this.microSketches = new Uint8Array(residentCopy.buffer, microOffset - SKETCH_HEADER_BYTES, count * microRowBytes);
        }
        this.residentBytes = this.vectorsOffset;
    }

    async _verifyFullResident(residentCopy, header) {
        const digest = await sha256BytesAsync(residentCopy);
        if (!digest) {
            // Verification was requested (callers gate this on verify) but no
            // crypto backend is available. Fail closed rather than admit
            // unverified bytes.
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID,
                'Sketch artifact verification requested but no crypto backend is available; pass verify:false to skip');
        }
        const expected = header.subarray(56, 88);
        for (let b = 0; b < 32; b++) {
            if (digest[b] !== expected[b]) {
                throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact resident prefix failed hash verification');
            }
        }
        this.residentVerified = true;
    }

    static async openFile(filePath, options = {}) {
        return PancakeSketchArtifact.open(new NodeFileRangeSource(filePath), options);
    }

    stats() {
        return {
            rangeRequests: this.rangeRequests,
            rangeBytes: this.rangeBytes,
            cachedRows: this.cache.size,
            cacheBytes: this.cacheBytes,
            residentBytes: this.residentBytes,
            residentVerified: this.residentVerified,
            vectorsVerified: this.vectorsVerified,
        };
    }

    // Verify the lazy vectors segment against the header's whole-segment
    // hash. Reads the segment in bounded chunks with a streaming hash (Node);
    // runtimes without one fall back to a single read bounded by the open
    // budget. Intended for producers, CI, and hosts that materialize
    // artifacts, not the query path. Verifying individual row ranges needs
    // per-chunk commitments, which arrive with the contract's
    // complete-profile manifest.
    async verifyVectors(options = {}) {
        const bytes = this.count * this.dim;
        await verifySegmentSha256(this.source, this.vectorsOffset, bytes, this.vectorsSha256,
            'Sketch artifact vectors segment', { chunkBytes: options.chunkBytes, oneShotLimit: this.maxReadBytes });
        this.vectorsVerified = true;
        return true;
    }

    sketchValue(id, sd) {
        if (this.sketchBits === 8) return this.sketches[id * this.sketchDims + sd];
        const byte = this.sketches[id * (this.sketchDims >> 1) + (sd >> 1)];
        const nibble = sd % 2 === 0 ? (byte & 0x0f) : (byte >> 4);
        return nibble * 17;
    }

    microValue(id, sd) {
        if (this.microBits === 8) return this.microSketches[id * this.microDims + sd];
        const byte = this.microSketches[id * (this.microDims >> 1) + (sd >> 1)];
        const nibble = sd % 2 === 0 ? (byte & 0x0f) : (byte >> 4);
        return nibble * 17;
    }

    // Active resident tier for candidate selection. 'full' whenever the full
    // sketches are adopted; 'micro' only during a staged boot window.
    activeTier() {
        return this.tier === 'micro'
            ? { name: 'micro', dims: this.microDims, value: (i, sd) => this.microValue(i, sd) }
            : { name: 'full', dims: this.sketchDims, value: (i, sd) => this.sketchValue(i, sd) };
    }

    async fetchRows(ids, options = {}) {
        const gap = Math.max(0, options.gap === undefined ? 2048 : options.gap);
        const parallelism = Math.max(1, Math.trunc(options.parallelism || 8));
        const rows = new Map();
        const missing = [];
        for (const id of ids) {
            if (rows.has(id)) continue;
            if (!Number.isInteger(id) || id < 0 || id >= this.count) {
                throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'row id out of range', { id });
            }
            const cached = this.cachedRow(id);
            if (cached !== undefined) rows.set(id, cached);
            else {
                rows.set(id, null);
                missing.push(id);
            }
        }
        if (!missing.length) return rows;
        missing.sort((a, b) => a - b);
        const dim = this.dim;
        // Coalesced runs are split at maxRangeBytes (row-aligned) so no gap
        // setting — or a bulk fetch of every row — turns into one unbounded
        // read and allocation.
        const maxRunRows = Math.max(1, Math.floor(Math.trunc(options.maxRangeBytes || MAX_COALESCED_RANGE_BYTES) / dim));
        const ranges = [];
        const pushRun = (startId, endId) => {
            for (let piece = startId; piece < endId; piece += maxRunRows) {
                ranges.push([piece, Math.min(piece + maxRunRows, endId)]);
            }
        };
        let runStartId = missing[0];
        let runEndId = missing[0] + 1;
        for (let i = 1; i < missing.length; i++) {
            const id = missing[i];
            if (id * dim <= runEndId * dim + gap) {
                runEndId = id + 1;
            } else {
                pushRun(runStartId, runEndId);
                runStartId = id;
                runEndId = id + 1;
            }
        }
        pushRun(runStartId, runEndId);
        for (let i = 0; i < ranges.length; i += parallelism) {
            const batch = ranges.slice(i, i + parallelism);
            const buffers = await Promise.all(batch.map(async ([startId, endId]) => {
                const offset = this.vectorsOffset + startId * dim;
                const length = (endId - startId) * dim;
                const bytes = await readChecked(this.source, offset, length, 'row');
                if (bytes.byteLength !== length) {
                    throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact row read returned a truncated range', { offset, length });
                }
                this.rangeRequests++;
                this.rangeBytes += length;
                return { startId, endId, bytes };
            }));
            for (const { startId, endId, bytes } of buffers) {
                for (let id = startId; id < endId; id++) {
                    // Copy each row out of the coalesced fetch buffer: a view
                    // would pin the whole range in the LRU, so cacheBytes
                    // would count dim bytes while retaining the full fetch.
                    // (Explicit copy constructor — .slice() is unreliable here
                    // because Node Buffers override it with view semantics.)
                    const row = new Uint8Array(bytes.subarray((id - startId) * dim, (id - startId + 1) * dim));
                    if (rows.has(id)) rows.set(id, row);
                    this.cacheRow(id, row);
                }
            }
        }
        return rows;
    }

    async search(query, k, options = {}) {
        if (!query || query.length !== this.dim) {
            throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'search() query must match artifact dim', { dim: this.dim });
        }
        if (!Number.isInteger(k) || k < 1) {
            throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'search() k must be a positive integer', { k });
        }
        const tier = this.activeTier();
        // The micro tier is coarser, so its candidate pool defaults wider:
        // an explicit options.rerank always wins; otherwise recommendedRerank
        // is scaled by microBoost while serving from the micro tier.
        const base = Math.trunc(options.rerank || 0);
        const boost = Math.max(1, Math.trunc(options.microBoost || 4));
        const rec = this.recommendedRerank || Math.max(100, k * 10);
        const C = Math.max(k, base > 0 ? base : tier.name === 'micro' ? rec * boost : rec);
        const { dim, count } = this;
        const tierDims = tier.dims;
        const pool = dim / tierDims;

        let q = query;
        if (this.metric === 1) {
            let norm = 0;
            for (let d = 0; d < dim; d++) norm += query[d] * query[d];
            norm = Math.sqrt(norm);
            if (!(norm > 0) || !Number.isFinite(norm)) {
                throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'cosine query must have a nonzero finite norm');
            }
            q = new Float32Array(dim);
            for (let d = 0; d < dim; d++) q[d] = query[d] / norm;
        }
        const qPool = new Float32Array(tierDims);
        for (let sd = 0; sd < tierDims; sd++) {
            let acc = 0;
            for (let j = 0; j < pool; j++) acc += q[sd * pool + j];
            qPool[sd] = acc / pool;
        }

        let ids;
        let scanner = options.scanner || null;
        if (scanner && scanner.sketchDims !== undefined && scanner.sketchDims !== tierDims) scanner = null;
        if (!scanner && options.microScanner && options.microScanner.sketchDims === tierDims) scanner = options.microScanner;
        if (scanner) {
            // A scanner scores the resident sketches itself, so it must
            // implement this artifact's metric. Cosine requires an explicit
            // declaration: a metric-blind scanner silently loses recall there.
            const scannerMetric = scanner.metric;
            if (scannerMetric !== undefined && scannerMetric !== this.metric) {
                throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'scanner metric does not match artifact metric', { scannerMetric, metric: this.metric });
            }
            if (this.metric === 1 && scannerMetric !== 1) {
                throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'cosine sketch artifacts require a metric-aware scanner (scanner.metric === 1)', { metric: this.metric });
            }
            ids = scanner.scan(qPool, C);
        } else {
            const candDist = new Float64Array(C).fill(Infinity);
            const candId = new Int32Array(C).fill(-1);
            let candMax = Infinity;
            for (let i = 0; i < count; i++) {
                const s = this.scales[i];
                const o = this.offsets[i];
                let metricAcc = 0;
                if (this.metric === 1) {
                    for (let sd = 0; sd < tierDims; sd++) {
                        metricAcc += qPool[sd] * (o + s * tier.value(i, sd));
                    }
                    metricAcc = 1 - Math.max(-1, Math.min(1, metricAcc * pool));
                } else {
                    for (let sd = 0; sd < tierDims; sd++) {
                        const diff = qPool[sd] - (o + s * tier.value(i, sd));
                        metricAcc += diff * diff;
                    }
                }
                if (metricAcc < candMax) {
                    let worst = 0;
                    for (let j = 1; j < C; j++) if (candDist[j] > candDist[worst]) worst = j;
                    candDist[worst] = metricAcc;
                    candId[worst] = i;
                    candMax = 0;
                    for (let j = 0; j < C; j++) if (candDist[j] > candMax) candMax = candDist[j];
                }
            }
            ids = [];
            for (let j = 0; j < C; j++) if (candId[j] >= 0) ids.push(candId[j]);
        }

        const rows = await this.fetchRows(ids, options);

        const exact = [];
        for (const id of ids) {
            const row = rows.get(id);
            const s = this.scales[id];
            const o = this.offsets[id];
            let acc = 0;
            if (this.metric === 1) {
                for (let d = 0; d < dim; d++) acc += q[d] * (o + s * row[d]);
                acc = 1 - Math.max(-1, Math.min(1, acc));
            } else {
                for (let d = 0; d < dim; d++) {
                    const diff = q[d] - (o + s * row[d]);
                    acc += diff * diff;
                }
            }
            exact.push([acc, id]);
        }
        exact.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        // Rerank accumulates squared L2; the API contract (README "Distance
        // values") reports Euclidean, matching PancakeIndex.search.
        const sqrtL2 = this.metric !== 1;
        return {
            results: exact.slice(0, k).map(([distance, id]) => ({ id, distance: sqrtL2 ? Math.sqrt(distance) : distance })),
            rerank: ids.length,
            tier: tier.name,
        };
    }

    async close() {
        if (this.source && typeof this.source.close === 'function') await this.source.close();
    }
}

// Build a WASM-backed scanner for a sketch artifact's resident tier, usable
// as the `scanner` option to PancakeSketchArtifact.search(). The engine's
// pancake_sketch_scan SIMD kernel runs the O(count*sketchDims) resident scan
// that is otherwise the browser query bottleneck. `loadEngine` is the
// entrypoint's own async engine loader (Node/web/workerd all supply one);
// the sketch tier is staged in the engine heap once at creation.
async function createSketchScanner(loadEngine, artifact, options = {}) {
    if (typeof loadEngine !== 'function') {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'createSketchScanner() requires an engine loader');
    }
    if (!artifact || !(artifact instanceof PancakeSketchArtifact)) {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'createSketchScanner() requires a sketch artifact');
    }
    const engine = await loadEngine();
    const { count, metric } = artifact;
    const tierName = options.tier === 'micro' ? 'micro' : 'full';
    if (tierName === 'micro' && !artifact.microDims) {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'artifact has no micro tier');
    }
    if (tierName === 'full' && !artifact.sketches) {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'full sketches are not resident yet (staged open still in stage 1; await artifact.fullyResident)');
    }
    const sketchDims = tierName === 'micro' ? artifact.microDims : artifact.sketchDims;
    const tierBits = tierName === 'micro' ? artifact.microBits : artifact.sketchBits;
    const maxC = Math.max(1, Math.trunc(options.maxRerank || 1024));

    // The kernel reads u8 sketch values; expand 4-bit nibbles to their
    // reconstructed u8 once so the scan needs no per-element unpacking.
    const expanded = new Uint8Array(count * sketchDims);
    if (tierBits === 4) {
        for (let i = 0; i < count; i++) {
            for (let sd = 0; sd < sketchDims; sd++) expanded[i * sketchDims + sd] = tierName === 'micro' ? artifact.microValue(i, sd) : artifact.sketchValue(i, sd);
        }
    } else {
        expanded.set(tierName === 'micro' ? artifact.microSketches : artifact.sketches);
    }

    const sketchesPtr = engine._emsc_malloc(expanded.length);
    const scalesPtr = engine._emsc_malloc(count * 4);
    const offsetsPtr = engine._emsc_malloc(count * 4);
    const queryPtr = engine._emsc_malloc(sketchDims * 4);
    const outIdsPtr = engine._emsc_malloc(maxC * 4);
    const outDistsPtr = engine._emsc_malloc(maxC * 4);
    if (!sketchesPtr || !scalesPtr || !offsetsPtr || !queryPtr || !outIdsPtr || !outDistsPtr) {
        // Free whichever allocations succeeded before one failed, so a
        // partial failure does not leak WASM heap (matches create()).
        for (const ptr of [sketchesPtr, scalesPtr, offsetsPtr, queryPtr, outIdsPtr, outDistsPtr]) {
            if (ptr) engine._emsc_free(ptr);
        }
        throw pancakeError(PANCAKE_ERROR_CODES.WASM_ALLOCATION_FAILED, 'sketch scanner heap allocation failed');
    }
    engine.HEAPU8.set(expanded, sketchesPtr);
    engine.HEAPF32.set(artifact.scales, scalesPtr >> 2);
    engine.HEAPF32.set(artifact.offsets, offsetsPtr >> 2);

    let disposed = false;
    return {
        maxRerank: maxC,
        metric,
        sketchDims,
        tier: tierName,
        scan(pooledQuery, c) {
            if (disposed) throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'sketch scanner disposed');
            const query = pooledQuery instanceof Float32Array ? pooledQuery : Float32Array.from(pooledQuery);
            // The query is copied into a queryPtr sized for exactly sketchDims
            // floats: an over- or under-sized input would read/write outside
            // that buffer in the WASM heap. Validate before the copy.
            if (query.length !== sketchDims) {
                throw pancakeError(PANCAKE_ERROR_CODES.DIMENSION_MISMATCH,
                    `scan() pooled query has ${query.length} values, expected ${sketchDims}`,
                    { expected: sketchDims, actual: query.length });
            }
            for (let i = 0; i < query.length; i++) {
                if (!Number.isFinite(query[i])) {
                    throw pancakeError(PANCAKE_ERROR_CODES.INVALID_VECTOR,
                        'scan() pooled query contains a non-finite value', { index: i });
                }
            }
            engine.HEAPF32.set(query, queryPtr >> 2);
            const topC = Math.min(Math.max(1, Math.trunc(c)), maxC);
            const n = engine._pancake_sketch_scan(
                sketchesPtr, scalesPtr, offsetsPtr, count, sketchDims, queryPtr, metric, topC, outIdsPtr, outDistsPtr
            );
            return Array.from(engine.HEAPU32.subarray(outIdsPtr >> 2, (outIdsPtr >> 2) + n));
        },
        dispose() {
            if (disposed) return;
            disposed = true;
            for (const ptr of [sketchesPtr, scalesPtr, offsetsPtr, queryPtr, outIdsPtr, outDistsPtr]) {
                engine._emsc_free(ptr);
            }
        },
    };
}

module.exports = {
    PancakeRangeArtifact,
    PancakeSketchArtifact,
    createSketchScanner,
    NodeFileRangeSource,
    buildRangeArtifact,
    buildRangeArtifactFile,
    buildSketchArtifact,
    buildSketchArtifactFile,
    exportSketchArtifact,
    parseUint8Snapshot,
};
