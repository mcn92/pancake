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
const HEADER_BYTES_V2 = 256;
const RANGE_KIND_U8 = 1;
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
        const buffer = Buffer.allocUnsafe(length);
        this.fs.readSync(this.fd, buffer, 0, length, offset);
        return buffer;
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
        this.currentRanges = [];
        this.routerResident = { records: 0, bytes: 0 };
        this.rangeRequests = 0;
        this.rangeBytes = 0;
        this.rangeNodesDecoded = 0;
        this.loadRouter = options.loadRouter !== false;
    }

    static async open(source, options = {}) {
        if (!source || typeof source.read !== 'function') {
            throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'PancakeRangeArtifact.open() requires a range source with read(offset, length)');
        }
        const headerBytes = asUint8Array(await source.read(0, HEADER_BYTES));
        if (headerBytes.byteLength < HEADER_BYTES) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact header is truncated');
        }
        const header = parseHeader(headerBytes);
        const idMapBytes = asUint8Array(await source.read(header.idMapOffset, header.count * 4));
        if (idMapBytes.byteLength !== header.count * 4) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact id map is truncated');
        }
        const copied = new Uint8Array(idMapBytes.byteLength);
        copied.set(idMapBytes);
        const idMap = new Uint32Array(copied.buffer);
        const artifact = new PancakeRangeArtifact(source, header, idMap, options);
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
            cachedNodes: this.cache.size,
            routerResident: { ...this.routerResident },
        };
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

    async readNode(id) {
        const cached = this.cache.get(id);
        if (cached) return cached;
        await this.prefetch([id]);
        return this.cache.get(id);
    }

    async prefetch(ids, options = {}) {
        const addresses = [];
        const seen = new Set();
        for (const id of ids) {
            if (this.cache.has(id) || seen.has(id)) continue;
            seen.add(id);
            addresses.push(this.recordAddressForId(id));
        }
        if (addresses.length === 0) return 0;
        addresses.sort((a, b) => a - b);
        const gap = Math.max(0, options.gap || 0);
        const gapBytes = this.version >= 2 ? gap : gap * this.recordBytes;
        const rangeParallelism = Math.max(1, Math.trunc(options.parallelism || 1));
        const ranges = [];
        let runStart = addresses[0];
        let runEnd = addresses[0] + this.recordBytes;
        const flush = () => {
            ranges.push([runStart, runEnd]);
        };
        const readRange = async ([start, end]) => {
            const bytes = end - start;
            const buffer = asUint8Array(await this.source.read(start, bytes));
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
                if (!this.cache.has(originalId)) {
                    this.cache.set(originalId, this.decodeNode(record));
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
        const buffer = asUint8Array(await this.source.read(this.routerRecordsOffset, bytes));
        if (buffer.byteLength !== bytes) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact router segment is truncated');
        }
        for (let i = 0; i < this.routerCount; i++) {
            const record = buffer.subarray(i * this.recordBytes, (i + 1) * this.recordBytes);
            const originalId = new DataView(record.buffer, record.byteOffset, record.byteLength).getUint32(0, true);
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
        const upperCounts = new Uint16Array(this.maxLevel);
        for (let i = 0; i < this.maxLevel; i++) upperCounts[i] = readU16(view, state);
        const qdata = new Uint8Array(this.dim);
        qdata.set(bytes.subarray(state.offset, state.offset + this.dim));
        state.offset += this.dim;
        const scale = readF32(view, state);
        const offset = readF32(view, state);
        const base = new Uint32Array(baseCount);
        for (let i = 0; i < this.M0; i++) {
            const neighbor = readU32(view, state);
            if (i < baseCount) base[i] = neighbor;
        }
        const upper = Array.from({ length: this.maxLevel }, () => new Uint32Array(0));
        for (let levelIndex = 0; levelIndex < this.maxLevel; levelIndex++) {
            const edges = new Uint32Array(upperCounts[levelIndex]);
            for (let i = 0; i < this.M; i++) {
                const neighbor = readU32(view, state);
                if (i < edges.length) edges[i] = neighbor;
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
            await this.prefetch(ids, { gap: options.gap || 0, parallelism: options.rangeParallelism || options.parallelism || 1 });
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

        return {
            results: results.items.sort(compareDistancesAsc).slice(0, k),
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
    if (version < 1 || version > 2) {
        throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported Pancake range artifact version', { version });
    }
    return header;
}

function unwrapSnapshot(bytes) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (bytes.byteLength < 4 || view.getUint32(0, true) !== PANCAKE_MAGIC) return bytes;
    const version = view.getUint32(4, true);
    if (version === 1) return bytes.subarray(V1_ENVELOPE_HEADER_SIZE);
    if (version === 2) return bytes.subarray(V2_ENVELOPE_HEADER_SIZE);
    if (version === 3) {
        const mappingCount = view.getUint32(24, true);
        const rawSize = view.getUint32(28, true);
        const rawOffset = V3_ENVELOPE_HEADER_SIZE + mappingCount * MAPPING_ENTRY_SIZE;
        return bytes.subarray(rawOffset, rawOffset + rawSize);
    }
    throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, `Unsupported Pancake envelope version ${version}`, { version });
}

function parseUint8Snapshot(bytes) {
    const raw = unwrapSnapshot(asUint8Array(bytes));
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    let offset = 0;
    const u32 = () => {
        const value = view.getUint32(offset, true);
        offset += 4;
        return value;
    };
    const f32 = () => {
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
    if (metric !== 0 && metric !== 1) {
        throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported uint8 snapshot metric', { metric });
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
    try {
        const header = Buffer.alloc(HEADER_BYTES_V2);
        let h = 0;
        header.writeUInt32LE(RANGE_MAGIC, h); h += 4;
        header.writeUInt32LE(2, h); h += 4;
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
        fs.writeSync(fd, header, 0, header.length, 0);
        fs.writeSync(fd, Buffer.from(locationMap.buffer), 0, index.count * 4, idMapOffset);

        const record = Buffer.alloc(recBytes);
        for (let i = 0; i < routerIds.length; i++) {
            writeNodeRecord(index, routerIds[i], record);
            fs.writeSync(fd, record, 0, record.length, routerRecordsOffset + i * recBytes);
        }
        for (let i = 0; i < baseIds.length; i++) {
            writeNodeRecord(index, baseIds[i], record);
            fs.writeSync(fd, record, 0, record.length, baseRecordsOffset + i * recBytes);
        }
    } finally {
        fs.closeSync(fd);
    }
    return {
        format: 'pancake-range-artifact',
        formatVersion: 2,
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

module.exports = {
    PancakeRangeArtifact,
    NodeFileRangeSource,
    buildRangeArtifact,
    buildRangeArtifactFile,
    parseUint8Snapshot,
};
