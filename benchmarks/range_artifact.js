#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { performance } = require('perf_hooks');

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
const OPERATIONAL_EF_SEARCH = 80;
const OPERATIONAL_EXPANSION_BATCH = 8;

function arg(name, fallback = null) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

function hasArg(name) {
  return process.argv.includes(`--${name}`);
}

function parseIntArg(name, fallback) {
  const raw = arg(name, fallback == null ? null : String(fallback));
  if (raw == null) return fallback;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 0) throw new Error(`--${name} must be a non-negative integer`);
  return n;
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
  throw new Error(`unsupported Pancake envelope version ${version}`);
}

function parseSnapshot(bytes) {
  const raw = unwrapSnapshot(bytes);
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  let off = 0;
  const u32 = () => { const v = view.getUint32(off, true); off += 4; return v; };
  const f32 = () => { const v = view.getFloat32(off, true); off += 4; return v; };

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
  if (magic !== UINT8_HNSW_MAGIC_V1) throw new Error('range artifact prototype currently supports u8 Pancake snapshots only');
  if (metric !== 0) throw new Error('range artifact prototype currently supports L2 snapshots only');

  const scales = new Float32Array(count);
  const offsets = new Float32Array(count);
  for (let i = 0; i < count; i++) scales[i] = f32();
  for (let i = 0; i < count; i++) offsets[i] = f32();
  const qdata = raw.subarray(off, off + count * dim);
  off += qdata.byteLength;

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
        off += 4; // cached distance in uint8 snapshots
      }
      if (l === 0) base[id] = edges;
      else upper[id][l - 1] = edges;
    }
    if (!base[id]) base[id] = new Uint32Array(0);
  }
  return { kind: 'u8', dim, count, entryPoint, maxLevel, M, M0, metric, efConstruction, scales, offsets, qdata, levels, base, upper };
}

function readMetisPartition(file, count) {
  const lines = fs.readFileSync(file, 'utf8').trim().split(/\r?\n/);
  if (lines.length !== count) throw new Error(`partition has ${lines.length} rows; expected ${count}`);
  const partition = new Int32Array(count);
  let maxPart = -1;
  for (let i = 0; i < count; i++) {
    const p = Number(lines[i].trim());
    if (!Number.isInteger(p) || p < 0) throw new Error(`bad partition row ${i + 1}`);
    partition[i] = p;
    if (p > maxPart) maxPart = p;
  }
  return { partition, parts: maxPart + 1 };
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
    let read = 0, write = 0;
    queue[write++] = start;
    visited[start] = 1;
    while (read < write) {
      const id = queue[read++];
      order[orderLen++] = id;
      const ns = Array.from(index.base[id]).filter((n) => !visited[n]);
      ns.sort((a, b) => degrees[a] - degrees[b] || a - b);
      for (const n of ns) {
        if (visited[n]) continue;
        visited[n] = 1;
        queue[write++] = n;
      }
    }
  }
  const perm = new Uint32Array(count);
  for (let pos = 0; pos < orderLen; pos++) perm[order[orderLen - 1 - pos]] = pos;
  return perm;
}

function partitionPermutation(partition, fallbackPermutation) {
  const ids = Array.from({ length: partition.length }, (_, id) => id);
  ids.sort((a, b) => partition[a] - partition[b] || fallbackPermutation[a] - fallbackPermutation[b]);
  const originalToOrdinal = new Uint32Array(partition.length);
  const ordinalToOriginal = new Uint32Array(partition.length);
  for (let pos = 0; pos < ids.length; pos++) {
    originalToOrdinal[ids[pos]] = pos;
    ordinalToOriginal[pos] = ids[pos];
  }
  return { originalToOrdinal, ordinalToOriginal };
}

function recordBytes(index) {
  return 4 + 2 + 2 + index.maxLevel * 2 + index.dim + 8 + index.M0 * 4 + index.maxLevel * index.M * 4;
}

function writeNodeRecord(index, id, record) {
  record.fill(0);
  let off = 0;
  record.writeUInt32LE(id, off); off += 4;
  record.writeUInt16LE(index.levels[id], off); off += 2;
  const baseEdges = index.base[id];
  record.writeUInt16LE(baseEdges.length, off); off += 2;
  for (let l = 1; l <= index.maxLevel; l++) {
    record.writeUInt16LE((index.upper[id][l - 1] || []).length, off);
    off += 2;
  }
  Buffer.from(index.qdata.buffer, index.qdata.byteOffset + id * index.dim, index.dim).copy(record, off); off += index.dim;
  record.writeFloatLE(index.scales[id], off); off += 4;
  record.writeFloatLE(index.offsets[id], off); off += 4;
  for (let i = 0; i < index.M0; i++) {
    record.writeUInt32LE(i < baseEdges.length ? baseEdges[i] : 0xFFFFFFFF, off);
    off += 4;
  }
  for (let l = 1; l <= index.maxLevel; l++) {
    const edges = index.upper[id][l - 1] || [];
    for (let i = 0; i < index.M; i++) {
      record.writeUInt32LE(i < edges.length ? edges[i] : 0xFFFFFFFF, off);
      off += 4;
    }
  }
}

function exportRangeArtifact(snapshotPath, partitionPath, outPath) {
  console.log(`Parsing ${snapshotPath}...`);
  const snapshot = fs.readFileSync(snapshotPath);
  const index = parseSnapshot(new Uint8Array(snapshot.buffer, snapshot.byteOffset, snapshot.byteLength));
  console.log(`Parsed u8 graph: ${index.count.toLocaleString()} nodes, ${index.dim}D, maxLevel=${index.maxLevel}, M=${index.M}, M0=${index.M0}`);

  const metis = readMetisPartition(partitionPath, index.count);
  console.log(`Loaded ${metis.parts.toLocaleString()} METIS parts`);
  console.log('Building RCM fallback order...');
  const rcm = reverseCuthillMckeePermutation(index);
  const layout = partitionPermutation(metis.partition, rcm);
  const recBytes = recordBytes(index);
  const idMapOffset = HEADER_BYTES;
  const recordsOffset = idMapOffset + index.count * 4;
  const totalBytes = recordsOffset + index.count * recBytes;

  const fd = fs.openSync(outPath, 'w');
  try {
    const header = Buffer.alloc(HEADER_BYTES);
    let h = 0;
    header.writeUInt32LE(RANGE_MAGIC, h); h += 4;
    header.writeUInt32LE(1, h); h += 4;
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
    header.writeUInt32LE(recordsOffset, h); h += 4;
    header.writeUInt32LE(metis.parts, h); h += 4;
    fs.writeSync(fd, header, 0, header.length, 0);
    fs.writeSync(fd, Buffer.from(layout.originalToOrdinal.buffer), 0, index.count * 4, idMapOffset);

    const record = Buffer.alloc(recBytes);
    for (let ordinal = 0; ordinal < index.count; ordinal++) {
      const id = layout.ordinalToOriginal[ordinal];
      writeNodeRecord(index, id, record);
      fs.writeSync(fd, record, 0, record.length, recordsOffset + ordinal * recBytes);
      if ((ordinal + 1) % 100000 === 0) console.log(`  wrote ${ordinal + 1}/${index.count} records`);
    }
  } finally {
    fs.closeSync(fd);
  }
  console.log(`Wrote ${outPath} (${(totalBytes / 1048576).toFixed(1)} MiB, record=${recBytes} bytes)`);
  const manifestOut = arg('manifest-out', null);
  if (manifestOut) writeManifest(outPath, manifestOut, { sourceSnapshot: snapshotPath, sourcePartition: partitionPath });
}

function exportSplitRangeArtifact(snapshotPath, partitionPath, outPath) {
  console.log(`Parsing ${snapshotPath}...`);
  const snapshot = fs.readFileSync(snapshotPath);
  const index = parseSnapshot(new Uint8Array(snapshot.buffer, snapshot.byteOffset, snapshot.byteLength));
  console.log(`Parsed u8 graph: ${index.count.toLocaleString()} nodes, ${index.dim}D, maxLevel=${index.maxLevel}, M=${index.M}, M0=${index.M0}`);

  const metis = readMetisPartition(partitionPath, index.count);
  console.log(`Loaded ${metis.parts.toLocaleString()} METIS parts`);
  console.log('Building RCM fallback order...');
  const rcm = reverseCuthillMckeePermutation(index);
  const layout = partitionPermutation(metis.partition, rcm);
  const recBytes = recordBytes(index);
  const locationMap = new Uint32Array(index.count);
  const routerIds = [];
  const baseIds = [];
  for (let ordinal = 0; ordinal < index.count; ordinal++) {
    const id = layout.ordinalToOriginal[ordinal];
    if (index.levels[id] > 0) {
      locationMap[id] = ROUTER_LOCATION_MASK | routerIds.length;
      routerIds.push(id);
    } else {
      locationMap[id] = baseIds.length;
      baseIds.push(id);
    }
  }

  const idMapOffset = HEADER_BYTES_V2;
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
    header.writeUInt32LE(metis.parts, h); h += 4;
    header.writeUInt32LE(routerIds.length, h); h += 4;
    header.writeUInt32LE(baseIds.length, h); h += 4;
    header.writeUInt32LE(routerRecordsOffset, h); h += 4;
    header.writeUInt32LE(baseRecordsOffset, h); h += 4;
    fs.writeSync(fd, header, 0, header.length, 0);
    fs.writeSync(fd, Buffer.from(locationMap.buffer), 0, index.count * 4, idMapOffset);

    const record = Buffer.alloc(recBytes);
    for (let ordinal = 0; ordinal < routerIds.length; ordinal++) {
      writeNodeRecord(index, routerIds[ordinal], record);
      fs.writeSync(fd, record, 0, record.length, routerRecordsOffset + ordinal * recBytes);
    }
    console.log(`  wrote ${routerIds.length.toLocaleString()} router records`);
    for (let ordinal = 0; ordinal < baseIds.length; ordinal++) {
      writeNodeRecord(index, baseIds[ordinal], record);
      fs.writeSync(fd, record, 0, record.length, baseRecordsOffset + ordinal * recBytes);
      if ((ordinal + 1) % 100000 === 0) console.log(`  wrote ${ordinal + 1}/${baseIds.length} base records`);
    }
  } finally {
    fs.closeSync(fd);
  }
  console.log(`Wrote ${outPath} (${(totalBytes / 1048576).toFixed(1)} MiB, router=${routerIds.length.toLocaleString()}, base=${baseIds.length.toLocaleString()}, record=${recBytes} bytes)`);
  const manifestOut = arg('manifest-out', null);
  if (manifestOut) writeManifest(outPath, manifestOut, { sourceSnapshot: snapshotPath, sourcePartition: partitionPath });
}

function hashFile(file) {
  const hash = crypto.createHash('sha256');
  const fd = fs.openSync(file, 'r');
  const buf = Buffer.allocUnsafe(4 * 1024 * 1024);
  try {
    while (true) {
      const read = fs.readSync(fd, buf, 0, buf.length, null);
      if (read === 0) break;
      hash.update(buf.subarray(0, read));
    }
  } finally {
    fs.closeSync(fd);
  }
  return hash.digest('hex');
}

class RangeArtifact {
  constructor(file) {
    this.file = file;
    this.fd = fs.openSync(file, 'r');
    const header = Buffer.alloc(HEADER_BYTES);
    fs.readSync(this.fd, header, 0, HEADER_BYTES, 0);
    let h = 0;
    const magic = header.readUInt32LE(h); h += 4;
    if (magic !== RANGE_MAGIC) throw new Error('not a Pancake range artifact');
    this.version = header.readUInt32LE(h); h += 4;
    this.kind = header.readUInt32LE(h); h += 4;
    this.dim = header.readUInt32LE(h); h += 4;
    this.count = header.readUInt32LE(h); h += 4;
    this.entryPoint = header.readUInt32LE(h); h += 4;
    this.maxLevel = header.readUInt32LE(h); h += 4;
    this.M = header.readUInt32LE(h); h += 4;
    this.M0 = header.readUInt32LE(h); h += 4;
    this.metric = header.readUInt32LE(h); h += 4;
    this.recordBytes = header.readUInt32LE(h); h += 4;
    this.idMapOffset = header.readUInt32LE(h); h += 4;
    this.recordsOffset = header.readUInt32LE(h); h += 4;
    this.parts = header.readUInt32LE(h); h += 4;
    if (this.version >= 2) {
      this.routerCount = header.readUInt32LE(h); h += 4;
      this.baseCount = header.readUInt32LE(h); h += 4;
      this.routerRecordsOffset = header.readUInt32LE(h); h += 4;
      this.baseRecordsOffset = header.readUInt32LE(h); h += 4;
    } else {
      this.routerCount = 0;
      this.baseCount = this.count;
      this.routerRecordsOffset = 0;
      this.baseRecordsOffset = this.recordsOffset;
    }
    const idMapBytes = Buffer.alloc(this.count * 4);
    fs.readSync(this.fd, idMapBytes, 0, idMapBytes.length, this.idMapOffset);
    this.originalToLocation = new Uint32Array(idMapBytes.buffer, idMapBytes.byteOffset, this.count);
    this.originalToOrdinal = this.originalToLocation;
    this.cache = new Map();
    this.pageCache = new Set();
    this.rangeRequests = 0;
    this.rangeBytes = 0;
    this.rangeNodesDecoded = 0;
    this.currentRanges = [];
    this.residentRouter = this.version >= 2 ? this.loadRouterSegment() : { records: 0, bytes: 0 };
    this.resetStats();
  }
  close() { fs.closeSync(this.fd); }
  markRanges() { return this.currentRanges.length; }
  rangesSince(mark) { return this.currentRanges.slice(mark); }
  resetStats() {
    this.rangeRequests = 0;
    this.rangeBytes = 0;
    this.rangeNodesDecoded = 0;
    this.currentRanges = [];
  }
  clearCache() {
    this.cache.clear();
    this.pageCache.clear();
  }
  readNode(id) {
    const cached = this.cache.get(id);
    if (cached) return cached;
    this.prefetch([id], 0);
    return this.cache.get(id);
  }
  recordAddressForId(id) {
    const location = this.originalToLocation[id];
    if (this.version >= 2) {
      const ordinal = location & LOCATION_ORDINAL_MASK;
      if ((location & ROUTER_LOCATION_MASK) !== 0) {
        if (ordinal >= this.routerCount) throw new Error(`router ordinal ${ordinal} outside artifact`);
        return this.routerRecordsOffset + ordinal * this.recordBytes;
      }
      if (ordinal >= this.baseCount) throw new Error(`base ordinal ${ordinal} outside artifact`);
      return this.baseRecordsOffset + ordinal * this.recordBytes;
    }
    if (location >= this.count) throw new Error(`node id ${id} is not addressable in artifact`);
    return this.recordsOffset + location * this.recordBytes;
  }
  prefetch(ids, gap = 0) {
    if (this.version >= 2) return this.prefetchLocations(ids, gap);
    const ordinals = [];
    const seen = new Set();
    for (const id of ids) {
      if (this.cache.has(id) || seen.has(id)) continue;
      seen.add(id);
      const ordinal = this.originalToOrdinal[id];
      if (ordinal === 0xffffffff || ordinal >= this.count) throw new Error(`node id ${id} is not addressable in artifact`);
      ordinals.push(ordinal);
    }
    if (!ordinals.length) return 0;
    ordinals.sort((a, b) => a - b);
    let requests = 0;
    let runStart = ordinals[0];
    let runEnd = ordinals[0];
    const flush = () => {
      const count = runEnd - runStart + 1;
      const bytes = count * this.recordBytes;
      const startByte = this.recordsOffset + runStart * this.recordBytes;
      const buf = Buffer.allocUnsafe(bytes);
      fs.readSync(this.fd, buf, 0, bytes, startByte);
      this.rangeRequests++;
      this.rangeBytes += bytes;
      this.currentRanges.push([startByte, startByte + bytes]);
      requests++;
      for (let i = 0; i < count; i++) {
        const off = i * this.recordBytes;
        const record = buf.subarray(off, off + this.recordBytes);
        const originalId = record.readUInt32LE(0);
        if (!this.cache.has(originalId)) {
          this.cache.set(originalId, this.decodeNode(record));
          this.rangeNodesDecoded++;
        }
      }
    };
    for (let i = 1; i < ordinals.length; i++) {
      const ordinal = ordinals[i];
      if (ordinal <= runEnd + gap + 1) {
        runEnd = ordinal;
      } else {
        flush();
        runStart = ordinal;
        runEnd = ordinal;
      }
    }
    flush();
    return requests;
  }
  prefetchLocations(ids, gap = 0) {
    const addresses = [];
    const seen = new Set();
    for (const id of ids) {
      if (this.cache.has(id) || seen.has(id)) continue;
      seen.add(id);
      addresses.push(this.recordAddressForId(id));
    }
    if (!addresses.length) return 0;
    addresses.sort((a, b) => a - b);
    let requests = 0;
    let runStart = addresses[0];
    let runEnd = addresses[0] + this.recordBytes;
    const flush = () => {
      const bytes = runEnd - runStart;
      const buf = Buffer.allocUnsafe(bytes);
      fs.readSync(this.fd, buf, 0, bytes, runStart);
      this.rangeRequests++;
      this.rangeBytes += bytes;
      this.currentRanges.push([runStart, runEnd]);
      requests++;
      for (let off = 0; off < bytes; off += this.recordBytes) {
        const record = buf.subarray(off, off + this.recordBytes);
        const originalId = record.readUInt32LE(0);
        if (!this.cache.has(originalId)) {
          this.cache.set(originalId, this.decodeNode(record));
          this.rangeNodesDecoded++;
        }
      }
    };
    for (let i = 1; i < addresses.length; i++) {
      const address = addresses[i];
      if (address <= runEnd + gap) {
        runEnd = address + this.recordBytes;
      } else {
        flush();
        runStart = address;
        runEnd = address + this.recordBytes;
      }
    }
    flush();
    return requests;
  }
  loadRouterSegment() {
    if (this.version < 2 || this.routerCount === 0) return { records: 0, bytes: 0 };
    const bytes = this.routerCount * this.recordBytes;
    const buf = Buffer.allocUnsafe(bytes);
    fs.readSync(this.fd, buf, 0, bytes, this.routerRecordsOffset);
    for (let i = 0; i < this.routerCount; i++) {
      const off = i * this.recordBytes;
      const record = buf.subarray(off, off + this.recordBytes);
      const originalId = record.readUInt32LE(0);
      if (!this.cache.has(originalId)) this.cache.set(originalId, this.decodeNode(record));
    }
    return { records: this.routerCount, bytes };
  }
  prefetchPages(ids, pageSize) {
    const recordsPerPage = Math.max(1, Math.floor(pageSize / this.recordBytes));
    const pages = [];
    const seenPages = new Set();
    for (const id of ids) {
      if (this.cache.has(id)) continue;
      const ordinal = this.originalToOrdinal[id];
      if (ordinal === 0xffffffff || ordinal >= this.count) throw new Error(`node id ${id} is not addressable in artifact`);
      const page = Math.floor(ordinal / recordsPerPage);
      if (this.pageCache.has(page) || seenPages.has(page)) continue;
      seenPages.add(page);
      pages.push(page);
    }
    if (!pages.length) return 0;
    pages.sort((a, b) => a - b);
    let requests = 0;
    for (const page of pages) {
      const startOrdinal = page * recordsPerPage;
      const count = Math.min(recordsPerPage, this.count - startOrdinal);
      const bytes = count * this.recordBytes;
      const startByte = this.recordsOffset + startOrdinal * this.recordBytes;
      const buf = Buffer.allocUnsafe(bytes);
      fs.readSync(this.fd, buf, 0, bytes, startByte);
      this.pageCache.add(page);
      this.rangeRequests++;
      this.rangeBytes += bytes;
      this.currentRanges.push([startByte, startByte + bytes]);
      requests++;
      for (let i = 0; i < count; i++) {
        const off = i * this.recordBytes;
        const record = buf.subarray(off, off + this.recordBytes);
        const originalId = record.readUInt32LE(0);
        if (!this.cache.has(originalId)) {
          this.cache.set(originalId, this.decodeNode(record));
          this.rangeNodesDecoded++;
        }
      }
    }
    return requests;
  }
  readNodeDirect(id) {
    const cached = this.cache.get(id);
    if (cached) return cached;
    const address = this.recordAddressForId(id);
    const buf = Buffer.allocUnsafe(this.recordBytes);
    fs.readSync(this.fd, buf, 0, this.recordBytes, address);
    return this.decodeNode(buf);
  }
  prefetchCustomPages(ids, layout, gap = 0) {
    const pages = [];
    const residualIds = [];
    const seenPages = new Set();
    for (const id of ids) {
      if (this.cache.has(id)) continue;
      const pageId = layout.nodeToPage.get(id);
      if (pageId === undefined) {
        residualIds.push(id);
        continue;
      }
      if (layout.pageCache.has(pageId) || seenPages.has(pageId)) continue;
      seenPages.add(pageId);
      pages.push(pageId);
    }
    for (const pageId of pages) {
      const page = layout.pages[pageId];
      layout.pageCache.add(pageId);
      this.rangeRequests++;
      this.rangeBytes += page.length * this.recordBytes;
      this.currentRanges.push([-(pageId + 1), -(pageId + 1)]);
      for (const id of page) {
        if (!this.cache.has(id)) {
          this.cache.set(id, this.readNodeDirect(id));
          this.rangeNodesDecoded++;
        }
      }
    }
    return pages.length + this.prefetchLocations(residualIds, gap);
  }
  loadOrdinalPrefix(prefixRecords) {
    const count = Math.min(prefixRecords, this.count);
    if (count <= 0) return { records: 0, bytes: 0 };
    const chunkRecords = Math.max(1, Math.floor((4 * 1024 * 1024) / this.recordBytes));
    for (let startOrdinal = 0; startOrdinal < count; startOrdinal += chunkRecords) {
      const n = Math.min(chunkRecords, count - startOrdinal);
      const bytes = n * this.recordBytes;
      const startByte = this.recordsOffset + startOrdinal * this.recordBytes;
      const buf = Buffer.allocUnsafe(bytes);
      fs.readSync(this.fd, buf, 0, bytes, startByte);
      for (let i = 0; i < n; i++) {
        const off = i * this.recordBytes;
        const record = buf.subarray(off, off + this.recordBytes);
        const originalId = record.readUInt32LE(0);
        if (!this.cache.has(originalId)) {
          this.cache.set(originalId, this.decodeNode(record));
        }
      }
    }
    return { records: count, bytes: count * this.recordBytes };
  }
  loadNodeSet(ids) {
    const resident = [];
    const seen = new Set();
    for (const id of ids) {
      if (this.cache.has(id) || seen.has(id)) continue;
      seen.add(id);
      resident.push(id);
    }
    if (resident.length > 0) {
      this.prefetch(resident, 0);
      this.resetStats();
    }
    return { records: resident.length, bytes: resident.length * this.recordBytes };
  }
  loadUpperLayerNodes() {
    const ids = [];
    const chunkRecords = Math.max(1, Math.floor((8 * 1024 * 1024) / this.recordBytes));
    for (let startOrdinal = 0; startOrdinal < this.count; startOrdinal += chunkRecords) {
      const n = Math.min(chunkRecords, this.count - startOrdinal);
      const bytes = n * this.recordBytes;
      const buf = Buffer.allocUnsafe(bytes);
      fs.readSync(this.fd, buf, 0, bytes, this.recordsOffset + startOrdinal * this.recordBytes);
      for (let i = 0; i < n; i++) {
        const off = i * this.recordBytes;
        const id = buf.readUInt32LE(off);
        const level = buf.readUInt16LE(off + 4);
        if (level > 0) ids.push(id);
      }
    }
    return this.loadNodeSet(ids);
  }
  decodeNode(buf) {
    let off = 0;
    const id = buf.readUInt32LE(off); off += 4;
    const level = buf.readUInt16LE(off); off += 2;
    const baseCount = buf.readUInt16LE(off); off += 2;
    const upperCounts = new Uint16Array(this.maxLevel);
    for (let i = 0; i < this.maxLevel; i++) {
      upperCounts[i] = buf.readUInt16LE(off);
      off += 2;
    }
    const qdata = new Uint8Array(buf.buffer, buf.byteOffset + off, this.dim);
    off += this.dim;
    const scale = buf.readFloatLE(off); off += 4;
    const offset = buf.readFloatLE(off); off += 4;
    const base = new Uint32Array(baseCount);
    for (let i = 0; i < this.M0; i++) {
      const n = buf.readUInt32LE(off); off += 4;
      if (i < baseCount) base[i] = n;
    }
    const upper = Array.from({ length: this.maxLevel }, () => []);
    for (let l = 0; l < this.maxLevel; l++) {
      const edges = new Uint32Array(upperCounts[l]);
      for (let i = 0; i < this.M; i++) {
        const n = buf.readUInt32LE(off); off += 4;
        if (i < edges.length) edges[i] = n;
      }
      upper[l] = edges;
    }
    return { id, level, base, upper, qdata: new Uint8Array(qdata), scale, offset };
  }
}

function inspectArtifact(file, options = {}) {
  const artifact = new RangeArtifact(file);
  try {
    const sizeBytes = fs.statSync(file).size;
    const expectedSizeBytes = artifact.recordsOffset + artifact.count * artifact.recordBytes;
    const manifest = {
      format: 'pancake-range-artifact',
      formatVersion: artifact.version,
      file: path.resolve(file),
      sizeBytes,
      expectedSizeBytes,
      complete: sizeBytes === expectedSizeBytes,
      kind: artifact.kind === RANGE_KIND_U8 ? 'u8-affine-l2' : `unknown-${artifact.kind}`,
      metric: artifact.metric === 0 ? 'l2' : `unknown-${artifact.metric}`,
      layout: {
        permutation: artifact.version >= 2 ? 'split_router_then_metis_partition_rcm_base' : 'metis_partition_rcm_within_partition',
        parts: artifact.parts,
      },
      graph: {
        count: artifact.count,
        dim: artifact.dim,
        entryPoint: artifact.entryPoint,
        maxLevel: artifact.maxLevel,
        M: artifact.M,
        M0: artifact.M0,
      },
      addressing: {
        headerBytes: artifact.version >= 2 ? HEADER_BYTES_V2 : HEADER_BYTES,
        idMapOffset: artifact.idMapOffset,
        idMapBytes: artifact.count * 4,
        recordsOffset: artifact.recordsOffset,
        recordBytes: artifact.recordBytes,
        routerCount: artifact.routerCount,
        routerBytes: artifact.routerCount * artifact.recordBytes,
        routerRecordsOffset: artifact.routerRecordsOffset,
        baseCount: artifact.baseCount,
        baseBytes: artifact.baseCount * artifact.recordBytes,
        baseRecordsOffset: artifact.baseRecordsOffset,
        ordinalFormula: 'ordinal = u32(idMapOffset + originalId * 4)',
        byteRangeFormula: 'start = recordsOffset + ordinal * recordBytes; length = recordBytes',
        coalescing: 'sort requested ordinals and merge runs separated by <= gap records',
      },
      execution: {
        immutable: true,
        lazyMaterialization: true,
        boundedMemoryReader: 'id map plus decoded-node cache; node records are range-addressable',
      },
      createdAt: new Date().toISOString(),
      ...options,
    };
    if (hasArg('hash')) manifest.sha256 = hashFile(file);
    return manifest;
  } finally {
    artifact.close();
  }
}

function writeManifest(artifactPath, manifestOut, options = {}) {
  const manifest = inspectArtifact(artifactPath, options);
  fs.writeFileSync(manifestOut, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`Wrote ${manifestOut}`);
  return manifest;
}

function l2Node(query, node) {
  let sum = 0;
  for (let d = 0; d < query.length; d++) {
    const decoded = node.offset + node.scale * node.qdata[d];
    const diff = query[d] - decoded;
    sum += diff * diff;
  }
  return sum;
}

function l2Snapshot(index, query, id) {
  const qoff = id * index.dim;
  const scale = index.scales[id];
  const offset = index.offsets[id];
  let sum = 0;
  for (let d = 0; d < index.dim; d++) {
    const decoded = offset + scale * index.qdata[qoff + d];
    const diff = query[d] - decoded;
    sum += diff * diff;
  }
  return sum;
}

class MinHeap {
  constructor(compare) { this.a = []; this.compare = compare; }
  get size() { return this.a.length; }
  push(x) { const a = this.a; a.push(x); this.up(a.length - 1); }
  up(i) { const a = this.a, x = a[i]; while (i > 0) { const p = (i - 1) >> 1; if (this.compare(a[p], x) <= 0) break; a[i] = a[p]; i = p; } a[i] = x; }
  pop() { const a = this.a; if (!a.length) return undefined; const top = a[0], x = a.pop(); if (a.length) { let i = 0; while (true) { let c = i * 2 + 1; if (c >= a.length) break; if (c + 1 < a.length && this.compare(a[c + 1], a[c]) < 0) c++; if (this.compare(a[c], x) >= 0) break; a[i] = a[c]; i = c; } a[i] = x; } return top; }
  peek() { return this.a[0]; }
  sorted() { return [...this.a].sort(this.compare); }
}

function snapshotSearch(index, query, k, efSearch) {
  let curr = index.entryPoint;
  let currDist = l2Snapshot(index, query, curr);
  for (let level = index.maxLevel; level > 0; level--) {
    let changed = true;
    while (changed) {
      changed = false;
      const edges = index.upper[curr][level - 1] || [];
      for (const n of edges) {
        const d = l2Snapshot(index, query, n);
        if (d < currDist) {
          curr = n;
          currDist = d;
          changed = true;
        }
      }
    }
  }
  const ef = Math.max(efSearch, k);
  const visited = new Uint8Array(index.count);
  const candidates = new MinHeap((a, b) => a[0] - b[0]);
  const results = new MinHeap((a, b) => b[0] - a[0]);
  const d0 = l2Snapshot(index, query, curr);
  candidates.push([d0, curr]);
  results.push([d0, curr]);
  visited[curr] = 1;
  while (candidates.size > 0) {
    const [candDist, candId] = candidates.pop();
    const worst = results.peek();
    if (results.size >= ef && worst && candDist > worst[0]) break;
    for (const n of index.base[candId]) {
      if (visited[n]) continue;
      visited[n] = 1;
      const nd = l2Snapshot(index, query, n);
      const currentWorst = results.peek();
      if (results.size < ef || nd < currentWorst[0]) {
        candidates.push([nd, n]);
        results.push([nd, n]);
        if (results.size > ef) results.pop();
      }
    }
  }
  return results.a.sort((a, b) => a[0] - b[0]).slice(0, k).map((x) => x[1]);
}

function lazySearch(artifact, query, k, efSearch, options = {}) {
  const gap = options.gap || 0;
  const pageSize = options.pageSize || 0;
  const customPages = options.customPages || null;
  const lookahead = options.lookahead || 0;
  const sameRoundLookahead = !!options.sameRoundLookahead;
  const expansionBatch = Math.max(1, options.expansionBatch || OPERATIONAL_EXPANSION_BATCH);
  const rounds = [];
  const prefetchRound = (ids, meta = {}) => {
    if (!ids.length) return;
    const beforeRequests = artifact.rangeRequests;
    const beforeBytes = artifact.rangeBytes;
    const beforeRanges = artifact.markRanges();
    if (customPages) artifact.prefetchCustomPages(ids, customPages, gap);
    else if (pageSize > 0) artifact.prefetchPages(ids, pageSize);
    else artifact.prefetch(ids, gap);
    const ranges = artifact.rangesSince(beforeRanges);
    rounds.push({
      ...meta,
      ids: ids.length,
      fetchedIds: options.traceRounds ? Array.from(ids) : undefined,
      requests: artifact.rangeRequests - beforeRequests,
      bytes: artifact.rangeBytes - beforeBytes,
      rangeBytes: ranges.map(([start, end]) => end - start),
    });
  };
  const read = (id) => artifact.readNode(id);
  let curr = artifact.entryPoint;
  prefetchRound([curr], { phase: 'entry', level: artifact.maxLevel, sourceIds: [] });
  let currNode = read(curr);
  let currDist = l2Node(query, currNode);
  for (let level = artifact.maxLevel; level > 0; level--) {
    let changed = true;
    while (changed) {
      changed = false;
      const edges = currNode.upper[level - 1] || [];
      prefetchRound(edges, { phase: 'upper', level, sourceIds: [curr] });
      for (const n of edges) {
        const node = read(n);
        const d = l2Node(query, node);
        if (d < currDist) {
          curr = n;
          currNode = node;
          currDist = d;
          changed = true;
        }
      }
    }
  }
  const ef = Math.max(efSearch, k);
  const visited = new Uint8Array(artifact.count);
  const candidates = new MinHeap((a, b) => a[0] - b[0]);
  const results = new MinHeap((a, b) => b[0] - a[0]);
  currNode = read(curr);
  const d0 = l2Node(query, currNode);
  candidates.push([d0, curr]);
  results.push([d0, curr]);
  visited[curr] = 1;
  while (candidates.size > 0) {
    const batch = [];
    while (batch.length < expansionBatch && candidates.size > 0) {
      const next = candidates.peek();
      const worst = results.peek();
      if (results.size >= ef && worst && next && next[0] > worst[0]) break;
      batch.push(candidates.pop());
    }
    if (!batch.length) break;

    const toVisit = [];
    for (const [, candId] of batch) {
      const node = read(candId);
      for (const n of node.base) {
        if (visited[n]) continue;
        visited[n] = 1;
        toVisit.push(n);
      }
    }
    const actualToVisitCount = toVisit.length;
    if (sameRoundLookahead && lookahead > 0 && candidates.size > 0) {
      const worstAfter = results.peek();
      const seenSpeculative = new Set(toVisit);
      let considered = 0;
      for (const [dist, id] of candidates.sorted()) {
        if (considered >= lookahead) break;
        if (results.size >= ef && worstAfter && dist > worstAfter[0]) break;
        const candidateNode = artifact.cache.get(id);
        if (!candidateNode) continue;
        considered++;
        for (const n of candidateNode.base) {
          if (!visited[n] && !artifact.cache.has(n) && !seenSpeculative.has(n)) {
            seenSpeculative.add(n);
            toVisit.push(n);
          }
        }
      }
    }
    prefetchRound(toVisit, { phase: 'base', level: 0, sourceIds: batch.map(([, id]) => id) });
    for (let i = 0; i < actualToVisitCount; i++) {
      const n = toVisit[i];
      const nn = read(n);
      const nd = l2Node(query, nn);
      const currentWorst = results.peek();
      if (results.size < ef || nd < currentWorst[0]) {
        candidates.push([nd, n]);
        results.push([nd, n]);
        if (results.size > ef) results.pop();
      }
    }
    if (!sameRoundLookahead && lookahead > 0 && candidates.size > 0) {
      const worstAfter = results.peek();
      const speculative = [];
      const seenSpeculative = new Set();
      let considered = 0;
      for (const [dist, id] of candidates.sorted()) {
        if (considered >= lookahead) break;
        if (results.size >= ef && worstAfter && dist > worstAfter[0]) break;
        const candidateNode = artifact.cache.get(id);
        if (!candidateNode) continue;
        considered++;
        for (const n of candidateNode.base) {
          if (!visited[n] && !artifact.cache.has(n) && !seenSpeculative.has(n)) {
            seenSpeculative.add(n);
            speculative.push(n);
          }
        }
      }
      prefetchRound(speculative, { phase: 'lookahead', level: 0, sourceIds: candidates.sorted().slice(0, lookahead).map(([, id]) => id) });
    }
  }
  return { rounds, results: results.a.sort((a, b) => a[0] - b[0]).slice(0, k).map((x) => x[1]) };
}

function readFvecs(file, limit = Infinity) {
  const fd = fs.openSync(file, 'r');
  const vectors = [];
  let dim = null, pos = 0;
  const header = Buffer.allocUnsafe(4);
  try {
    while (vectors.length < limit) {
      const got = fs.readSync(fd, header, 0, 4, pos);
      if (got === 0) break;
      pos += 4;
      const d = header.readInt32LE(0);
      if (dim == null) dim = d;
      const buf = Buffer.allocUnsafe(d * 4);
      fs.readSync(fd, buf, 0, buf.length, pos);
      pos += buf.length;
      const v = new Float32Array(d);
      for (let i = 0; i < d; i++) v[i] = buf.readFloatLE(i * 4);
      vectors.push(v);
    }
  } finally {
    fs.closeSync(fd);
  }
  return { vectors, dim };
}

function readIvecs(file, limit = Infinity) {
  const fd = fs.openSync(file, 'r');
  const vectors = [];
  let dim = null, pos = 0;
  const header = Buffer.allocUnsafe(4);
  try {
    while (vectors.length < limit) {
      const got = fs.readSync(fd, header, 0, 4, pos);
      if (got === 0) break;
      pos += 4;
      const d = header.readInt32LE(0);
      if (dim == null) dim = d;
      const buf = Buffer.allocUnsafe(d * 4);
      fs.readSync(fd, buf, 0, buf.length, pos);
      pos += buf.length;
      const ids = [];
      for (let i = 0; i < d; i++) ids.push(buf.readInt32LE(i * 4));
      vectors.push(ids);
    }
  } finally {
    fs.closeSync(fd);
  }
  return { vectors, dim };
}

function recallAt(predicted, truth, k) {
  const expected = new Set(truth.slice(0, k));
  let hits = 0;
  for (const id of predicted.slice(0, k)) {
    if (expected.has(id)) hits++;
  }
  return hits / k;
}

function runLocalSearch(artifactPath, dataDir, queries, k, efSearch, gap, clearCachePerQuery) {
  const artifact = new RangeArtifact(artifactPath);
  const queryData = readFvecs(path.join(dataDir, 'sift_query.fvecs'), queries);
  if (queryData.dim !== artifact.dim) throw new Error(`query dim ${queryData.dim} != artifact dim ${artifact.dim}`);
  const requestCounts = [];
  const byteCounts = [];
  const roundCounts = [];
  const requestRounds = [];
  const byteRounds = [];
  const t0 = performance.now();
  try {
    for (let i = 0; i < queryData.vectors.length; i++) {
      if (clearCachePerQuery) artifact.clearCache();
      artifact.resetStats();
      const res = lazySearch(artifact, queryData.vectors[i], k, efSearch, { gap });
      requestCounts.push(artifact.rangeRequests);
      byteCounts.push(artifact.rangeBytes);
      roundCounts.push(res.rounds.length);
      for (const round of res.rounds) {
        requestRounds.push(round.requests);
        byteRounds.push(round.bytes);
      }
      if ((i + 1) % 25 === 0 || i + 1 === queryData.vectors.length) console.log(`  searched ${i + 1}/${queryData.vectors.length}`);
    }
  } finally {
    artifact.close();
  }
  const summary = {
    artifact: artifactPath,
    queries: queryData.vectors.length,
    k,
    efSearch,
    gap,
    cacheMode: clearCachePerQuery ? 'cold_per_query' : 'warm_stream',
    recordBytes: artifact.recordBytes,
    cacheNodes: artifact.cache.size,
    requests: summarize(requestCounts),
    bytes: summarize(byteCounts),
    mibMean: summarize(byteCounts).mean / 1048576,
    rounds: summarize(roundCounts),
    requestsPerRound: summarize(requestRounds),
    bytesPerRound: summarize(byteRounds),
    elapsedSeconds: (performance.now() - t0) / 1000,
  };
  if (!hasArg('quiet')) console.log(JSON.stringify(summary, null, 2));
  const summaryOut = arg('summary-out', null);
  if (summaryOut) fs.writeFileSync(summaryOut, `${JSON.stringify(summary, null, 2)}\n`);
}

function uniqueRangeBytes(ranges) {
  if (ranges.length === 0) return 0;
  const sorted = [...ranges].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let total = 0;
  let start = sorted[0][0];
  let end = sorted[0][1];
  for (let i = 1; i < sorted.length; i++) {
    const [s, e] = sorted[i];
    if (s <= end) {
      if (e > end) end = e;
    } else {
      total += end - start;
      start = s;
      end = e;
    }
  }
  total += end - start;
  return total;
}

function runWorkingSet(artifactPath, dataDir, queries, k, efSearch, gap, checkpoints) {
  const artifact = new RangeArtifact(artifactPath);
  const artifactBytes = fs.statSync(artifactPath).size;
  const queryData = readFvecs(path.join(dataDir, 'sift_query.fvecs'), queries);
  if (queryData.dim !== artifact.dim) throw new Error(`query dim ${queryData.dim} != artifact dim ${artifact.dim}`);
  const wanted = new Set(checkpoints.filter((n) => n > 0 && n <= queryData.vectors.length));
  wanted.add(queryData.vectors.length);
  const ranges = [];
  const rows = [];
  const requestCounts = [];
  const byteCounts = [];
  const roundCounts = [];
  const t0 = performance.now();
  try {
    for (let i = 0; i < queryData.vectors.length; i++) {
      artifact.resetStats();
      const res = lazySearch(artifact, queryData.vectors[i], k, efSearch, { gap });
      for (const range of artifact.currentRanges) ranges.push(range);
      requestCounts.push(artifact.rangeRequests);
      byteCounts.push(artifact.rangeBytes);
      roundCounts.push(res.rounds.length);
      const seenQueries = i + 1;
      if (wanted.has(seenQueries)) {
        const uniqueBytes = uniqueRangeBytes(ranges);
        rows.push({
          queries: seenQueries,
          uniqueFetchedBytes: uniqueBytes,
          uniqueFetchedMiB: uniqueBytes / 1048576,
          artifactFraction: uniqueBytes / artifactBytes,
          cacheNodes: artifact.cache.size,
          cacheNodeFraction: artifact.cache.size / artifact.count,
          totalRangeRequests: requestCounts.reduce((a, b) => a + b, 0),
          totalFetchedBytes: byteCounts.reduce((a, b) => a + b, 0),
        });
      }
      if ((i + 1) % 25 === 0 || i + 1 === queryData.vectors.length) console.log(`  measured ${i + 1}/${queryData.vectors.length}`);
    }
  } finally {
    artifact.close();
  }
  const summary = {
    artifact: artifactPath,
    queries: queryData.vectors.length,
    k,
    efSearch,
    gap,
    cacheMode: 'warm_stream',
    artifactBytes,
    recordBytes: artifact.recordBytes,
    checkpoints: rows,
    requests: summarize(requestCounts),
    bytes: summarize(byteCounts),
    mibMean: summarize(byteCounts).mean / 1048576,
    rounds: summarize(roundCounts),
    elapsedSeconds: (performance.now() - t0) / 1000,
  };
  if (!hasArg('quiet')) console.log(JSON.stringify(summary, null, 2));
  const summaryOut = arg('summary-out', null);
  if (summaryOut) fs.writeFileSync(summaryOut, `${JSON.stringify(summary, null, 2)}\n`);
}

function parseNumberListArg(name, fallback) {
  const raw = arg(name, null);
  if (raw == null) return fallback;
  const values = raw.split(',').map((x) => Number(x.trim())).filter((x) => Number.isFinite(x) && x >= 0);
  if (values.length === 0) throw new Error(`--${name} must contain at least one non-negative number`);
  return Array.from(new Set(values)).sort((a, b) => a - b);
}

function roundLatencyMs(round, fixedMs, bandwidthMiBps, parallelism) {
  if (round.requests <= 0) return 0;
  const bandwidthBytesPerMs = bandwidthMiBps * 1048576 / 1000;
  const sizes = round.rangeBytes && round.rangeBytes.length
    ? [...round.rangeBytes].sort((a, b) => b - a)
    : Array.from({ length: round.requests }, () => round.bytes / round.requests);
  const lanes = new Array(Math.max(1, parallelism)).fill(0);
  for (const bytes of sizes) {
    let best = 0;
    for (let i = 1; i < lanes.length; i++) {
      if (lanes[i] < lanes[best]) best = i;
    }
    lanes[best] += fixedMs + bytes / bandwidthBytesPerMs;
  }
  return Math.max(...lanes);
}

function modelQueryLatency(rounds, fixedMs, bandwidthMiBps, parallelism) {
  let total = 0;
  for (const round of rounds) total += roundLatencyMs(round, fixedMs, bandwidthMiBps, parallelism);
  return total;
}

function runLatencyModel(artifactPath, dataDir, queries, k, efSearch, gap, clearCachePerQuery, fixedLatencies, bandwidths, parallelism) {
  const artifact = new RangeArtifact(artifactPath);
  const queryData = readFvecs(path.join(dataDir, 'sift_query.fvecs'), queries);
  if (queryData.dim !== artifact.dim) throw new Error(`query dim ${queryData.dim} != artifact dim ${artifact.dim}`);
  const perQueryRounds = [];
  const requestCounts = [];
  const byteCounts = [];
  const roundCounts = [];
  const t0 = performance.now();
  try {
    for (let i = 0; i < queryData.vectors.length; i++) {
      if (clearCachePerQuery) artifact.clearCache();
      artifact.resetStats();
      const res = lazySearch(artifact, queryData.vectors[i], k, efSearch, { gap });
      perQueryRounds.push(res.rounds);
      requestCounts.push(artifact.rangeRequests);
      byteCounts.push(artifact.rangeBytes);
      roundCounts.push(res.rounds.length);
      if ((i + 1) % 25 === 0 || i + 1 === queryData.vectors.length) console.log(`  modeled ${i + 1}/${queryData.vectors.length}`);
    }
  } finally {
    artifact.close();
  }

  const scenarios = [];
  for (const fixedMs of fixedLatencies) {
    for (const bandwidthMiBps of bandwidths) {
      const latencies = perQueryRounds.map((rounds) => modelQueryLatency(rounds, fixedMs, bandwidthMiBps, parallelism));
      scenarios.push({
        fixedMs,
        bandwidthMiBps,
        parallelism,
        latencyMs: summarize(latencies),
      });
    }
  }

  const summary = {
    artifact: artifactPath,
    queries: queryData.vectors.length,
    k,
    efSearch,
    gap,
    cacheMode: clearCachePerQuery ? 'cold_per_query' : 'warm_stream',
    recordBytes: artifact.recordBytes,
    requests: summarize(requestCounts),
    bytes: summarize(byteCounts),
    mibMean: summarize(byteCounts).mean / 1048576,
    rounds: summarize(roundCounts),
    scenarios,
    elapsedSeconds: (performance.now() - t0) / 1000,
  };
  if (!hasArg('quiet')) console.log(JSON.stringify(summary, null, 2));
  const summaryOut = arg('summary-out', null);
  if (summaryOut) fs.writeFileSync(summaryOut, `${JSON.stringify(summary, null, 2)}\n`);
}

function runPageSweep(artifactPath, dataDir, queries, k, efSearch, pageSizes, clearCachePerQuery, checkpoints, fixedLatencies, bandwidths, parallelism) {
  const queryData = readFvecs(path.join(dataDir, 'sift_query.fvecs'), queries);
  const artifactBytes = fs.statSync(artifactPath).size;
  const pages = [];
  const t0 = performance.now();
  for (const pageSize of pageSizes) {
    const artifact = new RangeArtifact(artifactPath);
    if (queryData.dim !== artifact.dim) throw new Error(`query dim ${queryData.dim} != artifact dim ${artifact.dim}`);
    const wanted = new Set(checkpoints.filter((n) => n > 0 && n <= queryData.vectors.length));
    wanted.add(queryData.vectors.length);
    const ranges = [];
    const rows = [];
    const perQueryRounds = [];
    const requestCounts = [];
    const byteCounts = [];
    const roundCounts = [];
    let totalRangeRequests = 0;
    let totalFetchedBytes = 0;
    const pageT0 = performance.now();
    try {
      for (let i = 0; i < queryData.vectors.length; i++) {
        if (clearCachePerQuery) artifact.clearCache();
        artifact.resetStats();
        const res = lazySearch(artifact, queryData.vectors[i], k, efSearch, { pageSize });
        for (const range of artifact.currentRanges) ranges.push(range);
        perQueryRounds.push(res.rounds);
        requestCounts.push(artifact.rangeRequests);
        byteCounts.push(artifact.rangeBytes);
        roundCounts.push(res.rounds.length);
        totalRangeRequests += artifact.rangeRequests;
        totalFetchedBytes += artifact.rangeBytes;
        const seenQueries = i + 1;
        if (!clearCachePerQuery && wanted.has(seenQueries)) {
          const uniqueBytes = uniqueRangeBytes(ranges);
          rows.push({
            queries: seenQueries,
            uniqueFetchedBytes: uniqueBytes,
            uniqueFetchedMiB: uniqueBytes / 1048576,
            artifactFraction: uniqueBytes / artifactBytes,
            cacheNodes: artifact.cache.size,
            cacheNodeFraction: artifact.cache.size / artifact.count,
            cachedPages: artifact.pageCache.size,
            totalRangeRequests,
            totalFetchedBytes,
          });
        }
        if ((i + 1) % 25 === 0 || i + 1 === queryData.vectors.length) {
          console.log(`  page ${pageSize}: ${i + 1}/${queryData.vectors.length}`);
        }
      }
    } finally {
      artifact.close();
    }

    const scenarios = [];
    for (const fixedMs of fixedLatencies) {
      for (const bandwidthMiBps of bandwidths) {
        const latencies = perQueryRounds.map((rounds) => modelQueryLatency(rounds, fixedMs, bandwidthMiBps, parallelism));
        scenarios.push({
          fixedMs,
          bandwidthMiBps,
          parallelism,
          latencyMs: summarize(latencies),
        });
      }
    }
    const recordsPerPage = Math.max(1, Math.floor(pageSize / artifact.recordBytes));
    pages.push({
      requestedPageSize: pageSize,
      recordsPerPage,
      effectivePageBytes: recordsPerPage * artifact.recordBytes,
      cacheMode: clearCachePerQuery ? 'cold_per_query' : 'warm_stream',
      requests: summarize(requestCounts),
      bytes: summarize(byteCounts),
      mibMean: summarize(byteCounts).mean / 1048576,
      rounds: summarize(roundCounts),
      checkpoints: clearCachePerQuery ? [] : rows,
      scenarios,
      elapsedSeconds: (performance.now() - pageT0) / 1000,
    });
  }
  const summary = {
    artifact: artifactPath,
    queries: queryData.vectors.length,
    k,
    efSearch,
    artifactBytes,
    pageSizes,
    cacheMode: clearCachePerQuery ? 'cold_per_query' : 'warm_stream',
    pages,
    elapsedSeconds: (performance.now() - t0) / 1000,
  };
  console.log(JSON.stringify(summary, null, 2));
  const summaryOut = arg('summary-out', null);
  if (summaryOut) fs.writeFileSync(summaryOut, `${JSON.stringify(summary, null, 2)}\n`);
}

function runHotPrefixSweep(artifactPath, dataDir, queries, k, efSearch, gap, prefixSizes, checkpoints, fixedLatencies, bandwidths, parallelism) {
  const queryData = readFvecs(path.join(dataDir, 'sift_query.fvecs'), queries);
  const artifactBytes = fs.statSync(artifactPath).size;
  const prefixes = [];
  const t0 = performance.now();
  for (const prefixRecords of prefixSizes) {
    const artifact = new RangeArtifact(artifactPath);
    if (queryData.dim !== artifact.dim) throw new Error(`query dim ${queryData.dim} != artifact dim ${artifact.dim}`);
    const resident = artifact.loadOrdinalPrefix(prefixRecords);
    const wanted = new Set(checkpoints.filter((n) => n > 0 && n <= queryData.vectors.length));
    wanted.add(queryData.vectors.length);
    const ranges = [];
    const rows = [];
    const perQueryRounds = [];
    const requestCounts = [];
    const byteCounts = [];
    const roundCounts = [];
    let totalRangeRequests = 0;
    let totalFetchedBytes = 0;
    const prefixT0 = performance.now();
    try {
      for (let i = 0; i < queryData.vectors.length; i++) {
        artifact.resetStats();
        const res = lazySearch(artifact, queryData.vectors[i], k, efSearch, { gap });
        for (const range of artifact.currentRanges) ranges.push(range);
        perQueryRounds.push(res.rounds);
        requestCounts.push(artifact.rangeRequests);
        byteCounts.push(artifact.rangeBytes);
        roundCounts.push(res.rounds.length);
        totalRangeRequests += artifact.rangeRequests;
        totalFetchedBytes += artifact.rangeBytes;
        const seenQueries = i + 1;
        if (wanted.has(seenQueries)) {
          const uniqueLazyBytes = uniqueRangeBytes(ranges);
          rows.push({
            queries: seenQueries,
            residentBytes: resident.bytes,
            residentMiB: resident.bytes / 1048576,
            uniqueLazyFetchedBytes: uniqueLazyBytes,
            uniqueLazyFetchedMiB: uniqueLazyBytes / 1048576,
            totalResidentPlusLazyBytes: resident.bytes + uniqueLazyBytes,
            totalResidentPlusLazyMiB: (resident.bytes + uniqueLazyBytes) / 1048576,
            artifactFraction: (resident.bytes + uniqueLazyBytes) / artifactBytes,
            cacheNodes: artifact.cache.size,
            cacheNodeFraction: artifact.cache.size / artifact.count,
            totalRangeRequests,
            totalFetchedBytes,
          });
        }
        if ((i + 1) % 25 === 0 || i + 1 === queryData.vectors.length) {
          console.log(`  hot ${prefixRecords}: ${i + 1}/${queryData.vectors.length}`);
        }
      }
    } finally {
      artifact.close();
    }

    const scenarios = [];
    for (const fixedMs of fixedLatencies) {
      for (const bandwidthMiBps of bandwidths) {
        const latencies = perQueryRounds.map((rounds) => modelQueryLatency(rounds, fixedMs, bandwidthMiBps, parallelism));
        scenarios.push({
          fixedMs,
          bandwidthMiBps,
          parallelism,
          latencyMs: summarize(latencies),
        });
      }
    }
    prefixes.push({
      prefixRecords: resident.records,
      residentBytes: resident.bytes,
      residentMiB: resident.bytes / 1048576,
      cacheMode: 'resident_prefix_warm_stream',
      requests: summarize(requestCounts),
      bytes: summarize(byteCounts),
      mibMean: summarize(byteCounts).mean / 1048576,
      rounds: summarize(roundCounts),
      checkpoints: rows,
      scenarios,
      elapsedSeconds: (performance.now() - prefixT0) / 1000,
    });
  }
  const summary = {
    artifact: artifactPath,
    queries: queryData.vectors.length,
    k,
    efSearch,
    gap,
    artifactBytes,
    prefixSizes,
    prefixes,
    elapsedSeconds: (performance.now() - t0) / 1000,
  };
  console.log(JSON.stringify(summary, null, 2));
  const summaryOut = arg('summary-out', null);
  if (summaryOut) fs.writeFileSync(summaryOut, `${JSON.stringify(summary, null, 2)}\n`);
}

function topTouchedNodes(artifactPath, vectors, k, efSearch, gap, trainQueries) {
  const artifact = new RangeArtifact(artifactPath);
  const counts = new Uint32Array(artifact.count);
  const t0 = performance.now();
  try {
    const n = Math.min(trainQueries, vectors.length);
    for (let i = 0; i < n; i++) {
      artifact.clearCache();
      artifact.resetStats();
      lazySearch(artifact, vectors[i], k, efSearch, { gap });
      for (const id of artifact.cache.keys()) counts[id]++;
      if ((i + 1) % 25 === 0 || i + 1 === n) console.log(`  trained trace ${i + 1}/${n}`);
    }
  } finally {
    artifact.close();
  }
  const ids = Array.from({ length: counts.length }, (_, id) => id);
  ids.sort((a, b) => counts[b] - counts[a] || a - b);
  return { ids, counts, elapsedSeconds: (performance.now() - t0) / 1000 };
}

function runTraceResidentSweep(artifactPath, dataDir, queries, trainQueries, evalStart, k, efSearch, gap, residentSizes, checkpoints, fixedLatencies, bandwidths, parallelism) {
  const queryData = readFvecs(path.join(dataDir, 'sift_query.fvecs'), queries);
  const trainN = Math.min(trainQueries, queryData.vectors.length);
  const start = Math.min(evalStart, queryData.vectors.length);
  const evalVectors = queryData.vectors.slice(start);
  if (evalVectors.length === 0) throw new Error('--eval-start leaves no queries to evaluate');
  console.log(`Training trace weights on ${trainN} queries; evaluating ${evalVectors.length} queries from offset ${start}`);
  const trace = topTouchedNodes(artifactPath, queryData.vectors, k, efSearch, gap, trainN);
  const artifactBytes = fs.statSync(artifactPath).size;
  const residents = [];
  const t0 = performance.now();
  for (const residentSize of residentSizes) {
    const artifact = new RangeArtifact(artifactPath);
    if (queryData.dim !== artifact.dim) throw new Error(`query dim ${queryData.dim} != artifact dim ${artifact.dim}`);
    const ids = trace.ids.slice(0, Math.min(residentSize, artifact.count));
    const resident = artifact.loadNodeSet(ids);
    const wanted = new Set(checkpoints.filter((n) => n > 0 && n <= evalVectors.length));
    wanted.add(evalVectors.length);
    const ranges = [];
    const rows = [];
    const perQueryRounds = [];
    const requestCounts = [];
    const byteCounts = [];
    const roundCounts = [];
    let totalRangeRequests = 0;
    let totalFetchedBytes = 0;
    const residentT0 = performance.now();
    try {
      for (let i = 0; i < evalVectors.length; i++) {
        artifact.resetStats();
        const res = lazySearch(artifact, evalVectors[i], k, efSearch, { gap });
        for (const range of artifact.currentRanges) ranges.push(range);
        perQueryRounds.push(res.rounds);
        requestCounts.push(artifact.rangeRequests);
        byteCounts.push(artifact.rangeBytes);
        roundCounts.push(res.rounds.length);
        totalRangeRequests += artifact.rangeRequests;
        totalFetchedBytes += artifact.rangeBytes;
        const seenQueries = i + 1;
        if (wanted.has(seenQueries)) {
          const uniqueLazyBytes = uniqueRangeBytes(ranges);
          rows.push({
            queries: seenQueries,
            residentBytes: resident.bytes,
            residentMiB: resident.bytes / 1048576,
            uniqueLazyFetchedBytes: uniqueLazyBytes,
            uniqueLazyFetchedMiB: uniqueLazyBytes / 1048576,
            totalResidentPlusLazyBytes: resident.bytes + uniqueLazyBytes,
            totalResidentPlusLazyMiB: (resident.bytes + uniqueLazyBytes) / 1048576,
            artifactFraction: (resident.bytes + uniqueLazyBytes) / artifactBytes,
            cacheNodes: artifact.cache.size,
            cacheNodeFraction: artifact.cache.size / artifact.count,
            totalRangeRequests,
            totalFetchedBytes,
          });
        }
        if ((i + 1) % 25 === 0 || i + 1 === evalVectors.length) {
          console.log(`  trace hot ${residentSize}: ${i + 1}/${evalVectors.length}`);
        }
      }
    } finally {
      artifact.close();
    }

    const scenarios = [];
    for (const fixedMs of fixedLatencies) {
      for (const bandwidthMiBps of bandwidths) {
        const latencies = perQueryRounds.map((rounds) => modelQueryLatency(rounds, fixedMs, bandwidthMiBps, parallelism));
        scenarios.push({
          fixedMs,
          bandwidthMiBps,
          parallelism,
          latencyMs: summarize(latencies),
        });
      }
    }
    residents.push({
      residentRecords: resident.records,
      residentBytes: resident.bytes,
      residentMiB: resident.bytes / 1048576,
      cacheMode: 'trace_weighted_resident_warm_stream',
      requests: summarize(requestCounts),
      bytes: summarize(byteCounts),
      mibMean: summarize(byteCounts).mean / 1048576,
      rounds: summarize(roundCounts),
      checkpoints: rows,
      scenarios,
      elapsedSeconds: (performance.now() - residentT0) / 1000,
    });
  }
  const summary = {
    artifact: artifactPath,
    queries: queryData.vectors.length,
    trainQueries: trainN,
    evalStart: start,
    evalQueries: evalVectors.length,
    k,
    efSearch,
    gap,
    artifactBytes,
    residentSizes,
    traceTrainingSeconds: trace.elapsedSeconds,
    residents,
    elapsedSeconds: (performance.now() - t0) / 1000,
  };
  console.log(JSON.stringify(summary, null, 2));
  const summaryOut = arg('summary-out', null);
  if (summaryOut) fs.writeFileSync(summaryOut, `${JSON.stringify(summary, null, 2)}\n`);
}

function runLookaheadSweep(artifactPath, dataDir, queries, k, efSearch, gap, lookaheadValues, checkpoints, fixedLatencies, bandwidths, parallelism, clearCachePerQuery, sameRoundLookahead = false) {
  const queryData = readFvecs(path.join(dataDir, 'sift_query.fvecs'), queries);
  const artifactBytes = fs.statSync(artifactPath).size;
  const variants = [];
  const t0 = performance.now();
  for (const lookahead of lookaheadValues) {
    const artifact = new RangeArtifact(artifactPath);
    if (queryData.dim !== artifact.dim) throw new Error(`query dim ${queryData.dim} != artifact dim ${artifact.dim}`);
    const wanted = new Set(checkpoints.filter((n) => n > 0 && n <= queryData.vectors.length));
    wanted.add(queryData.vectors.length);
    const ranges = [];
    const rows = [];
    const perQueryRounds = [];
    const requestCounts = [];
    const byteCounts = [];
    const roundCounts = [];
    const missRoundCounts = [];
    let totalRangeRequests = 0;
    let totalFetchedBytes = 0;
    const variantT0 = performance.now();
    try {
      for (let i = 0; i < queryData.vectors.length; i++) {
        if (clearCachePerQuery) artifact.clearCache();
        artifact.resetStats();
        const res = lazySearch(artifact, queryData.vectors[i], k, efSearch, { gap, lookahead, sameRoundLookahead });
        for (const range of artifact.currentRanges) ranges.push(range);
        perQueryRounds.push(res.rounds);
        requestCounts.push(artifact.rangeRequests);
        byteCounts.push(artifact.rangeBytes);
        roundCounts.push(res.rounds.length);
        missRoundCounts.push(res.rounds.filter((r) => r.requests > 0).length);
        totalRangeRequests += artifact.rangeRequests;
        totalFetchedBytes += artifact.rangeBytes;
        const seenQueries = i + 1;
        if (!clearCachePerQuery && wanted.has(seenQueries)) {
          const uniqueBytes = uniqueRangeBytes(ranges);
          rows.push({
            queries: seenQueries,
            uniqueFetchedBytes: uniqueBytes,
            uniqueFetchedMiB: uniqueBytes / 1048576,
            artifactFraction: uniqueBytes / artifactBytes,
            cacheNodes: artifact.cache.size,
            cacheNodeFraction: artifact.cache.size / artifact.count,
            totalRangeRequests,
            totalFetchedBytes,
          });
        }
        if ((i + 1) % 25 === 0 || i + 1 === queryData.vectors.length) {
          console.log(`  lookahead ${lookahead}: ${i + 1}/${queryData.vectors.length}`);
        }
      }
    } finally {
      artifact.close();
    }

    const scenarios = [];
    for (const fixedMs of fixedLatencies) {
      for (const bandwidthMiBps of bandwidths) {
        const latencies = perQueryRounds.map((rounds) => modelQueryLatency(rounds, fixedMs, bandwidthMiBps, parallelism));
        scenarios.push({
          fixedMs,
          bandwidthMiBps,
          parallelism,
          latencyMs: summarize(latencies),
        });
      }
    }
    variants.push({
      lookahead,
      cacheMode: clearCachePerQuery ? 'cold_per_query' : 'warm_stream',
      requests: summarize(requestCounts),
      bytes: summarize(byteCounts),
      mibMean: summarize(byteCounts).mean / 1048576,
      rounds: summarize(roundCounts),
      missRounds: summarize(missRoundCounts),
      checkpoints: clearCachePerQuery ? [] : rows,
      scenarios,
      elapsedSeconds: (performance.now() - variantT0) / 1000,
    });
  }
  const summary = {
    artifact: artifactPath,
    queries: queryData.vectors.length,
    k,
    efSearch,
    gap,
    artifactBytes,
    lookaheadValues,
    lookaheadMode: sameRoundLookahead ? 'same_round' : 'extra_round',
    cacheMode: clearCachePerQuery ? 'cold_per_query' : 'warm_stream',
    variants,
    elapsedSeconds: (performance.now() - t0) / 1000,
  };
  console.log(JSON.stringify(summary, null, 2));
  const summaryOut = arg('summary-out', null);
  if (summaryOut) fs.writeFileSync(summaryOut, `${JSON.stringify(summary, null, 2)}\n`);
}

function runBatchSweep(artifactPath, dataDir, queries, k, efSearch, gap, batchSizes, fixedLatencies, bandwidths, parallelism, clearCachePerQuery) {
  const queryData = readFvecs(path.join(dataDir, 'sift_query.fvecs'), queries);
  const truthData = readIvecs(path.join(dataDir, 'sift_groundtruth.ivecs'), queries);
  const variants = [];
  const t0 = performance.now();
  for (const expansionBatch of batchSizes) {
    const artifact = new RangeArtifact(artifactPath);
    if (queryData.dim !== artifact.dim) throw new Error(`query dim ${queryData.dim} != artifact dim ${artifact.dim}`);
    const recalls = [];
    const perQueryRounds = [];
    const requestCounts = [];
    const byteCounts = [];
    const roundCounts = [];
    const missRoundCounts = [];
    const variantT0 = performance.now();
    try {
      for (let i = 0; i < queryData.vectors.length; i++) {
        if (clearCachePerQuery) artifact.clearCache();
        artifact.resetStats();
        const res = lazySearch(artifact, queryData.vectors[i], k, efSearch, { gap, expansionBatch });
        recalls.push(recallAt(res.results, truthData.vectors[i], k));
        perQueryRounds.push(res.rounds);
        requestCounts.push(artifact.rangeRequests);
        byteCounts.push(artifact.rangeBytes);
        roundCounts.push(res.rounds.length);
        missRoundCounts.push(res.rounds.filter((r) => r.requests > 0).length);
        if ((i + 1) % 25 === 0 || i + 1 === queryData.vectors.length) {
          console.log(`  batch ${expansionBatch}: ${i + 1}/${queryData.vectors.length}`);
        }
      }
    } finally {
      artifact.close();
    }

    const scenarios = [];
    for (const fixedMs of fixedLatencies) {
      for (const bandwidthMiBps of bandwidths) {
        const latencies = perQueryRounds.map((rounds) => modelQueryLatency(rounds, fixedMs, bandwidthMiBps, parallelism));
        scenarios.push({
          fixedMs,
          bandwidthMiBps,
          parallelism,
          latencyMs: summarize(latencies),
        });
      }
    }
    variants.push({
      expansionBatch,
      cacheMode: clearCachePerQuery ? 'cold_per_query' : 'warm_stream',
      recallAtK: summarize(recalls),
      requests: summarize(requestCounts),
      bytes: summarize(byteCounts),
      mibMean: summarize(byteCounts).mean / 1048576,
      rounds: summarize(roundCounts),
      missRounds: summarize(missRoundCounts),
      scenarios,
      elapsedSeconds: (performance.now() - variantT0) / 1000,
    });
  }
  const summary = {
    artifact: artifactPath,
    queries: queryData.vectors.length,
    k,
    efSearch,
    gap,
    batchSizes,
    cacheMode: clearCachePerQuery ? 'cold_per_query' : 'warm_stream',
    variants,
    elapsedSeconds: (performance.now() - t0) / 1000,
  };
  console.log(JSON.stringify(summary, null, 2));
  const summaryOut = arg('summary-out', null);
  if (summaryOut) fs.writeFileSync(summaryOut, `${JSON.stringify(summary, null, 2)}\n`);
}

function runGapSweep(artifactPath, dataDir, queries, k, efSearch, expansionBatch, gaps, fixedLatencies, bandwidths, parallelism, clearCachePerQuery) {
  const queryData = readFvecs(path.join(dataDir, 'sift_query.fvecs'), queries);
  const truthData = readIvecs(path.join(dataDir, 'sift_groundtruth.ivecs'), queries);
  const variants = [];
  const t0 = performance.now();
  for (const gap of gaps) {
    const artifact = new RangeArtifact(artifactPath);
    if (queryData.dim !== artifact.dim) throw new Error(`query dim ${queryData.dim} != artifact dim ${artifact.dim}`);
    const recalls = [];
    const perQueryRounds = [];
    const requestCounts = [];
    const byteCounts = [];
    const roundCounts = [];
    const missRoundCounts = [];
    const variantT0 = performance.now();
    try {
      for (let i = 0; i < queryData.vectors.length; i++) {
        if (clearCachePerQuery) artifact.clearCache();
        artifact.resetStats();
        const res = lazySearch(artifact, queryData.vectors[i], k, efSearch, { gap, expansionBatch });
        recalls.push(recallAt(res.results, truthData.vectors[i], k));
        perQueryRounds.push(res.rounds);
        requestCounts.push(artifact.rangeRequests);
        byteCounts.push(artifact.rangeBytes);
        roundCounts.push(res.rounds.length);
        missRoundCounts.push(res.rounds.filter((r) => r.requests > 0).length);
        if ((i + 1) % 25 === 0 || i + 1 === queryData.vectors.length) {
          console.log(`  gap ${gap}: ${i + 1}/${queryData.vectors.length}`);
        }
      }
    } finally {
      artifact.close();
    }

    const scenarios = [];
    for (const fixedMs of fixedLatencies) {
      for (const bandwidthMiBps of bandwidths) {
        const latencies = perQueryRounds.map((rounds) => modelQueryLatency(rounds, fixedMs, bandwidthMiBps, parallelism));
        scenarios.push({
          fixedMs,
          bandwidthMiBps,
          parallelism,
          latencyMs: summarize(latencies),
        });
      }
    }
    variants.push({
      gap,
      expansionBatch,
      cacheMode: clearCachePerQuery ? 'cold_per_query' : 'warm_stream',
      recallAtK: summarize(recalls),
      requests: summarize(requestCounts),
      bytes: summarize(byteCounts),
      mibMean: summarize(byteCounts).mean / 1048576,
      rounds: summarize(roundCounts),
      missRounds: summarize(missRoundCounts),
      scenarios,
      elapsedSeconds: (performance.now() - variantT0) / 1000,
    });
  }
  const summary = {
    artifact: artifactPath,
    queries: queryData.vectors.length,
    k,
    efSearch,
    expansionBatch,
    gaps,
    cacheMode: clearCachePerQuery ? 'cold_per_query' : 'warm_stream',
    variants,
    elapsedSeconds: (performance.now() - t0) / 1000,
  };
  console.log(JSON.stringify(summary, null, 2));
  const summaryOut = arg('summary-out', null);
  if (summaryOut) fs.writeFileSync(summaryOut, `${JSON.stringify(summary, null, 2)}\n`);
}

function runStructuralResidentSweep(artifactPath, dataDir, queries, k, efSearch, gap, includeUpper, traceResidentSizes, checkpoints, fixedLatencies, bandwidths, parallelism, trainQueries) {
  const queryData = readFvecs(path.join(dataDir, 'sift_query.fvecs'), queries);
  const artifactBytes = fs.statSync(artifactPath).size;
  const trace = traceResidentSizes.some((n) => n > 0)
    ? topTouchedNodes(artifactPath, queryData.vectors, k, efSearch, gap, Math.min(trainQueries, queryData.vectors.length))
    : null;
  const variants = [];
  const t0 = performance.now();
  for (const traceResident of traceResidentSizes) {
    const artifact = new RangeArtifact(artifactPath);
    if (queryData.dim !== artifact.dim) throw new Error(`query dim ${queryData.dim} != artifact dim ${artifact.dim}`);
    let residentRecords = 0;
    let residentBytes = 0;
    if (includeUpper) {
      const upper = artifact.loadUpperLayerNodes();
      residentRecords += upper.records;
      residentBytes += upper.bytes;
    }
    if (traceResident > 0) {
      const ids = trace.ids.slice(0, Math.min(traceResident, artifact.count));
      const hot = artifact.loadNodeSet(ids);
      residentRecords += hot.records;
      residentBytes += hot.bytes;
    }

    const wanted = new Set(checkpoints.filter((n) => n > 0 && n <= queryData.vectors.length));
    wanted.add(queryData.vectors.length);
    const ranges = [];
    const rows = [];
    const perQueryRounds = [];
    const requestCounts = [];
    const byteCounts = [];
    const roundCounts = [];
    const missRoundCounts = [];
    let totalRangeRequests = 0;
    let totalFetchedBytes = 0;
    const variantT0 = performance.now();
    try {
      for (let i = 0; i < queryData.vectors.length; i++) {
        artifact.resetStats();
        const res = lazySearch(artifact, queryData.vectors[i], k, efSearch, { gap });
        for (const range of artifact.currentRanges) ranges.push(range);
        perQueryRounds.push(res.rounds);
        requestCounts.push(artifact.rangeRequests);
        byteCounts.push(artifact.rangeBytes);
        roundCounts.push(res.rounds.length);
        missRoundCounts.push(res.rounds.filter((r) => r.requests > 0).length);
        totalRangeRequests += artifact.rangeRequests;
        totalFetchedBytes += artifact.rangeBytes;
        const seenQueries = i + 1;
        if (wanted.has(seenQueries)) {
          const uniqueLazyBytes = uniqueRangeBytes(ranges);
          rows.push({
            queries: seenQueries,
            residentBytes,
            residentMiB: residentBytes / 1048576,
            uniqueLazyFetchedBytes: uniqueLazyBytes,
            uniqueLazyFetchedMiB: uniqueLazyBytes / 1048576,
            totalResidentPlusLazyBytes: residentBytes + uniqueLazyBytes,
            totalResidentPlusLazyMiB: (residentBytes + uniqueLazyBytes) / 1048576,
            artifactFraction: (residentBytes + uniqueLazyBytes) / artifactBytes,
            cacheNodes: artifact.cache.size,
            cacheNodeFraction: artifact.cache.size / artifact.count,
            totalRangeRequests,
            totalFetchedBytes,
          });
        }
        if ((i + 1) % 25 === 0 || i + 1 === queryData.vectors.length) {
          console.log(`  structural upper=${includeUpper} trace=${traceResident}: ${i + 1}/${queryData.vectors.length}`);
        }
      }
    } finally {
      artifact.close();
    }

    const scenarios = [];
    for (const fixedMs of fixedLatencies) {
      for (const bandwidthMiBps of bandwidths) {
        const latencies = perQueryRounds.map((rounds) => modelQueryLatency(rounds, fixedMs, bandwidthMiBps, parallelism));
        scenarios.push({
          fixedMs,
          bandwidthMiBps,
          parallelism,
          latencyMs: summarize(latencies),
        });
      }
    }
    variants.push({
      includeUpper,
      traceResident,
      residentRecords,
      residentBytes,
      residentMiB: residentBytes / 1048576,
      cacheMode: 'structural_resident_warm_stream',
      requests: summarize(requestCounts),
      bytes: summarize(byteCounts),
      mibMean: summarize(byteCounts).mean / 1048576,
      rounds: summarize(roundCounts),
      missRounds: summarize(missRoundCounts),
      checkpoints: rows,
      scenarios,
      elapsedSeconds: (performance.now() - variantT0) / 1000,
    });
  }
  const summary = {
    artifact: artifactPath,
    queries: queryData.vectors.length,
    trainQueries: trace ? Math.min(trainQueries, queryData.vectors.length) : 0,
    k,
    efSearch,
    gap,
    artifactBytes,
    includeUpper,
    traceResidentSizes,
    traceTrainingSeconds: trace ? trace.elapsedSeconds : 0,
    variants,
    elapsedSeconds: (performance.now() - t0) / 1000,
  };
  console.log(JSON.stringify(summary, null, 2));
  const summaryOut = arg('summary-out', null);
  if (summaryOut) fs.writeFileSync(summaryOut, `${JSON.stringify(summary, null, 2)}\n`);
}

function addMap(map, key, amount = 1) {
  map.set(key, (map.get(key) || 0) + amount);
}

function topMap(map, limit) {
  return [...map.entries()]
    .sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .slice(0, limit)
    .map(([id, weight]) => ({ id: Number(id), weight }));
}

function missWeightedResidentIds(tracePath, limit) {
  if (!tracePath || limit <= 0) return { ids: [], source: null, available: 0 };
  const trace = JSON.parse(fs.readFileSync(tracePath, 'utf8'));
  const weights = new Map();
  const addRows = (rows, multiplier) => {
    for (const row of rows || []) addMap(weights, row.id, row.weight * multiplier);
  };
  addRows(trace.topEarlyFetchedNodes, 8);
  addRows(trace.topEarlyGateNodes, 8);
  addRows(trace.topGateNodes, 4);
  addRows(trace.topFetchedNodes, 2);
  const ids = [...weights.entries()]
    .sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .slice(0, limit)
    .map(([id]) => Number(id));
  return { ids, source: tracePath, available: weights.size };
}

function runMissRoundTrace(artifactPath, dataDir, queries, k, efSearch, gap, expansionBatch, topN, tailN, residentFromTrace, residentSize) {
  const queryData = readFvecs(path.join(dataDir, 'sift_query.fvecs'), queries);
  const truthData = readIvecs(path.join(dataDir, 'sift_groundtruth.ivecs'), queries);
  const artifact = new RangeArtifact(artifactPath);
  if (queryData.dim !== artifact.dim) throw new Error(`query dim ${queryData.dim} != artifact dim ${artifact.dim}`);
  const residentSelection = missWeightedResidentIds(residentFromTrace, residentSize);
  const resident = residentSelection.ids.length > 0
    ? artifact.loadNodeSet(residentSelection.ids)
    : { records: 0, bytes: 0 };

  const recalls = [];
  const requestCounts = [];
  const byteCounts = [];
  const roundCounts = [];
  const missRoundCounts = [];
  const missRoundOrdinals = new Map();
  const missRoundIndexes = new Map();
  const missRoundPhases = new Map();
  const fetchedNodeWeights = new Map();
  const earlyFetchedNodeWeights = new Map();
  const gateNodeWeights = new Map();
  const earlyGateNodeWeights = new Map();
  const gateBytes = new Map();
  const gateRequests = new Map();
  const tailQueries = [];
  const t0 = performance.now();

  try {
    for (let i = 0; i < queryData.vectors.length; i++) {
      artifact.resetStats();
      const res = lazySearch(artifact, queryData.vectors[i], k, efSearch, {
        gap,
        expansionBatch,
        traceRounds: true,
      });
      const missRounds = res.rounds
        .map((round, roundIndex) => ({ ...round, roundIndex }))
        .filter((round) => round.requests > 0);

      recalls.push(recallAt(res.results, truthData.vectors[i], k));
      requestCounts.push(artifact.rangeRequests);
      byteCounts.push(artifact.rangeBytes);
      roundCounts.push(res.rounds.length);
      missRoundCounts.push(missRounds.length);

      for (let missIndex = 0; missIndex < missRounds.length; missIndex++) {
        const round = missRounds[missIndex];
        addMap(missRoundOrdinals, missIndex, 1);
        addMap(missRoundIndexes, round.roundIndex, 1);
        addMap(missRoundPhases, round.phase || 'unknown', 1);

        for (const id of round.fetchedIds || []) {
          addMap(fetchedNodeWeights, id, 1);
          if (missIndex < 4) addMap(earlyFetchedNodeWeights, id, 1);
        }
        for (const id of round.sourceIds || []) {
          addMap(gateNodeWeights, id, 1);
          addMap(gateBytes, id, round.bytes);
          addMap(gateRequests, id, round.requests);
          if (missIndex < 4) addMap(earlyGateNodeWeights, id, 1);
        }
      }

      tailQueries.push({
        query: i,
        recallAtK: recalls[recalls.length - 1],
        requests: artifact.rangeRequests,
        bytes: artifact.rangeBytes,
        rounds: res.rounds.length,
        missRounds: missRounds.length,
        firstMissRound: missRounds[0]?.roundIndex ?? null,
        maxRoundRequests: Math.max(0, ...missRounds.map((round) => round.requests)),
        maxRoundBytes: Math.max(0, ...missRounds.map((round) => round.bytes)),
      });

      if ((i + 1) % 25 === 0 || i + 1 === queryData.vectors.length) {
        console.log(`  traced ${i + 1}/${queryData.vectors.length}`);
      }
    }
  } finally {
    artifact.close();
  }

  const topGates = topMap(gateNodeWeights, topN).map((row) => ({
    ...row,
    requests: gateRequests.get(row.id) || 0,
    bytes: gateBytes.get(row.id) || 0,
  }));
  const summary = {
    artifact: artifactPath,
    queries: queryData.vectors.length,
    k,
    efSearch,
    gap,
    expansionBatch,
    cacheMode: 'warm_stream',
    resident: {
      strategy: residentSelection.source ? 'miss_weighted_from_trace' : 'none',
      source: residentSelection.source,
      requestedRecords: residentSize,
      availableRecords: residentSelection.available,
      loadedRecords: resident.records,
      loadedBytes: resident.bytes,
      loadedMiB: resident.bytes / 1048576,
    },
    recallAtK: summarize(recalls),
    requests: summarize(requestCounts),
    bytes: summarize(byteCounts),
    mibMean: summarize(byteCounts).mean / 1048576,
    rounds: summarize(roundCounts),
    missRounds: summarize(missRoundCounts),
    missRoundOrdinals: topMap(missRoundOrdinals, 32),
    missRoundIndexes: topMap(missRoundIndexes, 64),
    missRoundPhases: [...missRoundPhases.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map(([phase, count]) => ({ phase, count })),
    topFetchedNodes: topMap(fetchedNodeWeights, topN),
    topEarlyFetchedNodes: topMap(earlyFetchedNodeWeights, topN),
    topGateNodes: topGates,
    topEarlyGateNodes: topMap(earlyGateNodeWeights, topN),
    tailByRequests: [...tailQueries].sort((a, b) => b.requests - a.requests).slice(0, tailN),
    tailByBytes: [...tailQueries].sort((a, b) => b.bytes - a.bytes).slice(0, tailN),
    tailByMissRounds: [...tailQueries].sort((a, b) => b.missRounds - a.missRounds || b.requests - a.requests).slice(0, tailN),
    elapsedSeconds: (performance.now() - t0) / 1000,
  };
  if (!hasArg('quiet')) console.log(JSON.stringify(summary, null, 2));
  const summaryOut = arg('summary-out', null);
  if (summaryOut) fs.writeFileSync(summaryOut, `${JSON.stringify(summary, null, 2)}\n`);
}

function makeCustomPageLayout(pageRecords) {
  return {
    pageRecords,
    pages: [],
    nodeToPage: new Map(),
    pageCache: new Set(),
  };
}

function addCustomPage(layout, ids) {
  const page = [];
  const seen = new Set();
  for (const id of ids) {
    if (layout.nodeToPage.has(id) || seen.has(id)) continue;
    seen.add(id);
    page.push(id);
    if (page.length >= layout.pageRecords) break;
  }
  if (!page.length) return false;
  const pageId = layout.pages.length;
  layout.pages.push(page);
  for (const id of page) layout.nodeToPage.set(id, pageId);
  return true;
}

function buildMissCohortPages(artifactPath, vectors, trainQueries, k, efSearch, gap, expansionBatch, pageRecords) {
  const artifact = new RangeArtifact(artifactPath);
  const layout = makeCustomPageLayout(pageRecords);
  const t0 = performance.now();
  try {
    for (let i = 0; i < trainQueries; i++) {
      artifact.resetStats();
      const res = lazySearch(artifact, vectors[i], k, efSearch, { gap, expansionBatch, traceRounds: true });
      for (const round of res.rounds) {
        if (round.phase !== 'base' || round.requests <= 0) continue;
        const ids = (round.fetchedIds || []).filter((id) => !layout.nodeToPage.has(id));
        for (let start = 0; start < ids.length; start += pageRecords) {
          addCustomPage(layout, ids.slice(start, start + pageRecords));
        }
      }
      if ((i + 1) % 25 === 0 || i + 1 === trainQueries) {
        console.log(`  trained pages ${i + 1}/${trainQueries}`);
      }
    }
  } finally {
    artifact.close();
  }
  return {
    layout,
    elapsedSeconds: (performance.now() - t0) / 1000,
  };
}

function resetCustomPageCache(layout) {
  layout.pageCache = new Set();
}

function runMissPageSim(artifactPath, dataDir, queries, trainQueries, k, efSearch, gap, expansionBatch, pageSizes, fixedLatencies, bandwidths, parallelism, clearCachePerQuery) {
  const queryData = readFvecs(path.join(dataDir, 'sift_query.fvecs'), queries);
  const truthData = readIvecs(path.join(dataDir, 'sift_groundtruth.ivecs'), queries);
  const variants = [];
  const t0 = performance.now();

  for (const pageSize of pageSizes) {
    const recordBytesProbe = new RangeArtifact(artifactPath);
    const recordBytes = recordBytesProbe.recordBytes;
    recordBytesProbe.close();
    const pageRecords = Math.max(1, Math.floor(pageSize / recordBytes));
    const trained = buildMissCohortPages(
      artifactPath,
      queryData.vectors,
      Math.min(trainQueries, queryData.vectors.length),
      k,
      efSearch,
      gap,
      expansionBatch,
      pageRecords
    );

    const artifact = new RangeArtifact(artifactPath);
    const evalStart = Math.min(trainQueries, queryData.vectors.length);
    const recalls = [];
    const perQueryRounds = [];
    const requestCounts = [];
    const byteCounts = [];
    const roundCounts = [];
    const missRoundCounts = [];
    const pageT0 = performance.now();
    try {
      for (let i = evalStart; i < queryData.vectors.length; i++) {
        if (clearCachePerQuery) {
          artifact.clearCache();
          resetCustomPageCache(trained.layout);
        }
        artifact.resetStats();
        const res = lazySearch(artifact, queryData.vectors[i], k, efSearch, {
          gap,
          expansionBatch,
          customPages: trained.layout,
        });
        recalls.push(recallAt(res.results, truthData.vectors[i], k));
        perQueryRounds.push(res.rounds);
        requestCounts.push(artifact.rangeRequests);
        byteCounts.push(artifact.rangeBytes);
        roundCounts.push(res.rounds.length);
        missRoundCounts.push(res.rounds.filter((round) => round.requests > 0).length);
        const done = i - evalStart + 1;
        const total = queryData.vectors.length - evalStart;
        if (done % 25 === 0 || done === total) {
          console.log(`  page ${pageSize}: ${done}/${total}`);
        }
      }
    } finally {
      artifact.close();
    }

    const scenarios = [];
    for (const fixedMs of fixedLatencies) {
      for (const bandwidthMiBps of bandwidths) {
        const latencies = perQueryRounds.map((rounds) => modelQueryLatency(rounds, fixedMs, bandwidthMiBps, parallelism));
        scenarios.push({
          fixedMs,
          bandwidthMiBps,
          parallelism,
          latencyMs: summarize(latencies),
        });
      }
    }
    variants.push({
      requestedPageSize: pageSize,
      pageRecords,
      effectivePageBytes: pageRecords * recordBytes,
      pages: trained.layout.pages.length,
      pagedNodes: trained.layout.nodeToPage.size,
      pageBytes: trained.layout.nodeToPage.size * recordBytes,
      pageMiB: trained.layout.nodeToPage.size * recordBytes / 1048576,
      trainQueries: evalStart,
      evalQueries: recalls.length,
      cacheMode: clearCachePerQuery ? 'cold_per_query' : 'warm_stream',
      recallAtK: summarize(recalls),
      requests: summarize(requestCounts),
      bytes: summarize(byteCounts),
      mibMean: summarize(byteCounts).mean / 1048576,
      rounds: summarize(roundCounts),
      missRounds: summarize(missRoundCounts),
      scenarios,
      trainingSeconds: trained.elapsedSeconds,
      elapsedSeconds: (performance.now() - pageT0) / 1000,
    });
  }

  const summary = {
    artifact: artifactPath,
    queries: queryData.vectors.length,
    k,
    efSearch,
    gap,
    expansionBatch,
    pageSizes,
    trainQueries,
    cacheMode: clearCachePerQuery ? 'cold_per_query' : 'warm_stream',
    variants,
    elapsedSeconds: (performance.now() - t0) / 1000,
  };
  if (!hasArg('quiet')) console.log(JSON.stringify(summary, null, 2));
  const summaryOut = arg('summary-out', null);
  if (summaryOut) fs.writeFileSync(summaryOut, `${JSON.stringify(summary, null, 2)}\n`);
}

function overlapAtK(a, b, k) {
  const seen = new Set(a.slice(0, k));
  let hits = 0;
  for (const id of b.slice(0, k)) {
    if (seen.has(id)) hits++;
  }
  return hits / k;
}

function runParity(snapshotPath, artifactPath, dataDir, queries, k, efSearch, gap) {
  console.log(`Parsing ${snapshotPath}...`);
  const snapshot = fs.readFileSync(snapshotPath);
  const index = parseSnapshot(new Uint8Array(snapshot.buffer, snapshot.byteOffset, snapshot.byteLength));
  const artifact = new RangeArtifact(artifactPath);
  const queryData = readFvecs(path.join(dataDir, 'sift_query.fvecs'), queries);
  if (queryData.dim !== index.dim) throw new Error(`query dim ${queryData.dim} != snapshot dim ${index.dim}`);
  if (artifact.dim !== index.dim || artifact.count !== index.count || artifact.entryPoint !== index.entryPoint) {
    throw new Error('artifact metadata does not match snapshot');
  }

  const exact = [];
  const overlap = [];
  const requestCounts = [];
  const byteCounts = [];
  const roundCounts = [];
  const examples = [];
  const t0 = performance.now();
  try {
    for (let i = 0; i < queryData.vectors.length; i++) {
      artifact.clearCache();
      artifact.resetStats();
      const expected = snapshotSearch(index, queryData.vectors[i], k, efSearch);
      const actual = lazySearch(artifact, queryData.vectors[i], k, efSearch, { gap });
      const same = expected.length === actual.results.length && expected.every((id, j) => id === actual.results[j]);
      exact.push(same ? 1 : 0);
      overlap.push(overlapAtK(expected, actual.results, k));
      requestCounts.push(artifact.rangeRequests);
      byteCounts.push(artifact.rangeBytes);
      roundCounts.push(actual.rounds.length);
      if (!same && examples.length < 5) {
        examples.push({ query: i, expected, actual: actual.results });
      }
      if ((i + 1) % 10 === 0 || i + 1 === queryData.vectors.length) console.log(`  checked ${i + 1}/${queryData.vectors.length}`);
    }
  } finally {
    artifact.close();
  }

  const exactMatches = exact.reduce((a, b) => a + b, 0);
  const summary = {
    snapshot: snapshotPath,
    artifact: artifactPath,
    queries: queryData.vectors.length,
    k,
    efSearch,
    gap,
    exactMatches,
    exactMatchRate: exactMatches / queryData.vectors.length,
    overlapAtK: summarize(overlap),
    requests: summarize(requestCounts),
    bytes: summarize(byteCounts),
    mibMean: summarize(byteCounts).mean / 1048576,
    rounds: summarize(roundCounts),
    mismatchExamples: examples,
    elapsedSeconds: (performance.now() - t0) / 1000,
  };
  console.log(JSON.stringify(summary, null, 2));
  const summaryOut = arg('summary-out', null);
  if (summaryOut) fs.writeFileSync(summaryOut, `${JSON.stringify(summary, null, 2)}\n`);
}

function summarize(values) {
  const xs = [...values].sort((a, b) => a - b);
  if (xs.length === 0) return { min: null, mean: null, p50: null, p95: null, p99: null, max: null };
  const pick = (p) => xs[Math.min(xs.length - 1, Math.floor(xs.length * p))];
  const sum = xs.reduce((a, b) => a + b, 0);
  return { min: xs[0], mean: sum / xs.length, p50: pick(0.5), p95: pick(0.95), p99: pick(0.99), max: xs[xs.length - 1] };
}

function parseCheckpointArg(name, fallback) {
  const raw = arg(name, null);
  if (raw == null) return fallback;
  const values = raw.split(',').map((x) => Number(x.trim())).filter((x) => Number.isInteger(x) && x > 0);
  if (values.length === 0) throw new Error(`--${name} must contain at least one positive integer`);
  return Array.from(new Set(values)).sort((a, b) => a - b);
}

function main() {
  if (hasArg('export-split')) {
    const snapshot = arg('snapshot');
    const partition = arg('metis-partition');
    const out = arg('out');
    if (!snapshot || !partition || !out) throw new Error('--export-split requires --snapshot, --metis-partition, and --out');
    exportSplitRangeArtifact(path.resolve(snapshot), path.resolve(partition), path.resolve(out));
    return;
  }
  if (hasArg('export')) {
    const snapshot = arg('snapshot');
    const partition = arg('metis-partition');
    const out = arg('out');
    if (!snapshot || !partition || !out) throw new Error('--export requires --snapshot, --metis-partition, and --out');
    exportRangeArtifact(path.resolve(snapshot), path.resolve(partition), path.resolve(out));
    return;
  }
  if (hasArg('search')) {
    const artifact = arg('artifact');
    if (!artifact) throw new Error('--search requires --artifact');
    runLocalSearch(
      path.resolve(artifact),
      path.resolve(arg('data-dir', path.join(__dirname, '..', 'sift'))),
      parseIntArg('queries', 100),
      parseIntArg('k', 10),
      parseIntArg('ef-search', OPERATIONAL_EF_SEARCH),
      parseIntArg('gap', 0),
      hasArg('clear-cache-per-query')
    );
    return;
  }
  if (hasArg('parity')) {
    const snapshot = arg('snapshot');
    const artifact = arg('artifact');
    if (!snapshot || !artifact) throw new Error('--parity requires --snapshot and --artifact');
    runParity(
      path.resolve(snapshot),
      path.resolve(artifact),
      path.resolve(arg('data-dir', path.join(__dirname, '..', 'sift'))),
      parseIntArg('queries', 25),
      parseIntArg('k', 10),
      parseIntArg('ef-search', OPERATIONAL_EF_SEARCH),
      parseIntArg('gap', 0)
    );
    return;
  }
  if (hasArg('working-set')) {
    const artifact = arg('artifact');
    if (!artifact) throw new Error('--working-set requires --artifact');
    runWorkingSet(
      path.resolve(artifact),
      path.resolve(arg('data-dir', path.join(__dirname, '..', 'sift'))),
      parseIntArg('queries', 1000),
      parseIntArg('k', 10),
      parseIntArg('ef-search', OPERATIONAL_EF_SEARCH),
      parseIntArg('gap', 0),
      parseCheckpointArg('checkpoints', [1, 10, 100, 1000])
    );
    return;
  }
  if (hasArg('latency-model')) {
    const artifact = arg('artifact');
    if (!artifact) throw new Error('--latency-model requires --artifact');
    runLatencyModel(
      path.resolve(artifact),
      path.resolve(arg('data-dir', path.join(__dirname, '..', 'sift'))),
      parseIntArg('queries', 1000),
      parseIntArg('k', 10),
      parseIntArg('ef-search', OPERATIONAL_EF_SEARCH),
      parseIntArg('gap', 0),
      hasArg('clear-cache-per-query'),
      parseNumberListArg('fixed-ms', [2, 5, 10, 20]),
      parseNumberListArg('bandwidth-mibps', [25, 100, 500]),
      parseIntArg('parallelism', 32)
    );
    return;
  }
  if (hasArg('page-sweep')) {
    const artifact = arg('artifact');
    if (!artifact) throw new Error('--page-sweep requires --artifact');
    runPageSweep(
      path.resolve(artifact),
      path.resolve(arg('data-dir', path.join(__dirname, '..', 'sift'))),
      parseIntArg('queries', 1000),
      parseIntArg('k', 10),
      parseIntArg('ef-search', OPERATIONAL_EF_SEARCH),
      parseCheckpointArg('page-sizes', [4096, 16384, 65536]),
      hasArg('clear-cache-per-query'),
      parseCheckpointArg('checkpoints', [1, 10, 100, 1000]),
      parseNumberListArg('fixed-ms', [2, 5, 10, 20]),
      parseNumberListArg('bandwidth-mibps', [25, 100, 500]),
      parseIntArg('parallelism', 32)
    );
    return;
  }
  if (hasArg('hot-prefix-sweep')) {
    const artifact = arg('artifact');
    if (!artifact) throw new Error('--hot-prefix-sweep requires --artifact');
    runHotPrefixSweep(
      path.resolve(artifact),
      path.resolve(arg('data-dir', path.join(__dirname, '..', 'sift'))),
      parseIntArg('queries', 1000),
      parseIntArg('k', 10),
      parseIntArg('ef-search', OPERATIONAL_EF_SEARCH),
      parseIntArg('gap', 0),
      parseCheckpointArg('prefix-sizes', [1000, 4000, 16000, 64000]),
      parseCheckpointArg('checkpoints', [1, 10, 100, 1000]),
      parseNumberListArg('fixed-ms', [2, 5, 10]),
      parseNumberListArg('bandwidth-mibps', [100]),
      parseIntArg('parallelism', 32)
    );
    return;
  }
  if (hasArg('trace-resident-sweep')) {
    const artifact = arg('artifact');
    if (!artifact) throw new Error('--trace-resident-sweep requires --artifact');
    runTraceResidentSweep(
      path.resolve(artifact),
      path.resolve(arg('data-dir', path.join(__dirname, '..', 'sift'))),
      parseIntArg('queries', 1000),
      parseIntArg('train-queries', 500),
      parseIntArg('eval-start', 500),
      parseIntArg('k', 10),
      parseIntArg('ef-search', OPERATIONAL_EF_SEARCH),
      parseIntArg('gap', 0),
      parseCheckpointArg('resident-sizes', [1000, 4000, 16000, 64000]),
      parseCheckpointArg('checkpoints', [1, 10, 100, 500]),
      parseNumberListArg('fixed-ms', [2, 5, 10]),
      parseNumberListArg('bandwidth-mibps', [100]),
      parseIntArg('parallelism', 32)
    );
    return;
  }
  if (hasArg('lookahead-sweep')) {
    const artifact = arg('artifact');
    if (!artifact) throw new Error('--lookahead-sweep requires --artifact');
    runLookaheadSweep(
      path.resolve(artifact),
      path.resolve(arg('data-dir', path.join(__dirname, '..', 'sift'))),
      parseIntArg('queries', 1000),
      parseIntArg('k', 10),
      parseIntArg('ef-search', OPERATIONAL_EF_SEARCH),
      parseIntArg('gap', 0),
      parseNumberListArg('lookahead-values', [0, 2, 4, 8]).map((x) => {
        if (!Number.isInteger(x)) throw new Error('--lookahead-values must contain integers');
        return x;
      }),
      parseCheckpointArg('checkpoints', [1, 10, 100, 1000]),
      parseNumberListArg('fixed-ms', [2, 5, 10]),
      parseNumberListArg('bandwidth-mibps', [100]),
      parseIntArg('parallelism', 32),
      hasArg('clear-cache-per-query'),
      hasArg('same-round-lookahead')
    );
    return;
  }
  if (hasArg('batch-sweep')) {
    const artifact = arg('artifact');
    if (!artifact) throw new Error('--batch-sweep requires --artifact');
    runBatchSweep(
      path.resolve(artifact),
      path.resolve(arg('data-dir', path.join(__dirname, '..', 'sift'))),
      parseIntArg('queries', 1000),
      parseIntArg('k', 10),
      parseIntArg('ef-search', OPERATIONAL_EF_SEARCH),
      parseIntArg('gap', 0),
      parseNumberListArg('batch-sizes', [OPERATIONAL_EXPANSION_BATCH]).map((x) => {
        if (!Number.isInteger(x) || x <= 0) throw new Error('--batch-sizes must contain positive integers');
        return x;
      }),
      parseNumberListArg('fixed-ms', [2, 5, 10]),
      parseNumberListArg('bandwidth-mibps', [100]),
      parseIntArg('parallelism', 32),
      hasArg('clear-cache-per-query')
    );
    return;
  }
  if (hasArg('gap-sweep')) {
    const artifact = arg('artifact');
    if (!artifact) throw new Error('--gap-sweep requires --artifact');
    runGapSweep(
      path.resolve(artifact),
      path.resolve(arg('data-dir', path.join(__dirname, '..', 'sift'))),
      parseIntArg('queries', 1000),
      parseIntArg('k', 10),
      parseIntArg('ef-search', OPERATIONAL_EF_SEARCH),
      parseIntArg('expansion-batch', OPERATIONAL_EXPANSION_BATCH),
      parseNumberListArg('gaps', [0, 1, 2, 4, 8, 16, 32, 64]).map((x) => {
        if (!Number.isInteger(x) || x < 0) throw new Error('--gaps must contain non-negative integers');
        return x;
      }),
      parseNumberListArg('fixed-ms', [1, 10, 30]),
      parseNumberListArg('bandwidth-mibps', [100]),
      parseIntArg('parallelism', 32),
      hasArg('clear-cache-per-query')
    );
    return;
  }
  if (hasArg('structural-resident-sweep')) {
    const artifact = arg('artifact');
    if (!artifact) throw new Error('--structural-resident-sweep requires --artifact');
    runStructuralResidentSweep(
      path.resolve(artifact),
      path.resolve(arg('data-dir', path.join(__dirname, '..', 'sift'))),
      parseIntArg('queries', 1000),
      parseIntArg('k', 10),
      parseIntArg('ef-search', OPERATIONAL_EF_SEARCH),
      parseIntArg('gap', 0),
      !hasArg('no-upper'),
      parseNumberListArg('trace-resident-sizes', [0, 4000, 16000]).map((x) => {
        if (!Number.isInteger(x)) throw new Error('--trace-resident-sizes must contain integers');
        return x;
      }),
      parseCheckpointArg('checkpoints', [1, 10, 100, 1000]),
      parseNumberListArg('fixed-ms', [2, 5, 10]),
      parseNumberListArg('bandwidth-mibps', [100]),
      parseIntArg('parallelism', 32),
      parseIntArg('train-queries', 500)
    );
    return;
  }
  if (hasArg('miss-round-trace')) {
    const artifact = arg('artifact');
    if (!artifact) throw new Error('--miss-round-trace requires --artifact');
    runMissRoundTrace(
      path.resolve(artifact),
      path.resolve(arg('data-dir', path.join(__dirname, '..', 'sift'))),
      parseIntArg('queries', 1000),
      parseIntArg('k', 10),
      parseIntArg('ef-search', OPERATIONAL_EF_SEARCH),
      parseIntArg('gap', 65536),
      parseIntArg('expansion-batch', OPERATIONAL_EXPANSION_BATCH),
      parseIntArg('top-n', 100),
      parseIntArg('tail-n', 20),
      arg('resident-from-trace', null),
      parseIntArg('resident-size', 0)
    );
    return;
  }
  if (hasArg('miss-page-sim')) {
    const artifact = arg('artifact');
    if (!artifact) throw new Error('--miss-page-sim requires --artifact');
    runMissPageSim(
      path.resolve(artifact),
      path.resolve(arg('data-dir', path.join(__dirname, '..', 'sift'))),
      parseIntArg('queries', 1000),
      parseIntArg('train-queries', 500),
      parseIntArg('k', 10),
      parseIntArg('ef-search', OPERATIONAL_EF_SEARCH),
      parseIntArg('gap', 65536),
      parseIntArg('expansion-batch', OPERATIONAL_EXPANSION_BATCH),
      parseNumberListArg('page-sizes', [65536, 131072]).map((x) => {
        if (!Number.isInteger(x) || x <= 0) throw new Error('--page-sizes must contain positive integers');
        return x;
      }),
      parseNumberListArg('fixed-ms', [1, 10, 30]),
      parseNumberListArg('bandwidth-mibps', [100]),
      parseIntArg('parallelism', 6),
      hasArg('clear-cache-per-query')
    );
    return;
  }
  if (hasArg('inspect')) {
    const artifact = arg('artifact');
    if (!artifact) throw new Error('--inspect requires --artifact');
    const manifest = inspectArtifact(path.resolve(artifact));
    const manifestOut = arg('manifest-out', null);
    if (manifestOut) {
      fs.writeFileSync(manifestOut, `${JSON.stringify(manifest, null, 2)}\n`);
      console.log(`Wrote ${manifestOut}`);
    } else {
      console.log(JSON.stringify(manifest, null, 2));
    }
    return;
  }
  throw new Error('use --export, --export-split, --search, --parity, --working-set, --latency-model, --page-sweep, --hot-prefix-sweep, --trace-resident-sweep, --lookahead-sweep, --batch-sweep, --gap-sweep, --structural-resident-sweep, --miss-round-trace, --miss-page-sim, or --inspect');
}

main();
