#!/usr/bin/env node
'use strict';

/**
 * Offline SIFT layout-touch analyzer.
 *
 * Builds or loads a Pancake SIFT snapshot, replays HNSW search from the raw
 * snapshot graph, records node ids touched by each query, applies a candidate
 * node-layout permutation, and reports unique layout blocks touched per query.
 *
 * The block model is intentionally hypothetical: after permutation each node is
 * treated as one contiguous record containing vector payload, scale/offset, base
 * edge storage, and per-node size metadata. This lets us compare candidate node
 * layouts without changing Pancake's current struct-of-arrays engine layout.
 *
 * Example:
 *   node benchmarks/sift_layout_trace.js \
 *     --snapshot /tmp/sift1m-u8.pnck --queries 100 --ef-search 100 \
 *     --trace-out /tmp/sift-trace.jsonl
 *
 *   node benchmarks/sift_layout_trace.js \
 *     --snapshot /tmp/sift1m-u8.pnck --queries 100 \
 *     --ef-search-values 10,20,40,80,100 \
 *     --trace-train-queries 50 \
 *     --hilbert-dims 0,1 \
 *     --metis-partition /tmp/sift1m.graph.part.5036 \
 *     --hot-prefix-sizes 1000,4000,16000,64000 \
 *     --summary-out /tmp/sift-layout-by-beam.json
 *
 *   node benchmarks/sift_layout_trace.js \
 *     --build-snapshot /tmp/sift1m-u8.pnck --count 1000000 --queries 100
 */

const fs = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');

const Pancake = require('../pancake.js');

const PANCAKE_MAGIC = 0x504E434B;
const V1_ENVELOPE_HEADER_SIZE = 24;
const V2_ENVELOPE_HEADER_SIZE = 20;
const V3_ENVELOPE_HEADER_SIZE = 32;
const MAPPING_ENTRY_SIZE = 8;
const UINT8_HNSW_MAGIC_V1 = 0x49384831;
const FLOAT_HNSW_MAGIC_V1 = 0x464C4831;

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

function parseIntListArg(name, fallback) {
  const raw = arg(name, null);
  if (raw == null) return fallback;
  const values = raw.split(',')
    .map((x) => Number(x.trim()))
    .filter((x) => Number.isInteger(x) && x >= 0);
  if (values.length === 0) throw new Error(`--${name} must contain at least one non-negative integer`);
  return Array.from(new Set(values)).sort((a, b) => a - b);
}

function parseBoolArg(name) {
  return process.argv.includes(`--${name}`);
}

function readFvecs(file, limit = Infinity) {
  const fd = fs.openSync(file, 'r');
  const vectors = [];
  let dim = null;
  let pos = 0;
  const header = Buffer.allocUnsafe(4);
  try {
    while (vectors.length < limit) {
      const got = fs.readSync(fd, header, 0, 4, pos);
      if (got === 0) break;
      if (got !== 4) throw new Error(`truncated fvecs header in ${file}`);
      pos += 4;
      const d = header.readInt32LE(0);
      if (d <= 0) throw new Error(`invalid fvecs dimension ${d} in ${file}`);
      if (dim == null) dim = d;
      if (d !== dim) throw new Error(`dimension changed in ${file}: ${d} != ${dim}`);
      const bytes = d * 4;
      const buf = Buffer.allocUnsafe(bytes);
      const read = fs.readSync(fd, buf, 0, bytes, pos);
      if (read !== bytes) throw new Error(`truncated fvecs record in ${file}`);
      pos += bytes;
      const v = new Float32Array(d);
      for (let i = 0; i < d; i++) v[i] = buf.readFloatLE(i * 4);
      vectors.push(v);
    }
  } finally {
    fs.closeSync(fd);
  }
  return { vectors, dim };
}

function unwrapSnapshot(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.byteLength < 4 || view.getUint32(0, true) !== PANCAKE_MAGIC) return bytes;
  if (bytes.byteLength < 8) throw new Error('truncated Pancake snapshot envelope');
  const version = view.getUint32(4, true);
  if (version === 1) return bytes.subarray(V1_ENVELOPE_HEADER_SIZE);
  if (version === 2) return bytes.subarray(V2_ENVELOPE_HEADER_SIZE);
  if (version === 3) {
    if (bytes.byteLength < V3_ENVELOPE_HEADER_SIZE) throw new Error('truncated v3 Pancake snapshot envelope');
    const mappingCount = view.getUint32(24, true);
    const rawSize = view.getUint32(28, true);
    const rawOffset = V3_ENVELOPE_HEADER_SIZE + mappingCount * MAPPING_ENTRY_SIZE;
    if (rawOffset > bytes.byteLength || rawSize > bytes.byteLength - rawOffset) {
      throw new Error('truncated v3 Pancake snapshot payload');
    }
    return bytes.subarray(rawOffset, rawOffset + rawSize);
  }
  throw new Error(`unsupported Pancake snapshot envelope version ${version}`);
}

function parseSnapshot(bytes) {
  const raw = unwrapSnapshot(bytes);
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  let off = 0;
  const u32 = () => {
    if (off + 4 > raw.byteLength) throw new Error('truncated snapshot');
    const v = view.getUint32(off, true);
    off += 4;
    return v;
  };
  const f32 = () => {
    if (off + 4 > raw.byteLength) throw new Error('truncated snapshot');
    const v = view.getFloat32(off, true);
    off += 4;
    return v;
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
  if (metric !== 0) throw new Error('this analyzer currently expects SIFT/L2 snapshots');

  if (magic === UINT8_HNSW_MAGIC_V1) {
    if (version < 1) throw new Error(`unsupported uint8 snapshot version ${version}`);
    const scales = new Float32Array(count);
    const offsets = new Float32Array(count);
    for (let i = 0; i < count; i++) scales[i] = f32();
    for (let i = 0; i < count; i++) offsets[i] = f32();
    const qdata = raw.subarray(off, off + count * dim);
    if (qdata.byteLength !== count * dim) throw new Error('truncated qdata');
    off += qdata.byteLength;
    const graph = readGraph({ view, raw, off, count, maxLevel, M0, edgeBytes: 8, edgeHasDist: true });
    return { kind: 'u8', dim, count, entryPoint, maxLevel, M, M0, metric, efConstruction, scales, offsets, qdata, ...graph };
  }

  if (magic === FLOAT_HNSW_MAGIC_V1) {
    const vecBytes = count * dim * 4;
    if (off + vecBytes > raw.byteLength) throw new Error('truncated f32 vectors');
    const vectors = new Float32Array(raw.buffer, raw.byteOffset + off, count * dim);
    off += vecBytes;
    const graph = readGraph({ view, raw, off, count, maxLevel, M0, edgeBytes: 4, edgeHasDist: false });
    return { kind: 'f32', dim, count, entryPoint, maxLevel, M, M0, metric, efConstruction, vectors, ...graph };
  }

  throw new Error(`unsupported raw snapshot magic 0x${magic.toString(16)}`);
}

function readGraph({ view, raw, off, count, maxLevel, M0, edgeBytes, edgeHasDist }) {
  const levels = new Int32Array(count);
  const base = new Array(count);
  const upper = Array.from({ length: count }, () => []);
  const u32 = () => {
    if (off + 4 > raw.byteLength) throw new Error('truncated graph');
    const v = view.getUint32(off, true);
    off += 4;
    return v;
  };
  for (let i = 0; i < count; i++) {
    const level = u32();
    if (level > maxLevel) throw new Error(`node ${i} level ${level} exceeds maxLevel ${maxLevel}`);
    levels[i] = level;
    for (let l = 0; l <= level; l++) {
      const size = u32();
      const out = new Uint32Array(size);
      for (let e = 0; e < size; e++) {
        out[e] = u32();
        if (out[e] >= count) throw new Error(`edge points outside graph: ${out[e]}`);
        if (edgeHasDist) off += 4;
      }
      if (edgeBytes === 8 && !edgeHasDist) off += size * 4;
      if (l === 0) base[i] = out;
      else upper[i][l - 1] = out;
    }
    if (!base[i]) base[i] = new Uint32Array(0);
  }
  return { levels, base, upper, graphBytesRead: off };
}

function l2U8(index, query, id) {
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

function l2F32(index, query, id) {
  const off = id * index.dim;
  let sum = 0;
  for (let d = 0; d < index.dim; d++) {
    const diff = query[d] - index.vectors[off + d];
    sum += diff * diff;
  }
  return sum;
}

function distance(index, query, id) {
  return index.kind === 'u8' ? l2U8(index, query, id) : l2F32(index, query, id);
}

class MinHeap {
  constructor(compare) {
    this.a = [];
    this.compare = compare;
  }
  get size() { return this.a.length; }
  push(x) {
    const a = this.a;
    a.push(x);
    this._siftUp(a.length - 1);
  }
  _siftUp(i) {
    const a = this.a;
    const x = a[i];
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.compare(a[p], x) <= 0) break;
      a[i] = a[p];
      i = p;
    }
    a[i] = x;
  }
  pop() {
    const a = this.a;
    if (a.length === 0) return undefined;
    const top = a[0];
    const x = a.pop();
    if (a.length > 0) {
      let i = 0;
      while (true) {
        let c = i * 2 + 1;
        if (c >= a.length) break;
        if (c + 1 < a.length && this.compare(a[c + 1], a[c]) < 0) c++;
        if (this.compare(a[c], x) >= 0) break;
        a[i] = a[c];
        i = c;
      }
      a[i] = x;
    }
    return top;
  }
  peek() { return this.a[0]; }
}

class MaxHeap extends MinHeap {
  constructor() {
    super((a, b) => b[0] - a[0]);
  }
}

function replaySearch(index, query, k, efSearch) {
  const touched = [];
  const touchedSet = new Set();
  const rounds = [];
  const touch = (id) => {
    touched.push(id);
    touchedSet.add(id);
  };

  let curr = index.entryPoint;
  let currDist = distance(index, query, curr);
  touch(curr);

  for (let level = index.maxLevel; level > 0; level--) {
    let changed = true;
    while (changed) {
      changed = false;
      const edges = index.upper[curr]?.[level - 1] || [];
      const round = [];
      for (let i = 0; i < edges.length; i++) {
        const neighbor = edges[i];
        const d = distance(index, query, neighbor);
        touch(neighbor);
        round.push(neighbor);
        if (d < currDist) {
          curr = neighbor;
          currDist = d;
          changed = true;
        }
      }
      if (round.length > 0) rounds.push(round);
    }
  }

  const ef = Math.max(efSearch, k);
  const visited = new Uint8Array(index.count);
  const candidates = new MinHeap((a, b) => a[0] - b[0]);
  const results = new MaxHeap();
  const d0 = distance(index, query, curr);
  touch(curr);
  candidates.push([d0, curr]);
  results.push([d0, curr]);
  visited[curr] = 1;

  while (candidates.size > 0) {
    const [candDist, candId] = candidates.pop();
    const worst = results.peek();
    if (results.size >= ef && worst && candDist > worst[0]) break;
    const edges = index.base[candId];
    const round = [];
    for (let i = 0; i < edges.length; i++) {
      const neighbor = edges[i];
      if (visited[neighbor]) continue;
      visited[neighbor] = 1;
      const nd = distance(index, query, neighbor);
      touch(neighbor);
      round.push(neighbor);
      const currentWorst = results.peek();
      if (results.size < ef || nd < currentWorst[0]) {
        candidates.push([nd, neighbor]);
        results.push([nd, neighbor]);
        if (results.size > ef) results.pop();
      }
    }
    if (round.length > 0) rounds.push(round);
  }

  return { touched, uniqueTouched: Array.from(touchedSet), rounds };
}

function bfsPermutation(index) {
  const perm = new Uint32Array(index.count);
  perm.fill(0xFFFFFFFF);
  const queue = [index.entryPoint];
  let read = 0;
  let write = 0;
  perm[index.entryPoint] = write++;
  while (read < queue.length) {
    const id = queue[read++];
    const edges = index.base[id];
    for (let i = 0; i < edges.length; i++) {
      const n = edges[i];
      if (perm[n] !== 0xFFFFFFFF) continue;
      perm[n] = write++;
      queue.push(n);
    }
  }
  for (let id = 0; id < index.count; id++) {
    if (perm[id] === 0xFFFFFFFF) perm[id] = write++;
  }
  return perm;
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
      const neighbors = Array.from(index.base[id]).filter((n) => !visited[n]);
      neighbors.sort((a, b) => degrees[a] - degrees[b] || a - b);
      for (const n of neighbors) {
        if (visited[n]) continue;
        visited[n] = 1;
        queue[write++] = n;
      }
    }
  }

  const perm = new Uint32Array(count);
  for (let pos = 0; pos < orderLen; pos++) {
    perm[order[orderLen - 1 - pos]] = pos;
  }
  return perm;
}

function identityPermutation(count) {
  const p = new Uint32Array(count);
  for (let i = 0; i < count; i++) p[i] = i;
  return p;
}

function traceWeightedPermutation(touchCounts, fallbackPermutation) {
  const ids = Array.from({ length: touchCounts.length }, (_, id) => id);
  ids.sort((a, b) => touchCounts[b] - touchCounts[a] || fallbackPermutation[a] - fallbackPermutation[b]);
  const perm = new Uint32Array(touchCounts.length);
  for (let pos = 0; pos < ids.length; pos++) perm[ids[pos]] = pos;
  return perm;
}

function partitionPermutation(partition, fallbackPermutation) {
  const ids = Array.from({ length: partition.length }, (_, id) => id);
  ids.sort((a, b) => partition[a] - partition[b] || fallbackPermutation[a] - fallbackPermutation[b]);
  const perm = new Uint32Array(partition.length);
  for (let pos = 0; pos < ids.length; pos++) perm[ids[pos]] = pos;
  return perm;
}

function vectorValue(index, id, dim) {
  if (index.kind === 'u8') {
    return index.offsets[id] + index.scales[id] * index.qdata[id * index.dim + dim];
  }
  return index.vectors[id * index.dim + dim];
}

function hilbert2dIndex(x, y, bits) {
  let d = 0;
  for (let s = 1 << (bits - 1); s > 0; s >>= 1) {
    const rx = (x & s) > 0 ? 1 : 0;
    const ry = (y & s) > 0 ? 1 : 0;
    d += s * s * ((3 * rx) ^ ry);
    if (ry === 0) {
      if (rx === 1) {
        x = s - 1 - x;
        y = s - 1 - y;
      }
      const t = x;
      x = y;
      y = t;
    }
  }
  return d;
}

function hilbertPermutation(index, dims, bits, fallbackPermutation) {
  if (dims.length !== 2) throw new Error('--hilbert-dims currently expects exactly two dimensions');
  for (const dim of dims) {
    if (dim >= index.dim) throw new Error(`Hilbert dimension ${dim} is outside index dim ${index.dim}`);
  }
  if (bits < 1 || bits > 16) throw new Error('--hilbert-bits must be between 1 and 16');

  const mins = [Infinity, Infinity];
  const maxs = [-Infinity, -Infinity];
  for (let id = 0; id < index.count; id++) {
    for (let j = 0; j < 2; j++) {
      const v = vectorValue(index, id, dims[j]);
      if (v < mins[j]) mins[j] = v;
      if (v > maxs[j]) maxs[j] = v;
    }
  }

  const maxCoord = (1 << bits) - 1;
  const keys = new Float64Array(index.count);
  for (let id = 0; id < index.count; id++) {
    keys[id] = hilbertKeyForNode(index, id, dims, mins, maxs, maxCoord, bits);
  }
  const ids = Array.from({ length: index.count }, (_, id) => id);
  ids.sort((a, b) => {
    return keys[a] - keys[b] || fallbackPermutation[a] - fallbackPermutation[b];
  });

  const perm = new Uint32Array(index.count);
  for (let pos = 0; pos < ids.length; pos++) perm[ids[pos]] = pos;
  return perm;
}

function hilbertKeyForNode(index, id, dims, mins, maxs, maxCoord, bits) {
  const coords = [0, 0];
  for (let j = 0; j < 2; j++) {
    const span = maxs[j] - mins[j];
    const raw = span > 0 ? (vectorValue(index, id, dims[j]) - mins[j]) / span : 0;
    coords[j] = Math.max(0, Math.min(maxCoord, Math.round(raw * maxCoord)));
  }
  return hilbert2dIndex(coords[0], coords[1], bits);
}

function hotPrefixMask(touchCounts, size) {
  const keep = Math.min(size, touchCounts.length);
  const ids = Array.from({ length: touchCounts.length }, (_, id) => id);
  ids.sort((a, b) => touchCounts[b] - touchCounts[a] || a - b);
  const mask = new Uint8Array(touchCounts.length);
  for (let i = 0; i < keep; i++) {
    if (touchCounts[ids[i]] === 0) break;
    mask[ids[i]] = 1;
  }
  return mask;
}

function nodeStrideBytes(index) {
  if (arg('node-stride')) return parseIntArg('node-stride', null);
  if (index.kind === 'u8') return index.dim + 8 + 2 + index.M0 * 8;
  return index.dim * 4 + 2 + index.M0 * 4;
}

function summarize(values) {
  const xs = [...values].sort((a, b) => a - b);
  const pick = (p) => xs[Math.min(xs.length - 1, Math.floor(xs.length * p))];
  const sum = xs.reduce((a, b) => a + b, 0);
  return {
    min: xs[0],
    mean: sum / xs.length,
    p50: pick(0.50),
    p95: pick(0.95),
    p99: pick(0.99),
    max: xs[xs.length - 1],
  };
}

function countBlocks(ids, permutation, stride, blockSize) {
  const blocks = new Set();
  for (const id of ids) {
    const pos = permutation[id];
    blocks.add(Math.floor((pos * stride) / blockSize));
  }
  return blocks.size;
}

function countColdBlocks(ids, hotMask, permutation, stride, blockSize) {
  const blocks = new Set();
  for (const id of ids) {
    if (hotMask[id]) continue;
    const pos = permutation[id];
    blocks.add(Math.floor((pos * stride) / blockSize));
  }
  return blocks.size;
}

function workingSetByPrefix(traces, layouts, stride, blockSizes, prefixes, totalBytes) {
  const out = {};
  const cappedPrefixes = prefixes.filter((n) => n > 0 && n <= traces.length);
  for (const blockSize of blockSizes) {
    const blockOut = {};
    for (const [name, permutation] of Object.entries(layouts)) {
      const seen = new Set();
      const rows = {};
      let nextPrefix = 0;
      for (let qi = 0; qi < traces.length && nextPrefix < cappedPrefixes.length; qi++) {
        const ids = traces[qi];
        for (const id of ids) {
          const pos = permutation[id];
          seen.add(Math.floor((pos * stride) / blockSize));
        }
        while (nextPrefix < cappedPrefixes.length && qi + 1 === cappedPrefixes[nextPrefix]) {
          const bytes = seen.size * blockSize;
          rows[cappedPrefixes[nextPrefix]] = {
            uniqueBlocks: seen.size,
            bytes,
            mib: bytes / 1048576,
            fractionOfArtifact: totalBytes > 0 ? bytes / totalBytes : null,
          };
          nextPrefix++;
        }
      }
      blockOut[name] = rows;
    }
    out[blockSize] = blockOut;
  }
  return out;
}

function coalescedRuns(ids, permutation, stride, gap) {
  if (ids.length === 0) return { requests: 0, bytes: 0, coveredNodes: 0 };
  const seen = new Set();
  for (const id of ids) seen.add(permutation[id]);
  const positions = Array.from(seen).sort((a, b) => a - b);
  let requests = 0;
  let coveredNodes = 0;
  let start = positions[0];
  let prev = positions[0];
  for (let i = 1; i < positions.length; i++) {
    const pos = positions[i];
    if (pos - prev - 1 <= gap) {
      prev = pos;
      continue;
    }
    requests++;
    coveredNodes += prev - start + 1;
    start = pos;
    prev = pos;
  }
  requests++;
  coveredNodes += prev - start + 1;
  return { requests, bytes: coveredNodes * stride, coveredNodes };
}

function summarizeCoalescing(traces, roundTraces, permutation, stride, gaps) {
  const byGap = {};
  for (const gap of gaps) {
    const queryRequests = [];
    const queryBytes = [];
    const queryRounds = [];
    const maxRoundRequests = [];
    const meanRoundRequests = [];
    for (let qi = 0; qi < traces.length; qi++) {
      const query = coalescedRuns(traces[qi], permutation, stride, gap);
      queryRequests.push(query.requests);
      queryBytes.push(query.bytes);

      const rounds = roundTraces[qi] || [];
      let nonEmptyRounds = 0;
      let totalRoundRequests = 0;
      let maxRequests = 0;
      for (const roundIds of rounds) {
        const round = coalescedRuns(roundIds, permutation, stride, gap);
        if (round.requests === 0) continue;
        nonEmptyRounds++;
        totalRoundRequests += round.requests;
        if (round.requests > maxRequests) maxRequests = round.requests;
      }
      queryRounds.push(nonEmptyRounds);
      maxRoundRequests.push(maxRequests);
      meanRoundRequests.push(nonEmptyRounds > 0 ? totalRoundRequests / nonEmptyRounds : 0);
    }
    byGap[gap] = {
      query: {
        requests: summarize(queryRequests),
        bytes: summarize(queryBytes),
        mibMean: summarize(queryBytes).mean / 1048576,
      },
      rounds: {
        count: summarize(queryRounds),
        maxRequests: summarize(maxRoundRequests),
        meanRequests: summarize(meanRoundRequests),
      },
    };
  }
  return byGap;
}

function countTokens(ids, partition) {
  const tokens = new Set();
  for (const id of ids) tokens.add(partition[id]);
  return tokens.size;
}

function readMetisPartition(file, count) {
  const lines = fs.readFileSync(file, 'utf8').trim().split(/\r?\n/);
  if (lines.length !== count) throw new Error(`METIS partition has ${lines.length} rows; expected ${count}`);
  const partition = new Int32Array(count);
  let maxPart = -1;
  for (let i = 0; i < count; i++) {
    const part = Number(lines[i].trim());
    if (!Number.isInteger(part) || part < 0) throw new Error(`invalid METIS part at row ${i + 1}: ${lines[i]}`);
    partition[i] = part;
    if (part > maxPart) maxPart = part;
  }
  return { partition, parts: maxPart + 1 };
}

function buildSymmetricAdjacency(index) {
  const count = index.count;
  const degrees = new Uint32Array(count);
  for (let id = 0; id < count; id++) {
    const edges = index.base[id];
    for (let i = 0; i < edges.length; i++) {
      const n = edges[i];
      if (n === id) continue;
      degrees[id]++;
      degrees[n]++;
    }
  }

  const offsets = new Uint32Array(count + 1);
  for (let id = 0; id < count; id++) offsets[id + 1] = offsets[id] + degrees[id];
  const adj = new Uint32Array(offsets[count]);
  const cursor = new Uint32Array(offsets);
  for (let id = 0; id < count; id++) {
    const edges = index.base[id];
    for (let i = 0; i < edges.length; i++) {
      const n = edges[i];
      if (n === id) continue;
      adj[cursor[id]++] = n;
      adj[cursor[n]++] = id;
    }
  }

  let undirectedEdges = 0;
  const uniqueDegrees = new Uint32Array(count);
  for (let id = 0; id < count; id++) {
    const start = offsets[id];
    const end = offsets[id + 1];
    const neighbors = adj.subarray(start, end);
    neighbors.sort();
    let write = start;
    let prev = -1;
    for (let p = start; p < end; p++) {
      const n = adj[p];
      if (n === prev) continue;
      adj[write++] = n;
      prev = n;
    }
    uniqueDegrees[id] = write - start;
    undirectedEdges += uniqueDegrees[id];
  }
  return { offsets, adj, uniqueDegrees, undirectedEdges: undirectedEdges / 2 };
}

async function writeMetisGraph(index, file) {
  console.log('Building symmetrized base graph for METIS export...');
  const graph = buildSymmetricAdjacency(index);
  const out = fs.createWriteStream(file);
  out.write(`${index.count} ${graph.undirectedEdges}\n`);
  for (let id = 0; id < index.count; id++) {
    const start = graph.offsets[id];
    const degree = graph.uniqueDegrees[id];
    const parts = new Array(degree);
    for (let i = 0; i < degree; i++) parts[i] = String(graph.adj[start + i] + 1);
    out.write(`${parts.join(' ')}\n`);
    if ((id + 1) % 100000 === 0) console.log(`  wrote ${id + 1}/${index.count} nodes`);
  }
  await new Promise((resolve) => out.end(resolve));
  console.log(`Wrote METIS graph ${file}`);
}

async function buildSnapshot(snapshotPath, dataDir, count, opts) {
  console.log(`Loading ${count.toLocaleString()} SIFT base vectors...`);
  const { vectors, dim } = readFvecs(path.join(dataDir, 'sift_base.fvecs'), count);
  console.log(`Building Pancake u8 index (${vectors.length.toLocaleString()} x ${dim}D)...`);
  const index = await Pancake.create({
    dim,
    maxElements: vectors.length,
    metric: 'l2',
    quantized: true,
    M: opts.M,
    efConstruction: opts.efConstruction,
    efSearch: opts.efSearch,
  });
  try {
    const t0 = performance.now();
    index.addBatch(vectors);
    console.log(`Built in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
    const snapshot = index.export();
    fs.writeFileSync(snapshotPath, snapshot);
    console.log(`Wrote ${snapshotPath} (${(snapshot.byteLength / 1048576).toFixed(1)} MB)`);
  } finally {
    index.dispose();
  }
}

async function main() {
  const dataDir = path.resolve(arg('data-dir', path.join(__dirname, '..', 'sift')));
  const snapshotPath = path.resolve(arg('snapshot', arg('build-snapshot', '/tmp/pancake-sift1m-u8.pnck')));
  const count = parseIntArg('count', 1000000);
  const queries = parseIntArg('queries', 1000);
  const k = parseIntArg('k', 10);
  const efSearch = parseIntArg('ef-search', 100);
  const efSearchValues = parseIntListArg('ef-search-values', [efSearch]);
  const traceTrainQueries = parseIntArg('trace-train-queries', 0);
  const M = parseIntArg('m', 12);
  const efConstruction = parseIntArg('ef-construction', 75);
  const blockSizes = String(arg('block-sizes', '4096,16384,65536'))
    .split(',')
    .map((x) => Number(x.trim()))
    .filter((x) => Number.isInteger(x) && x > 0);
  const workingSetPrefixes = parseIntListArg('working-set-prefixes', [1, 10, 100, 1000]);
  const runGapTolerances = parseIntListArg('run-gap-tolerances', [0, 1, 2, 4, 8, 16]);
  const hotPrefixSizes = parseIntListArg('hot-prefix-sizes', []);
  const hilbertDims = parseIntListArg('hilbert-dims', [0, 1]);
  const hilbertBits = parseIntArg('hilbert-bits', 16);
  const metisGraphOut = arg('metis-graph-out', null);
  const metisExportOnly = parseBoolArg('metis-export-only');
  const metisPartitionPath = arg('metis-partition', null);
  const traceOut = arg('trace-out', null);
  const summaryOut = arg('summary-out', null);

  if (hasArg('build-snapshot') || !fs.existsSync(snapshotPath)) {
    await buildSnapshot(snapshotPath, dataDir, count, { M, efConstruction, efSearch });
  }

  console.log(`Parsing snapshot ${snapshotPath}...`);
  const snapshot = fs.readFileSync(snapshotPath);
  const parsed = parseSnapshot(new Uint8Array(snapshot.buffer, snapshot.byteOffset, snapshot.byteLength));
  console.log(`Parsed ${parsed.kind} graph: ${parsed.count.toLocaleString()} nodes, ${parsed.dim}D, entry=${parsed.entryPoint}, M=${parsed.M}, M0=${parsed.M0}`);

  if (metisGraphOut) {
    await writeMetisGraph(parsed, path.resolve(metisGraphOut));
    if (metisExportOnly) return;
  }

  let metis = null;
  let metisLayout = null;
  if (metisPartitionPath) {
    console.log(`Loading METIS partition ${metisPartitionPath}...`);
    metis = readMetisPartition(path.resolve(metisPartitionPath), parsed.count);
    console.log(`Loaded METIS partition with ${metis.parts.toLocaleString()} parts`);
  }

  console.log(`Loading ${queries.toLocaleString()} SIFT queries...`);
  const queryData = readFvecs(path.join(dataDir, 'sift_query.fvecs'), queries);
  if (queryData.dim !== parsed.dim) throw new Error(`query dim ${queryData.dim} != index dim ${parsed.dim}`);

  console.log('Building BFS layout permutation from entry point...');
  const identity = identityPermutation(parsed.count);
  const bfs = bfsPermutation(parsed);
  console.log('Building Reverse Cuthill-McKee layout permutation...');
  const rcm = reverseCuthillMckeePermutation(parsed);
  if (metis) {
    console.log('Building METIS partition-grouped layout permutation...');
    metisLayout = partitionPermutation(metis.partition, rcm);
  }
  console.log(`Building 2D Hilbert layout permutation from dims ${hilbertDims.join(',')}...`);
  const hilbert = hilbertPermutation(parsed, hilbertDims, hilbertBits, bfs);
  const stride = nodeStrideBytes(parsed);
  const totalLayoutBytes = parsed.count * stride;
  console.log(`Hypothetical node stride: ${stride} bytes`);

  const traceStream = traceOut ? fs.createWriteStream(traceOut) : null;
  const makeStats = () => ({
    identity: Object.fromEntries(blockSizes.map((b) => [b, []])),
    bfs: Object.fromEntries(blockSizes.map((b) => [b, []])),
    rcm: Object.fromEntries(blockSizes.map((b) => [b, []])),
    metis: Object.fromEntries(blockSizes.map((b) => [b, []])),
    hilbert: Object.fromEntries(blockSizes.map((b) => [b, []])),
    trace: Object.fromEntries(blockSizes.map((b) => [b, []])),
    hot: Object.fromEntries(hotPrefixSizes.map((n) => [n, {
      identity: Object.fromEntries(blockSizes.map((b) => [b, []])),
      bfs: Object.fromEntries(blockSizes.map((b) => [b, []])),
      rcm: Object.fromEntries(blockSizes.map((b) => [b, []])),
      metis: Object.fromEntries(blockSizes.map((b) => [b, []])),
      hilbert: Object.fromEntries(blockSizes.map((b) => [b, []])),
      trace: Object.fromEntries(blockSizes.map((b) => [b, []])),
    }])),
    heldOutTrace: {
      trace: Object.fromEntries(blockSizes.map((b) => [b, []])),
      hot: Object.fromEntries(hotPrefixSizes.map((n) => [n, {
        trace: Object.fromEntries(blockSizes.map((b) => [b, []])),
      }])),
      touched: [],
      evalQueries: 0,
    },
    touched: [],
    tokens: {
      metis: [],
    },
    traces: [],
    roundTraces: [],
    touchCounts: new Uint32Array(parsed.count),
    trainTouchCounts: new Uint32Array(parsed.count),
  });
  const statsByBeam = new Map(efSearchValues.map((beam) => [beam, makeStats()]));

  const t0 = performance.now();
  for (const beam of efSearchValues) {
    console.log(`Tracing efSearch=${beam}...`);
    const stats = statsByBeam.get(beam);
    for (let qi = 0; qi < queryData.vectors.length; qi++) {
      const replay = replaySearch(parsed, queryData.vectors[qi], k, beam);
      stats.traces.push(replay.uniqueTouched);
      stats.roundTraces.push(replay.rounds);
      stats.touched.push(replay.uniqueTouched.length);
      for (const id of replay.uniqueTouched) stats.touchCounts[id]++;
      if (traceTrainQueries > 0 && qi < traceTrainQueries) {
        for (const id of replay.uniqueTouched) stats.trainTouchCounts[id]++;
      }
      for (const block of blockSizes) {
        const idBlocks = countBlocks(replay.uniqueTouched, identity, stride, block);
        const bfsBlocks = countBlocks(replay.uniqueTouched, bfs, stride, block);
        const rcmBlocks = countBlocks(replay.uniqueTouched, rcm, stride, block);
        const hilbertBlocks = countBlocks(replay.uniqueTouched, hilbert, stride, block);
        stats.identity[block].push(idBlocks);
        stats.bfs[block].push(bfsBlocks);
        stats.rcm[block].push(rcmBlocks);
        if (metisLayout) stats.metis[block].push(countBlocks(replay.uniqueTouched, metisLayout, stride, block));
        stats.hilbert[block].push(hilbertBlocks);
      }
      if (metis) stats.tokens.metis.push(countTokens(replay.uniqueTouched, metis.partition));
      if ((qi + 1) % 25 === 0 || qi + 1 === queryData.vectors.length) {
        console.log(`  traced ${qi + 1}/${queryData.vectors.length}`);
      }
    }

    console.log('  building trace-weighted layout...');
    const trace = traceWeightedPermutation(stats.touchCounts, bfs);
    const hasHeldOutTrace = traceTrainQueries > 0 && traceTrainQueries < stats.traces.length;
    const heldOutTrace = hasHeldOutTrace ? traceWeightedPermutation(stats.trainTouchCounts, bfs) : null;
    stats.workingSet = workingSetByPrefix(
      stats.traces,
      {
        identity,
        bfs,
        rcm,
        ...(metisLayout ? { metis: metisLayout } : {}),
        hilbert,
        trace,
      },
      stride,
      blockSizes,
      workingSetPrefixes,
      totalLayoutBytes
    );
    stats.runCoalescing = {};
    const coalescingLayouts = {
      rcm,
      ...(metisLayout ? { metis: metisLayout } : {}),
      trace,
    };
    for (const [name, permutation] of Object.entries(coalescingLayouts)) {
      stats.runCoalescing[name] = summarizeCoalescing(
        stats.traces,
        stats.roundTraces,
        permutation,
        stride,
        runGapTolerances
      );
    }
    for (let qi = 0; qi < stats.traces.length; qi++) {
      const touched = stats.traces[qi];
      const row = traceStream ? { query: qi, efSearch: beam, touched } : null;
      for (const block of blockSizes) {
        const traceBlocks = countBlocks(touched, trace, stride, block);
        stats.trace[block].push(traceBlocks);
        if (hasHeldOutTrace && qi >= traceTrainQueries) {
          stats.heldOutTrace.trace[block].push(countBlocks(touched, heldOutTrace, stride, block));
        }
        if (row) {
          row[`identity_${block}`] = stats.identity[block][qi];
          row[`bfs_${block}`] = stats.bfs[block][qi];
          row[`rcm_${block}`] = stats.rcm[block][qi];
          if (metisLayout) row[`metis_${block}`] = stats.metis[block][qi];
          row[`hilbert_${block}`] = stats.hilbert[block][qi];
          row[`trace_${block}`] = traceBlocks;
          if (hasHeldOutTrace && qi >= traceTrainQueries) {
            row[`heldout_trace_${block}`] = stats.heldOutTrace.trace[block][stats.heldOutTrace.trace[block].length - 1];
          }
        }
      }
      if (hasHeldOutTrace && qi >= traceTrainQueries) {
        stats.heldOutTrace.touched.push(touched.length);
        stats.heldOutTrace.evalQueries++;
      }
      if (traceStream) traceStream.write(`${JSON.stringify(row)}\n`);
    }

    for (const prefixSize of hotPrefixSizes) {
      console.log(`  counting cold blocks with hotPrefix=${prefixSize.toLocaleString()}...`);
      const hotMask = hotPrefixMask(stats.touchCounts, prefixSize);
      const heldOutHotMask = hasHeldOutTrace ? hotPrefixMask(stats.trainTouchCounts, prefixSize) : null;
      const hotStats = stats.hot[prefixSize];
      const heldOutHotStats = stats.heldOutTrace.hot[prefixSize];
      let qi = 0;
      for (const touched of stats.traces) {
        for (const block of blockSizes) {
          hotStats.identity[block].push(countColdBlocks(touched, hotMask, identity, stride, block));
          hotStats.bfs[block].push(countColdBlocks(touched, hotMask, bfs, stride, block));
          hotStats.rcm[block].push(countColdBlocks(touched, hotMask, rcm, stride, block));
          if (metisLayout) hotStats.metis[block].push(countColdBlocks(touched, hotMask, metisLayout, stride, block));
          hotStats.hilbert[block].push(countColdBlocks(touched, hotMask, hilbert, stride, block));
          hotStats.trace[block].push(countColdBlocks(touched, hotMask, trace, stride, block));
          if (hasHeldOutTrace && qi >= traceTrainQueries) {
            heldOutHotStats.trace[block].push(countColdBlocks(touched, heldOutHotMask, heldOutTrace, stride, block));
          }
        }
        qi++;
      }
    }
  }
  if (traceStream) await new Promise((resolve) => traceStream.end(resolve));

  const summary = {
    snapshot: snapshotPath,
    dataset: 'sift',
    count: parsed.count,
    queries: queryData.vectors.length,
    dim: parsed.dim,
    k,
    efSearchValues,
    layout: {
      permutations: ['identity', 'bfs_from_entry_point', 'reverse_cuthill_mckee_base', ...(metis ? ['metis_partition_grouped'] : []), 'hilbert_2d', 'trace_weighted_by_beam'],
      entryPoint: parsed.entryPoint,
      nodeStrideBytes: stride,
      blockSizes,
      workingSetPrefixes,
      runGapTolerances,
      hotPrefixSizes,
      traceTrainQueries,
      hilbertDims,
      hilbertBits,
      metisPartition: metis ? {
        path: path.resolve(metisPartitionPath),
        parts: metis.parts,
      } : null,
    },
    byBeamWidth: {},
    elapsedSeconds: (performance.now() - t0) / 1000,
  };
  for (const beam of efSearchValues) {
    const stats = statsByBeam.get(beam);
    summary.byBeamWidth[beam] = {
      touchedNodes: summarize(stats.touched),
      blocks: {},
      hotPrefix: {},
      workingSet: stats.workingSet,
      runCoalescing: stats.runCoalescing,
    };
    if (metis) {
      summary.byBeamWidth[beam].tokens = {
        metis: summarize(stats.tokens.metis),
      };
    }
    if (stats.heldOutTrace.evalQueries > 0) {
      summary.byBeamWidth[beam].heldOutTrace = {
        trainQueries: traceTrainQueries,
        evalQueries: stats.heldOutTrace.evalQueries,
        touchedNodes: summarize(stats.heldOutTrace.touched),
        blocks: {},
        hotPrefix: {},
      };
    }
    for (const block of blockSizes) {
      summary.byBeamWidth[beam].blocks[block] = {
        identity: summarize(stats.identity[block]),
        bfs: summarize(stats.bfs[block]),
        rcm: summarize(stats.rcm[block]),
        ...(metisLayout ? { metis: summarize(stats.metis[block]) } : {}),
        hilbert: summarize(stats.hilbert[block]),
        trace: summarize(stats.trace[block]),
      };
      if (stats.heldOutTrace.evalQueries > 0) {
        summary.byBeamWidth[beam].heldOutTrace.blocks[block] = {
          trace: summarize(stats.heldOutTrace.trace[block]),
        };
      }
    }
    for (const prefixSize of hotPrefixSizes) {
      summary.byBeamWidth[beam].hotPrefix[prefixSize] = {
        blocks: {},
      };
      const hotStats = stats.hot[prefixSize];
      for (const block of blockSizes) {
        summary.byBeamWidth[beam].hotPrefix[prefixSize].blocks[block] = {
          identity: summarize(hotStats.identity[block]),
          bfs: summarize(hotStats.bfs[block]),
          rcm: summarize(hotStats.rcm[block]),
          ...(metisLayout ? { metis: summarize(hotStats.metis[block]) } : {}),
          hilbert: summarize(hotStats.hilbert[block]),
          trace: summarize(hotStats.trace[block]),
        };
        if (stats.heldOutTrace.evalQueries > 0) {
          if (!summary.byBeamWidth[beam].heldOutTrace.hotPrefix[prefixSize]) {
            summary.byBeamWidth[beam].heldOutTrace.hotPrefix[prefixSize] = { blocks: {} };
          }
          summary.byBeamWidth[beam].heldOutTrace.hotPrefix[prefixSize].blocks[block] = {
            trace: summarize(stats.heldOutTrace.hot[prefixSize].trace[block]),
          };
        }
      }
    }
  }

  const text = JSON.stringify(summary, null, 2);
  console.log(text);
  if (summaryOut) fs.writeFileSync(summaryOut, `${text}\n`);
}

main().catch((err) => {
  console.error(err && err.stack || err);
  process.exit(1);
});
