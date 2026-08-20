import crypto from 'node:crypto';
import fs from 'node:fs';
import { expectedBlobBytes } from './inline-transformer.mjs';
import { MAGIC, HEADER_BYTES, TABLE_ENTRY_BYTES, KINDS, KIND_NAMES } from './format.mjs';

export { MAGIC, HEADER_BYTES, TABLE_ENTRY_BYTES, KINDS, KIND_NAMES };

// The inline-transformer kernel via the web glue with an explicitly read
// wasm binary — the form that works under both native ESM and jiti's CJS
// transform (the node glue's createRequire bootstrap breaks there). Node
// builders only; the reader picks its own kernel.
export async function loadInlineEncoderKernel() {
  const { default: createEncoderModule } = await import('./encoder-kernels/encoder.mjs');
  const wasmBinary = fs.readFileSync(new URL('./encoder-kernels/encoder.wasm', import.meta.url));
  return () => createEncoderModule({ wasmBinary });
}

export const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest();
export const align16 = (n) => Math.ceil(n / 16) * 16;

export function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') {
    const keys = Object.keys(value).sort();
    return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalJson(value[k])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

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

export function buildInlineTransformerEncoderSegment({ declaration, vocabBytes, weightBytes }) {
  if (declaration.layout) {
    const expected = expectedBlobBytes(declaration.layout);
    if (weightBytes.length !== expected) {
      throw new Error(`inline-encoder weights are ${weightBytes.length} bytes but the declared layout implies ${expected}`);
    }
  }
  const declarationBytes = Buffer.from(canonicalJson(declaration), 'utf8');
  const out = Buffer.alloc(12 + declarationBytes.length + vocabBytes.length + weightBytes.length);
  out.writeUInt32LE(declarationBytes.length, 0);
  out.writeUInt32LE(vocabBytes.length, 4);
  out.writeUInt32LE(weightBytes.length, 8);
  declarationBytes.copy(out, 12);
  vocabBytes.copy(out, 12 + declarationBytes.length);
  weightBytes.copy(out, 12 + declarationBytes.length + vocabBytes.length);
  return out;
}

// The spec's measured operating point (SKETCH_PROFILE.md section 5, mandatory
// in the complete profile's evaluation segment): open the just-built sketch
// bytes in memory, replay a query set at increasing rerank C against the
// exact full-rerank top-k, and return the smallest C whose recall reaches
// targetRecall — plus the whole curve for the evaluation segment. Queries
// are the corpus's own float embeddings when the caller has them, else rows
// dequantized out of the snapshot.
export async function measureRecommendedRerank({
  artifactModule,
  sketchBytes,
  queryVectors = null,
  snapshotBytes = null,
  k = 10,
  targetRecall = 0.98,
  maxQueries = 256,
  sweep = null,
}) {
  const bytes = sketchBytes instanceof Uint8Array ? sketchBytes : new Uint8Array(sketchBytes);
  const source = {
    size: bytes.length,
    preferredParallelism: Infinity,
    preferredGapBytes: 0,
    async read(offset, length) { return bytes.subarray(offset, offset + length); },
    async close() {},
  };
  const sketch = await artifactModule.PancakeSketchArtifact.open(source);
  try {
    const count = sketch.count;
    if (count === 0) throw new Error('cannot measure rerank on an empty sketch');
    const topK = Math.min(k, count);
    let queries;
    let querySource;
    if (queryVectors && queryVectors.length > 0) {
      queries = sampleEvenly(queryVectors, maxQueries);
      querySource = 'corpus-embeddings';
    } else {
      queries = selfQueriesFromSnapshot(artifactModule.parseUint8Snapshot(snapshotBytes), maxQueries);
      querySource = 'dequantized-rows';
    }
    const truth = [];
    for (const q of queries) {
      truth.push(new Set((await sketch.search(q, topK, { rerank: count })).results.map((r) => r.id)));
    }
    const ladder = (sweep || [10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480, 640])
      .filter((c) => c >= topK && c < count);
    ladder.push(count);
    const curve = [];
    let recommendedRerank = null;
    let recommendedRecall = null;
    for (const C of ladder) {
      let hits = 0;
      let total = 0;
      for (let i = 0; i < queries.length; i++) {
        for (const r of (await sketch.search(queries[i], topK, { rerank: C })).results) {
          if (truth[i].has(r.id)) hits++;
        }
        total += truth[i].size;
      }
      const recall = total > 0 ? Number((hits / total).toFixed(4)) : 1;
      curve.push({ rerank: C, recall });
      if (recommendedRerank === null && recall >= targetRecall) {
        recommendedRerank = C;
        recommendedRecall = recall;
      }
    }
    if (recommendedRerank === null) {
      recommendedRerank = count;
      recommendedRecall = curve[curve.length - 1].recall;
    }
    return {
      recommendedRerank,
      recall: recommendedRecall,
      k: topK,
      targetRecall,
      queries: queries.length,
      querySource,
      curve,
    };
  } finally {
    await sketch.close();
  }
}

function sampleEvenly(items, max) {
  if (items.length <= max) return items;
  const stride = items.length / max;
  const out = [];
  for (let i = 0; i < max; i++) out.push(items[Math.floor(i * stride)]);
  return out;
}

function selfQueriesFromSnapshot(graph, maxQueries) {
  const n = Math.min(graph.count, maxQueries);
  const stride = graph.count / n;
  const queries = [];
  for (let i = 0; i < n; i++) {
    const id = Math.floor(i * stride);
    const v = new Float32Array(graph.dim);
    const base = id * graph.dim;
    for (let d = 0; d < graph.dim; d++) v[d] = graph.offsets[id] + graph.scales[id] * graph.qdata[base + d];
    queries.push(v);
  }
  return queries;
}

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
