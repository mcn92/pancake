import { createInlineTransformerEmbedder, parseInlineTransformerEncoder } from '../../src/inline-transformer.mjs';
import createEncoder from '../../src/encoder-kernels/encoder.mjs';

const MAGIC = 0x31465350;
const HEADER_BYTES = 64;
const TABLE_ENTRY_BYTES = 48;
const KIND_NAMES = { 1: 'index', 2: 'corpus', 3: 'query-interp', 4: 'evaluation' };
const decoder = new TextDecoder();

function viewOf(bytes) {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function toHex(bytes) {
  let out = '';
  for (const b of bytes) out += b.toString(16).padStart(2, '0');
  return out;
}

function align16(n) {
  return Math.ceil(n / 16) * 16;
}

async function sha256hex(bytes) {
  return toHex(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)));
}

function memorySource(bytes) {
  return {
    size: bytes.length,
    preferredParallelism: Infinity,
    preferredGapBytes: 2048,
    async read(offset, length) {
      return bytes.subarray(offset, offset + length);
    },
    async close() {},
  };
}

function windowSource(source, offset, length) {
  return {
    size: length,
    preferredParallelism: source.preferredParallelism,
    preferredGapBytes: source.preferredGapBytes,
    async read(off, len) { return source.read(offset + off, len); },
    async close() {},
  };
}

async function loadArtifactContract() {
  const mod = await import('pancake-wasm/artifact');
  const contract = mod.default || mod;
  if (contract.PancakeSketchArtifact) return contract;
  if (globalThis.process?.versions?.node) {
    const fallback = await import(new URL('../../../pancake-artifact.js', import.meta.url).href);
    return fallback.default || fallback;
  }
  throw new Error('pancake-wasm/artifact does not expose PancakeSketchArtifact');
}

function createAbstentionScorer(asset, bloomBytes) {
  if (!asset || !Array.isArray(asset.weights) || !asset.thresholds) return null;
  const bloom = new Uint8Array(bloomBytes);
  const bits = asset.vocabBloom.bits;
  const seeds = [0, 0x9e3779b9];
  const fnv1a = (str, seed) => {
    let h = 0x811c9dc5 ^ seed;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return (h >>> 0) % bits;
  };
  const knownFrac = (text) => {
    const words = text.toLowerCase().match(/[a-z0-9']+/g) || [];
    if (!words.length) return 0;
    let known = 0;
    for (const word of words) {
      if (seeds.every((seed) => {
        const bit = fnv1a(word, seed);
        return (bloom[bit >> 3] >> (bit & 7)) & 1;
      })) known++;
    }
    return known / words.length;
  };
  return {
    score(queryText, results) {
      const d0 = results.length ? results[0].distance : 1;
      const margin = results.length > 1 ? results[Math.min(4, results.length - 1)].distance - d0 : 0;
      const mean10 = results.length ? results.reduce((s, r) => s + r.distance, 0) / results.length : 1;
      const signals = { d0, margin, mean10, known_frac: knownFrac(queryText) };
      let z = asset.bias;
      asset.features.forEach((f, j) => {
        z += ((signals[f] - asset.standardize.mean[f]) / asset.standardize.std[f]) * asset.weights[j];
      });
      const p = 1 / (1 + Math.exp(-z));
      const verdict = p < asset.thresholds.hard ? 'abstain' : p < asset.thresholds.weak ? 'weak' : 'answer';
      return { p, verdict, signals };
    },
  };
}

export async function openCompletePancake(bytes, options = {}) {
  const { PancakeSketchArtifact } = await loadArtifactContract();
  const source = memorySource(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes));
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
  if (source.size !== fileBytes) {
    throw new Error(`.pancake is truncated or padded: header says ${fileBytes} bytes, source has ${source.size}`);
  }
  const identity = toHex(header.subarray(24, 56));
  const manifestBuf = new Uint8Array(await source.read(HEADER_BYTES, manifestBytes));
  if (await sha256hex(manifestBuf) !== identity) throw new Error('.pancake manifest failed identity verification');
  const manifest = JSON.parse(decoder.decode(manifestBuf));
  if (manifest.profile !== 'pancake-complete-v1') throw new Error(`unsupported profile ${manifest.profile}`);

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
    if (!declared || declared.kind !== kind || declared.bytes !== length || offset !== expectedOffset || offset + length > fileBytes) {
      throw new Error(`.pancake segment table disagrees with manifest at entry ${i}`);
    }
    segments.set(kind, { offset, length, sha256: declared.sha256 });
    expectedOffset = align16(offset + length);
  }

  const qi = segments.get('query-interp');
  const qiBytes = new Uint8Array(await source.read(qi.offset, qi.length));
  if (await sha256hex(qiBytes) !== qi.sha256) throw new Error('.pancake query-interp segment failed hash verification');
  const qiView = viewOf(qiBytes);
  const qiKind = qiView.getUint32(4, true);
  if (qiKind !== 3) throw new Error(`expected inline-transformer-v1 query-interp kind 3, got ${qiKind}`);
  const encoderLen = qiView.getUint32(8, true);
  const calibrationLen = qiView.getUint32(12, true);
  if (16 + encoderLen + calibrationLen !== qi.length) throw new Error('.pancake query-interp segment layout is inconsistent');
  const encoderBytes = qiBytes.subarray(16, 16 + encoderLen);
  const calibrationJson = JSON.parse(decoder.decode(qiBytes.subarray(16 + encoderLen)));

  const { declaration, vocabText, blob } = parseInlineTransformerEncoder(encoderBytes);
  const embedder = await createInlineTransformerEmbedder({ declaration, vocabText, blob, createEncoder });

  const bloomBytes = Uint8Array.from(atob(calibrationJson.vocabBloomBase64 || ''), (c) => c.charCodeAt(0));
  const scorer = createAbstentionScorer(calibrationJson.asset, bloomBytes);
  const verdicts = { answer: 'strong', weak: 'weak', abstain: 'none' };

  const idx = segments.get('index');
  const sketch = await PancakeSketchArtifact.open(windowSource(source, idx.offset, idx.length));
  const corpus = segments.get('corpus');
  const countBuf = new Uint8Array(await source.read(corpus.offset, 4));
  const recordCount = viewOf(countBuf).getUint32(0, true);
  const offsetsBuf = new Uint8Array(await source.read(corpus.offset + 4, 8 * (recordCount + 1)));
  const offsetsView = viewOf(offsetsBuf);
  const recordOffsets = new Array(recordCount + 1);
  for (let i = 0; i <= recordCount; i++) recordOffsets[i] = Number(offsetsView.getBigUint64(8 * i, true));

  const hydrate = async (id) => {
    const start = recordOffsets[id];
    const end = recordOffsets[id + 1];
    const record = new Uint8Array(await source.read(corpus.offset + start, end - start));
    return JSON.parse(decoder.decode(record));
  };

  return {
    info() {
      return { identity, records: recordCount, dim: manifest.dim, metric: manifest.metric, encoder: declaration, fileBytes };
    },
    async query(text, queryOptions = {}) {
      const k = Number.isInteger(queryOptions.k) && queryOptions.k > 0 ? queryOptions.k : 8;
      const raw = String(text || '').trim();
      const context = await embedder.embed(`${declaration.prefixPolicy?.query || ''}${raw}`);
      const hits = (await sketch.search(context.vector, k, {
        rerank: queryOptions.rerank ?? options.rerank,
        parallelism: queryOptions.parallelism ?? options.rerankParallelism,
        gap: queryOptions.gap ?? options.rerankGap,
      })).results;
      const scored = scorer ? scorer.score(raw, hits) : null;
      const quality = scored ? { matchQuality: verdicts[scored.verdict] || scored.verdict, confidence: scored.p } : { matchQuality: 'unscored' };
      const returned = quality.matchQuality === 'none' ? [] : hits;
      const results = await Promise.all(returned.map(async (hit) => ({ id: hit.id, distance: hit.distance, ...await hydrate(hit.id) })));
      return { ...quality, results };
    },
    async close() {
      embedder.dispose();
      await sketch.close();
    },
  };
}
