export const DEMO_DIM = 256;

const TOKEN_RE = /[a-z0-9]+/g;

const SYNONYM_GROUPS = [
  ['worker', ['worker', 'workers', 'workerd', 'cloudflare', 'edge', 'isolate', 'isolates']],
  ['snapshot', ['snapshot', 'snapshots', 'checkpoint', 'checkpoints']],
  ['restore', ['restore', 'restores', 'restored', 'restoring', 'reload', 'reloads', 'rebuild', 'rebuilds']],
  ['persist', ['persist', 'persistence', 'persisted', 'persisting', 'durable', 'durability', 'saved', 'saving']],
  ['search', ['search', 'searches', 'searching', 'query', 'queries', 'retrieval', 'retrieve', 'retrieved']],
  ['vector', ['vector', 'vectors', 'embedding', 'embeddings']],
  ['delete', ['delete', 'deletes', 'deleted', 'deletion', 'remove', 'removed', 'removing']],
  ['ghost', ['ghost', 'ghosts', 'tombstone', 'tombstones']],
  ['compact', ['compact', 'compaction', 'compacted', 'compacts']],
  ['export', ['export', 'exports', 'exported', 'serialize', 'serialized', 'serialization']],
  ['import', ['import', 'imports', 'imported', 'deserialize', 'deserialized', 'deserialization']],
  ['filter', ['filter', 'filtered', 'filtering', 'allowlist', 'allowed', 'tenant']],
  ['quantized', ['quantized', 'quantize', 'int8', 'quantization']],
  ['float', ['float', 'float32', 'fp32']],
  ['metric', ['metric', 'cosine', 'l2', 'distance']],
  ['docs', ['docs', 'documentation', 'readme', 'guide', 'guides', 'quickstart', 'manual']],
  ['api', ['api', 'apis', 'endpoint', 'endpoints', 'http', 'request', 'response']],
  ['latency', ['latency', 'speed', 'fast', 'performance', 'throughput', 'qps']],
];

const ALIAS_TO_CANONICAL = new Map();
for (const [canonical, terms] of SYNONYM_GROUPS) {
  for (const term of terms) {
    ALIAS_TO_CANONICAL.set(term, canonical);
  }
}

function hash32(text, seed = 2166136261) {
  let h = seed >>> 0;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function normalizeToken(raw) {
  let token = raw.toLowerCase();
  if (token.endsWith('ies') && token.length > 4) token = `${token.slice(0, -3)}y`;
  else if (token.endsWith('ing') && token.length > 5) token = token.slice(0, -3);
  else if (token.endsWith('ed') && token.length > 4) token = token.slice(0, -2);
  else if (token.endsWith('es') && token.length > 4) token = token.slice(0, -2);
  else if (token.endsWith('s') && token.length > 3) token = token.slice(0, -1);
  return ALIAS_TO_CANONICAL.get(token) || token;
}

export function tokenize(text) {
  const lowered = String(text || '').toLowerCase();
  const matches = lowered.match(TOKEN_RE) || [];
  return matches.map(normalizeToken).filter((token) => token.length > 1);
}

export function embedText(text, dim = DEMO_DIM) {
  const tokens = tokenize(text);
  const vec = new Float32Array(dim);
  if (tokens.length === 0) return vec;

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    const h = hash32(token);
    const idx = h % dim;
    const sign = (hash32(token, 31337) & 1) === 0 ? 1 : -1;
    vec[idx] += sign;

    if (i + 1 < tokens.length) {
      const bigram = `${token}:${tokens[i + 1]}`;
      const bh = hash32(bigram, 97);
      const bIdx = bh % dim;
      const bSign = (hash32(bigram, 193) & 1) === 0 ? 1 : -1;
      vec[bIdx] += 0.65 * bSign;
    }
  }

  let norm = 0;
  for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm === 0) return vec;
  for (let i = 0; i < dim; i++) vec[i] /= norm;
  return vec;
}
