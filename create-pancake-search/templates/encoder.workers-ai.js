// Query encoder: Cloudflare Workers AI (@cf/baai/bge-small-en-v1.5).
// LOCAL_STUB_AI=1 swaps in deterministic hash embeddings for endpoint testing
// only; stub query vectors do not live in the index's semantic space.

export function assertEncoderManifest(manifest) {
  if (manifest.workersAiModel !== '@cf/baai/bge-small-en-v1.5') {
    throw Object.assign(new Error('Manifest Workers AI model mismatch'), { code: 'MANIFEST_MISMATCH' });
  }
}

export async function embedQuery(query, manifest, env) {
  const prefixed = `${manifest.prefixPolicy.query}${query}`;
  if (isLocalStubAi(env)) {
    return { vector: await hashEmbedding(prefixed, manifest.dims), embedded: null };
  }
  const response = await env.AI.run(manifest.workersAiModel, {
    text: [prefixed],
    pooling: manifest.pooling || 'mean',
  }).catch((error) => {
    throw Object.assign(new Error(`Workers AI embedding failed: ${error.message || String(error)}`), { code: 'EMBED_UNAVAILABLE' });
  });
  const raw = response?.data?.[0] || response?.data || response?.embeddings?.[0] || response?.embedding;
  if (!raw || raw.length !== manifest.dims) {
    throw Object.assign(new Error('Workers AI returned an unexpected embedding shape'), { code: 'EMBED_UNAVAILABLE' });
  }
  return { vector: Float32Array.from(raw), embedded: null };
}

export function scoreHits() {
  return null;
}

export function encoderInfo(env, manifest) {
  return {
    model: manifest?.workersAiModel || null,
    encoder_mode: 'workers-ai',
    local_stub_ai: isLocalStubAi(env),
  };
}

function isLocalStubAi(env) {
  const value = String(env?.LOCAL_STUB_AI || '').trim().toLowerCase();
  return value === '1' || value === 'true' || value === 'yes' || value === 'on';
}

async function sha256Bytes(text) {
  const bytes = new TextEncoder().encode(text);
  return new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
}

async function hashEmbedding(text, dims) {
  const vector = new Float32Array(dims);
  const words = String(text || '').toLowerCase().match(/[a-z0-9_'-]+/g) || [];
  for (const word of words) {
    const hash = await sha256Bytes(word);
    const index = ((hash[0] | (hash[1] << 8) | (hash[2] << 16) | (hash[3] << 24)) >>> 0) % dims;
    vector[index] += (hash[4] / 255) * 2 - 1;
  }
  let norm = 0;
  for (let i = 0; i < dims; i++) norm += vector[i] * vector[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < dims; i++) vector[i] /= norm;
  return vector;
}
