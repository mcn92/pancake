/**
 * Pancake Search - Cloudflare Workers Entrypoint
 *
 * This example intentionally uses the public Pancake API. It should be useful
 * as deployment documentation and as a benchmark of the package surface users
 * actually call, not as a copy of the raw WASM ABI.
 */

import Pancake from '../../pancake.workerd.mjs';

let index = null;
let indexConfig = null;
let localSnapshot = null;

const MAX_RESULTS = 100;
const MAX_DIMS = 4096;
const DEFAULT_MAX_ELEMENTS = 5_000;
const DEFAULT_MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024;
const DEFAULT_MAX_JSON_BYTES = 1 * 1024 * 1024;
const MAX_EF = 2000;
const MAX_M = 128;
const SNAPSHOT_KEY_PREFIX = 'pancake-index-';
const LEGACY_SNAPSHOT_KEY = 'pancake-index.bin';
const ADMIN_ROUTES = new Set([
  '/init',
  '/add',
  '/add_batch',
  '/delete',
  '/compact',
  '/import',
  '/export',
  '/reset_cache',
  '/search_debug'
]);

const rateLimitMap = new Map();
const RATE_LIMIT_WINDOW_MS = 60_000;
const restoreState = {
  restoreCount: 0,
  restoredAt: null,
  lastRestoreMs: null,
  lastFetchMs: null,
  lastDeserializeMs: null,
  lastSnapshotBytes: null,
  lastSnapshotKey: null
};

function isPositiveInteger(value) {
  return Number.isInteger(value) && value > 0;
}

function jsonResponse(body, status = 200, headers = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...headers
    }
  });
}

function binaryResponse(bytes, headers = {}) {
  return new Response(bytes, {
    headers: {
      'Content-Type': 'application/octet-stream',
      ...headers
    }
  });
}

function corsHeaders(env) {
  const origin = env?.ALLOWED_ORIGIN || '*';
  return {
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
  };
}

function withCors(response, env) {
  const headers = new Headers(response.headers);
  for (const [key, value] of Object.entries(corsHeaders(env))) {
    headers.set(key, value);
  }
  return new Response(response.body, { status: response.status, headers });
}

function isReadOnly(env) {
  return String(env?.READ_ONLY || '') === '1';
}

function isAdminAllowed(request, env) {
  if (!env?.API_KEY) return true;
  const expected = `Bearer ${env.API_KEY}`;
  return request.headers.get('Authorization') === expected;
}

function getRateLimit(env) {
  const raw = env?.RATE_LIMIT_RPM;
  if (raw === undefined || raw === null || raw === '') return null;
  const parsed = Number.parseInt(raw, 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function isRateLimited(request, env) {
  const maxRpm = getRateLimit(env);
  if (maxRpm === null) return false;

  const ip = request.headers.get('CF-Connecting-IP') || 'local';
  const now = Date.now();
  let timestamps = rateLimitMap.get(ip);
  if (!timestamps) {
    timestamps = [];
    rateLimitMap.set(ip, timestamps);
  }

  while (timestamps.length > 0 && now - timestamps[0] >= RATE_LIMIT_WINDOW_MS) {
    timestamps.shift();
  }
  if (timestamps.length >= maxRpm) return true;
  timestamps.push(now);
  return false;
}

async function readJson(request, env) {
  const maxBytes = positiveEnvInt(env, 'MAX_JSON_BYTES', DEFAULT_MAX_JSON_BYTES);
  const text = await request.text();
  if (text.length > maxBytes) {
    throw new Error(`JSON body exceeds MAX_JSON_BYTES (${text.length} > ${maxBytes})`);
  }
  return text.length === 0 ? {} : JSON.parse(text);
}

async function readBytes(request, env) {
  const maxBytes = positiveEnvInt(env, 'MAX_SNAPSHOT_BYTES', DEFAULT_MAX_SNAPSHOT_BYTES);
  const buffer = await request.arrayBuffer();
  if (buffer.byteLength > maxBytes) {
    throw new Error(`Snapshot exceeds MAX_SNAPSHOT_BYTES (${buffer.byteLength} > ${maxBytes})`);
  }
  return new Uint8Array(buffer);
}

function positiveEnvInt(env, name, fallback) {
  const raw = env?.[name];
  if (raw === undefined || raw === null || raw === '') return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function validateCreateConfig(config, label, maxElementsLimit) {
  if (!isPositiveInteger(config.dim) || config.dim > MAX_DIMS) {
    throw new Error(`${label}: dims must be an integer between 1 and ${MAX_DIMS}`);
  }
  if (!isPositiveInteger(config.maxElements) || config.maxElements > maxElementsLimit) {
    throw new Error(`${label}: maxElements must be an integer between 1 and ${maxElementsLimit}`);
  }
  if (!isPositiveInteger(config.M) || config.M > MAX_M) {
    throw new Error(`${label}: M must be an integer between 1 and ${MAX_M}`);
  }
  if (!isPositiveInteger(config.efConstruction) || config.efConstruction > MAX_EF) {
    throw new Error(`${label}: efConstruction must be an integer between 1 and ${MAX_EF}`);
  }
  if (!isPositiveInteger(config.efSearch) || config.efSearch > MAX_EF) {
    throw new Error(`${label}: efSearch must be an integer between 1 and ${MAX_EF}`);
  }
}

function makeCreateConfig(body, env) {
  const config = {
    dim: body.dims ?? body.dim,
    maxElements: body.maxElements ?? DEFAULT_MAX_ELEMENTS,
    metric: body.metric || 'cosine',
    quantized: body.quantized !== false,
    M: body.M ?? 16,
    efConstruction: body.efConstruction ?? body.efC ?? 200,
    efSearch: body.efSearch ?? body.efS ?? 100
  };
  validateCreateConfig(config, 'Invalid index config', positiveEnvInt(env, 'MAX_ELEMENTS_LIMIT', DEFAULT_MAX_ELEMENTS));
  return config;
}

function configForMetadata(config) {
  return {
    dims: config.dim,
    maxElements: config.maxElements,
    metric: config.metric,
    quantized: config.quantized,
    initParams: {
      M: config.M,
      efC: config.efConstruction,
      efS: config.efSearch
    }
  };
}

function configFromMetadata(metadata, fallbackDims, env) {
  const initParams = metadata?.initParams || {};
  const config = {
    dim: metadata?.dims ?? fallbackDims,
    maxElements: metadata?.maxElements ?? DEFAULT_MAX_ELEMENTS,
    metric: metadata?.metric || 'cosine',
    quantized: metadata?.quantized !== false,
    M: initParams.M ?? 16,
    efConstruction: initParams.efC ?? initParams.efConstruction ?? 200,
    efSearch: initParams.efS ?? initParams.efSearch ?? 100
  };
  validateCreateConfig(config, 'Invalid snapshot config', positiveEnvInt(env, 'MAX_ELEMENTS_LIMIT', DEFAULT_MAX_ELEMENTS));
  return config;
}

function parsePancakeEnvelopeMetadata(bytes) {
  if (bytes.byteLength < 20) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(0, true) !== 0x504E434B) return null;
  const version = view.getUint32(4, true);
  if (version !== 2 && version !== 3) return null;
  return {
    dims: view.getUint32(8, true),
    metric: view.getUint32(12, true) === 0 ? 'l2' : 'cosine',
    quantized: view.getUint32(16, true) !== 0
  };
}

function nextSnapshotKey() {
  return `${SNAPSHOT_KEY_PREFIX}${String(Date.now()).padStart(13, '0')}.pnck`;
}

async function findLatestSnapshot(env) {
  if (!env?.INDEX_BUCKET) return null;

  const listed = await env.INDEX_BUCKET.list({ prefix: SNAPSHOT_KEY_PREFIX });
  if (listed.objects && listed.objects.length > 0) {
    let latest = listed.objects[0];
    for (const obj of listed.objects) {
      if (obj.key > latest.key) latest = obj;
    }
    return latest.key;
  }

  const legacy = await env.INDEX_BUCKET.head(LEGACY_SNAPSHOT_KEY);
  return legacy ? LEGACY_SNAPSHOT_KEY : null;
}

async function persistIndex(env) {
  if (!index || !indexConfig) return;
  if (index.ghostCount > 0) index.compact();
  const snapshot = index.export();
  localSnapshot = {
    bytes: snapshot,
    config: { ...indexConfig },
    key: 'local-memory-snapshot'
  };
  if (!env?.INDEX_BUCKET) return;
  const key = nextSnapshotKey();
  await env.INDEX_BUCKET.put(key, snapshot, {
    customMetadata: Object.fromEntries(
      Object.entries(configForMetadata(indexConfig)).map(([k, v]) => [k, typeof v === 'string' ? v : JSON.stringify(v)])
    ),
    httpMetadata: { contentType: 'application/octet-stream' }
  });
  restoreState.lastSnapshotKey = key;
}

async function restoreIndex(env) {
  if (index) return false;
  if (!env?.INDEX_BUCKET && localSnapshot) {
    const start = performance.now();
    const restored = await Pancake.create(localSnapshot.config);
    try {
      restored.import(localSnapshot.bytes);
    } catch (error) {
      restored.dispose();
      throw error;
    }
    index = restored;
    indexConfig = { ...localSnapshot.config };
    restoreState.restoreCount += 1;
    restoreState.restoredAt = new Date().toISOString();
    restoreState.lastRestoreMs = performance.now() - start;
    restoreState.lastFetchMs = 0;
    restoreState.lastDeserializeMs = restoreState.lastRestoreMs;
    restoreState.lastSnapshotBytes = localSnapshot.bytes.byteLength;
    restoreState.lastSnapshotKey = localSnapshot.key;
    return true;
  }
  if (!env?.INDEX_BUCKET) return false;
  const key = await findLatestSnapshot(env);
  if (!key) return false;

  const fetchStart = performance.now();
  const obj = await env.INDEX_BUCKET.get(key);
  if (!obj) return false;
  const bytes = new Uint8Array(await obj.arrayBuffer());
  const fetchMs = performance.now() - fetchStart;
  const maxBytes = positiveEnvInt(env, 'MAX_SNAPSHOT_BYTES', DEFAULT_MAX_SNAPSHOT_BYTES);
  if (bytes.byteLength > maxBytes) throw new Error(`Snapshot exceeds MAX_SNAPSHOT_BYTES (${bytes.byteLength} > ${maxBytes})`);

  const metadata = {};
  for (const [key, value] of Object.entries(obj.customMetadata || {})) {
    try { metadata[key] = JSON.parse(value); } catch { metadata[key] = value; }
  }
  const envelopeMetadata = parsePancakeEnvelopeMetadata(bytes);
  const config = configFromMetadata(metadata, envelopeMetadata?.dims || 0, env);
  if (envelopeMetadata) {
    config.dim = envelopeMetadata.dims;
    config.metric = envelopeMetadata.metric;
    config.quantized = envelopeMetadata.quantized;
  }

  const deserializeStart = performance.now();
  const restored = await Pancake.create(config);
  try {
    restored.import(bytes);
  } catch (error) {
    restored.dispose();
    throw error;
  }

  index = restored;
  indexConfig = config;
  restoreState.restoreCount += 1;
  restoreState.restoredAt = new Date().toISOString();
  restoreState.lastRestoreMs = performance.now() - fetchStart;
  restoreState.lastFetchMs = fetchMs;
  restoreState.lastDeserializeMs = performance.now() - deserializeStart;
  restoreState.lastSnapshotBytes = bytes.byteLength;
  restoreState.lastSnapshotKey = key;
  return true;
}

async function ensureIndex(env) {
  if (index) return true;
  return restoreIndex(env);
}

function healthBody(env) {
  return {
    ok: true,
    initialized: !!index,
    dims: indexConfig?.dim ?? null,
    count: index ? index.count : 0,
    read_only: isReadOnly(env),
    restore: restoreState
  };
}

function statsBody() {
  return {
    dims: indexConfig.dim,
    count: index.count,
    memory: index.memory,
    ghost_count: index.ghostCount,
    ghost_ratio: index.ghostRatio
  };
}

function applyEf(index, ef) {
  if (ef === undefined || ef === null) return;
  const parsed = Number.parseInt(ef, 10);
  if (Number.isInteger(parsed) && parsed > 0 && typeof index._setEfSearch === 'function') {
    index._setEfSearch(Math.min(parsed, MAX_EF));
  }
}

async function handleRequest(request, env, ctx) {
  const url = new URL(request.url);
  const method = request.method.toUpperCase();

  if (method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: corsHeaders(env) });
  }

  const adminRoute = ADMIN_ROUTES.has(url.pathname);
  if (adminRoute && isReadOnly(env)) {
    return jsonResponse({ error: 'Read-only mode' }, 403);
  }
  if (url.pathname !== '/health' && !isAdminAllowed(request, env)) {
    return jsonResponse({ error: 'Unauthorized' }, 401);
  }
  if (url.pathname !== '/health' && isRateLimited(request, env)) {
    return jsonResponse({ error: 'Rate limit exceeded' }, 429);
  }

  if (url.pathname === '/health' && method === 'GET') {
    return jsonResponse(healthBody(env));
  }

  if (url.pathname === '/readiness' && method === 'GET') {
    const snapshotKey = await findLatestSnapshot(env);
    return jsonResponse({ ...healthBody(env), snapshot_available: !!snapshotKey, snapshot_key: snapshotKey });
  }

  if (url.pathname === '/reset_cache' && method === 'POST') {
    if (index) index.dispose();
    index = null;
    indexConfig = null;
    return jsonResponse({ ok: true, reset: true });
  }

  if (url.pathname === '/init' && method === 'POST') {
    const body = await readJson(request, env);
    const config = makeCreateConfig(body, env);
    const next = await Pancake.create(config);
    try {
      if (Array.isArray(body.vectors) && body.vectors.length > 0) {
        next.addBatch(body.vectors);
      }
    } catch (error) {
      next.dispose();
      throw error;
    }
    if (index) index.dispose();
    index = next;
    indexConfig = config;
    await persistIndex(env);
    return jsonResponse({ ok: true, dims: config.dim, count: index.count, inserted: Array.isArray(body.vectors) ? body.vectors.length : 0 });
  }

  if (url.pathname === '/import' && method === 'POST') {
    const bytes = await readBytes(request, env);
    const envelopeMetadata = parsePancakeEnvelopeMetadata(bytes);
    const dims = envelopeMetadata?.dims || Number.parseInt(url.searchParams.get('dims') || '0', 10);
    const config = makeCreateConfig({
      dims,
      maxElements: Number.parseInt(url.searchParams.get('maxElements') || String(DEFAULT_MAX_ELEMENTS), 10),
      metric: envelopeMetadata?.metric || url.searchParams.get('metric') || 'cosine',
      quantized: envelopeMetadata?.quantized ?? true,
      M: Number.parseInt(url.searchParams.get('M') || '16', 10),
      efConstruction: Number.parseInt(url.searchParams.get('efConstruction') || '200', 10),
      efSearch: Number.parseInt(url.searchParams.get('efSearch') || '100', 10)
    }, env);
    const next = await Pancake.create(config);
    try {
      next.import(bytes);
    } catch (error) {
      next.dispose();
      throw error;
    }
    if (index) index.dispose();
    index = next;
    indexConfig = config;
    await persistIndex(env);
    return jsonResponse({ ok: true, count: index.count, dims: config.dim });
  }

  if (url.pathname === '/stats' && method === 'GET') {
    if (!await ensureIndex(env)) return jsonResponse({ error: 'Index not initialized. Call /init first.' }, 503);
    return jsonResponse(statsBody());
  }

  if (url.pathname === '/add' && method === 'POST') {
    if (!await ensureIndex(env)) return jsonResponse({ error: 'Index not initialized. Call /init first.' }, 503);
    const body = await readJson(request, env);
    const id = index.add(body.vector);
    await persistIndex(env);
    return jsonResponse({ ok: true, id, count: index.count });
  }

  if (url.pathname === '/add_batch' && method === 'POST') {
    if (!await ensureIndex(env)) return jsonResponse({ error: 'Index not initialized. Call /init first.' }, 503);
    const body = await readJson(request, env);
    if (!Array.isArray(body.vectors)) throw new Error('add_batch requires vectors');
    const ids = index.addBatch(body.vectors);
    await persistIndex(env);
    return jsonResponse({ ok: true, ids, inserted: ids.length, count: index.count });
  }

  if (url.pathname === '/delete' && method === 'POST') {
    if (!await ensureIndex(env)) return jsonResponse({ error: 'Index not initialized. Call /init first.' }, 503);
    const body = await readJson(request, env);
    index.delete(body.id);
    await persistIndex(env);
    return jsonResponse({ ok: true, count: index.count, ghost_count: index.ghostCount });
  }

  if (url.pathname === '/compact' && method === 'POST') {
    if (!await ensureIndex(env)) return jsonResponse({ error: 'Index not initialized. Call /init first.' }, 503);
    index.compact();
    await persistIndex(env);
    return jsonResponse({ ok: true, count: index.count, ghost_count: index.ghostCount });
  }

  if ((url.pathname === '/search' || url.pathname === '/search_debug') && method === 'POST') {
    if (!await ensureIndex(env)) return jsonResponse({ error: 'Index not initialized. Call /init first.' }, 503);
    const body = await readJson(request, env);
    const k = Math.min(Number.parseInt(body.k ?? '10', 10), MAX_RESULTS);
    applyEf(index, body.ef);
    const start = performance.now();
    const neighbors = body.allowedIds
      ? index.searchFiltered(body.query, k, new Set(body.allowedIds))
      : index.search(body.query, k);
    return jsonResponse({
      ok: true,
      neighbors,
      results: neighbors,
      search_ms: performance.now() - start,
      count: index.count
    });
  }

  if (url.pathname === '/export' && method === 'GET') {
    if (!await ensureIndex(env)) return jsonResponse({ error: 'Index not initialized. Call /init first.' }, 503);
    if (index.ghostCount > 0) index.compact();
    return binaryResponse(index.export());
  }

  return jsonResponse({
    ok: true,
    routes: {
      'GET /health': 'Public health check',
      'GET /readiness': 'Authenticated snapshot/readiness check',
      'GET /stats': 'Index stats',
      'POST /init': '{ dims, maxElements, M?, efConstruction?, efSearch?, vectors? }',
      'POST /search': '{ query: float[], k?, ef?, allowedIds? }',
      'POST /add': '{ vector: float[] }',
      'POST /add_batch': '{ vectors: float[][] }',
      'POST /delete': '{ id: number }',
      'POST /compact': 'Compact deleted entries',
      'GET /export': 'Export Pancake snapshot',
      'POST /import': 'Import Pancake snapshot',
      'POST /reset_cache': 'Drop warm in-memory index'
    }
  });
}

export default {
  async fetch(request, env, ctx) {
    try {
      return withCors(await handleRequest(request, env, ctx), env);
    } catch (error) {
      const message = error && error.message ? error.message : String(error);
      return withCors(jsonResponse({ error: message }, 400), env);
    }
  }
};
