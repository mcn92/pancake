/**
 * Pancake Search - Cloudflare Workers Entrypoint
 *
 * API:
 *   POST /search       { query: float[], k?: int, ef?: int }
 *
 *   POST /init         { dims, maxElements, M?, efConstruction?, efSearch?, vectors? }
 *   POST /add          { vector: float[] }
 *   POST /import       <binary body>  ?dims=<n>
 *
 *   GET  /export
 *   GET  /health
 *
 * All dims route through the unified pancake_init handle-based API.
 * Backend dispatch (QuantizedHNSW template vs Int8FloatHNSW runtime) is
 * handled internally by the WASM engine.
 */

import P from '../../dist/engine.js';
import wasmModule from '../../dist/engine.wasm';

let pancake = null;
let index = null;
const MAX_RESULTS = 100;
const MAX_DIMS = 4096;
const MAX_ELEMENTS = 5_000;
const MAX_EF = 2000;
const MAX_M = 128;

// ---------------------------------------------------------------------------
// Rate limiting — per-IP sliding window (in-memory, resets on cold start)
// ---------------------------------------------------------------------------

const rateLimitMap = new Map();
const RATE_LIMIT_WINDOW_MS = 60_000;

function isRateLimited(ip, maxRpm) {
  const now = Date.now();
  let entry = rateLimitMap.get(ip);
  if (!entry) {
    entry = { timestamps: [] };
    rateLimitMap.set(ip, entry);
  }
  entry.timestamps = entry.timestamps.filter(t => now - t < RATE_LIMIT_WINDOW_MS);
  if (entry.timestamps.length >= maxRpm) return true;
  entry.timestamps.push(now);
  return false;
}

let lastCleanup = Date.now();
function cleanupRateLimits() {
  const now = Date.now();
  if (now - lastCleanup < RATE_LIMIT_WINDOW_MS) return;
  lastCleanup = now;
  for (const [ip, entry] of rateLimitMap) {
    entry.timestamps = entry.timestamps.filter(t => now - t < RATE_LIMIT_WINDOW_MS);
    if (entry.timestamps.length === 0) rateLimitMap.delete(ip);
  }
}

// ---------------------------------------------------------------------------
// R2 persistence helpers
// ---------------------------------------------------------------------------

let persistTimer = null;
const PERSIST_DEBOUNCE_MS = 2000;

async function persistIndex(env) {
  console.log("persistIndex called, INDEX_BUCKET=" + !!env.INDEX_BUCKET + " index=" + !!index);
  if (!env.INDEX_BUCKET || !index) return;
  try {
    const bytes = index.exportBinary();
    const metadata = { dims: String(index.dims), count: String(index.count), savedAt: new Date().toISOString() };
    await env.INDEX_BUCKET.put('pancake-index.bin', bytes, {
      customMetadata: metadata,
      httpMetadata: { contentType: 'application/octet-stream' }
    });
  } catch (e) {
    console.error('R2 persist failed:', e);
  }
}

function schedulePersist(env, ctx) {
  if (!env.INDEX_BUCKET) return;
  if (persistTimer) clearTimeout(persistTimer);
  persistTimer = setTimeout(() => {
    persistTimer = null;
    ctx.waitUntil(persistIndex(env));
  }, PERSIST_DEBOUNCE_MS);
}

async function restoreIndex(env) {
  if (!env.INDEX_BUCKET || index) return false;
  try {
    const obj = await env.INDEX_BUCKET.get('pancake-index.bin');
    if (!obj) return false;

    const dims = parseInt(obj.customMetadata?.dims ?? '0', 10);
    if (!dims) return false;

    const buffer = await obj.arrayBuffer();
    const engine = await initializePancake();

    const handle = engine._pancake_init(dims, MAX_ELEMENTS, 1, 1, 8, 150, 100);
    if (handle === 0xFFFFFFFF) return false;

    const buf = engine._emsc_malloc(buffer.byteLength);
    if (!buf) { engine._pancake_dispose(handle); return false; }

    try {
      engine.HEAPU8.set(new Uint8Array(buffer), buf);
      const status = engine._pancake_import(handle, buf, buffer.byteLength);
      if (status !== 0) { engine._pancake_dispose(handle); return false; }
    } finally {
      engine._emsc_free(buf);
    }

    const count = engine._pancake_count(handle);
    index = buildIndexWrapper(engine, dims, MAX_ELEMENTS, handle);
    for (let i = 0; i < count; i++) index._seedId(i);
    console.log(`Restored index from R2: dims=${dims}, count=${count}`);
    return true;
  } catch (e) {
    console.error('R2 restore failed:', e);
    return false;
  }
}

// ---------------------------------------------------------------------------
// WASM bootstrap
// ---------------------------------------------------------------------------

async function initializePancake() {
  if (pancake) return pancake;

  pancake = await P({
    instantiateWasm(imports, successCallback) {
      WebAssembly.instantiate(wasmModule, imports)
        .then(instance => successCallback(instance))
        .catch(err => { throw err; });
      return {};
    }
  });

  const required = [
    '_pancake_init', '_pancake_add', '_pancake_query', '_pancake_count',
    '_pancake_memory', '_pancake_set_ef', '_pancake_export', '_pancake_import',
    '_pancake_delete', '_pancake_compact', '_pancake_ghost_count',
    '_pancake_ghost_ratio', '_pancake_dispose',
    '_emsc_malloc', '_emsc_free'
  ];
  for (const fn of required) {
    if (typeof pancake[fn] !== 'function') {
      throw new Error(`WASM binding missing: ${fn}`);
    }
  }

  return pancake;
}

// ---------------------------------------------------------------------------
// Index wrapper
// ---------------------------------------------------------------------------

function buildIndexWrapper(engine, dims, maxElements, handle, initParams = {}) {
  const vecBuffer   = engine._emsc_malloc(dims * 4);
  const idsBuffer   = engine._emsc_malloc(MAX_RESULTS * 8);
  const distsBuffer = engine._emsc_malloc(MAX_RESULTS * 4);

  if (!vecBuffer || !idsBuffer || !distsBuffer) {
    throw new Error('WASM heap allocation failed');
  }

  const _initParams = { M: initParams.M || 8, efC: initParams.efC || 150, efS: initParams.efS || 100 };

  const _extToInt = new Map();
  const _intToExt = new Map();
  const _deletedExt = new Set();
  let _nextExtId = 0;

  const _vectors = new Map();

  return {
    engine,
    dims,
    maxElements,
    handle,
    _vecBuffer: vecBuffer,
    _idsBuffer: idsBuffer,
    _distsBuffer: distsBuffer,

    add(vec) {
      engine.HEAPF32.set(vec, this._vecBuffer >> 2);
      const intId = engine._pancake_add(this.handle, this._vecBuffer);
      if (intId === 0xFFFFFFFF || intId < 0) {
        throw new Error('Insert failed (index full or not initialized)');
      }
      const extId = _nextExtId++;
      _extToInt.set(extId, intId);
      _intToExt.set(intId, extId);
      _vectors.set(extId, new Float32Array(vec));
      return extId;
    },

    search(vec, k, ef = 50) {
      engine._pancake_set_ef(this.handle, ef);
      engine.HEAPF32.set(vec, this._vecBuffer >> 2);

      const found = engine._pancake_query(
        this.handle, this._vecBuffer, k, this._idsBuffer, this._distsBuffer);

      const neighbors = [];
      const distances = [];
      const dv = new DataView(engine.HEAPU8.buffer);

      for (let i = 0; i < found; i++) {
        const lo = dv.getUint32(this._idsBuffer + i * 8, true);
        const hi = dv.getUint32(this._idsBuffer + i * 8 + 4, true);
        const intId = hi * 0x100000000 + lo;
        const extId = _intToExt.get(intId);
        if (extId !== undefined) {
          neighbors.push(extId);
          distances.push(engine.HEAPF32[this._distsBuffer / 4 + i]);
        }
      }

      return { neighbors, distances };
    },

    searchDebug(vec, k, ef = 50) {
      engine._pancake_set_ef(this.handle, ef);
      engine.HEAPF32.set(vec, this._vecBuffer >> 2);
      const found = engine._pancake_query(
        this.handle, this._vecBuffer, k, this._idsBuffer, this._distsBuffer);
      const raw = [], translated = [], dists = [];
      const dv = new DataView(engine.HEAPU8.buffer);
      for (let i = 0; i < found; i++) {
        const lo = dv.getUint32(this._idsBuffer + i * 8, true);
        const hi = dv.getUint32(this._idsBuffer + i * 8 + 4, true);
        const intId = hi * 0x100000000 + lo;
        raw.push(intId);
        translated.push(_intToExt.get(intId));
        dists.push(engine.HEAPF32[this._distsBuffer / 4 + i]);
      }
      return { raw, translated, dists, mapSize: _intToExt.size, nextExtId: _nextExtId };
    },

    exportBinary() {
      const sizePtr = engine._emsc_malloc(8);
      if (!sizePtr) throw new Error('Allocation failed');
      try {
        const dataPtr = engine._pancake_export(this.handle, sizePtr);
        if (!dataPtr) throw new Error('export returned null');

        const size = engine.HEAPU8[sizePtr]       |
                    (engine.HEAPU8[sizePtr + 1] << 8)  |
                    (engine.HEAPU8[sizePtr + 2] << 16) |
                    (engine.HEAPU8[sizePtr + 3] << 24);

        if (size === 0) throw new Error('Export produced 0 bytes');
        return engine.HEAPU8.slice(dataPtr, dataPtr + size);
      } finally {
        engine._emsc_free(sizePtr);
      }
    },

    delete(extId) {
      const intId = _extToInt.get(extId);
      if (intId === undefined) return;
      engine._pancake_delete(this.handle, intId);
      _deletedExt.add(extId);
      _vectors.delete(extId);
    },

    compact() {
      const survivors = [];
      for (const [extId, vec] of _vectors) {
        if (!_deletedExt.has(extId)) {
          survivors.push({ extId, vec });
        }
      }

      if (_vectors.size === 0) {
        const oldSurvivors = [];
        for (const [extId, intId] of _extToInt) {
          if (!_deletedExt.has(extId)) oldSurvivors.push({ extId, intId });
        }
        oldSurvivors.sort((a, b) => a.intId - b.intId);
        engine._pancake_compact(this.handle);
        _extToInt.clear();
        _intToExt.clear();
        _deletedExt.clear();
        for (let newInt = 0; newInt < oldSurvivors.length; newInt++) {
          _extToInt.set(oldSurvivors[newInt].extId, newInt);
          _intToExt.set(newInt, oldSurvivors[newInt].extId);
        }
        return;
      }

      if (survivors.length === 0) {
        engine._pancake_compact(this.handle);
        _extToInt.clear();
        _intToExt.clear();
        _deletedExt.clear();
        _vectors.clear();
        return;
      }

      // Dispose old handle and create fresh one for rebuild
      engine._pancake_dispose(this.handle);
      this.handle = engine._pancake_init(
        dims, maxElements, 1, 1, _initParams.M, _initParams.efC, _initParams.efS);

      _extToInt.clear();
      _intToExt.clear();
      _deletedExt.clear();

      for (const { extId, vec } of survivors) {
        engine.HEAPF32.set(vec, this._vecBuffer >> 2);
        const newIntId = engine._pancake_add(this.handle, this._vecBuffer);
        _extToInt.set(extId, newIntId);
        _intToExt.set(newIntId, extId);
      }

      _vectors.clear();
      for (const { extId, vec } of survivors) {
        _vectors.set(extId, vec);
      }
    },

    _seedId(intId) {
      const extId = _nextExtId++;
      _extToInt.set(extId, intId);
      _intToExt.set(intId, extId);
    },

    ghostCount() { return engine._pancake_ghost_count(this.handle); },
    ghostRatio() { return engine._pancake_ghost_ratio(this.handle); },

    get count() { return engine._pancake_count(this.handle); },
    get memory() { return engine._pancake_memory(this.handle); },
    get nextId() { return _nextExtId; },

    destroy() {
      engine._pancake_dispose(this.handle);
      engine._emsc_free(this._vecBuffer);
      engine._emsc_free(this._idsBuffer);
      engine._emsc_free(this._distsBuffer);
    }
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function jsonResponse(body, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...extraHeaders }
  });
}

function isPositiveInteger(v) {
  return Number.isInteger(v) && v > 0;
}

async function safeParseJson(request) {
  try {
    return await request.json();
  } catch {
    return null;
  }
}

function validateVector(vector, dims, fieldName) {
  if (!Array.isArray(vector))
    return `${fieldName} must be an array`;
  if (vector.length !== dims)
    return `${fieldName} must have exactly ${dims} elements`;
  for (const v of vector) {
    if (typeof v !== 'number' || !Number.isFinite(v))
      return `${fieldName} must contain only finite numbers`;
  }
  return null;
}

function getCorsOrigin(env) {
  return env.ALLOWED_ORIGIN || '*';
}

function corsHeaders(env) {
  return { 'Access-Control-Allow-Origin': getCorsOrigin(env) };
}

// ---------------------------------------------------------------------------
// Request handler
// ---------------------------------------------------------------------------

async function handleRequest(request, env, ctx) {
  const url = new URL(request.url);
  const method = request.method;
  const origin = getCorsOrigin(env);

  if (method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: {
        'Access-Control-Allow-Origin': origin,
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization'
      }
    });
  }

  if (env.API_KEY && url.pathname !== '/health') {
    const authHeader = request.headers.get('Authorization') || '';
    const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : '';
    if (token !== env.API_KEY) {
      return jsonResponse({ error: 'Unauthorized' }, 401);
    }
  }

  const maxRpm = parseInt(env.RATE_LIMIT_RPM || '0', 10);
  if (maxRpm > 0 && url.pathname !== '/health') {
    cleanupRateLimits();
    const ip = request.headers.get('CF-Connecting-IP') || request.headers.get('X-Forwarded-For') || 'unknown';
    if (isRateLimited(ip, maxRpm)) {
      return jsonResponse({ error: 'Rate limit exceeded. Try again later.' }, 429);
    }
  }

  if (!index && env.INDEX_BUCKET) {
    await restoreIndex(env);
  }

  if (url.pathname === '/health' && method === 'GET') {
    return jsonResponse({
      status: 'ok',
      initialized: index !== null,
      count: index ? index.count : 0,
      memory_bytes: index ? index.memory : 0,
      dims: index ? index.dims : null
    });
  }

  if (url.pathname === '/init' && method === 'POST') {
    const body = await safeParseJson(request);
    if (!body) return jsonResponse({ error: 'Invalid JSON' }, 400);
    const { dims, maxElements, M = 8, efConstruction = 150, efSearch = 100, vectors = [] } = body;

    if (!isPositiveInteger(dims) || dims > MAX_DIMS)
      return jsonResponse({ error: `dims must be an integer between 1 and ${MAX_DIMS}` }, 400);
    if (!isPositiveInteger(maxElements) || maxElements > MAX_ELEMENTS)
      return jsonResponse({ error: `maxElements must be an integer between 1 and ${MAX_ELEMENTS}` }, 400);
    if (!isPositiveInteger(M) || M > MAX_M)
      return jsonResponse({ error: `M must be an integer between 1 and ${MAX_M}` }, 400);
    if (!isPositiveInteger(efConstruction) || efConstruction > MAX_EF)
      return jsonResponse({ error: `efConstruction must be an integer between 1 and ${MAX_EF}` }, 400);
    if (!isPositiveInteger(efSearch) || efSearch > MAX_EF)
      return jsonResponse({ error: `efSearch must be an integer between 1 and ${MAX_EF}` }, 400);
    if (!Array.isArray(vectors))
      return jsonResponse({ error: 'vectors must be an array' }, 400);
    if (vectors.length > maxElements)
      return jsonResponse({ error: 'vectors length cannot exceed maxElements' }, 400);

    const engine = await initializePancake();
    if (index) { index.destroy(); index = null; }

    // quantized=1, metric=1 (cosine)
    const handle = engine._pancake_init(dims, maxElements, 1, 1, M, efConstruction, efSearch);
    if (handle === 0xFFFFFFFF) {
      return jsonResponse({ error: 'Backend init failed' }, 500);
    }

    index = buildIndexWrapper(engine, dims, maxElements, handle, { M, efC: efConstruction, efS: efSearch });

    let inserted = 0;
    for (const vec of vectors) {
      const err = validateVector(vec, dims, 'vectors[]');
      if (err) { index.destroy(); index = null; return jsonResponse({ error: err }, 400); }
      index.add(new Float32Array(vec));
      inserted++;
    }

    await persistIndex(env);
    return jsonResponse({
      status: 'initialized',
      dims,
      maxElements,
      M,
      efConstruction,
      efSearch,
      inserted,
      memory_bytes: index.memory
    });
  }

  if (url.pathname === '/add' && method === 'POST') {
    if (!index) return jsonResponse({ error: 'Index not initialized. Call /init first.' }, 503);
    if (index.count >= index.maxElements)
      return jsonResponse({ error: 'Index is at capacity' }, 409);

    const body = await safeParseJson(request);
    if (!body) return jsonResponse({ error: 'Invalid JSON' }, 400);
    const err = validateVector(body.vector, index.dims, 'vector');
    if (err) return jsonResponse({ error: err }, 400);

    let id;
    try {
      id = index.add(new Float32Array(body.vector));
    } catch (e) {
      if (e.message.includes('Insert failed')) {
        return jsonResponse({ error: 'Index is at capacity' }, 409);
      }
      throw e;
    }
    await persistIndex(env);
    return jsonResponse({ id, count: index.count });
  }

  if (url.pathname === '/delete' && method === 'POST') {
    if (!index) return jsonResponse({ error: 'Index not initialized.' }, 503);
    const body = await safeParseJson(request);
    if (!body) return jsonResponse({ error: 'Invalid JSON' }, 400);
    const { id } = body;
    if (typeof id !== 'number' || !Number.isInteger(id) || id < 0)
      return jsonResponse({ error: 'id must be a non-negative integer' }, 400);
    index.delete(id);
    await persistIndex(env);
    return jsonResponse({ deleted: id, ghost_count: index.ghostCount(), ghost_ratio: index.ghostRatio() });
  }

  if (url.pathname === '/compact' && method === 'POST') {
    if (!index) return jsonResponse({ error: 'Index not initialized.' }, 503);
    const t0 = performance.now();
    index.compact();
    const elapsed = performance.now() - t0;
    await persistIndex(env);
    return jsonResponse({ compacted: true, elapsed_ms: elapsed, count: index.count, memory_bytes: index.memory });
  }

  if (url.pathname === '/stats' && method === 'GET') {
    if (!index) return jsonResponse({ error: 'Index not initialized.' }, 503);
    return jsonResponse({
      count: index.count,
      memory_bytes: index.memory,
      ghost_count: index.ghostCount(),
      ghost_ratio: index.ghostRatio(),
      next_id: index.nextId,
      dims: index.dims
    });
  }

  if (url.pathname === '/add_batch' && method === 'POST') {
    if (!index) return jsonResponse({ error: 'Index not initialized. Call /init first.' }, 503);

    const body = await safeParseJson(request);
    if (!body) return jsonResponse({ error: 'Invalid JSON' }, 400);
    const { vectors } = body;
    if (!Array.isArray(vectors) || vectors.length === 0)
      return jsonResponse({ error: 'vectors must be a non-empty array' }, 400);

    const ids = [];
    for (let i = 0; i < vectors.length; i++) {
      const err = validateVector(vectors[i], index.dims, `vectors[${i}]`);
      if (err) return jsonResponse({ error: err }, 400);
      try {
        ids.push(index.add(new Float32Array(vectors[i])));
      } catch (e) {
        if (e.message.includes('Insert failed')) {
          return jsonResponse({
            error: 'Index is at capacity',
            inserted: ids.length,
            ids,
            count: index.count
          }, 409);
        }
        throw e;
      }
    }

    await persistIndex(env);
    return jsonResponse({ inserted: ids.length, ids, count: index.count });
  }

  if (url.pathname === '/search' && method === 'POST') {
    if (!index) return jsonResponse({ error: 'Index not initialized. Call /init first.' }, 503);

    const t0 = performance.now();
    const body = await safeParseJson(request);
    if (!body) return jsonResponse({ error: 'Invalid JSON' }, 400);
    const { query, k = 10, ef = 100 } = body;

    const err = validateVector(query, index.dims, 'query');
    if (err) return jsonResponse({ error: err }, 400);

    if (!Number.isInteger(k) || k < 1 || k > MAX_RESULTS)
      return jsonResponse({ error: `k must be an integer between 1 and ${MAX_RESULTS}` }, 400);
    if (!isPositiveInteger(ef) || ef > MAX_EF)
      return jsonResponse({ error: `ef must be an integer between 1 and ${MAX_EF}` }, 400);

    const results = index.search(new Float32Array(query), k, ef);
    const latency_ms = performance.now() - t0;

    return jsonResponse({ ...results, latency_ms }, 200, corsHeaders(env));
  }

  if (url.pathname === '/search_debug' && method === 'POST') {
    if (!index) return jsonResponse({ error: 'Index not initialized.' }, 503);
    const body = await safeParseJson(request);
    if (!body) return jsonResponse({ error: 'Invalid JSON' }, 400);
    const { query, k = 5 } = body;

    const err = validateVector(query, index.dims, 'query');
    if (err) return jsonResponse({ error: err }, 400);
    if (!Number.isInteger(k) || k < 1 || k > MAX_RESULTS)
      return jsonResponse({ error: `k must be an integer between 1 and ${MAX_RESULTS}` }, 400);

    const results = index.searchDebug(new Float32Array(query), k);
    return jsonResponse(results);
  }

  if (url.pathname === '/export' && method === 'GET') {
    if (!index) return jsonResponse({ error: 'No index to export' }, 503);

    const bytes = index.exportBinary();
    return new Response(bytes, {
      status: 200,
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Length': String(bytes.byteLength),
        'X-Pancake-Dims': String(index.dims),
        'X-Pancake-Count': String(index.count)
      }
    });
  }

  if (url.pathname === '/import' && method === 'POST') {
    const dims = parseInt(url.searchParams.get('dims') ?? '', 10);
    if (!isPositiveInteger(dims) || dims > MAX_DIMS)
      return jsonResponse({ error: `dims query param required and must be between 1 and ${MAX_DIMS}` }, 400);

    const buffer = await request.arrayBuffer();
    const engine = await initializePancake();

    if (index) { index.destroy(); index = null; }

    const handle = engine._pancake_init(dims, MAX_ELEMENTS, 1, 1, 8, 150, 100);
    if (handle === 0xFFFFFFFF) {
      return jsonResponse({ error: 'Backend init failed' }, 500);
    }

    const buf = engine._emsc_malloc(buffer.byteLength);
    if (!buf) {
      engine._pancake_dispose(handle);
      return jsonResponse({ error: 'WASM heap allocation failed' }, 500);
    }

    try {
      engine.HEAPU8.set(new Uint8Array(buffer), buf);
      const status = engine._pancake_import(handle, buf, buffer.byteLength);
      if (status !== 0) {
        engine._pancake_dispose(handle);
        return jsonResponse({ error: 'Import failed' }, 500);
      }
    } finally {
      engine._emsc_free(buf);
    }

    index = buildIndexWrapper(engine, dims, MAX_ELEMENTS, handle);
    const importedCount = index.count;
    for (let i = 0; i < importedCount; i++) index._seedId(i);

    await persistIndex(env);
    return jsonResponse({
      status: 'imported',
      dims,
      count: index.count,
      memory_bytes: index.memory
    });
  }

  return jsonResponse({
    name: 'Pancake Search API',
    version: '1.0.0',
    endpoints: {
      'GET  /health':       'Health check and stats',
      'GET  /export':       'Export index as binary blob',
      'POST /init':         '{ dims, maxElements, M?, efConstruction?, efSearch?, vectors? }',
      'POST /import':       'Binary body from /export — ?dims=<n> required',
      'POST /add':          '{ vector: float[] }',
      'POST /search':       '{ query: float[], k?, ef? }'
    }
  });
}

export default {
  async fetch(request, env, ctx) {
    try {
      return await handleRequest(request, env, ctx);
    } catch (error) {
      console.error('Unhandled error:', error);
      return jsonResponse({ error: 'Internal server error' }, 500);
    }
  }
};
