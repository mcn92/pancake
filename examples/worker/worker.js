/**
 * Pancake Search - Cloudflare Workers Entrypoint
 *
 * API:
 *   POST /search       { query: float[], k?: int, ef?: int }
 *   POST /search_text  { text: string, k?: int, ef?: int }
 *   POST /init         { dims, maxElements, M?, efConstruction?, efSearch?, vectors? }
 *   POST /add          { vector: float[] }
 *   POST /import       <binary body>  ?dims=<n>
 *   POST /upload_model <multipart: vocab, model, ngram, embeddings, metadata>
 *   GET  /export
 *   GET  /health
 *
 * Backend routing by dims:
 *   dims=1536 → p1536_* (QuantizedHNSW<1536>, dedicated SIMD path)
 *   dims=384  → pi/pa/pq/pc/pm (QuantizedHNSW<384>, original path)
 *   other     → i8_* (Int8FloatHNSW runtime path)
 */

import P from '../../dist/engine.js';
import wasmModule from '../../dist/engine.wasm';

let pancake = null;
let index = null;
const MAX_RESULTS = 100;
const MAX_DIMS = 4096;
const MAX_ELEMENTS = 1_000_000;
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
  // Evict timestamps outside the window
  entry.timestamps = entry.timestamps.filter(t => now - t < RATE_LIMIT_WINDOW_MS);
  if (entry.timestamps.length >= maxRpm) return true;
  entry.timestamps.push(now);
  return false;
}

// Periodic cleanup to avoid unbounded memory growth
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
// Hybrid embedding model — TF-IDF weighted word + n-gram embeddings
// Ported from demo_hybrid_search.js
// ---------------------------------------------------------------------------

let hybridModel = null; // cached after first load

function extractNgrams(word, minN = 3, maxN = 4) {
  const ngrams = [];
  const padded = `^${word}$`;
  for (let n = minN; n <= maxN; n++) {
    for (let i = 0; i <= padded.length - n; i++) {
      ngrams.push(padded.slice(i, i + n));
    }
  }
  return ngrams;
}

function computeTFIDF_Hybrid(text, vocabData) {
  const { vocab, idf, ngram_vocab, ngram_idf } = vocabData;
  const tokens = text.toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 0);

  const tf = {};
  const tfNgram = {};
  const matchedWords = [];
  const unmatchedWords = [];

  for (const token of tokens) {
    if (vocab[token] !== undefined) {
      tf[token] = (tf[token] || 0) + 1;
      matchedWords.push(token);
    } else if (!/^\d+$/.test(token) && token.length >= 3) {
      unmatchedWords.push(token);
      const ngrams = extractNgrams(token, 3, 4);
      for (const ng of ngrams) {
        if (ngram_vocab[ng] !== undefined) {
          tfNgram[ng] = (tfNgram[ng] || 0) + 1;
        }
      }
    } else {
      unmatchedWords.push(token);
    }
  }

  const wordIndices = [];
  const wordValues = [];
  const totalTokens = tokens.length;

  for (const [word, count] of Object.entries(tf)) {
    const idx = vocab[word];
    const val = (count / totalTokens) * idf[idx];
    wordIndices.push(idx);
    wordValues.push(val);
  }

  const ngramIndices = [];
  const ngramValues = [];
  const NGRAM_WEIGHT = 0.5;

  for (const [ngram, count] of Object.entries(tfNgram)) {
    const idx = ngram_vocab[ngram];
    const val = (count / totalTokens) * ngram_idf[idx] * NGRAM_WEIGHT;
    ngramIndices.push(idx);
    ngramValues.push(val);
  }

  return { wordIndices, wordValues, ngramIndices, ngramValues, matchedWords, unmatchedWords };
}

function sparseMatmul(matrix, indices, values, bias, rows, cols) {
  const result = new Float32Array(rows);
  for (let i = 0; i < rows; i++) result[i] = bias ? bias[i] : 0;
  for (let k = 0; k < indices.length; k++) {
    const j = indices[k];
    const val = values[k];
    for (let i = 0; i < rows; i++) {
      result[i] += matrix[i * cols + j] * val;
    }
  }
  return result;
}

function hybridEmbed(text, model) {
  const { vocabData, wordModel, ngramModel } = model;
  const { wordIndices, wordValues, ngramIndices, ngramValues, matchedWords, unmatchedWords } =
    computeTFIDF_Hybrid(text, vocabData);

  if (wordIndices.length === 0 && ngramIndices.length === 0) return null;

  // Parallel embedding layers
  const hiddenWords  = sparseMatmul(wordModel.w1, wordIndices, wordValues, wordModel.b1, wordModel.hiddenDim, wordModel.vocabSize);
  const hiddenNgrams = sparseMatmul(ngramModel.w1_ngrams, ngramIndices, ngramValues, null, wordModel.hiddenDim, ngramModel.numNgrams);

  // Combine: hidden = hidden_words + alpha * hidden_ngrams
  const hidden = new Float32Array(wordModel.hiddenDim);
  for (let i = 0; i < wordModel.hiddenDim; i++) {
    hidden[i] = hiddenWords[i] + hiddenNgrams[i];
    if (hidden[i] < 0) hidden[i] = 0; // ReLU
  }

  // Dense matmul: W2 @ hidden + b2
  const output = new Float32Array(wordModel.outputDim);
  for (let i = 0; i < wordModel.outputDim; i++) {
    let sum = wordModel.b2[i];
    for (let j = 0; j < wordModel.hiddenDim; j++) sum += wordModel.w2[i * wordModel.hiddenDim + j] * hidden[j];
    output[i] = sum;
  }

  // L2 normalize
  let norm = 0;
  for (let i = 0; i < output.length; i++) norm += output[i] * output[i];
  norm = Math.sqrt(norm);
  if (norm < 1e-6) return null;
  for (let i = 0; i < output.length; i++) output[i] /= norm;

  return output;
}

function parseModelBin(buffer) {
  let offset = 0;
  const view = new DataView(buffer);
  const vocabSize = view.getUint32(offset, true); offset += 4;
  const hiddenDim = view.getUint32(offset, true); offset += 4;
  const outputDim = view.getUint32(offset, true); offset += 4;

  const w1 = new Float32Array(buffer, offset, vocabSize * hiddenDim); offset += vocabSize * hiddenDim * 4;
  const b1 = new Float32Array(buffer, offset, hiddenDim); offset += hiddenDim * 4;
  const w2 = new Float32Array(buffer, offset, outputDim * hiddenDim); offset += outputDim * hiddenDim * 4;
  const b2 = new Float32Array(buffer, offset, outputDim);

  return { vocabSize, hiddenDim, outputDim, w1, b1, w2, b2 };
}

function parseNgramBin(buffer) {
  const view = new DataView(buffer);
  let offset = 0;
  const numNgrams = view.getUint32(offset, true); offset += 4;
  const hiddenDim = view.getUint32(offset, true); offset += 4;
  offset += 4; // reserved

  const w1_ngrams = new Float32Array(buffer, offset, numNgrams * hiddenDim);
  return { numNgrams, hiddenDim, w1_ngrams };
}

function parseEmbeddingsNpy(buffer) {
  // NPY format: 128-byte header then raw float32 data
  return new Float32Array(buffer, 128);
}

async function loadHybridModel(env) {
  if (hybridModel) return hybridModel;
  if (!env.INDEX_BUCKET) throw new Error('INDEX_BUCKET required for hybrid model');

  const [vocabObj, modelObj, ngramObj, embObj, metaObj] = await Promise.all([
    env.INDEX_BUCKET.get('model/vocab_augmented.json'),
    env.INDEX_BUCKET.get('model/model_distilled.bin'),
    env.INDEX_BUCKET.get('model/ngram_embeddings.bin'),
    env.INDEX_BUCKET.get('model/embeddings_distilled.npy'),
    env.INDEX_BUCKET.get('model/metadata.json'),
  ]);

  if (!vocabObj || !modelObj || !ngramObj || !embObj || !metaObj) {
    const missing = [];
    if (!vocabObj) missing.push('vocab_augmented.json');
    if (!modelObj) missing.push('model_distilled.bin');
    if (!ngramObj) missing.push('ngram_embeddings.bin');
    if (!embObj) missing.push('embeddings_distilled.npy');
    if (!metaObj) missing.push('metadata.json');
    throw new Error(`Model files missing from R2 (model/ prefix): ${missing.join(', ')}. Use POST /upload_model to upload them.`);
  }

  const [vocabData, modelBuf, ngramBuf, embBuf, metadataArr] = await Promise.all([
    vocabObj.json(),
    modelObj.arrayBuffer(),
    ngramObj.arrayBuffer(),
    embObj.arrayBuffer(),
    metaObj.json(),
  ]);

  const wordModel = parseModelBin(modelBuf);
  const ngramModel = parseNgramBin(ngramBuf);
  const embeddings = parseEmbeddingsNpy(embBuf);

  hybridModel = { vocabData, wordModel, ngramModel, embeddings, metadata: metadataArr };
  return hybridModel;
}

// ---------------------------------------------------------------------------
// R2 persistence helpers
// ---------------------------------------------------------------------------

let persistTimer = null;
const PERSIST_DEBOUNCE_MS = 2000;

async function persistIndex(env) {
  if (!env.INDEX_BUCKET || !index) return;
  try {
    const bytes = index.exportBinary();
    const metadata = { dims: index.dims, count: index.count, savedAt: new Date().toISOString() };
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
    const buf = engine._emsc_malloc(buffer.byteLength);
    if (!buf) return false;

    try {
      engine.HEAPU8.set(new Uint8Array(buffer), buf);
      index = importIndex(engine, dims, buf, buffer.byteLength);
      console.log(`Restored index from R2: dims=${dims}, count=${index.count}`);
      return true;
    } finally {
      engine._emsc_free(buf);
    }
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
    '_p1536_init', '_p1536_add', '_p1536_query', '_p1536_count', '_p1536_memory', '_p1536_bulk_insert', '_p1536_export_index', '_p1536_import_index',
    '_pi', '_pa', '_pq', '_pc', '_pm', '_export_index', '_import_index',
    '_i8_init', '_i8_add', '_i8_query', '_i8_count', '_i8_memory', '_i8_set_ef',
    '_i8_export_index', '_i8_import_index',
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
// Backend implementations
// ---------------------------------------------------------------------------

function make1536Backend(engine, maxElements) {
  // p1536_init takes (max_elem, M, ef_construction, ef_search, metric)
  // metric: 0=L2, 1=Cosine
  return {
    init(maxElem, M, efC, efS) {
      return engine.ccall('p1536_init', 'number',
        ['number', 'number', 'number', 'number', 'number'],
        [maxElem, M, efC, efS, 1 /* cosine */]);
    },
    add(vecPtr) {
      return engine._p1536_add(vecPtr);
    },
    query(vecPtr, k, idsPtr, distsPtr) {
      return engine.ccall('p1536_query', 'number',
        ['number', 'number', 'number', 'number'],
        [vecPtr, k, idsPtr, distsPtr]);
    },
    count() { return engine._p1536_count(); },
    memory() { return engine._p1536_memory(); },
    setEf(ef) { /* p1536 uses ef from init */ },
    supportsDelete() { return false; },
    supportsCompact() { return false; },
    supportsGhosts() { return false; },
    delete(id) { /* p1536 has no delete export */ },
    ghostCount() { return 0; },
    ghostRatio() { return 0; },
    compact() { /* no compact for p1536 */ },
    exportBinary(sizePtr) {
      return engine._p1536_export_index(sizePtr);
    },
    importBinary(buf, size, dims) {
      return engine.ccall('p1536_import_index', 'number',
        ['number', 'number'],
        [buf, size]);
    }
  };
}

function make384Backend(engine) {
  return {
    init(maxElem, M, efC, efS) {
      // pi only takes max_elem — M/ef are hardcoded in pi()
      engine._pi(maxElem);
      return 0;
    },
    add(vecPtr) { return engine._pa(vecPtr); },
    query(vecPtr, k, idsPtr, distsPtr) {
      return engine._pq(vecPtr, k, idsPtr, distsPtr);
    },
    count() { return engine._pc(); },
    memory() { return engine._pm(); },
    setEf(ef) { /* not configurable in pi/pq */ },
    supportsDelete() { return true; },
    supportsCompact() { return true; },
    supportsGhosts() { return true; },
    delete(id) { engine._pd(id); },
    ghostCount() { return engine._p_ghost_count(); },
    ghostRatio() { return engine._p_ghost_ratio(); },
    compact() { engine._p_compact(); },
    exportBinary(sizePtr) { return engine._export_index(sizePtr); },
    importBinary(buf, size, dims) {
      return engine.ccall('import_index', 'number',
        ['number', 'number'],
        [buf, size]);
    }
  };
}

function makeRuntimeBackend(engine) {
  return {
    init(maxElem, M, efC, efS) {
      return engine.ccall('i8_init', 'number',
        ['number', 'number', 'number', 'number', 'number', 'number'],
        [0 /* dims set separately */, maxElem, 1, M, efC, efS]);
    },
    add(vecPtr) { return engine.ccall('i8_add', 'number', ['number'], [vecPtr]); },
    query(vecPtr, k, idsPtr, distsPtr) {
      return engine.ccall('i8_query', 'number',
        ['number', 'number', 'number', 'number'],
        [vecPtr, k, idsPtr, distsPtr]);
    },
    count() { return engine.ccall('i8_count', 'number', [], []); },
    memory() { return engine.ccall('i8_memory', 'number', [], []); },
    setEf(ef) { engine.ccall('i8_set_ef', null, ['number'], [ef]); },
    supportsDelete() { return true; },
    supportsCompact() { return true; },
    supportsGhosts() { return true; },
    delete(id) { engine.ccall('i8_delete', null, ['number'], [id]); },
    ghostCount() { return engine.ccall('i8_ghost_count', 'number', [], []); },
    ghostRatio() { return engine.ccall('i8_ghost_ratio', 'number', [], []); },
    compact() { engine.ccall('i8_compact', null, [], []); },
    exportBinary(sizePtr) { return engine._i8_export_index(sizePtr); },
    importBinary(buf, size, dims) {
      return engine.ccall('i8_import_index', 'number',
        ['number', 'number', 'number'],
        [buf, size, dims]);
    }
  };
}

// ---------------------------------------------------------------------------
// Index wrapper
// ---------------------------------------------------------------------------

function buildIndexWrapper(engine, dims, maxElements, backend, initParams = {}) {
  const vecBuffer   = engine._emsc_malloc(dims * 4);
  const idsBuffer   = engine._emsc_malloc(MAX_RESULTS * 8);
  const distsBuffer = engine._emsc_malloc(MAX_RESULTS * 4);

  if (!vecBuffer || !idsBuffer || !distsBuffer) {
    throw new Error('WASM heap allocation failed');
  }

  // Store init params so compact can rebuild with the same configuration
  const _initParams = { M: initParams.M || 8, efC: initParams.efC || 150, efS: initParams.efS || 100 };

  // ID translation: external IDs (stable, returned to callers) <-> internal IDs (WASM engine)
  const _extToInt = new Map();   // external -> internal
  const _intToExt = new Map();   // internal -> external
  const _deletedExt = new Set(); // external IDs that are soft-deleted
  let _nextExtId = 0;

  // Vector stash: keeps a copy of each inserted vector (as Float32Array) so
  // compact can rebuild the graph from scratch instead of just removing ghosts.
  // Memory cost: dims * 4 bytes per live vector.
  const _vectors = new Map();    // external ID -> Float32Array

  return {
    engine,
    dims,
    maxElements,
    backend,
    _vecBuffer: vecBuffer,
    _idsBuffer: idsBuffer,
    _distsBuffer: distsBuffer,

    add(vec) {
      engine.HEAPF32.set(vec, this._vecBuffer >> 2);
      const intId = this.backend.add(this._vecBuffer);
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
      this.backend.setEf(ef);
      engine.HEAPF32.set(vec, this._vecBuffer >> 2);

      const found = this.backend.query(
        this._vecBuffer, k, this._idsBuffer, this._distsBuffer);

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

    // Diagnostic: return both raw internal IDs and translated external IDs
    searchDebug(vec, k, ef = 50) {
      this.backend.setEf(ef);
      engine.HEAPF32.set(vec, this._vecBuffer >> 2);
      const found = this.backend.query(
        this._vecBuffer, k, this._idsBuffer, this._distsBuffer);
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
        const dataPtr = this.backend.exportBinary(sizePtr);
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
      this.backend.delete(intId);
      _deletedExt.add(extId);
      _vectors.delete(extId);
    },

    compact() {
      // Rebuild the entire HNSW graph from scratch using stashed vectors.
      // This produces a fresh, fully-connected graph with no recall loss,
      // unlike in-place compaction which strips edges and degrades quality.
      //
      // The old index is destroyed and replaced atomically — searches that
      // started before compact see the old graph, searches after see the new one.

      // Collect surviving vectors (not deleted)
      const survivors = [];
      for (const [extId, vec] of _vectors) {
        if (!_deletedExt.has(extId)) {
          survivors.push({ extId, vec });
        }
      }

      // If we don't have stashed vectors (e.g., after import), fall back
      // to in-place compaction (strips ghosts but may degrade recall).
      if (_vectors.size === 0) {
        const oldSurvivors = [];
        for (const [extId, intId] of _extToInt) {
          if (!_deletedExt.has(extId)) oldSurvivors.push({ extId, intId });
        }
        oldSurvivors.sort((a, b) => a.intId - b.intId);
        this.backend.compact();
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
        // Everything was deleted — just clear the engine
        this.backend.compact();
        _extToInt.clear();
        _intToExt.clear();
        _deletedExt.clear();
        _vectors.clear();
        return;
      }

      // Re-initialize the engine backend (destroys old graph, builds fresh)
      this.backend.init(maxElements, _initParams.M, _initParams.efC, _initParams.efS);

      // Re-insert all surviving vectors
      _extToInt.clear();
      _intToExt.clear();
      _deletedExt.clear();

      for (const { extId, vec } of survivors) {
        engine.HEAPF32.set(vec, this._vecBuffer >> 2);
        const newIntId = this.backend.add(this._vecBuffer);
        _extToInt.set(extId, newIntId);
        _intToExt.set(newIntId, extId);
      }

      // Clean up stash to only contain survivors
      _vectors.clear();
      for (const { extId, vec } of survivors) {
        _vectors.set(extId, vec);
      }
    },

    // Seed a 1:1 external↔internal mapping for an already-inserted vector.
    // Used after import to bootstrap the ID maps.
    _seedId(intId) {
      const extId = _nextExtId++;
      _extToInt.set(extId, intId);
      _intToExt.set(intId, extId);
    },

    ghostCount() { return this.backend.ghostCount(); },
    ghostRatio() { return this.backend.ghostRatio(); },

    get count() { return this.backend.count(); },
    get memory() { return this.backend.memory(); },
    get nextId() { return _nextExtId; },

    destroy() {
      engine._emsc_free(this._vecBuffer);
      engine._emsc_free(this._idsBuffer);
      engine._emsc_free(this._distsBuffer);
    }
  };
}

function initializeIndex(engine, dims, maxElements, M = 8, efConstruction = 150, efSearch = 100) {
  let backend;

  if (dims === 1536) {
    backend = make1536Backend(engine, maxElements);
    const status = backend.init(maxElements, M, efConstruction, efSearch);
    if (status !== 0) throw new Error(`p1536_init failed: ${status}`);
  } else if (dims === 384) {
    backend = make384Backend(engine);
    backend.init(maxElements, M, efConstruction, efSearch);
  } else {
    backend = makeRuntimeBackend(engine);
    engine.__pancake_i8_dim = dims;
    const status = engine.ccall('i8_init', 'number',
      ['number', 'number', 'number', 'number', 'number', 'number'],
      [dims, maxElements, 1, M, efConstruction, efSearch]);
    if (status !== 0) throw new Error(`i8_init failed: ${status}`);
  }

  return buildIndexWrapper(engine, dims, maxElements, backend, { M, efC: efConstruction, efS: efSearch });
}

function importIndex(engine, dims, buf, size) {
  let backend;
  let status;

  if (dims === 1536) {
    backend = make1536Backend(engine, 0);
    status = engine.ccall('p1536_import_index', 'number',
      ['number', 'number'], [buf, size]);
  } else if (dims === 384) {
    backend = make384Backend(engine);
    status = engine.ccall('import_index', 'number',
      ['number', 'number'], [buf, size]);
  } else {
    backend = makeRuntimeBackend(engine);
    engine.__pancake_i8_dim = dims;
    status = engine.ccall('i8_import_index', 'number',
      ['number', 'number', 'number'], [buf, size, dims]);
  }

  if (status !== 0) throw new Error(`import failed: ${status}`);
  const wrapper = buildIndexWrapper(engine, dims, Number.MAX_SAFE_INTEGER, backend);
  // Seed ID maps: imported vectors have internal IDs 0..count-1.
  // Assign matching external IDs so the mapping layer is consistent.
  const importedCount = wrapper.count;
  for (let i = 0; i < importedCount; i++) {
    wrapper._seedId(i);
  }
  return wrapper;
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

function backendCapabilities(activeIndex) {
  if (!activeIndex) {
    return {
      supports_delete: false,
      supports_compact: false,
      supports_ghosts: false
    };
  }
  return {
    supports_delete: !!activeIndex.backend.supportsDelete?.(),
    supports_compact: !!activeIndex.backend.supportsCompact?.(),
    supports_ghosts: !!activeIndex.backend.supportsGhosts?.()
  };
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

  // --- Authentication (skip for /health and OPTIONS) ---
  if (env.API_KEY && url.pathname !== '/health') {
    const authHeader = request.headers.get('Authorization') || '';
    const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : '';
    if (token !== env.API_KEY) {
      return jsonResponse({ error: 'Unauthorized' }, 401);
    }
  }

  // --- Rate limiting (skip for /health — monitoring endpoints shouldn't be throttled) ---
  const maxRpm = parseInt(env.RATE_LIMIT_RPM || '0', 10);
  if (maxRpm > 0 && url.pathname !== '/health') {
    cleanupRateLimits();
    const ip = request.headers.get('CF-Connecting-IP') || request.headers.get('X-Forwarded-For') || 'unknown';
    if (isRateLimited(ip, maxRpm)) {
      return jsonResponse({ error: 'Rate limit exceeded. Try again later.' }, 429);
    }
  }

  // --- Restore index from R2 on cold start ---
  if (!index && env.INDEX_BUCKET) {
    await restoreIndex(env);
  }

  if (url.pathname === '/health' && method === 'GET') {
    return jsonResponse({
      status: 'ok',
      initialized: index !== null,
      count: index ? index.count : 0,
      memory_bytes: index ? index.memory : 0,
      dims: index ? index.dims : null,
      ...backendCapabilities(index)
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
    index = initializeIndex(engine, dims, maxElements, M, efConstruction, efSearch);

    let inserted = 0;
    for (const vec of vectors) {
      const err = validateVector(vec, dims, 'vectors[]');
      if (err) { index.destroy(); index = null; return jsonResponse({ error: err }, 400); }
      index.add(new Float32Array(vec));
      inserted++;
    }

    schedulePersist(env, ctx);
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
    schedulePersist(env, ctx);
    return jsonResponse({ id, count: index.count });
  }

  if (url.pathname === '/delete' && method === 'POST') {
    if (!index) return jsonResponse({ error: 'Index not initialized.' }, 503);
    if (!index.backend.supportsDelete?.()) {
      return jsonResponse({ error: `Delete is not supported for dims=${index.dims}` }, 405);
    }
    const body = await safeParseJson(request);
    if (!body) return jsonResponse({ error: 'Invalid JSON' }, 400);
    const { id } = body;
    if (typeof id !== 'number' || !Number.isInteger(id) || id < 0)
      return jsonResponse({ error: 'id must be a non-negative integer' }, 400);
    index.delete(id);
    schedulePersist(env, ctx);
    return jsonResponse({ deleted: id, ghost_count: index.ghostCount(), ghost_ratio: index.ghostRatio() });
  }

  if (url.pathname === '/compact' && method === 'POST') {
    if (!index) return jsonResponse({ error: 'Index not initialized.' }, 503);
    if (!index.backend.supportsCompact?.()) {
      return jsonResponse({ error: `Compaction is not supported for dims=${index.dims}` }, 405);
    }
    const t0 = performance.now();
    index.compact();
    const elapsed = performance.now() - t0;
    schedulePersist(env, ctx);
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
      dims: index.dims,
      ...backendCapabilities(index)
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

    schedulePersist(env, ctx);
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

  if (url.pathname === '/search_text' && method === 'POST') {
    if (!index) return jsonResponse({ error: 'Index not initialized. Call /init first.' }, 503);

    const t0 = performance.now();
    const body = await safeParseJson(request);
    if (!body) return jsonResponse({ error: 'Invalid JSON' }, 400);
    const { text, k = 10, ef = 100 } = body;

    if (typeof text !== 'string' || text.trim().length === 0)
      return jsonResponse({ error: 'text must be a non-empty string' }, 400);
    if (!Number.isInteger(k) || k < 1 || k > MAX_RESULTS)
      return jsonResponse({ error: `k must be an integer between 1 and ${MAX_RESULTS}` }, 400);
    if (!isPositiveInteger(ef) || ef > MAX_EF)
      return jsonResponse({ error: `ef must be an integer between 1 and ${MAX_EF}` }, 400);

    let model;
    try {
      model = await loadHybridModel(env);
    } catch (e) {
      return jsonResponse({ error: e.message }, 503);
    }

    const embedStart = performance.now();
    const queryVec = hybridEmbed(text, model);
    const embed_ms = performance.now() - embedStart;

    if (!queryVec)
      return jsonResponse({ error: 'Query produced no usable features. Try different terms.' }, 400);

    const searchStart = performance.now();
    const results = index.search(queryVec, k, ef);
    const search_ms = performance.now() - searchStart;

    // Attach metadata if available
    const enriched = results.neighbors.map((id, i) => {
      const entry = { id, distance: results.distances[i] };
      if (model.metadata && model.metadata[id]) {
        const meta = model.metadata[id];
        if (meta.text) entry.text = meta.text;
        if (meta.clause_type) entry.clause_type = meta.clause_type;
        if (meta.filename) entry.filename = meta.filename;
      }
      return entry;
    });

    const latency_ms = performance.now() - t0;
    return jsonResponse({ results: enriched, embed_ms, search_ms, latency_ms }, 200, corsHeaders(env));
  }

  if (url.pathname === '/upload_model' && method === 'POST') {
    if (!env.INDEX_BUCKET)
      return jsonResponse({ error: 'INDEX_BUCKET (R2) not configured' }, 503);

    const formData = await request.formData();
    const files = {
      'vocab_augmented.json': formData.get('vocab'),
      'model_distilled.bin': formData.get('model'),
      'ngram_embeddings.bin': formData.get('ngram'),
      'embeddings_distilled.npy': formData.get('embeddings'),
      'metadata.json': formData.get('metadata'),
    };

    const uploaded = [];
    const missing = [];
    for (const [name, file] of Object.entries(files)) {
      if (!file) { missing.push(name); continue; }
      const buf = await file.arrayBuffer();
      await env.INDEX_BUCKET.put(`model/${name}`, buf, {
        httpMetadata: { contentType: file.type || 'application/octet-stream' }
      });
      uploaded.push({ name, size: buf.byteLength });
    }

    // Invalidate cached model so next search_text reloads
    hybridModel = null;

    return jsonResponse({ uploaded, missing: missing.length > 0 ? missing : undefined });
  }

  if (url.pathname === '/init_hybrid' && method === 'POST') {
    // Initialize the index from pre-computed embeddings stored in R2
    let model;
    try {
      model = await loadHybridModel(env);
    } catch (e) {
      return jsonResponse({ error: e.message }, 503);
    }

    const body = await safeParseJson(request) || {};
    const { M = 16, efConstruction = 200, efSearch = 100 } = body;
    if (!isPositiveInteger(M) || M > MAX_M)
      return jsonResponse({ error: `M must be an integer between 1 and ${MAX_M}` }, 400);
    if (!isPositiveInteger(efConstruction) || efConstruction > MAX_EF)
      return jsonResponse({ error: `efConstruction must be an integer between 1 and ${MAX_EF}` }, 400);
    if (!isPositiveInteger(efSearch) || efSearch > MAX_EF)
      return jsonResponse({ error: `efSearch must be an integer between 1 and ${MAX_EF}` }, 400);
    const dims = model.wordModel.outputDim;
    const count = model.metadata.length;

    const engine = await initializePancake();
    if (index) { index.destroy(); index = null; }

    // Use i8 runtime backend (matches demo_hybrid_search.js)
    const backend = makeRuntimeBackend(engine);
    const status = engine.ccall('i8_init', 'number',
      ['number', 'number', 'number', 'number', 'number', 'number'],
      [dims, count, 1, M, efConstruction, efSearch]);
    if (status !== 0) return jsonResponse({ error: `i8_init failed: ${status}` }, 500);

    index = buildIndexWrapper(engine, dims, count, backend);

    // Bulk insert pre-computed embeddings
    const t0 = performance.now();
    for (let i = 0; i < count; i++) {
      index.add(model.embeddings.subarray(i * dims, (i + 1) * dims));
    }
    const build_ms = performance.now() - t0;

    schedulePersist(env, ctx);
    return jsonResponse({
      status: 'initialized',
      dims,
      count: index.count,
      M,
      efConstruction,
      efSearch,
      build_ms,
      memory_bytes: index.memory
    });
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

    const buf = engine._emsc_malloc(buffer.byteLength);
    if (!buf) return jsonResponse({ error: 'WASM heap allocation failed' }, 500);

    try {
      engine.HEAPU8.set(new Uint8Array(buffer), buf);
      if (index) { index.destroy(); index = null; }
      index = importIndex(engine, dims, buf, buffer.byteLength);
    } catch (e) {
      return jsonResponse({ error: e.message }, 500);
    } finally {
      engine._emsc_free(buf);
    }

    schedulePersist(env, ctx);
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
      'POST /init_hybrid':  '{ M?, efConstruction?, efSearch? } — build index from R2 model embeddings',
      'POST /import':       'Binary body from /export — ?dims=<n> required',
      'POST /add':          '{ vector: float[] }',
      'POST /search':       '{ query: float[], k?, ef? }',
      'POST /search_text':  '{ text: string, k?, ef? } — hybrid word+ngram embedding search',
      'POST /upload_model': 'Multipart: vocab, model, ngram, embeddings, metadata files'
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
