'use strict';

// Envelope header for validated export/import
const PANCAKE_MAGIC = 0x504E434B; // "PNCK"
const ENVELOPE_VERSION = 2;
const ENVELOPE_HEADER_SIZE = 20; // magic(4) + version(4) + dim(4) + metric(4) + quantized(4)

class PancakeIndex {
    constructor(engine, opts, vecPtr, idPtr, distPtr, bufferCapacity) {
        this._e = engine;
        this._dim = opts.dim;
        this._useInt8 = !!opts.quantized;
        this._quantized = !!opts.quantized;
        this._isL2 = (opts.metric === 'l2');
        this._vecPtr = vecPtr;
        this._idPtr = idPtr;
        this._distPtr = distPtr;
        this._bufferCapacity = bufferCapacity;
        this._disposed = false;

        // ID translation layer -- WASM compact() reassigns sequential IDs,
        // so we maintain a stable external ID space for the consumer.
        this._nextExtId = 0;
        this._intToExt = new Map(); // internal (WASM) id -> external (user) id
        this._extToInt = new Map(); // external (user) id -> internal (WASM) id
        this._deletedExt = new Set(); // external ids that have been soft-deleted
    }

    add(vec) {
        this._checkDisposed();
        const f32 = vec instanceof Float32Array ? vec : new Float32Array(vec);
        if (f32.length !== this._dim) {
            throw new Error(`Expected vector of length ${this._dim}, got ${f32.length}`);
        }
        this._e.HEAPF32.set(f32, this._vecPtr >> 2);
        const intId = this._useInt8
            ? this._e._i8_add(this._vecPtr)
            : this._e._float_add(this._vecPtr);
        if (intId === 0xFFFFFFFF || intId < 0) {
            throw new Error('Insert failed (index full or not initialized)');
        }

        const extId = this._nextExtId++;
        this._intToExt.set(intId, extId);
        this._extToInt.set(extId, intId);
        return extId;
    }

    addBatch(vectors) {
        this._checkDisposed();
        if (!Array.isArray(vectors)) {
            throw new Error('addBatch() requires an array of vectors');
        }
        if (vectors.length === 0) return [];

        const bulkInsert = this._bulkInsertFn();
        if (!bulkInsert) {
            const ids = new Array(vectors.length);
            for (let i = 0; i < vectors.length; i++) {
                ids[i] = this.add(vectors[i]);
            }
            return ids;
        }

        const flat = new Float32Array(vectors.length * this._dim);
        let validCount = 0;
        for (let i = 0; i < vectors.length; i++) {
            const f32 = vectors[i] instanceof Float32Array ? vectors[i] : new Float32Array(vectors[i]);
            if (f32.length !== this._dim) {
                if (validCount === 0) {
                    throw new Error(`Expected vector of length ${this._dim}, got ${f32.length}`);
                }
                const ids = this._bulkInsert(flat.subarray(0, validCount * this._dim), validCount, bulkInsert);
                throw new Error(`Expected vector of length ${this._dim}, got ${f32.length}`);
            }
            flat.set(f32, i * this._dim);
            validCount++;
        }
        return this._bulkInsert(flat, validCount, bulkInsert);
    }

    search(query, k) {
        this._checkDisposed();
        if (!Number.isInteger(k) || k < 0) {
            throw new Error('search() requires a non-negative integer k');
        }
        const f32 = query instanceof Float32Array ? query : new Float32Array(query);
        if (f32.length !== this._dim) {
            throw new Error(`Expected query of length ${this._dim}, got ${f32.length}`);
        }
        if (k === 0) return [];
        this._ensureSearchCapacity(k);
        this._e.HEAPF32.set(f32, this._vecPtr >> 2);
        const found = this._useInt8
            ? this._e._i8_query(this._vecPtr, k, this._idPtr, this._distPtr)
            : this._e._float_query(this._vecPtr, k, this._idPtr, this._distPtr);
        return this._readResults(found);
    }

    delete(id) {
        this._checkDisposed();
        const intId = this._extToInt.get(id);
        if (intId === undefined) return;
        if (this._useInt8) {
            this._e._i8_delete(intId);
        } else {
            this._e._float_delete(intId);
        }
        this._deletedExt.add(id);
    }

    compact() {
        this._checkDisposed();

        const survivors = [];
        for (const [extId, intId] of this._extToInt) {
            if (!this._deletedExt.has(extId)) {
                survivors.push({ extId, intId });
            }
        }
        survivors.sort((a, b) => a.intId - b.intId);

        if (this._useInt8) {
            this._e._i8_compact();
        } else {
            this._e._float_compact();
        }

        this._intToExt.clear();
        this._extToInt.clear();
        this._deletedExt.clear();
        for (let newInt = 0; newInt < survivors.length; newInt++) {
            const extId = survivors[newInt].extId;
            this._intToExt.set(newInt, extId);
            this._extToInt.set(extId, newInt);
        }
    }

    export() {
        this._checkDisposed();
        const sizePtr = this._e._emsc_malloc(4);
        if (!sizePtr) throw new Error('WASM malloc failed for export');

        const dataPtr = this._useInt8
            ? this._e._i8_export_index(sizePtr)
            : this._e._float_export_index(sizePtr);

        if (!dataPtr) {
            this._e._emsc_free(sizePtr);
            throw new Error('Export failed');
        }

        const size = this._e.HEAPU32[sizePtr >> 2];
        const result = new Uint8Array(ENVELOPE_HEADER_SIZE + size);
        const view = new DataView(result.buffer);
        view.setUint32(0, PANCAKE_MAGIC, true);
        view.setUint32(4, ENVELOPE_VERSION, true);
        view.setUint32(8, this._dim, true);
        view.setUint32(12, this._isL2 ? 0 : 1, true);
        view.setUint32(16, this._quantized ? 1 : 0, true);
        result.set(this._e.HEAPU8.subarray(dataPtr, dataPtr + size), ENVELOPE_HEADER_SIZE);

        this._e._emsc_free(sizePtr);
        return result;
    }

    import(data) {
        this._checkDisposed();
        const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);

        let wasmBytes = bytes;

        if (bytes.length >= ENVELOPE_HEADER_SIZE) {
            const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
            if (view.getUint32(0, true) === PANCAKE_MAGIC) {
                const dim = view.getUint32(8, true);
                const version = view.getUint32(4, true);
                let metricVal;
                let quantizedVal;

                if (dim !== this._dim) {
                    throw new Error(`Import failed: dim mismatch (exported ${dim}, expected ${this._dim})`);
                }
                if (version === 1) {
                    const compressed = view.getUint32(12, true);
                    if (compressed !== dim) {
                        throw new Error('Import failed: DCT/PCA indexes are no longer supported');
                    }
                    metricVal = view.getUint32(16, true);
                    quantizedVal = view.getUint32(20, true);
                } else {
                    metricVal = view.getUint32(12, true);
                    quantizedVal = view.getUint32(16, true);
                }
                const exportedL2 = metricVal === 0;
                if (exportedL2 !== this._isL2) {
                    const got = exportedL2 ? 'l2' : 'cosine';
                    const want = this._isL2 ? 'l2' : 'cosine';
                    throw new Error(`Import failed: metric mismatch (exported ${got}, expected ${want})`);
                }
                if (!!quantizedVal !== this._quantized) {
                    throw new Error('Import failed: quantized mismatch');
                }

                wasmBytes = bytes.subarray(ENVELOPE_HEADER_SIZE);
            }
        }

        const dataPtr = this._e._emsc_malloc(wasmBytes.length);
        if (!dataPtr) throw new Error('WASM malloc failed for import');

        this._e.HEAPU8.set(wasmBytes, dataPtr);

        const status = this._useInt8
            ? this._e._i8_import_index(dataPtr, wasmBytes.length, this._dim)
            : this._e._float_import_index(dataPtr, wasmBytes.length);

        this._e._emsc_free(dataPtr);

        if (status !== 0) throw new Error('Import failed');

        const count = this._useInt8
            ? this._e._i8_count()
            : this._e._float_count();
        this._intToExt.clear();
        this._extToInt.clear();
        this._deletedExt.clear();
        for (let i = 0; i < count; i++) {
            this._intToExt.set(i, i);
            this._extToInt.set(i, i);
        }
        this._nextExtId = count;
    }

    get count() {
        this._checkDisposed();
        return this._useInt8
            ? this._e._i8_count()
            : this._e._float_count();
    }

    get ghostCount() {
        this._checkDisposed();
        return this._useInt8
            ? this._e._i8_ghost_count()
            : this._e._float_ghost_count();
    }

    get ghostRatio() {
        this._checkDisposed();
        return this._useInt8
            ? this._e._i8_ghost_ratio()
            : this._e._float_ghost_ratio();
    }

    get memory() {
        this._checkDisposed();
        return this._useInt8
            ? this._e._i8_memory()
            : this._e._float_memory();
    }

    get dim() { return this._dim; }

    dispose() {
        if (this._disposed) return;
        this._e._emsc_free(this._vecPtr);
        this._e._emsc_free(this._idPtr);
        this._e._emsc_free(this._distPtr);
        this._disposed = true;
    }

    _checkDisposed() {
        if (this._disposed) throw new Error('PancakeIndex has been disposed');
    }

    _ensureSearchCapacity(k) {
        if (k <= this._bufferCapacity) return;

        const newIdPtr = this._e._emsc_malloc(k * 8);
        const newDistPtr = this._e._emsc_malloc(k * 4);
        if (!newIdPtr || !newDistPtr) {
            if (newIdPtr) this._e._emsc_free(newIdPtr);
            if (newDistPtr) this._e._emsc_free(newDistPtr);
            throw new Error(`WASM malloc failed while growing search buffers to k=${k}`);
        }

        this._e._emsc_free(this._idPtr);
        this._e._emsc_free(this._distPtr);
        this._idPtr = newIdPtr;
        this._distPtr = newDistPtr;
        this._bufferCapacity = k;
    }

    _bulkInsert(flatVectors, count, bulkInsert) {
        const dataPtr = this._e._emsc_malloc(flatVectors.length * 4);
        if (!dataPtr) throw new Error('WASM malloc failed for bulk insert');
        try {
            this._e.HEAPF32.set(flatVectors, dataPtr >> 2);
            const countBefore = this.count;
            const inserted = bulkInsert.call(this._e, dataPtr, count);
            const ids = this._recordInsertedRange(countBefore, inserted);
            if (inserted !== count) {
                throw new Error('Insert failed (index full or not initialized)');
            }
            return ids;
        } finally {
            this._e._emsc_free(dataPtr);
        }
    }

    _bulkInsertFn() {
        if (this._useInt8) return this._e._i8_bulk_insert;
        return null;
    }

    _recordInsertedRange(firstIntId, count) {
        const ids = new Array(count);
        for (let i = 0; i < count; i++) {
            const intId = firstIntId + i;
            const extId = this._nextExtId++;
            this._intToExt.set(intId, extId);
            this._extToInt.set(extId, intId);
            ids[i] = extId;
        }
        return ids;
    }

    _readResults(n) {
        const dv = new DataView(this._e.HEAPU8.buffer);
        const results = new Array(n);
        for (let i = 0; i < n; i++) {
            const lo = dv.getUint32(this._idPtr + i * 8, true);
            const hi = dv.getUint32(this._idPtr + i * 8 + 4, true);
            const intId = hi * 0x100000000 + lo;
            let distance = this._e.HEAPF32[(this._distPtr >> 2) + i];
            if (this._isL2) distance = Math.sqrt(distance);
            const extId = this._intToExt.get(intId);
            results[i] = { id: extId !== undefined ? extId : intId, distance };
        }
        return results;
    }
}

function createPancakeApi(loadEngineImpl) {
    async function create(opts) {
        if (!opts || !opts.dim) throw new Error('opts.dim is required');
        if (!Number.isInteger(opts.dim) || opts.dim <= 0) {
            throw new Error('opts.dim must be a positive integer');
        }
        if ('compressed' in opts) {
            throw new Error('opts.compressed has been removed');
        }
        if ('varianceSample' in opts) {
            throw new Error('opts.varianceSample has been removed');
        }

        const dim = opts.dim;
        const metric = (opts.metric === 'l2') ? 0 : 1;
        const maxElements = opts.maxElements || 100000;
        const isQuantized = opts.quantized !== undefined ? !!opts.quantized : true;
        const quantized = isQuantized ? 1 : 0;
        const M = opts.M || 16;
        const efConstruction = opts.efConstruction || 200;
        const efSearch = opts.efSearch || 100;

        const resolvedOpts = { ...opts, quantized: isQuantized, M, efConstruction, efSearch };

        const e = await loadEngineImpl();
        if (quantized) {
            e.__pancake_i8_dim = dim;
        }

        const vecPtr = e._emsc_malloc(dim * 4);
        const initialBufferCapacity = 16;
        const idPtr = e._emsc_malloc(initialBufferCapacity * 8);
        const distPtr = e._emsc_malloc(initialBufferCapacity * 4);

        if (!vecPtr || !idPtr || !distPtr) {
            throw new Error('WASM malloc failed');
        }

        const status = quantized
            ? e._i8_init(dim, maxElements, metric, M, efConstruction, efSearch)
            : e._float_init(dim, maxElements, M, efConstruction, efSearch, metric);

        if (status !== 0) {
            e._emsc_free(vecPtr);
            e._emsc_free(idPtr);
            e._emsc_free(distPtr);
            throw new Error(`Backend init failed (status=${status})`);
        }

        return new PancakeIndex(e, resolvedOpts, vecPtr, idPtr, distPtr, initialBufferCapacity);
    }

    return { create };
}

module.exports = createPancakeApi;
module.exports.default = createPancakeApi;
