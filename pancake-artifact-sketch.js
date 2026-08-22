'use strict';
// Sketch artifact profile (.pancake-sketch) — spec/SKETCH_PROFILE.md:
// PancakeSketchArtifact reader, the sketch-artifact builders, and the
// engine-backed resident scanner.
// Split out of pancake-artifact.js (the public entry, which re-exports the
// three parts); see that file for the module map.

const { pancakeError, PANCAKE_ERROR_CODES } = require('./pancake-errors.js');
const {
    DEFAULT_OPEN_READ_BYTES,
    MAX_COALESCED_RANGE_BYTES,
    mapLimit,
    readChecked,
    resolveMaxReadBytes,
    resolveMaxCacheBytes,
    NodeFileRangeSource,
    parseUint8Snapshot,
    sha256Bytes,
    sha256BytesAsync,
    verifySegmentSha256,
} = require('./pancake-artifact-common.js');
const {
    normalizeQuery: normalizeQueryVector,
    resolveSearchK,
    resolveOptionalPositiveInt,
} = require('./pancake-artifact-common.js');

// ============================================================================
// Sketch artifact profile (.pancake-sketch) — spec/SKETCH_PROFILE.md
// ============================================================================
//
// Layout: 256-byte header | scales f32[count] | offsets f32[count] |
// packed pooled sketches | zero padding to 16-byte alignment | raw u8 rows.
// Everything before vectorsOffset is the resident prefix; row addresses are
// computed (vectorsOffset + id*dim), never looked up.

const SKETCH_MAGIC = 0x31415350; // PSA1
const SKETCH_HEADER_BYTES = 256;
const SKETCH_KIND_U8 = 1;

function buildSketchArtifact(snapshotBytes, outPath, options = {}) {
    if (typeof outPath !== 'string' || outPath.length === 0) {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'buildSketchArtifact() requires an output path');
    }
    return exportSketchArtifact(parseUint8Snapshot(snapshotBytes), outPath, options);
}

// Accepts any index-like source of quantized rows ({ dim, count, metric,
// qdata, scales, offsets }) — the profile is derivable from a snapshot, a
// range artifact, or any producer that emits row-affine u8 vectors.
function exportSketchArtifact(index, outPath, options = {}) {
    const dim = index.dim;
    const count = index.count;
    const sketchDims = options.sketchDims || (dim % 2 === 0 ? dim / 2 : dim);
    const sketchBits = options.sketchBits || 4;
    const recommendedRerank = options.recommendedRerank || 0;
    if (![4, 8].includes(sketchBits)) {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'sketchBits must be 4 or 8', { sketchBits });
    }
    if (!Number.isInteger(sketchDims) || sketchDims < 1 || dim % sketchDims !== 0) {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'sketchDims must divide dim', { sketchDims, dim });
    }
    if (sketchBits === 4 && sketchDims % 2 !== 0) {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'sketchDims must be even for 4-bit sketches', { sketchDims });
    }
    // Optional staged-boot micro tier: a second, coarser pooling of the same
    // quantized rows, stored after the full sketches so v1 readers see it
    // only as opaque resident tail bytes (their layout check permits a gap
    // between sketches end and vectorsOffset, and the resident hash already
    // covers it). Pooling commutes with the per-row affine map, so the micro
    // tier reuses scales/offsets unchanged, exactly like the full tier.
    const microDims = options.microDims || 0;
    // 4-bit default: the 2026-08-04 sweep on the 456k wiki corpus measured
    // bit depth as worthless at fixed bytes (48d/4b == 48d/8b within noise)
    // while halved pooling at the same bytes gained +28 recall points
    // (96d/4b 87.8% vs 48d/8b 59.4% capture at C=800). Micro tiers should
    // spend their byte budget on dims, never on bits.
    const microBits = microDims ? (options.microBits || 4) : 0;
    if (microDims) {
        if (!Number.isInteger(microDims) || microDims < 1 || sketchDims % microDims !== 0 || microDims >= sketchDims) {
            throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'microDims must divide sketchDims and be smaller', { microDims, sketchDims });
        }
        if (![4, 8].includes(microBits) || (microBits === 4 && microDims % 2 !== 0)) {
            throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'invalid micro sketch encoding', { microBits, microDims });
        }
    }
    const pool = dim / sketchDims;
    const sketchRowBytes = (sketchDims * sketchBits) / 8;
    const microPool = microDims ? dim / microDims : 0;
    const microRowBytes = microDims ? (microDims * microBits) / 8 : 0;

    const scalesOffset = SKETCH_HEADER_BYTES;
    const offsetsOffset = scalesOffset + count * 4;
    const sketchesOffset = offsetsOffset + count * 4;
    const sketchesEnd = sketchesOffset + count * sketchRowBytes;
    const microOffset = microDims ? sketchesEnd : 0;
    const residentEnd = microDims ? microOffset + count * microRowBytes : sketchesEnd;
    const vectorsOffset = Math.ceil(residentEnd / 16) * 16;
    const fileBytes = vectorsOffset + count * dim;

    const out = Buffer.alloc(fileBytes);
    out.writeUInt32LE(SKETCH_MAGIC, 0);
    out.writeUInt32LE(1, 4);
    out.writeUInt32LE(SKETCH_KIND_U8, 8);
    out.writeUInt32LE(index.metric, 12);
    out.writeUInt32LE(dim, 16);
    out.writeUInt32LE(count, 20);
    out.writeUInt32LE(sketchDims, 24);
    out.writeUInt32LE(sketchBits, 28);
    out.writeUInt32LE(scalesOffset, 32);
    out.writeUInt32LE(offsetsOffset, 36);
    out.writeUInt32LE(sketchesOffset, 40);
    out.writeUInt32LE(vectorsOffset, 44);
    out.writeBigUInt64LE(BigInt(fileBytes), 48);
    out.writeUInt32LE(recommendedRerank, 120);
    if (microDims) {
        out.writeUInt32LE(microDims, 124);
        out.writeUInt32LE(microBits, 128);
        out.writeUInt32LE(microOffset, 132);
    }

    for (let i = 0; i < count; i++) out.writeFloatLE(index.scales[i], scalesOffset + i * 4);
    for (let i = 0; i < count; i++) out.writeFloatLE(index.offsets[i], offsetsOffset + i * 4);

    for (let i = 0; i < count; i++) {
        const rowBase = i * dim;
        for (let sd = 0; sd < sketchDims; sd++) {
            let acc = 0;
            for (let j = 0; j < pool; j++) acc += index.qdata[rowBase + sd * pool + j];
            const pooled = Math.round(acc / pool);
            if (sketchBits === 8) {
                out[sketchesOffset + i * sketchRowBytes + sd] = pooled;
            } else {
                const q = Math.min(15, Math.round(pooled / 17));
                const byteIndex = sketchesOffset + i * sketchRowBytes + (sd >> 1);
                if (sd % 2 === 0) out[byteIndex] = (out[byteIndex] & 0xf0) | q;
                else out[byteIndex] = (out[byteIndex] & 0x0f) | (q << 4);
            }
        }
        if (microDims) {
            for (let sd = 0; sd < microDims; sd++) {
                let acc = 0;
                for (let j = 0; j < microPool; j++) acc += index.qdata[rowBase + sd * microPool + j];
                const pooled = Math.round(acc / microPool);
                if (microBits === 8) {
                    out[microOffset + i * microRowBytes + sd] = pooled;
                } else {
                    const q = Math.min(15, Math.round(pooled / 17));
                    const byteIndex = microOffset + i * microRowBytes + (sd >> 1);
                    if (sd % 2 === 0) out[byteIndex] = (out[byteIndex] & 0xf0) | q;
                    else out[byteIndex] = (out[byteIndex] & 0x0f) | (q << 4);
                }
            }
        }
    }

    Buffer.from(index.qdata.buffer, index.qdata.byteOffset, count * dim).copy(out, vectorsOffset);

    sha256Bytes(out.subarray(SKETCH_HEADER_BYTES, vectorsOffset)).copy(out, 56);
    sha256Bytes(out.subarray(vectorsOffset)).copy(out, 88);
    if (microDims) {
        // Stage-1 hash: scales+offsets plus the micro segment, exactly the
        // bytes a staged open serves from, verifiable before the full
        // sketches arrive. The full resident hash still covers everything.
        const stage1 = Buffer.concat([
            out.subarray(SKETCH_HEADER_BYTES, sketchesOffset),
            out.subarray(microOffset, microOffset + count * microRowBytes),
        ]);
        sha256Bytes(stage1).copy(out, 136);
    }

    const manifest = {
        format: 'pancake-sketch-artifact',
        formatVersion: 1,
        sizeBytes: fileBytes,
        metric: index.metric === 1 ? 'cosine' : 'l2',
        graph: { count, dim },
        sketch: { sketchDims, sketchBits, pool, residentBytes: vectorsOffset },
        micro: microDims ? { microDims, microBits, microPool, stage1Bytes: sketchesOffset + count * microRowBytes } : null,
        addressing: { scalesOffset, offsetsOffset, sketchesOffset, vectorsOffset },
        recommendedRerank,
    };

    if (outPath === null) return { bytes: out, manifest };

    const fs = require('fs');
    fs.writeFileSync(outPath, out);
    return { ...manifest, file: outPath };
}

// Bytes-in/bytes-out variant for producers that assemble segments in memory
// (e.g. the complete-profile compiler): identical output to
// buildSketchArtifact, no filesystem involved. Returns { bytes, manifest }.
function buildSketchArtifactBytes(snapshotBytes, options = {}) {
    return exportSketchArtifact(parseUint8Snapshot(snapshotBytes), null, options);
}

function buildSketchArtifactFile(snapshotPath, outPath, options = {}) {
    const fs = require('fs');
    return buildSketchArtifact(fs.readFileSync(snapshotPath), outPath, options);
}

class PancakeSketchArtifact {
    constructor(source) {
        this.source = source;
        // Fetched rows live in a byte-budgeted LRU; search correctness never
        // depends on retention because fetchRows returns rows directly.
        this.cache = new Map();
        this.cacheBytes = 0;
        this.maxCacheBytes = 64 * 1024 * 1024;
        this.rangeRequests = 0;
        this.rangeBytes = 0;
        this.maxReadBytes = DEFAULT_OPEN_READ_BYTES;
        this.residentVerified = false;
        this.vectorsVerified = false;
    }

    cachedRow(id) {
        const row = this.cache.get(id);
        if (row !== undefined) {
            this.cache.delete(id);
            this.cache.set(id, row);
        }
        return row;
    }

    // Drop all cached rows. Cache state never affects result semantics, so
    // this only changes fetch behavior (every subsequent search starts cold).
    clearCache() {
        this.cache.clear();
        this.cacheBytes = 0;
    }

    cacheRow(id, row) {
        if (this.cache.has(id)) return;
        this.cache.set(id, row);
        this.cacheBytes += this.dim;
        while (this.cacheBytes > this.maxCacheBytes && this.cache.size > 1) {
            const oldest = this.cache.keys().next().value;
            this.cache.delete(oldest);
            this.cacheBytes -= this.dim;
        }
    }

    static async open(source, options = {}) {
        if (!source || typeof source.read !== 'function') {
            throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'SketchArtifact.open() requires a range source with read(offset, length)');
        }
        const artifact = new PancakeSketchArtifact(source);
        artifact.maxReadBytes = resolveMaxReadBytes(options.maxReadBytes);
        const header = await readChecked(source, 0, SKETCH_HEADER_BYTES, 'header');
        if (header.byteLength !== SKETCH_HEADER_BYTES) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact header is truncated');
        }
        const view = new DataView(header.buffer, header.byteOffset, header.byteLength);
        if (view.getUint32(0, true) !== SKETCH_MAGIC) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Not a Pancake sketch artifact (bad magic)');
        }
        const version = view.getUint32(4, true);
        if (version !== 1) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported sketch artifact version', { version });
        }
        if (view.getUint32(8, true) !== SKETCH_KIND_U8) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported sketch artifact kind');
        }
        artifact.metric = view.getUint32(12, true);
        artifact.dim = view.getUint32(16, true);
        artifact.count = view.getUint32(20, true);
        artifact.sketchDims = view.getUint32(24, true);
        artifact.sketchBits = view.getUint32(28, true);
        const scalesOffset = view.getUint32(32, true);
        const offsetsOffset = view.getUint32(36, true);
        const sketchesOffset = view.getUint32(40, true);
        artifact.vectorsOffset = view.getUint32(44, true);
        const fileBytes = Number(view.getBigUint64(48, true));
        artifact.recommendedRerank = view.getUint32(120, true);
        // Retained so the lazy tier stays verifiable after open — see
        // verifyVectors(). Copied out of the header read buffer.
        artifact.vectorsSha256 = new Uint8Array(header.subarray(88, 120));
        artifact.microDims = view.getUint32(124, true);
        artifact.microBits = artifact.microDims ? view.getUint32(128, true) : 0;
        const microOffset = artifact.microDims ? view.getUint32(132, true) : 0;
        artifact.maxCacheBytes = resolveMaxCacheBytes(options.maxCacheBytes, 256 * artifact.dim, artifact.maxCacheBytes);

        const { metric, dim, count, sketchDims, sketchBits, vectorsOffset } = artifact;
        if (metric !== 0 && metric !== 1) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported sketch artifact metric', { metric });
        }
        if (dim < 1 || count < 1 || sketchDims < 1 || dim % sketchDims !== 0) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Invalid sketch artifact geometry', { dim, count, sketchDims });
        }
        if (![4, 8].includes(sketchBits) || (sketchBits === 4 && sketchDims % 2 !== 0)) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Invalid sketch encoding', { sketchBits, sketchDims });
        }
        const sketchRowBytes = (sketchDims * sketchBits) / 8;
        const expectVectors = count * dim;
        if (!Number.isSafeInteger(fileBytes) || !Number.isSafeInteger(expectVectors)
            || scalesOffset !== SKETCH_HEADER_BYTES
            || offsetsOffset !== scalesOffset + count * 4
            || sketchesOffset !== offsetsOffset + count * 4
            || vectorsOffset < sketchesOffset + count * sketchRowBytes
            || vectorsOffset % 16 !== 0
            || fileBytes !== vectorsOffset + expectVectors) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact layout is inconsistent');
        }
        let microRowBytes = 0;
        if (artifact.microDims) {
            microRowBytes = (artifact.microDims * artifact.microBits) / 8;
            if (sketchDims % artifact.microDims !== 0 || artifact.microDims >= sketchDims
                || ![4, 8].includes(artifact.microBits) || (artifact.microBits === 4 && artifact.microDims % 2 !== 0)
                || microOffset !== sketchesOffset + count * sketchRowBytes
                || microOffset + count * microRowBytes > vectorsOffset) {
                throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact micro tier layout is inconsistent', { microDims: artifact.microDims });
            }
        }
        artifact.microOffset = microOffset;

        const verify = options.verify !== false;
        const staged = options.staged === true && artifact.microDims > 0;
        artifact.tier = 'full';
        artifact.fullyResident = Promise.resolve(artifact);

        if (!staged) {
            const resident = await readChecked(source, SKETCH_HEADER_BYTES, vectorsOffset - SKETCH_HEADER_BYTES, 'resident prefix', artifact.maxReadBytes);
            if (resident.byteLength !== vectorsOffset - SKETCH_HEADER_BYTES) {
                throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact resident prefix is truncated');
            }
            // Copy into an aligned buffer so the typed-array views are valid
            // regardless of the source's byteOffset.
            const residentCopy = new Uint8Array(resident.byteLength);
            residentCopy.set(resident);
            artifact._adoptResident(residentCopy, count, sketchRowBytes, microRowBytes, microOffset);
            if (verify) await artifact._verifyFullResident(residentCopy, header);
            return artifact;
        }

        // Staged boot: serve queries from the coarse micro tier after
        // fetching only scales+offsets and the micro segment (one parallel
        // wave after the header), then complete residency in the background.
        // Stage 1 is independently hash-verified; stage 2 re-verifies the
        // full resident hash before the tier swap, so both states honor the
        // integrity contract. Result semantics differ between tiers, so the
        // tier is reported on every search result rather than hidden.
        artifact.tier = 'micro';
        const [affineBytes, microBytes] = await Promise.all([
            readChecked(source, SKETCH_HEADER_BYTES, sketchesOffset - SKETCH_HEADER_BYTES, 'stage-1 affine', artifact.maxReadBytes),
            readChecked(source, microOffset, count * microRowBytes, 'stage-1 micro', artifact.maxReadBytes),
        ]);
        if (affineBytes.byteLength !== sketchesOffset - SKETCH_HEADER_BYTES || microBytes.byteLength !== count * microRowBytes) {
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact stage-1 read is truncated');
        }
        const affineCopy = new Uint8Array(affineBytes.byteLength);
        affineCopy.set(affineBytes);
        const microCopy = new Uint8Array(microBytes.byteLength);
        microCopy.set(microBytes);
        if (verify) {
            const stage1 = new Uint8Array(affineCopy.byteLength + microCopy.byteLength);
            stage1.set(affineCopy, 0);
            stage1.set(microCopy, affineCopy.byteLength);
            const digest = await sha256BytesAsync(stage1);
            if (!digest) {
                // Verification requested but no crypto backend: fail closed.
                throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID,
                    'Sketch artifact verification requested but no crypto backend is available; pass verify:false to skip');
            }
            const expected = header.subarray(136, 168);
            for (let b = 0; b < 32; b++) {
                if (digest[b] !== expected[b]) {
                    throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact stage-1 prefix failed hash verification');
                }
            }
            artifact.residentVerified = true;
        }
        artifact.scales = new Float32Array(affineCopy.buffer, 0, count);
        artifact.offsets = new Float32Array(affineCopy.buffer, count * 4, count);
        artifact.sketches = null;
        artifact.microSketches = microCopy;
        artifact.residentBytes = SKETCH_HEADER_BYTES + affineCopy.byteLength + microCopy.byteLength;

        const onStage = typeof options.onStage === 'function' ? options.onStage : null;
        if (onStage) onStage({ tier: 'micro', residentBytes: artifact.residentBytes });
        artifact.fullyResident = (async () => {
            const rest = await readChecked(source, sketchesOffset, vectorsOffset - sketchesOffset, 'stage-2 sketches', artifact.maxReadBytes);
            if (rest.byteLength !== vectorsOffset - sketchesOffset) {
                throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact stage-2 read is truncated');
            }
            const residentCopy = new Uint8Array((sketchesOffset - SKETCH_HEADER_BYTES) + rest.byteLength);
            residentCopy.set(affineCopy, 0);
            residentCopy.set(rest, sketchesOffset - SKETCH_HEADER_BYTES);
            if (verify) await artifact._verifyFullResident(residentCopy, header);
            artifact._adoptResident(residentCopy, count, sketchRowBytes, microRowBytes, microOffset);
            artifact.tier = 'full';
            if (onStage) onStage({ tier: 'full', residentBytes: artifact.residentBytes });
            return artifact;
        })();
        // Surfacing failures is the caller's job via the promise; an
        // unobserved rejection must not crash the process mid-boot.
        artifact.fullyResident.catch(() => {});
        return artifact;
    }

    _adoptResident(residentCopy, count, sketchRowBytes, microRowBytes, microOffset) {
        this.scales = new Float32Array(residentCopy.buffer, 0, count);
        this.offsets = new Float32Array(residentCopy.buffer, count * 4, count);
        this.sketches = new Uint8Array(residentCopy.buffer, count * 8, count * sketchRowBytes);
        if (this.microDims) {
            this.microSketches = new Uint8Array(residentCopy.buffer, microOffset - SKETCH_HEADER_BYTES, count * microRowBytes);
        }
        this.residentBytes = this.vectorsOffset;
    }

    async _verifyFullResident(residentCopy, header) {
        const digest = await sha256BytesAsync(residentCopy);
        if (!digest) {
            // Verification was requested (callers gate this on verify) but no
            // crypto backend is available. Fail closed rather than admit
            // unverified bytes.
            throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID,
                'Sketch artifact verification requested but no crypto backend is available; pass verify:false to skip');
        }
        const expected = header.subarray(56, 88);
        for (let b = 0; b < 32; b++) {
            if (digest[b] !== expected[b]) {
                throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact resident prefix failed hash verification');
            }
        }
        this.residentVerified = true;
    }

    static async openFile(filePath, options = {}) {
        // The source owns a file descriptor from construction; a rejected
        // open must release it, or every corrupt artifact leaks an fd.
        const source = new NodeFileRangeSource(filePath);
        try {
            return await PancakeSketchArtifact.open(source, options);
        } catch (err) {
            await source.close().catch(() => {});
            throw err;
        }
    }

    stats() {
        return {
            rangeRequests: this.rangeRequests,
            rangeBytes: this.rangeBytes,
            cachedRows: this.cache.size,
            cacheBytes: this.cacheBytes,
            residentBytes: this.residentBytes,
            residentVerified: this.residentVerified,
            vectorsVerified: this.vectorsVerified,
        };
    }

    // Verify the lazy vectors segment against the header's whole-segment
    // hash. Reads the segment in bounded chunks with a streaming hash (Node);
    // runtimes without one fall back to a single read bounded by the open
    // budget. Intended for producers, CI, and hosts that materialize
    // artifacts, not the query path. Verifying individual row ranges needs
    // per-chunk commitments, which arrive with the contract's
    // complete-profile manifest.
    async verifyVectors(options = {}) {
        const bytes = this.count * this.dim;
        await verifySegmentSha256(this.source, this.vectorsOffset, bytes, this.vectorsSha256,
            'Sketch artifact vectors segment', { chunkBytes: options.chunkBytes, oneShotLimit: this.maxReadBytes });
        this.vectorsVerified = true;
        return true;
    }

    sketchValue(id, sd) {
        if (this.sketchBits === 8) return this.sketches[id * this.sketchDims + sd];
        const byte = this.sketches[id * (this.sketchDims >> 1) + (sd >> 1)];
        const nibble = sd % 2 === 0 ? (byte & 0x0f) : (byte >> 4);
        return nibble * 17;
    }

    microValue(id, sd) {
        if (this.microBits === 8) return this.microSketches[id * this.microDims + sd];
        const byte = this.microSketches[id * (this.microDims >> 1) + (sd >> 1)];
        const nibble = sd % 2 === 0 ? (byte & 0x0f) : (byte >> 4);
        return nibble * 17;
    }

    // Active resident tier for candidate selection. 'full' whenever the full
    // sketches are adopted; 'micro' only during a staged boot window.
    activeTier() {
        return this.tier === 'micro'
            ? { name: 'micro', dims: this.microDims, value: (i, sd) => this.microValue(i, sd) }
            : { name: 'full', dims: this.sketchDims, value: (i, sd) => this.sketchValue(i, sd) };
    }

    async fetchRows(ids, options = {}) {
        const defaultGap = this.source.preferredGapBytes === undefined ? 2048 : this.source.preferredGapBytes;
        const gap = Math.max(0, options.gap === undefined ? defaultGap : options.gap);
        const preferredParallelism = this.source.preferredParallelism === undefined ? 32 : this.source.preferredParallelism;
        const requestedParallelism = options.parallelism === undefined ? preferredParallelism : options.parallelism;
        const parallelism = requestedParallelism === 0 ? Infinity : Math.max(1, Math.trunc(requestedParallelism || 32));
        const rows = new Map();
        const missing = [];
        for (const id of ids) {
            if (rows.has(id)) continue;
            if (!Number.isInteger(id) || id < 0 || id >= this.count) {
                throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'row id out of range', { id });
            }
            const cached = this.cachedRow(id);
            if (cached !== undefined) rows.set(id, cached);
            else {
                rows.set(id, null);
                missing.push(id);
            }
        }
        if (!missing.length) return rows;
        missing.sort((a, b) => a - b);
        const dim = this.dim;
        // Coalesced runs are split at maxRangeBytes (row-aligned) so no gap
        // setting — or a bulk fetch of every row — turns into one unbounded
        // read and allocation.
        const maxRunRows = Math.max(1, Math.floor(Math.trunc(options.maxRangeBytes || MAX_COALESCED_RANGE_BYTES) / dim));
        const ranges = [];
        const pushRun = (startId, endId) => {
            for (let piece = startId; piece < endId; piece += maxRunRows) {
                ranges.push([piece, Math.min(piece + maxRunRows, endId)]);
            }
        };
        let runStartId = missing[0];
        let runEndId = missing[0] + 1;
        for (let i = 1; i < missing.length; i++) {
            const id = missing[i];
            if (id * dim <= runEndId * dim + gap) {
                runEndId = id + 1;
            } else {
                pushRun(runStartId, runEndId);
                runStartId = id;
                runEndId = id + 1;
            }
        }
        pushRun(runStartId, runEndId);
        const buffers = await mapLimit(ranges, parallelism, async ([startId, endId]) => {
            const offset = this.vectorsOffset + startId * dim;
            const length = (endId - startId) * dim;
            const bytes = await readChecked(this.source, offset, length, 'row');
            if (bytes.byteLength !== length) {
                throw pancakeError(PANCAKE_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact row read returned a truncated range', { offset, length });
            }
            this.rangeRequests++;
            this.rangeBytes += length;
            return { startId, endId, bytes };
        });
        for (const { startId, endId, bytes } of buffers) {
            for (let id = startId; id < endId; id++) {
                    // Copy each row out of the coalesced fetch buffer: a view
                    // would pin the whole range in the LRU, so cacheBytes
                    // would count dim bytes while retaining the full fetch.
                    // (Explicit copy constructor — .slice() is unreliable here
                    // because Node Buffers override it with view semantics.)
                    const row = new Uint8Array(bytes.subarray((id - startId) * dim, (id - startId + 1) * dim));
                if (rows.has(id)) rows.set(id, row);
                this.cacheRow(id, row);
            }
        }
        return rows;
    }

    async search(queryInput, k, options = {}) {
        // Same query contract as the core engine and the range reader:
        // numeric, finite, right dimension, usable cosine norm (normalized
        // here for cosine; the pooling below then sees a unit vector).
        const query = normalizeQueryVector(queryInput, this.dim, this.metric);
        k = resolveSearchK(k, this.count);
        if (k === 0) return { results: [], stats: this.stats() };
        const tier = this.activeTier();
        // The micro tier is coarser, so its candidate pool defaults wider:
        // an explicit options.rerank always wins; otherwise recommendedRerank
        // is scaled by microBoost while serving from the micro tier. The
        // candidate pool C can never usefully exceed the row count, and it
        // sizes the selection buffers below, so it is bounded by count before
        // anything is allocated — a k or rerank of 1e9 over a small artifact
        // must not try to allocate 1e9 slots.
        const base = resolveOptionalPositiveInt(options.rerank, 'rerank', 0);
        const boost = Math.max(1, Math.trunc(options.microBoost || 4));
        const rec = this.recommendedRerank || Math.max(100, k * 10);
        const C = Math.min(this.count, Math.max(k, base > 0 ? base : tier.name === 'micro' ? rec * boost : rec));
        const { dim, count } = this;
        const tierDims = tier.dims;
        const pool = dim / tierDims;

        let q = query;
        if (this.metric === 1) {
            let norm = 0;
            for (let d = 0; d < dim; d++) norm += query[d] * query[d];
            norm = Math.sqrt(norm);
            if (!(norm > 0) || !Number.isFinite(norm)) {
                throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'cosine query must have a nonzero finite norm');
            }
            q = new Float32Array(dim);
            for (let d = 0; d < dim; d++) q[d] = query[d] / norm;
        }
        const qPool = new Float32Array(tierDims);
        for (let sd = 0; sd < tierDims; sd++) {
            let acc = 0;
            for (let j = 0; j < pool; j++) acc += q[sd * pool + j];
            qPool[sd] = acc / pool;
        }

        let ids;
        let scanner = options.scanner || null;
        if (scanner && scanner.sketchDims !== undefined && scanner.sketchDims !== tierDims) scanner = null;
        if (!scanner && options.microScanner && options.microScanner.sketchDims === tierDims) scanner = options.microScanner;
        if (scanner) {
            // A scanner scores the resident sketches itself, so it must
            // implement this artifact's metric. Cosine requires an explicit
            // declaration: a metric-blind scanner silently loses recall there.
            const scannerMetric = scanner.metric;
            if (scannerMetric !== undefined && scannerMetric !== this.metric) {
                throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'scanner metric does not match artifact metric', { scannerMetric, metric: this.metric });
            }
            if (this.metric === 1 && scannerMetric !== 1) {
                throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'cosine sketch artifacts require a metric-aware scanner (scanner.metric === 1)', { metric: this.metric });
            }
            ids = scanner.scan(qPool, C);
        } else {
            const candDist = new Float64Array(C).fill(Infinity);
            const candId = new Int32Array(C).fill(-1);
            let candMax = Infinity;
            for (let i = 0; i < count; i++) {
                const s = this.scales[i];
                const o = this.offsets[i];
                let metricAcc = 0;
                if (this.metric === 1) {
                    for (let sd = 0; sd < tierDims; sd++) {
                        metricAcc += qPool[sd] * (o + s * tier.value(i, sd));
                    }
                    metricAcc = 1 - Math.max(-1, Math.min(1, metricAcc * pool));
                } else {
                    for (let sd = 0; sd < tierDims; sd++) {
                        const diff = qPool[sd] - (o + s * tier.value(i, sd));
                        metricAcc += diff * diff;
                    }
                }
                if (metricAcc < candMax) {
                    let worst = 0;
                    for (let j = 1; j < C; j++) if (candDist[j] > candDist[worst]) worst = j;
                    candDist[worst] = metricAcc;
                    candId[worst] = i;
                    candMax = 0;
                    for (let j = 0; j < C; j++) if (candDist[j] > candMax) candMax = candDist[j];
                }
            }
            ids = [];
            for (let j = 0; j < C; j++) if (candId[j] >= 0) ids.push(candId[j]);
        }

        const rows = await this.fetchRows(ids, options);

        const exact = [];
        for (const id of ids) {
            const row = rows.get(id);
            const s = this.scales[id];
            const o = this.offsets[id];
            let acc = 0;
            if (this.metric === 1) {
                for (let d = 0; d < dim; d++) acc += q[d] * (o + s * row[d]);
                acc = 1 - Math.max(-1, Math.min(1, acc));
            } else {
                for (let d = 0; d < dim; d++) {
                    const diff = q[d] - (o + s * row[d]);
                    acc += diff * diff;
                }
            }
            exact.push([acc, id]);
        }
        exact.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        // Rerank accumulates squared L2; the API contract (README "Distance
        // values") reports Euclidean, matching PancakeIndex.search.
        const sqrtL2 = this.metric !== 1;
        return {
            results: exact.slice(0, k).map(([distance, id]) => ({ id, distance: sqrtL2 ? Math.sqrt(distance) : distance })),
            rerank: ids.length,
            tier: tier.name,
        };
    }

    async close() {
        if (this.source && typeof this.source.close === 'function') await this.source.close();
    }
}

// Build a WASM-backed scanner for a sketch artifact's resident tier, usable
// as the `scanner` option to PancakeSketchArtifact.search(). The engine's
// pancake_sketch_scan SIMD kernel runs the O(count*sketchDims) resident scan
// that is otherwise the browser query bottleneck. `loadEngine` is the
// entrypoint's own async engine loader (Node/web/workerd all supply one);
// the sketch tier is staged in the engine heap once at creation.
async function createSketchScanner(loadEngine, artifact, options = {}) {
    if (typeof loadEngine !== 'function') {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'createSketchScanner() requires an engine loader');
    }
    if (!artifact || !(artifact instanceof PancakeSketchArtifact)) {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'createSketchScanner() requires a sketch artifact');
    }
    const engine = await loadEngine();
    const { count, metric } = artifact;
    const tierName = options.tier === 'micro' ? 'micro' : 'full';
    if (tierName === 'micro' && !artifact.microDims) {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'artifact has no micro tier');
    }
    if (tierName === 'full' && !artifact.sketches) {
        throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'full sketches are not resident yet (staged open still in stage 1; await artifact.fullyResident)');
    }
    const sketchDims = tierName === 'micro' ? artifact.microDims : artifact.sketchDims;
    const tierBits = tierName === 'micro' ? artifact.microBits : artifact.sketchBits;
    const maxC = Math.max(1, Math.trunc(options.maxRerank || 1024));

    // The kernel reads u8 sketch values; expand 4-bit nibbles to their
    // reconstructed u8 once so the scan needs no per-element unpacking.
    const expanded = new Uint8Array(count * sketchDims);
    if (tierBits === 4) {
        for (let i = 0; i < count; i++) {
            for (let sd = 0; sd < sketchDims; sd++) expanded[i * sketchDims + sd] = tierName === 'micro' ? artifact.microValue(i, sd) : artifact.sketchValue(i, sd);
        }
    } else {
        expanded.set(tierName === 'micro' ? artifact.microSketches : artifact.sketches);
    }

    const sketchesPtr = engine._emsc_malloc(expanded.length);
    const scalesPtr = engine._emsc_malloc(count * 4);
    const offsetsPtr = engine._emsc_malloc(count * 4);
    const queryPtr = engine._emsc_malloc(sketchDims * 4);
    const outIdsPtr = engine._emsc_malloc(maxC * 4);
    const outDistsPtr = engine._emsc_malloc(maxC * 4);
    if (!sketchesPtr || !scalesPtr || !offsetsPtr || !queryPtr || !outIdsPtr || !outDistsPtr) {
        // Free whichever allocations succeeded before one failed, so a
        // partial failure does not leak WASM heap (matches create()).
        for (const ptr of [sketchesPtr, scalesPtr, offsetsPtr, queryPtr, outIdsPtr, outDistsPtr]) {
            if (ptr) engine._emsc_free(ptr);
        }
        throw pancakeError(PANCAKE_ERROR_CODES.WASM_ALLOCATION_FAILED, 'sketch scanner heap allocation failed');
    }
    engine.HEAPU8.set(expanded, sketchesPtr);
    engine.HEAPF32.set(artifact.scales, scalesPtr >> 2);
    engine.HEAPF32.set(artifact.offsets, offsetsPtr >> 2);

    let disposed = false;
    return {
        maxRerank: maxC,
        metric,
        sketchDims,
        tier: tierName,
        scan(pooledQuery, c) {
            if (disposed) throw pancakeError(PANCAKE_ERROR_CODES.INVALID_ARGUMENT, 'sketch scanner disposed');
            const query = pooledQuery instanceof Float32Array ? pooledQuery : Float32Array.from(pooledQuery);
            // The query is copied into a queryPtr sized for exactly sketchDims
            // floats: an over- or under-sized input would read/write outside
            // that buffer in the WASM heap. Validate before the copy.
            if (query.length !== sketchDims) {
                throw pancakeError(PANCAKE_ERROR_CODES.DIMENSION_MISMATCH,
                    `scan() pooled query has ${query.length} values, expected ${sketchDims}`,
                    { expected: sketchDims, actual: query.length });
            }
            for (let i = 0; i < query.length; i++) {
                if (!Number.isFinite(query[i])) {
                    throw pancakeError(PANCAKE_ERROR_CODES.INVALID_VECTOR,
                        'scan() pooled query contains a non-finite value', { index: i });
                }
            }
            engine.HEAPF32.set(query, queryPtr >> 2);
            const topC = Math.min(Math.max(1, Math.trunc(c)), maxC);
            const n = engine._pancake_sketch_scan(
                sketchesPtr, scalesPtr, offsetsPtr, count, sketchDims, queryPtr, metric, topC, outIdsPtr, outDistsPtr
            );
            return Array.from(engine.HEAPU32.subarray(outIdsPtr >> 2, (outIdsPtr >> 2) + n));
        },
        dispose() {
            if (disposed) return;
            disposed = true;
            for (const ptr of [sketchesPtr, scalesPtr, offsetsPtr, queryPtr, outIdsPtr, outDistsPtr]) {
                engine._emsc_free(ptr);
            }
        },
    };
}

module.exports = {
    PancakeSketchArtifact,
    createSketchScanner,
    buildSketchArtifact,
    buildSketchArtifactBytes,
    buildSketchArtifactFile,
    exportSketchArtifact,
};
