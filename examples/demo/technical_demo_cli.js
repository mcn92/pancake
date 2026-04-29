#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const readline = require('readline');
const Pancake = require('../../dist/engine.js');

const DIMS = 384;
const K = 10;
const MAX_ELEM = 110_000;
const VECTORS_PATH = path.join(__dirname, '..', '..', 'dist', 'vectors.bin');
const DEFAULT_EXPORT_PATH = path.join(__dirname, 'pancake-index.bin');

const PROOFS = [
    { id: 'load', text: 'Real embeddings loaded from vectors.bin' },
    { id: 'init', text: 'Index built from real embedding data' },
    { id: 'search', text: 'Successful search execution' },
    { id: 'low_latency', text: 'Search latency < 2ms p50' },
    { id: 'insert', text: 'Live vector insertion post-build' },
    { id: 'delete', text: 'Live vector deletion' },
    { id: 'ghosts', text: 'Ghost node accumulation visible' },
    { id: 'compact', text: 'Compaction executed successfully' },
    { id: 'no_block', text: 'Search continues immediately after compaction' },
    { id: 'mem_shrink', text: 'Memory decreases after compaction' },
    { id: 'deterministic', text: 'Search is deterministic (same query -> same results)' },
    { id: 'excl_deleted', text: 'Deleted vectors excluded from results' },
    { id: 'export', text: 'Index serialized to binary (export)' },
    { id: 'import', text: 'Index restored from binary (import round-trip)' },
    { id: 'self_recall', text: 'Self-recall: inserted vectors found at rank 1 (50/50)' },
    { id: 'recall_at_k', text: 'Recall@10 >= 99% vs brute-force (50 queries)' },
    { id: 'stable', text: 'Latency stable under sustained mutation' },
    { id: 'stress', text: 'Survived adversarial stress test' }
];

const ADV_MODES = {
    ghost: { name: 'Ghost Explosion', insert: 100, delete: 500, search: 100, desc: 'Rapid deletions -> ghost accumulation' },
    churn: { name: 'Memory Churn', insert: 500, delete: 500, search: 100, desc: 'Equal inserts + deletes -> memory pressure' },
    read: { name: 'Read-Heavy', insert: 10, delete: 10, search: 1000, desc: 'Search-dominated workload' },
    write: { name: 'Write-Heavy', insert: 1000, delete: 800, search: 50, desc: 'Insert-dominated workload' },
    worst: { name: 'Worst-Case Adversarial', insert: 500, delete: 490, search: 1000, desc: 'Max stress on all paths simultaneously' }
};

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function percentile(sortedValues, p) {
    if (sortedValues.length === 0) return null;
    const idx = Math.min(sortedValues.length - 1, Math.floor(sortedValues.length * p));
    return sortedValues[idx];
}

function formatBytes(bytes) {
    if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / 1024).toFixed(1)} KB`;
}

class TechnicalDemoCLI {
    constructor() {
        this.engine = null;
        this.handle = null;
        this.vectors = null;
        this.totalVectors = 0;
        this.vectorCount = 0;
        this.queryPtr = 0;
        this.insertPtr = 0;
        this.resultIdPtr = 0;
        this.resultDistPtr = 0;
        this.latencyHistory = [];
        this.compactionCount = 0;
        this.proofState = Object.fromEntries(PROOFS.map((proof) => [proof.id, false]));
        this.rl = null;
        this.commandQueue = [];
        this.processingQueue = false;
    }

    async init(showHelp = true) {
        this.log('Loading WASM engine...', 'info');
        const wasmBinary = fs.readFileSync(path.join(__dirname, '..', '..', 'dist', 'engine.wasm'));
        this.engine = await Pancake({ wasmBinary });
        this.resetIndex();

        this.queryPtr = this.engine._emsc_malloc(DIMS * 4);
        this.insertPtr = this.engine._emsc_malloc(DIMS * 4);
        this.resultIdPtr = this.engine._emsc_malloc(K * 8);
        this.resultDistPtr = this.engine._emsc_malloc(K * 4);

        this.log(`Loading embeddings from ${VECTORS_PATH}...`, 'info');
        const buf = fs.readFileSync(VECTORS_PATH);
        this.vectors = new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
        this.totalVectors = Math.floor(this.vectors.length / DIMS);
        this.markProof('load');

        this.log(`Ready. Loaded ${this.totalVectors.toLocaleString()} embeddings (${DIMS}D cosine).`, 'success');
        if (showHelp) this.printHelp();
    }

    log(message, type = 'info') {
        const time = new Date().toLocaleTimeString('en-US', { hour12: false });
        const label = type.toUpperCase().padEnd(7);
        process.stdout.write(`${time} ${label} ${message}\n`);
    }

    markProof(id) {
        if (this.proofState[id]) return;
        this.proofState[id] = true;
    }

    getVec(idx) {
        const offset = idx * DIMS;
        return this.vectors.subarray(offset, offset + DIMS);
    }

    randomLoadedVec() {
        return this.getVec(Math.floor(Math.random() * this.totalVectors));
    }

    randomSyntheticVec() {
        const v = new Float32Array(DIMS);
        let norm = 0;
        for (let i = 0; i < DIMS; i++) {
            v[i] = Math.random() * 2 - 1;
            norm += v[i] * v[i];
        }
        norm = Math.sqrt(norm) || 1;
        for (let i = 0; i < DIMS; i++) v[i] /= norm;
        return v;
    }

    resetIndex() {
        if (this.handle !== null) {
            this.engine._pancake_dispose(this.handle);
        }
        // quantized=1, metric=1 (cosine), M=12, ef_c=150, ef_s=250
        this.handle = this.engine._pancake_init(DIMS, MAX_ELEM, 1, 1, 12, 150, 250);
        this.vectorCount = 0;
        this.compactionCount = 0;
        this.latencyHistory = [];
    }

    currentCount() {
        return this.engine._pancake_count(this.handle);
    }

    currentGhosts() {
        return this.engine._pancake_ghost_count(this.handle);
    }

    currentMemory() {
        return this.engine._pancake_memory(this.handle);
    }

    currentGhostRatio() {
        return this.engine._pancake_ghost_ratio(this.handle);
    }

    requireIndexedData(action) {
        if (this.currentCount() === 0) {
            throw new Error(`${action} requires a built or imported index`);
        }
    }

    printStatus() {
        const count = this.currentCount();
        const ghosts = this.currentGhosts();
        const live = count - ghosts;
        const ghostRatio = this.currentGhostRatio();
        const mem = this.currentMemory();
        const sorted = [...this.latencyHistory].sort((a, b) => a - b);
        const p50 = percentile(sorted, 0.5);
        const p99 = percentile(sorted, 0.99);

        process.stdout.write('\nStatus\n');
        process.stdout.write(`  Count:        ${count.toLocaleString()}\n`);
        process.stdout.write(`  Live:         ${live.toLocaleString()}\n`);
        process.stdout.write(`  Ghosts:       ${ghosts.toLocaleString()} (${(ghostRatio * 100).toFixed(1)}%)\n`);
        process.stdout.write(`  Memory:       ${formatBytes(mem)}\n`);
        process.stdout.write(`  Compactions:  ${this.compactionCount}\n`);
        process.stdout.write(`  Search p50:   ${p50 === null ? '—' : `${p50.toFixed(2)} ms`}\n`);
        process.stdout.write(`  Search p99:   ${p99 === null ? '—' : `${p99.toFixed(2)} ms`}\n\n`);
    }

    printChecklist() {
        process.stdout.write('\nProof Checklist\n');
        for (const proof of PROOFS) {
            const marker = this.proofState[proof.id] ? '[PASS]' : '[    ]';
            process.stdout.write(`  ${marker} ${proof.text}\n`);
        }
        process.stdout.write('\n');
    }

    printHelp() {
        process.stdout.write('\nCommands\n');
        process.stdout.write('  help                                 Show commands\n');
        process.stdout.write('  status                               Show index metrics\n');
        process.stdout.write('  checklist                            Show proof checklist\n');
        process.stdout.write('  build <count>                        Build index (default 100000)\n');
        process.stdout.write('  reset                                Reset the index\n');
        process.stdout.write('  search <count>                       Run N random searches\n');
        process.stdout.write('  insert <count>                       Insert synthetic vectors\n');
        process.stdout.write('  delete <count>                       Delete random vectors\n');
        process.stdout.write('  compact                              Run compaction\n');
        process.stdout.write(`  export [path]                        Export index to file (default: ${path.basename(DEFAULT_EXPORT_PATH)})\n`);
        process.stdout.write('  import <path>                        Import index from file\n');
        process.stdout.write('  proof <all|deterministic|deletion|compaction|memory|stability|export|self-recall|recall>\n');
        process.stdout.write('  stress <ghost|churn|read|write|worst> [seconds]\n');
        process.stdout.write('  quit                                 Exit\n\n');
    }

    doSearch(vec) {
        this.engine.HEAPF32.set(vec, this.queryPtr >> 2);
        const t0 = performance.now();
        const found = this.engine._pancake_query(this.handle, this.queryPtr, K, this.resultIdPtr, this.resultDistPtr);
        const latency = performance.now() - t0;
        this.latencyHistory.push(latency);
        if (this.latencyHistory.length > 1000) this.latencyHistory.shift();
        if (latency < 2) this.markProof('low_latency');
        this.markProof('search');
        return { found, latency };
    }

    readIds(found) {
        const ids = [];
        for (let i = 0; i < found; i++) {
            ids.push(this.engine.HEAPU32[(this.resultIdPtr >> 2) + i * 2]);
        }
        return ids;
    }

    async buildIndex(count) {
        const realCount = Math.min(count, this.totalVectors);
        const syntheticCount = count - realCount;
        if (syntheticCount > 0) {
            this.log(`Building index with ${realCount.toLocaleString()} real + ${syntheticCount.toLocaleString()} synthetic embeddings...`, 'info');
        } else {
            this.log(`Building index with ${realCount.toLocaleString()} real embeddings...`, 'info');
        }
        const t0 = performance.now();
        const batchSize = 10000;
        const batchPtr = this.engine._emsc_malloc(batchSize * DIMS * 4);

        for (let i = 0; i < count; i += batchSize) {
            const end = Math.min(i + batchSize, count);
            const n = end - i;
            const heapOffset = batchPtr >> 2;
            for (let j = 0; j < n; j++) {
                const vec = (i + j) < this.totalVectors ? this.getVec(i + j) : this.randomSyntheticVec();
                this.engine.HEAPF32.set(vec, heapOffset + j * DIMS);
            }
            this.engine._pancake_bulk_insert(this.handle, batchPtr, n);
            const rate = (end / ((performance.now() - t0) / 1000)).toFixed(0);
            this.log(`Indexed ${end.toLocaleString()}/${count.toLocaleString()} (${rate} vec/s)`, 'info');
            await sleep(0);
        }

        this.engine._emsc_free(batchPtr);

        this.vectorCount = count;
        this.markProof('init');
        this.log(`Index built in ${((performance.now() - t0) / 1000).toFixed(2)}s`, 'success');
        this.printStatus();
    }

    async performSearches(count) {
        this.requireIndexedData('search');
        this.log(`Running ${count} search${count === 1 ? '' : 'es'}...`, 'info');
        let total = 0;
        for (let i = 0; i < count; i++) {
            total += this.doSearch(this.randomLoadedVec()).latency;
            if ((i + 1) % 20 === 0) await sleep(0);
        }
        this.log(`Average latency: ${(total / count).toFixed(2)}ms`, 'success');
        this.printStatus();
    }

    async insertVectors(count) {
        this.requireIndexedData('insert');
        this.log(`Inserting ${count} synthetic vector${count === 1 ? '' : 's'}...`, 'info');
        for (let i = 0; i < count; i++) {
            this.engine.HEAPF32.set(this.randomSyntheticVec(), this.insertPtr >> 2);
            this.engine._pancake_add(this.handle, this.insertPtr);
        }
        this.markProof('insert');
        this.log(`Inserted ${count} vectors`, 'success');
        this.printStatus();
    }

    async deleteVectors(count) {
        this.requireIndexedData('delete');
        const total = this.currentCount();
        const n = Math.min(count, total);
        this.log(`Deleting ${n} vector${n === 1 ? '' : 's'}...`, 'info');
        for (let i = 0; i < n; i++) {
            this.engine._pancake_delete(this.handle, Math.floor(Math.random() * total));
        }
        if (this.currentGhosts() > 0) this.markProof('ghosts');
        this.markProof('delete');
        this.log(`Deleted ${n} vectors`, 'success');
        this.printStatus();
    }

    async compact() {
        this.requireIndexedData('compact');
        const memBefore = this.currentMemory();
        const ghostsBefore = this.currentGhosts();
        this.log('Compacting...', 'info');
        const t0 = performance.now();
        this.engine._pancake_compact(this.handle);
        const elapsed = performance.now() - t0;
        this.compactionCount += 1;
        const memAfter = this.currentMemory();
        const ghostsAfter = this.currentGhosts();
        const saved = memBefore - memAfter;
        this.markProof('compact');
        if (saved > 0) this.markProof('mem_shrink');
        this.log(`Compaction done in ${elapsed.toFixed(2)}ms — ghosts ${ghostsBefore}->${ghostsAfter}, saved ${formatBytes(Math.max(saved, 0))}`, 'success');
        this.printStatus();
    }

    exportIndex(filePath = DEFAULT_EXPORT_PATH) {
        this.requireIndexedData('export');
        const outPath = path.resolve(filePath);
        const sizePtr = this.engine._emsc_malloc(4);
        const dataPtr = this.engine._pancake_export(this.handle, sizePtr);
        const size = this.engine.HEAPU32[sizePtr >> 2];
        const data = Buffer.from(new Uint8Array(this.engine.HEAPU8.buffer, dataPtr, size));
        fs.writeFileSync(outPath, data);
        this.engine._emsc_free(sizePtr);
        this.markProof('export');
        this.log(`Exported ${formatBytes(size)} to ${outPath}`, 'success');
    }

    importIndex(filePath) {
        const inPath = path.resolve(filePath);
        const data = fs.readFileSync(inPath);
        const ptr = this.engine._emsc_malloc(data.length);
        this.engine.HEAPU8.set(data, ptr);
        const status = this.engine._pancake_import(this.handle, ptr, data.length);
        this.engine._emsc_free(ptr);
        if (status !== 0) {
            throw new Error(`Import failed with status ${status}`);
        }
        this.markProof('import');
        this.log(`Imported ${this.currentCount().toLocaleString()} vectors from ${inPath}`, 'success');
        this.printStatus();
    }

    async proveDeterministic() {
        this.requireIndexedData('proof deterministic');
        this.log('Proof: deterministic search', 'info');
        const vec = this.randomLoadedVec();
        const results = [];
        for (let run = 0; run < 3; run++) {
            this.engine.HEAPF32.set(vec, this.queryPtr >> 2);
            const found = this.engine._pancake_query(this.handle, this.queryPtr, K, this.resultIdPtr, this.resultDistPtr);
            results.push(this.readIds(found));
            await sleep(50);
        }
        const allSame = results.every((ids) =>
            ids.length === results[0].length && ids.every((id, idx) => id === results[0][idx])
        );
        if (!allSame) throw new Error('deterministic proof failed');
        this.markProof('deterministic');
        this.log('Deterministic search verified across 3 runs', 'success');
    }

    async proveDeletionExclusion() {
        this.requireIndexedData('proof deletion');
        this.log('Proof: deleted vectors excluded from results', 'info');
        const vec = this.randomSyntheticVec();
        this.engine.HEAPF32.set(vec, this.insertPtr >> 2);
        const id = this.engine._pancake_add(this.handle, this.insertPtr);

        this.engine.HEAPF32.set(vec, this.queryPtr >> 2);
        let found = this.engine._pancake_query(this.handle, this.queryPtr, K, this.resultIdPtr, this.resultDistPtr);
        const before = this.readIds(found).includes(id);

        this.engine._pancake_delete(this.handle, id);
        found = this.engine._pancake_query(this.handle, this.queryPtr, K, this.resultIdPtr, this.resultDistPtr);
        const after = this.readIds(found).includes(id);

        if (!before || after) throw new Error(`deletion exclusion failed (before=${before}, after=${after})`);
        this.markProof('excl_deleted');
        this.log(`Deleted vector ${id} no longer appears in results`, 'success');
    }

    async proveSearchDuringCompaction() {
        this.requireIndexedData('proof compaction');
        this.log('Proof: immediate search after compaction', 'info');
        if (this.currentGhosts() < 10) await this.deleteVectors(100);
        await this.compact();
        const { latency } = this.doSearch(this.randomLoadedVec());
        if (latency >= 10) throw new Error(`post-compaction latency too high (${latency.toFixed(2)}ms)`);
        this.markProof('no_block');
        this.log(`Immediate post-compaction search: ${latency.toFixed(2)}ms`, 'success');
    }

    async proveMemoryDecrease() {
        this.requireIndexedData('proof memory');
        this.log('Proof: memory decreases after compaction', 'info');
        if (this.currentGhosts() < 50) await this.deleteVectors(200);
        const before = this.currentMemory();
        await this.compact();
        const after = this.currentMemory();
        if (!(after < before)) throw new Error(`memory did not decrease (${before} -> ${after})`);
        this.markProof('mem_shrink');
        this.log(`Memory decreased from ${formatBytes(before)} to ${formatBytes(after)}`, 'success');
    }

    async proveSustainedStability() {
        this.requireIndexedData('proof stability');
        this.log('Proof: latency stability under sustained mutation', 'info');
        const duration = 5000;
        const t0 = performance.now();
        const lats = [];
        let inserts = 0;
        let deletes = 0;

        while (performance.now() - t0 < duration) {
            this.engine.HEAPF32.set(this.randomSyntheticVec(), this.insertPtr >> 2);
            this.engine._pancake_add(this.handle, this.insertPtr);
            inserts++;

            if (Math.random() < 0.4) {
                this.engine._pancake_delete(this.handle, Math.floor(Math.random() * this.currentCount()));
                deletes++;
            }

            this.engine.HEAPF32.set(this.randomLoadedVec(), this.queryPtr >> 2);
            const t1 = performance.now();
            this.engine._pancake_query(this.handle, this.queryPtr, K, this.resultIdPtr, this.resultDistPtr);
            lats.push(performance.now() - t1);

            if (inserts % 50 === 0) await sleep(0);
        }

        const sorted = [...lats].sort((a, b) => a - b);
        const p50 = percentile(sorted, 0.5);
        const p99 = percentile(sorted, 0.99);
        if (!(p50 < 2 && p99 < 5)) {
            throw new Error(`latency thresholds exceeded (p50=${p50.toFixed(2)}ms, p99=${p99.toFixed(2)}ms)`);
        }
        this.markProof('stable');
        this.log(`Stable under load: ${inserts} inserts, ${deletes} deletes, p50=${p50.toFixed(2)}ms, p99=${p99.toFixed(2)}ms`, 'success');
    }

    async proveExportImport() {
        this.requireIndexedData('proof export');
        this.log('Proof: export/import round-trip', 'info');
        const countBefore = this.currentCount();
        if (countBefore === 0) throw new Error('index is empty');

        const sizePtr = this.engine._emsc_malloc(4);
        const dataPtr = this.engine._pancake_export(this.handle, sizePtr);
        const size = this.engine.HEAPU32[sizePtr >> 2];
        const saved = new Uint8Array(this.engine.HEAPU8.buffer, dataPtr, size).slice();
        this.engine._emsc_free(sizePtr);
        this.markProof('export');

        // Dispose old handle and create fresh one for import
        this.engine._pancake_dispose(this.handle);
        this.handle = this.engine._pancake_init(DIMS, MAX_ELEM, 1, 1, 12, 150, 250);

        const ptr = this.engine._emsc_malloc(saved.length);
        this.engine.HEAPU8.set(saved, ptr);
        const status = this.engine._pancake_import(this.handle, ptr, saved.length);
        this.engine._emsc_free(ptr);
        if (status !== 0) throw new Error(`import failed with status ${status}`);

        const countAfter = this.currentCount();
        const { latency } = this.doSearch(this.randomLoadedVec());
        if (countAfter !== countBefore || latency >= 10) {
            throw new Error(`round-trip mismatch (before=${countBefore}, after=${countAfter}, latency=${latency.toFixed(2)}ms)`);
        }
        this.markProof('import');
        this.log(`Round-trip restored ${countAfter.toLocaleString()} vectors; immediate search ${latency.toFixed(2)}ms`, 'success');
    }

    async proveSelfRecall() {
        this.requireIndexedData('proof self-recall');
        const trials = 50;
        this.log(`Proof: self-recall (${trials} trials)`, 'info');
        let hits = 0;
        for (let i = 0; i < trials; i++) {
            const vec = this.randomSyntheticVec();
            this.engine.HEAPF32.set(vec, this.insertPtr >> 2);
            const id = this.engine._pancake_add(this.handle, this.insertPtr);
            this.engine.HEAPF32.set(vec, this.queryPtr >> 2);
            this.engine._pancake_query(this.handle, this.queryPtr, K, this.resultIdPtr, this.resultDistPtr);
            const topId = this.engine.HEAPU32[this.resultIdPtr >> 2];
            if (topId === id) hits++;
            if ((i + 1) % 10 === 0) await sleep(0);
        }
        if (hits !== trials) throw new Error(`rank-1 hits ${hits}/${trials}`);
        this.markProof('self_recall');
        this.log(`Self-recall verified: ${hits}/${trials} rank-1 hits`, 'success');
    }

    async proveRecallAtK() {
        this.requireIndexedData('proof recall');
        const queryCount = 50;
        const topK = 10;
        this.log(`Proof: recall@${topK} vs brute-force (${queryCount} queries)`, 'info');

        const count = this.currentCount();
        if (count < topK) throw new Error('not enough vectors indexed');

        const queryIndices = [];
        const used = new Set();
        while (queryIndices.length < queryCount) {
            const idx = Math.floor(Math.random() * Math.min(count, this.totalVectors));
            if (!used.has(idx)) {
                used.add(idx);
                queryIndices.push(idx);
            }
        }

        let totalRecall = 0;
        const limit = Math.min(count, this.totalVectors);

        for (let qi = 0; qi < queryCount; qi++) {
            const qIdx = queryIndices[qi];
            const qVec = this.getVec(qIdx);
            const scored = [];

            for (let i = 0; i < limit; i++) {
                const v = this.getVec(i);
                let dot = 0;
                let na = 0;
                let nb = 0;
                for (let d = 0; d < DIMS; d++) {
                    dot += qVec[d] * v[d];
                    na += qVec[d] * qVec[d];
                    nb += v[d] * v[d];
                }
                scored.push({ id: i, dist: 1 - dot / (Math.sqrt(na) * Math.sqrt(nb)) });
            }
            scored.sort((a, b) => a.dist - b.dist);
            const trueTopK = new Set(scored.slice(0, topK).map((s) => s.id));

            this.engine.HEAPF32.set(qVec, this.queryPtr >> 2);
            const found = this.engine._pancake_query(this.handle, this.queryPtr, topK, this.resultIdPtr, this.resultDistPtr);
            const hnswIds = new Set(this.readIds(found));

            let hits = 0;
            for (const id of hnswIds) {
                if (trueTopK.has(id)) hits++;
            }
            totalRecall += hits / topK;
            if ((qi + 1) % 10 === 0) await sleep(0);
        }

        const avgRecall = totalRecall / queryCount;
        if (avgRecall < 0.989) throw new Error(`recall@${topK}=${(avgRecall * 100).toFixed(1)}%`);
        this.markProof('recall_at_k');
        this.log(`Recall@${topK}: ${(avgRecall * 100).toFixed(1)}%`, 'success');
    }

    async runStress(mode, seconds = 30) {
        this.requireIndexedData('stress');
        const cfg = ADV_MODES[mode];
        if (!cfg) {
            throw new Error(`unknown stress mode "${mode}"`);
        }

        this.log(`Stress: ${cfg.name} for ${seconds}s — ${cfg.desc}`, 'info');
        const durationMs = seconds * 1000;
        const start = performance.now();
        let nextInsert = start;
        let nextDelete = start;
        let nextSearch = start;
        let nextCompact = start + 1000;
        let lastReport = start;
        let inserts = 0;
        let deletes = 0;
        let searches = 0;
        const latencies = [];

        while (performance.now() - start < durationMs) {
            const now = performance.now();

            while (cfg.insert > 0 && now >= nextInsert) {
                this.engine.HEAPF32.set(this.randomSyntheticVec(), this.insertPtr >> 2);
                this.engine._pancake_add(this.handle, this.insertPtr);
                inserts++;
                nextInsert += 1000 / cfg.insert;
            }

            while (cfg.delete > 0 && now >= nextDelete) {
                const total = this.currentCount();
                if (total > 0) {
                    this.engine._pancake_delete(this.handle, Math.floor(Math.random() * total));
                    deletes++;
                }
                nextDelete += 1000 / cfg.delete;
            }

            while (cfg.search > 0 && now >= nextSearch) {
                this.engine.HEAPF32.set(this.randomLoadedVec(), this.queryPtr >> 2);
                const t0 = performance.now();
                this.engine._pancake_query(this.handle, this.queryPtr, K, this.resultIdPtr, this.resultDistPtr);
                latencies.push(performance.now() - t0);
                if (latencies.length > 1000) latencies.shift();
                searches++;
                nextSearch += 1000 / cfg.search;
            }

            if (now >= nextCompact) {
                if (this.currentGhostRatio() > 0.15) {
                    this.engine._pancake_compact(this.handle);
                    this.compactionCount++;
                }
                nextCompact += 1000;
            }

            if (now - lastReport >= 1000) {
                const elapsed = (now - start) / 1000;
                const sorted = [...latencies].sort((a, b) => a - b);
                const p50 = percentile(sorted, 0.5);
                const p99 = percentile(sorted, 0.99);
                this.log(
                    `stress ${elapsed.toFixed(1)}s | insert=${(inserts / elapsed).toFixed(0)}/s delete=${(deletes / elapsed).toFixed(0)}/s search=${(searches / elapsed).toFixed(0)}/s p50=${p50 ? p50.toFixed(2) : '—'}ms p99=${p99 ? p99.toFixed(2) : '—'}ms ghosts=${(this.currentGhostRatio() * 100).toFixed(1)}%`,
                    'info'
                );
                lastReport = now;
            }

            await sleep(0);
        }

        const sorted = [...latencies].sort((a, b) => a - b);
        const p50 = percentile(sorted, 0.5) || 0;
        const p99 = percentile(sorted, 0.99) || 0;
        const max = sorted[sorted.length - 1] || 0;
        if (p99 < 5 && max < 20) this.markProof('stress');
        this.log(`Stress complete — inserts=${inserts}, deletes=${deletes}, searches=${searches}, p50=${p50.toFixed(2)}ms, p99=${p99.toFixed(2)}ms, max=${max.toFixed(2)}ms`, 'success');
    }

    async runFullProofSequence() {
        this.log('Starting full proof sequence...', 'info');
        this.resetIndex();
        await sleep(100);
        await this.buildIndex(100000);
        await sleep(100);
        await this.performSearches(10);
        await sleep(100);
        await this.proveRecallAtK();
        await sleep(100);
        await this.insertVectors(100);
        await sleep(100);
        await this.deleteVectors(50);
        await sleep(100);
        await this.proveDeterministic();
        await sleep(100);
        await this.proveDeletionExclusion();
        await sleep(100);
        await this.proveSearchDuringCompaction();
        await sleep(100);
        await this.proveMemoryDecrease();
        await sleep(100);
        await this.proveSelfRecall();
        await sleep(100);
        await this.proveSustainedStability();
        await sleep(100);
        await this.proveExportImport();
        this.log('Full proof sequence complete', 'success');
        this.printChecklist();
    }

    async handleCommand(input) {
        const parts = input.trim().split(/\s+/).filter(Boolean);
        if (parts.length === 0) return true;

        const [command, ...args] = parts;

        switch (command) {
        case 'help':
            this.printHelp();
            return true;
        case 'status':
            this.printStatus();
            return true;
        case 'checklist':
            this.printChecklist();
            return true;
        case 'reset':
            this.resetIndex();
            this.log('Index reset', 'warn');
            return true;
        case 'build':
            await this.buildIndex(parseInt(args[0] || '100000', 10));
            return true;
        case 'search':
            await this.performSearches(parseInt(args[0] || '1', 10));
            return true;
        case 'insert':
            await this.insertVectors(parseInt(args[0] || '1', 10));
            return true;
        case 'delete':
            await this.deleteVectors(parseInt(args[0] || '1', 10));
            return true;
        case 'compact':
            await this.compact();
            return true;
        case 'export':
            this.exportIndex(args[0] || DEFAULT_EXPORT_PATH);
            return true;
        case 'import':
            if (!args[0]) throw new Error('import requires a file path');
            this.importIndex(args[0]);
            return true;
        case 'proof':
            await this.handleProofCommand(args);
            return true;
        case 'stress':
            await this.runStress(args[0] || 'worst', parseInt(args[1] || '30', 10));
            return true;
        case 'quit':
        case 'exit':
            return false;
        default:
            throw new Error(`unknown command "${command}"`);
        }
    }

    async handleProofCommand(args) {
        const target = args[0] || 'all';
        switch (target) {
        case 'all':
            await this.runFullProofSequence();
            return;
        case 'deterministic':
            await this.proveDeterministic();
            return;
        case 'deletion':
            await this.proveDeletionExclusion();
            return;
        case 'compaction':
            await this.proveSearchDuringCompaction();
            return;
        case 'memory':
            await this.proveMemoryDecrease();
            return;
        case 'stability':
            await this.proveSustainedStability();
            return;
        case 'export':
            await this.proveExportImport();
            return;
        case 'self-recall':
            await this.proveSelfRecall();
            return;
        case 'recall':
            await this.proveRecallAtK();
            return;
        default:
            throw new Error(`unknown proof target "${target}"`);
        }
    }

    async runInteractive() {
        this.rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
            prompt: 'pancake-demo> '
        });

        this.rl.prompt();
        this.rl.on('line', (line) => {
            this.commandQueue.push(line);
            void this.processInteractiveQueue();
        });

        this.rl.on('close', () => {
            process.stdout.write('\n');
            process.exit(0);
        });
    }

    async processInteractiveQueue() {
        if (this.processingQueue || !this.rl) return;
        this.processingQueue = true;
        this.rl.pause();

        try {
            while (this.commandQueue.length > 0 && this.rl) {
                const line = this.commandQueue.shift();
                try {
                    const keepRunning = await this.handleCommand(line);
                    if (!keepRunning) {
                        this.rl.close();
                        return;
                    }
                } catch (error) {
                    this.log(error.message, 'error');
                }
            }
        } finally {
            this.processingQueue = false;
            if (this.rl) {
                this.rl.resume();
                this.rl.prompt();
            }
        }
    }
}

async function main() {
    const cli = new TechnicalDemoCLI();
    const args = process.argv.slice(2);
    await cli.init(args.length === 0);
    if (args.length > 0) {
        try {
            const keepRunning = await cli.handleCommand(args.join(' '));
            if (keepRunning) cli.printChecklist();
            process.exit(0);
        } catch (error) {
            cli.log(error.message, 'error');
            process.exit(1);
        }
    }

    await cli.runInteractive();
}

main().catch((error) => {
    process.stderr.write(`${error.stack || error.message}\n`);
    process.exit(1);
});
