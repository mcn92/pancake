#!/usr/bin/env node
/**
 * Technical Demo — Worker Edition
 *
 * Runs the same proof suite as technical_demo_cli.js but against
 * the Pancake Search Cloudflare Worker REST API (localhost:8787).
 * This is the primary worker demo for the 384D real-embedding flow using
 * `dist/vectors.bin` over the HTTP API.
 *
 * Checks: init, bulk insert, search, recall, deterministic search,
 * export/import round-trip, latency, and sustained load.
 *
 * Usage:
 *   1. Start the worker:  npx wrangler dev --port 8787
 *   2. Run this demo:     node technical_demo_worker.js [command]
 */

'use strict';

const fs = require('fs');
const path = require('path');
const readline = require('readline');
const http = require('http');

const BASE_URL = process.env.PANCAKE_URL || 'http://localhost:8787';
const DIMS = 384;
const K = 10;
const MAX_ELEM = 5_000;
const LATENCY_CHECK_QUERIES = 50;
const AVG_LATENCY_THRESHOLD_MS = 10;
const P99_LATENCY_THRESHOLD_MS = 25;
const VECTORS_PATH = path.join(__dirname, '..', '..', 'dist', 'vectors.bin');
const DEFAULT_EXPORT_PATH = path.join(__dirname, 'pancake-index.bin');
const STRESS_BUILD_COUNT = 4000;
const STRESS_RECALL_QUERIES = 25;

const PROOFS = [
    { id: 'load', text: 'Real embeddings loaded from vectors.bin' },
    { id: 'init', text: 'Index initialized via worker API' },
    { id: 'search', text: 'Successful search via worker API' },
    { id: 'avg_latency', text: `Average end-to-end search latency under ${AVG_LATENCY_THRESHOLD_MS}ms over ${LATENCY_CHECK_QUERIES} HTTP queries` },
    { id: 'p99_latency', text: `P99 end-to-end search latency under ${P99_LATENCY_THRESHOLD_MS}ms over ${LATENCY_CHECK_QUERIES} HTTP queries` },
    { id: 'insert', text: 'Live vector insertion post-build' },
    { id: 'delete', text: 'Live vector deletion via API' },
    { id: 'ghosts', text: 'Ghost node accumulation visible' },
    { id: 'compact', text: 'Compaction executed successfully' },
    { id: 'no_block', text: 'Search works immediately after compaction' },
    { id: 'mem_shrink', text: 'Memory decreases after compaction' },
    { id: 'deterministic', text: 'Search is deterministic (same query -> same results)' },
    { id: 'excl_deleted', text: 'Deleted vectors excluded from results' },
    { id: 'export', text: 'Index serialized to binary (export)' },
    { id: 'import', text: 'Index restored from binary (import round-trip)' },
    { id: 'self_recall', text: 'Self-recall: inserted vectors found at rank 1 (50/50)' },
    { id: 'recall_at_k', text: 'Recall@10 >= 95% vs brute-force (50 queries)' },
    { id: 'stable', text: 'Met sustained-mutation latency thresholds' },
    { id: 'stress', text: 'Completed mixed-workload stress run within demo thresholds' }
];

const ADV_MODES = {
    ghost: { name: 'Ghost Explosion', insert: 100, delete: 500, search: 100, desc: 'Rapid deletions -> ghost accumulation' },
    churn: { name: 'Memory Churn', insert: 500, delete: 500, search: 100, desc: 'Equal inserts + deletes -> memory pressure' },
    read: { name: 'Read-Heavy', insert: 10, delete: 10, search: 1000, desc: 'Search-dominated workload' },
    write: { name: 'Write-Heavy', insert: 1000, delete: 800, search: 50, desc: 'Insert-dominated workload' },
    worst: { name: 'Max-Load Mixed', insert: 500, delete: 490, search: 1000, desc: 'High combined load across insert, delete, and search' }
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

// ============================================================================
// HTTP client (zero dependencies)
// ============================================================================

function request(method, urlPath, body = null, isBinary = false) {
    return new Promise((resolve, reject) => {
        const url = new URL(urlPath, BASE_URL);
        const opts = {
            hostname: url.hostname,
            port: url.port,
            path: url.pathname + url.search,
            method,
            headers: {}
        };

        if (body && !isBinary) {
            const json = JSON.stringify(body);
            opts.headers['Content-Type'] = 'application/json';
            opts.headers['Content-Length'] = Buffer.byteLength(json);
            body = json;
        } else if (body && isBinary) {
            opts.headers['Content-Type'] = 'application/octet-stream';
            opts.headers['Content-Length'] = body.length;
        }

        const req = http.request(opts, res => {
            const chunks = [];
            res.on('data', chunk => chunks.push(chunk));
            res.on('end', () => {
                const buf = Buffer.concat(chunks);
                const ct = res.headers['content-type'] || '';
                if (ct.includes('application/json')) {
                    try { resolve({ status: res.statusCode, data: JSON.parse(buf.toString()), headers: res.headers, raw: buf }); }
                    catch { resolve({ status: res.statusCode, data: buf.toString(), headers: res.headers, raw: buf }); }
                } else {
                    resolve({ status: res.statusCode, data: null, headers: res.headers, raw: buf });
                }
            });
        });

        req.on('error', reject);
        if (body) req.write(body);
        req.end();
    });
}

async function apiGet(path) { return request('GET', path); }
async function apiPost(path, body) { return request('POST', path, body); }
async function apiPostBinary(path, buf) { return request('POST', path, buf, true); }

// ============================================================================
// Demo
// ============================================================================

class TechnicalDemoWorker {
    constructor() {
        this.vectors = null;
        this.totalVectors = 0;
        this.latencyHistory = [];
        this.compactionCount = 0;
        this.liveVectors = new Map();
        this.proofState = Object.fromEntries(PROOFS.map((p) => [p.id, false]));
        this.rl = null;
        this.commandQueue = [];
        this.processingQueue = false;
    }

    async init(showHelp = true) {
        this.log(`Loading embeddings from ${VECTORS_PATH}...`, 'info');
        const buf = fs.readFileSync(VECTORS_PATH);
        this.vectors = new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
        this.totalVectors = Math.floor(this.vectors.length / DIMS);
        this.markProof('load');

        // Check worker is reachable
        try {
            await this.checkHealth();
        } catch {
            this.log(`Cannot reach worker at ${BASE_URL}. Start it with: npx wrangler dev --port 8787`, 'error');
            process.exit(1);
        }
        this.log(`Connected to worker at ${BASE_URL}`, 'success');
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
        return Array.from(this.vectors.subarray(offset, offset + DIMS));
    }

    randomLoadedVec() {
        return this.getVec(Math.floor(Math.random() * this.totalVectors));
    }

    randomSyntheticVec() {
        const v = new Array(DIMS);
        let norm = 0;
        for (let i = 0; i < DIMS; i++) {
            v[i] = Math.random() * 2 - 1;
            norm += v[i] * v[i];
        }
        norm = Math.sqrt(norm) || 1;
        for (let i = 0; i < DIMS; i++) v[i] /= norm;
        return v;
    }

    randomLiveId() {
        if (this.liveVectors.size === 0) return null;
        const ids = Array.from(this.liveVectors.keys());
        return ids[Math.floor(Math.random() * ids.length)];
    }

    // ---- Core operations via HTTP ----

    async checkHealth() {
        const res = await apiGet('/health');
        if (res.status !== 200) throw new Error(`Worker not reachable: ${res.status}`);
        return res.data;
    }

    async initIndex(maxElements = MAX_ELEM, M = 12, efC = 150, efS = 250) {
        const res = await apiPost('/init', {
            dims: DIMS,
            maxElements,
            M,
            efConstruction: efC,
            efSearch: efS
        });
        if (res.status !== 200) throw new Error(`/init failed: ${JSON.stringify(res.data)}`);
        this.markProof('init');
        this.compactionCount = 0;
        this.latencyHistory = [];
        this.liveVectors.clear();
        return res.data;
    }

    async addVector(vec, throwOnFull = true) {
        const res = await apiPost('/add', { vector: vec });
        if (res.status === 409 && !throwOnFull) return null;
        if (res.status !== 200) throw new Error(`/add failed: ${JSON.stringify(res.data)}`);
        this.markProof('insert');
        this.liveVectors.set(res.data.id, Float32Array.from(vec));
        return res.data;
    }

    async addBatch(vecs) {
        const res = await apiPost('/add_batch', { vectors: vecs });
        if (res.status !== 200) throw new Error(`/add_batch failed: ${JSON.stringify(res.data)}`);
        this.markProof('insert');
        for (let i = 0; i < res.data.ids.length; i++) {
            this.liveVectors.set(res.data.ids[i], Float32Array.from(vecs[i]));
        }
        return res.data;
    }

    async search(query, k = K, ef = 250) {
        const t0 = performance.now();
        const res = await apiPost('/search', { query, k, ef });
        const latency = performance.now() - t0;
        if (res.status !== 200) throw new Error(`/search failed: ${JSON.stringify(res.data)}`);
        this.latencyHistory.push(latency);
        if (this.latencyHistory.length > 1000) this.latencyHistory.shift();
        this.markProof('search');
        return { ...res.data, client_latency: latency };
    }

    evaluateLatencyChecks(latencies) {
        if (latencies.length < LATENCY_CHECK_QUERIES) return;
        const avg = latencies.reduce((sum, value) => sum + value, 0) / latencies.length;
        const sorted = [...latencies].sort((a, b) => a - b);
        const p99 = percentile(sorted, 0.99) ?? Infinity;

        if (avg < AVG_LATENCY_THRESHOLD_MS) this.markProof('avg_latency');
        if (p99 < P99_LATENCY_THRESHOLD_MS) this.markProof('p99_latency');
    }

    async deleteVector(id) {
        const res = await apiPost('/delete', { id });
        if (res.status !== 200) throw new Error(`/delete failed: ${JSON.stringify(res.data)}`);
        this.markProof('delete');
        this.liveVectors.delete(id);
        return res.data;
    }

    async compactIndex() {
        const res = await apiPost('/compact', {});
        if (res.status !== 200) throw new Error(`/compact failed: ${JSON.stringify(res.data)}`);
        this.compactionCount += 1;
        this.markProof('compact');
        return res.data;
    }

    async getStats() {
        const res = await apiGet('/stats');
        if (res.status !== 200) throw new Error(`/stats failed: ${res.status}`);
        return res.data;
    }

    async exportIndex() {
        const res = await apiGet('/export');
        if (res.status !== 200) throw new Error(`/export failed: ${res.status}`);
        this.markProof('export');
        return res.raw;
    }

    async importIndex(binary, dims = DIMS) {
        const res = await apiPostBinary(`/import?dims=${dims}`, binary);
        if (res.status !== 200) throw new Error(`/import failed: ${JSON.stringify(res.data)}`);
        this.markProof('import');
        return res.data;
    }

    requireIndexedData(action) {
        // For worker, we check via stats; throw synchronously if clearly empty
        // Most commands will fail at the API level anyway, so this is a convenience check
    }

    printStatus() {
        const sorted = [...this.latencyHistory].sort((a, b) => a - b);
        const p50 = percentile(sorted, 0.5);
        const p99 = percentile(sorted, 0.99);

        process.stdout.write('\nStatus\n');
        process.stdout.write(`  Compactions:  ${this.compactionCount}\n`);
        process.stdout.write(`  Search p50:   ${p50 === null ? '—' : `${p50.toFixed(2)} ms`}\n`);
        process.stdout.write(`  Search p99:   ${p99 === null ? '—' : `${p99.toFixed(2)} ms`}\n\n`);
    }

    async printFullStatus() {
        const stats = await this.getStats();
        const count = stats.count;
        const ghosts = stats.ghost_count;
        const live = count - ghosts;
        const ghostRatio = stats.ghost_ratio;
        const mem = stats.memory_bytes;
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
        process.stdout.write('\nValidation Checklist\n');
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
        process.stdout.write('  checklist                            Show validation checklist\n');
        process.stdout.write('  build <count>                        Build index (default 5000)\n');
        process.stdout.write('  reset                                Reset the index\n');
        process.stdout.write('  search <count>                       Run N random searches\n');
        process.stdout.write('  insert <count>                       Insert synthetic vectors\n');
        process.stdout.write('  delete <count>                       Delete random vectors\n');
        process.stdout.write('  compact                              Run compaction\n');
        process.stdout.write(`  export [path]                        Export index to file (default: ${path.basename(DEFAULT_EXPORT_PATH)})\n`);
        process.stdout.write('  import <path>                        Import index from file\n');
        process.stdout.write('  validate <all|deterministic|deletion|compaction|memory|stability|export|self-recall|recall>\n');
        process.stdout.write('  proof <...>                          Alias for validate\n');
        process.stdout.write('  stress <ghost|churn|read|write|worst> [seconds]\n');
        process.stdout.write('  quit                                 Exit\n\n');
    }

    // ---- Build ----

    async buildIndex(count) {
        const realCount = Math.min(count, this.totalVectors);
        const syntheticCount = count - realCount;
        if (syntheticCount > 0) {
            this.log(`Building index with ${realCount.toLocaleString()} real + ${syntheticCount.toLocaleString()} synthetic embeddings...`, 'info');
        } else {
            this.log(`Building index with ${realCount.toLocaleString()} real embeddings...`, 'info');
        }

        await this.initIndex();

        const t0 = performance.now();
        const batchSize = 500;

        for (let i = 0; i < count; i += batchSize) {
            const end = Math.min(i + batchSize, count);
            const batch = [];
            for (let j = i; j < end; j++) {
                const vec = j < this.totalVectors ? this.getVec(j) : this.randomSyntheticVec();
                batch.push(vec);
            }
            await this.addBatch(batch);

            const rate = (end / ((performance.now() - t0) / 1000)).toFixed(0);
            this.log(`Indexed ${end.toLocaleString()}/${count.toLocaleString()} (${rate} vec/s)`, 'info');
        }

        this.markProof('init');
        this.log(`Index built in ${((performance.now() - t0) / 1000).toFixed(2)}s`, 'success');
        await this.printFullStatus();
    }

    // ---- Searches ----

    async performSearches(count) {
        this.log(`Running ${count} search${count === 1 ? '' : 'es'}...`, 'info');
        let total = 0;
        const latencies = [];
        for (let i = 0; i < count; i++) {
            const res = await this.search(this.randomLoadedVec());
            total += res.client_latency;
            latencies.push(res.client_latency);
            if ((i + 1) % 20 === 0) await sleep(0);
        }
        this.evaluateLatencyChecks(latencies);
        const sorted = [...latencies].sort((a, b) => a - b);
        const p99 = percentile(sorted, 0.99) ?? 0;
        this.log(`Average latency: ${(total / count).toFixed(2)}ms`, 'success');
        this.log(`P99 latency: ${p99.toFixed(2)}ms`, 'info');
        await this.printFullStatus();
    }

    async insertVectors(count) {
        this.log(`Inserting ${count} synthetic vector${count === 1 ? '' : 's'}...`, 'info');
        for (let i = 0; i < count; i++) {
            await this.addVector(this.randomSyntheticVec());
        }
        this.markProof('insert');
        this.log(`Inserted ${count} vectors`, 'success');
        await this.printFullStatus();
    }

    async deleteVectors(count) {
        const total = this.liveVectors.size;
        const n = Math.min(count, total);
        this.log(`Deleting ${n} vector${n === 1 ? '' : 's'}...`, 'info');
        for (let i = 0; i < n; i++) {
            const id = this.randomLiveId();
            if (id === null) break;
            await this.deleteVector(id);
        }
        const after = await this.getStats();
        if (after.ghost_count > 0) this.markProof('ghosts');
        this.markProof('delete');
        this.log(`Deleted ${n} vectors`, 'success');
        await this.printFullStatus();
    }

    async compact() {
        const statsBefore = await this.getStats();
        const memBefore = statsBefore.memory_bytes;
        const ghostsBefore = statsBefore.ghost_count;
        this.log('Compacting...', 'info');
        const res = await this.compactIndex();
        const statsAfter = await this.getStats();
        const memAfter = statsAfter.memory_bytes;
        const ghostsAfter = statsAfter.ghost_count;
        const saved = memBefore - memAfter;
        if (saved > 0) this.markProof('mem_shrink');
        this.log(`Compaction done in ${res.elapsed_ms.toFixed(2)}ms — ghosts ${ghostsBefore}->${ghostsAfter}, saved ${formatBytes(Math.max(saved, 0))}`, 'success');
        await this.printFullStatus();
    }

    async exportToFile(filePath = DEFAULT_EXPORT_PATH) {
        const outPath = path.resolve(filePath);
        const binary = await this.exportIndex();
        fs.writeFileSync(outPath, binary);
        this.log(`Exported ${formatBytes(binary.length)} to ${outPath}`, 'success');
    }

    async importFromFile(filePath) {
        const inPath = path.resolve(filePath);
        const data = fs.readFileSync(inPath);
        const res = await this.importIndex(data);
        this.log(`Imported ${res.count.toLocaleString()} vectors from ${inPath}`, 'success');
        await this.printFullStatus();
    }

    // ---- Proofs ----

    async proveDeterministic() {
        this.log('Check: deterministic search', 'info');
        const vec = this.randomLoadedVec();
        const results = [];
        for (let run = 0; run < 3; run++) {
            const res = await this.search(vec);
            results.push(res.neighbors);
            await sleep(50);
        }
        const allSame = results.every((ids) =>
            ids.length === results[0].length && ids.every((id, idx) => id === results[0][idx])
        );
        if (!allSame) throw new Error('deterministic check failed');
        this.markProof('deterministic');
        this.log('Deterministic search verified across 3 runs', 'success');
    }

    async proveDeletionExclusion() {
        this.log('Check: deleted vectors excluded from results', 'info');
        const vec = this.randomSyntheticVec();
        const addRes = await this.addVector(vec);
        const id = addRes.id;

        const before = await this.search(vec);
        const foundBefore = before.neighbors.includes(id);

        await this.deleteVector(id);
        const after = await this.search(vec);
        const foundAfter = after.neighbors.includes(id);

        if (!foundBefore || foundAfter) {
            throw new Error(`deletion exclusion failed (before=${foundBefore}, after=${foundAfter})`);
        }
        this.markProof('excl_deleted');
        this.log(`Deleted vector ${id} no longer appears in results`, 'success');
    }

    async proveSearchDuringCompaction() {
        this.log('Check: immediate search after compaction', 'info');
        const stats = await this.getStats();
        if (stats.ghost_count < 10) await this.deleteVectors(100);
        await this.compact();
        const res = await this.search(this.randomLoadedVec());
        if (res.client_latency >= 100) throw new Error(`post-compaction latency too high (${res.client_latency.toFixed(2)}ms)`);
        this.markProof('no_block');
        this.log(`Immediate post-compaction search: ${res.client_latency.toFixed(2)}ms`, 'success');
    }

    async proveMemoryDecrease() {
        this.log('Check: memory decreases after compaction', 'info');
        const stats = await this.getStats();
        if (stats.ghost_count < 50) await this.deleteVectors(200);
        const before = (await this.getStats()).memory_bytes;
        await this.compact();
        const after = (await this.getStats()).memory_bytes;
        if (!(after < before)) throw new Error(`memory did not decrease (${before} -> ${after})`);
        this.markProof('mem_shrink');
        this.log(`Memory decreased from ${formatBytes(before)} to ${formatBytes(after)}`, 'success');
    }

    async proveSustainedStability() {
        this.log('Check: sustained-mutation latency thresholds', 'info');
        const duration = 5000;
        const t0 = performance.now();
        const lats = [];
        let inserts = 0;
        let deletes = 0;

        while (performance.now() - t0 < duration) {
            await this.addVector(this.randomSyntheticVec());
            inserts++;

            if (Math.random() < 0.4) {
                if (this.liveVectors.size > 0) {
                    const id = this.randomLiveId();
                    if (id !== null) await this.deleteVector(id);
                    deletes++;
                }
            }

            const res = await this.search(this.randomLoadedVec());
            lats.push(res.client_latency);

            if (inserts % 50 === 0) await sleep(0);
        }

        const sorted = [...lats].sort((a, b) => a - b);
        const p50 = percentile(sorted, 0.5);
        const p99 = percentile(sorted, 0.99);
        if (!(p50 < 20 && p99 < 50)) {
            throw new Error(`latency thresholds exceeded (p50=${p50.toFixed(2)}ms, p99=${p99.toFixed(2)}ms)`);
        }
        this.markProof('stable');
        this.log(`Stable under load: ${inserts} inserts, ${deletes} deletes, p50=${p50.toFixed(2)}ms, p99=${p99.toFixed(2)}ms`, 'success');
    }

    async proveExportImport() {
        this.log('Check: export/import round-trip', 'info');
        // A clean export contains only live survivors — the engine does not
        // serialize soft-deleted ghosts, so /export compacts them away first.
        // Compact here too, so countBefore reflects the survivor set the export
        // actually carries; otherwise the count would drop by the ghost count
        // across the round-trip and the equality below would spuriously fail.
        await this.compact();
        const healthBefore = await this.checkHealth();
        const countBefore = healthBefore.count;
        if (countBefore === 0) throw new Error('index is empty');

        const binary = await this.exportIndex();
        this.log(`  Exported ${formatBytes(binary.length)}`, 'info');

        const importRes = await this.importIndex(binary);
        const countAfter = importRes.count;

        const searchRes = await this.search(this.randomLoadedVec());

        if (countAfter !== countBefore) {
            throw new Error(`round-trip count mismatch (before=${countBefore}, after=${countAfter})`);
        }
        this.markProof('import');
        this.log(`Round-trip restored ${countAfter.toLocaleString()} vectors; immediate search ${searchRes.client_latency.toFixed(2)}ms`, 'success');
    }

    async proveSelfRecall() {
        const trials = 50;
        this.log(`Check: self-recall (${trials} trials)`, 'info');
        let hits = 0;
        for (let i = 0; i < trials; i++) {
            const vec = this.randomSyntheticVec();
            const addRes = await this.addVector(vec);
            const searchRes = await this.search(vec);
            if (searchRes.neighbors.length > 0 && searchRes.neighbors[0] === addRes.id) {
                hits++;
            }
            if ((i + 1) % 10 === 0) await sleep(0);
        }
        if (hits !== trials) throw new Error(`rank-1 hits ${hits}/${trials}`);
        this.markProof('self_recall');
        this.log(`Self-recall verified: ${hits}/${trials} rank-1 hits`, 'success');
    }

    async proveRecallAtK(queryCount = 50, topK = 10, label = null) {
        const prefix = label ? `${label}: ` : 'Check: ';
        this.log(`${prefix}recall@${topK} vs brute-force (${queryCount} queries)`, 'info');

        const liveIds = Array.from(this.liveVectors.keys());
        if (liveIds.length < topK) throw new Error('not enough live vectors indexed');

        const queryIds = [];
        const used = new Set();
        const maxQueries = Math.min(queryCount, liveIds.length);
        while (queryIds.length < maxQueries) {
            const id = liveIds[Math.floor(Math.random() * liveIds.length)];
            if (!used.has(id)) {
                used.add(id);
                queryIds.push(id);
            }
        }

        let totalRecall = 0;
        const corpus = Array.from(this.liveVectors.entries());

        for (let qi = 0; qi < queryIds.length; qi++) {
            const qId = queryIds[qi];
            const qVec = this.liveVectors.get(qId);
            const scored = [];

            for (const [candidateId, v] of corpus) {
                let dot = 0;
                let na = 0;
                let nb = 0;
                for (let d = 0; d < DIMS; d++) {
                    dot += qVec[d] * v[d];
                    na += qVec[d] * qVec[d];
                    nb += v[d] * v[d];
                }
                scored.push({ id: candidateId, dist: 1 - dot / (Math.sqrt(na) * Math.sqrt(nb)) });
            }
            scored.sort((a, b) => a.dist - b.dist);
            const trueTopK = new Set(scored.slice(0, topK).map((s) => s.id));

            const res = await this.search(Array.from(qVec), topK);
            const hnswIds = new Set(res.neighbors);

            let hits = 0;
            for (const id of hnswIds) {
                if (trueTopK.has(id)) hits++;
            }
            totalRecall += hits / topK;
            if ((qi + 1) % 10 === 0) await sleep(0);
        }

        const avgRecall = totalRecall / queryIds.length;
        if (avgRecall < 0.95) throw new Error(`recall@${topK}=${(avgRecall * 100).toFixed(1)}%`);
        this.markProof('recall_at_k');
        this.log(`Recall@${topK}: ${(avgRecall * 100).toFixed(1)}%`, 'success');
        return avgRecall;
    }

    async runStress(mode, seconds = 30) {
        const cfg = ADV_MODES[mode];
        if (!cfg) {
            throw new Error(`unknown stress mode "${mode}"`);
        }

        this.log(`Preparing fresh baseline index for stress mode "${mode}"...`, 'info');
        await this.buildIndex(STRESS_BUILD_COUNT);
        this.log(`Stress: ${cfg.name} for ${seconds}s — ${cfg.desc}`, 'info');
        const durationMs = seconds * 1000;
        const start = performance.now();
        let inserts = 0;
        let deletes = 0;
        let searches = 0;
        const latencies = [];
        let lastReport = start;
        let lastCompact = start;

        while (performance.now() - start < durationMs) {
            const now = performance.now();

            // Interleave operations sequentially to avoid overwhelming single-threaded worker
            const opsPerRound = 10;
            for (let op = 0; op < opsPerRound; op++) {
                const roll = Math.random() * (cfg.insert + cfg.delete + cfg.search);

                if (roll < cfg.insert) {
                    const r = await this.addVector(this.randomSyntheticVec(), false);
                    if (r) inserts++;
                } else if (roll < cfg.insert + cfg.delete) {
                    if (this.liveVectors.size > 100) {
                        const id = this.randomLiveId();
                        if (id !== null) {
                            try {
                                await this.deleteVector(id);
                            } catch {}
                        }
                        deletes++;
                    }
                } else {
                    const res = await this.search(this.randomLoadedVec());
                    latencies.push(res.client_latency);
                    if (latencies.length > 1000) latencies.shift();
                    searches++;
                }
            }

            if (now - lastCompact > 1000) {
                const st = await this.getStats();
                if (st.ghost_ratio > 0.15) {
                    await this.compactIndex();
                }
                lastCompact = now;
            }

            if (now - lastReport >= 1000) {
                const elapsed = (now - start) / 1000;
                const sorted = [...latencies].sort((a, b) => a - b);
                const p50 = percentile(sorted, 0.5);
                const p99 = percentile(sorted, 0.99);
                const st = await this.getStats();
                this.log(
                    `stress ${elapsed.toFixed(1)}s | insert=${(inserts / elapsed).toFixed(0)}/s delete=${(deletes / elapsed).toFixed(0)}/s search=${(searches / elapsed).toFixed(0)}/s p50=${p50 ? p50.toFixed(2) : '—'}ms p99=${p99 ? p99.toFixed(2) : '—'}ms ghosts=${(st.ghost_ratio * 100).toFixed(1)}%`,
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
        const recall = await this.proveRecallAtK(STRESS_RECALL_QUERIES, K, `Post-stress recall (${mode})`);
        if (p99 < 100 && max < 300) this.markProof('stress');
        this.log(`Stress complete — inserts=${inserts}, deletes=${deletes}, searches=${searches}, p50=${p50.toFixed(2)}ms, p99=${p99.toFixed(2)}ms, max=${max.toFixed(2)}ms, recall@${K}=${(recall * 100).toFixed(1)}%`, 'success');
    }

    async runFullProofSequence() {
        this.log('Starting full validation run against worker API...', 'info');
        this.log(`Target: ${BASE_URL}`, 'info');
        await sleep(100);

        const health = await this.checkHealth();
        this.log(`Worker healthy: ${JSON.stringify(health)}`, 'info');

        await this.buildIndex(4000);
        await sleep(100);
        await this.performSearches(LATENCY_CHECK_QUERIES);
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
        this.log('Full validation run complete', 'success');
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
            await this.printFullStatus();
            return true;
        case 'checklist':
            this.printChecklist();
            return true;
        case 'reset':
            await this.initIndex();
            this.log('Index reset', 'warn');
            return true;
        case 'build':
            await this.buildIndex(parseInt(args[0] || '5000', 10));
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
            await this.exportToFile(args[0] || DEFAULT_EXPORT_PATH);
            return true;
        case 'import':
            if (!args[0]) throw new Error('import requires a file path');
            await this.importFromFile(args[0]);
            return true;
        case 'validate':
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
            prompt: 'pancake-worker> '
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
    const demo = new TechnicalDemoWorker();
    const args = process.argv.slice(2);
    await demo.init(args.length === 0);
    if (args.length > 0) {
        try {
            const keepRunning = await demo.handleCommand(args.join(' '));
            if (keepRunning) demo.printChecklist();
            process.exit(0);
        } catch (error) {
            demo.log(error.message, 'error');
            process.exit(1);
        }
    }

    await demo.runInteractive();
}

main().catch((error) => {
    process.stderr.write(`${error.stack || error.message}\n`);
    process.exit(1);
});
