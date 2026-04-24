#!/usr/bin/env node
/**
 * Technical Demo — Worker Edition
 *
 * Runs the same proof suite as technical_demo_cli.js but against
 * the Pancake Search Cloudflare Worker REST API (localhost:8787).
 * This is the primary worker demo for the 384D real-embedding flow using
 * `dist/vectors.bin` over the HTTP API.
 *
 * Proves: init, bulk insert, search, recall, deterministic search,
 * export/import round-trip, latency, and sustained load.
 *
 * Usage:
 *   1. Start the worker:  npx wrangler dev --port 8787
 *   2. Run this demo:     node technical_demo_worker.js [--auto]
 */

'use strict';

const fs = require('fs');
const path = require('path');
const readline = require('readline');
const http = require('http');

const BASE_URL = process.env.PANCAKE_URL || 'http://localhost:8787';
const DIMS = 384;
const K = 10;
const MAX_ELEM = 50_000;
const VECTORS_PATH = path.join(__dirname, '..', '..', 'dist', 'vectors.bin');

const PROOFS = [
    { id: 'load', text: 'Real embeddings loaded from vectors.bin' },
    { id: 'init', text: 'Index initialized via worker API' },
    { id: 'search', text: 'Successful search via worker API' },
    { id: 'low_latency', text: 'Search latency < 10ms (including HTTP)' },
    { id: 'insert', text: 'Live vector insertion post-build' },
    { id: 'delete', text: 'Live vector deletion via API' },
    { id: 'ghosts', text: 'Ghost node accumulation visible' },
    { id: 'compact', text: 'Compaction executed successfully' },
    { id: 'mem_shrink', text: 'Memory decreases after compaction' },
    { id: 'no_block', text: 'Search works immediately after compaction' },
    { id: 'excl_deleted', text: 'Deleted vectors excluded from results' },
    { id: 'deterministic', text: 'Search is deterministic (same query -> same results)' },
    { id: 'export', text: 'Index serialized to binary (export)' },
    { id: 'import', text: 'Index restored from binary (import round-trip)' },
    { id: 'self_recall', text: 'Self-recall: inserted vectors found at rank 1 (20/20)' },
    { id: 'recall_at_k', text: 'Recall@10 >= 95% vs brute-force (20 queries)' },
    { id: 'throughput', text: 'Sustained insert throughput > 50 vec/s via API' },
    { id: 'stable', text: 'Latency stable under sustained search load' },
    { id: 'stress', text: 'Survived adversarial stress test' },
];

const ADV_MODES = {
    ghost:  { name: 'Ghost Explosion',  insert: 20,  delete: 100, search: 50,  desc: 'Rapid deletions -> ghost accumulation' },
    churn:  { name: 'Memory Churn',     insert: 100, delete: 100, search: 50,  desc: 'Equal inserts + deletes -> memory pressure' },
    read:   { name: 'Read-Heavy',       insert: 5,   delete: 5,   search: 200, desc: 'Search-dominated workload' },
    write:  { name: 'Write-Heavy',      insert: 200, delete: 150, search: 20,  desc: 'Insert-dominated workload' },
    worst:  { name: 'Worst-Case',       insert: 100, delete: 100, search: 100, desc: 'Max stress on all paths simultaneously' }
};

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
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
        this.proofState = Object.fromEntries(PROOFS.map(p => [p.id, false]));
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

    async loadEmbeddings() {
        this.log(`Loading real ${DIMS}D embeddings from ${VECTORS_PATH}...`, 'info');
        const buf = fs.readFileSync(VECTORS_PATH);
        this.vectors = new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
        this.totalVectors = Math.floor(this.vectors.length / DIMS);
        this.markProof('load');
        this.log(`Loaded ${this.totalVectors.toLocaleString()} embeddings (${DIMS}D)`, 'success');
    }

    async checkHealth() {
        const res = await apiGet('/health');
        if (res.status !== 200) throw new Error(`Worker not reachable: ${res.status}`);
        return res.data;
    }

    printStatus() {
        const sorted = [...this.latencyHistory].sort((a, b) => a - b);
        const p50 = percentile(sorted, 0.5);
        const p99 = percentile(sorted, 0.99);
        process.stdout.write(`  Search p50: ${p50 === null ? '—' : `${p50.toFixed(2)} ms`}\n`);
        process.stdout.write(`  Search p99: ${p99 === null ? '—' : `${p99.toFixed(2)} ms`}\n`);
    }

    printChecklist() {
        process.stdout.write('\nProof Checklist\n');
        for (const proof of PROOFS) {
            const marker = this.proofState[proof.id] ? '[PASS]' : '[    ]';
            process.stdout.write(`  ${marker} ${proof.text}\n`);
        }
        process.stdout.write('\n');
    }

    // ---- Core operations via HTTP ----

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
        return res.data;
    }

    async addVector(vec, throwOnFull = true) {
        const res = await apiPost('/add', { vector: vec });
        if (res.status === 409 && !throwOnFull) return null;  // At capacity
        if (res.status !== 200) throw new Error(`/add failed: ${JSON.stringify(res.data)}`);
        this.markProof('insert');
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
        if (latency < 10) this.markProof('low_latency');
        return { ...res.data, client_latency: latency };
    }

    async deleteVector(id) {
        const res = await apiPost('/delete', { id });
        if (res.status !== 200) throw new Error(`/delete failed: ${JSON.stringify(res.data)}`);
        this.markProof('delete');
        return res.data;
    }

    async compactIndex() {
        const res = await apiPost('/compact', {});
        if (res.status !== 200) throw new Error(`/compact failed: ${JSON.stringify(res.data)}`);
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

    // ---- Build ----

    async addBatch(vecs) {
        const res = await apiPost('/add_batch', { vectors: vecs });
        if (res.status !== 200) throw new Error(`/add_batch failed: ${JSON.stringify(res.data)}`);
        this.markProof('insert');
        return res.data;
    }

    async buildIndex(count) {
        const n = Math.min(count, this.totalVectors);
        this.log(`Initializing index...`, 'info');
        await this.initIndex();

        this.log(`Inserting ${n.toLocaleString()} real embeddings (batched)...`, 'info');
        const t0 = performance.now();
        const batchSize = 500;

        for (let i = 0; i < n; i += batchSize) {
            const end = Math.min(i + batchSize, n);
            const batch = [];
            for (let j = i; j < end; j++) {
                batch.push(this.getVec(j));
            }
            await this.addBatch(batch);

            const rate = (end / ((performance.now() - t0) / 1000)).toFixed(0);
            this.log(`Indexed ${end.toLocaleString()}/${n.toLocaleString()} (${rate} vec/s)`, 'info');
        }

        const elapsed = (performance.now() - t0) / 1000;
        const rate = (n / elapsed).toFixed(0);
        this.log(`Index built: ${n.toLocaleString()} vectors in ${elapsed.toFixed(1)}s (${rate} vec/s)`, 'success');

        const health = await this.checkHealth();
        this.log(`  Count: ${health.count}, Memory: ${formatBytes(health.memory_bytes)}`, 'info');
    }

    // ---- Proofs ----

    async proveDeterministic() {
        this.log('Proof: deterministic search', 'info');
        const vec = this.randomLoadedVec();
        const results = [];
        for (let run = 0; run < 3; run++) {
            const res = await this.search(vec);
            results.push(res.neighbors);
            await sleep(50);
        }
        const allSame = results.every(ids =>
            ids.length === results[0].length && ids.every((id, idx) => id === results[0][idx])
        );
        if (!allSame) throw new Error('deterministic proof failed');
        this.markProof('deterministic');
        this.log('Deterministic search verified across 3 runs', 'success');
    }

    async proveSelfRecall() {
        const trials = 20;
        this.log(`Proof: self-recall (${trials} trials)`, 'info');
        let hits = 0;
        for (let i = 0; i < trials; i++) {
            const vec = this.randomSyntheticVec();
            const addRes = await this.addVector(vec);
            const searchRes = await this.search(vec);
            if (searchRes.neighbors.length > 0 && searchRes.neighbors[0] === addRes.id) {
                hits++;
            }
        }
        if (hits !== trials) throw new Error(`rank-1 hits ${hits}/${trials}`);
        this.markProof('self_recall');
        this.log(`Self-recall verified: ${hits}/${trials} rank-1 hits`, 'success');
    }

    async measureRecallAtK(excludeIds = null, queryCount = 20) {
        const topK = 10;

        // Build list of alive loaded-vector IDs
        const alive = [];
        for (let i = 0; i < this.totalVectors; i++) {
            if (!excludeIds || !excludeIds.has(i)) alive.push(i);
        }
        if (alive.length < topK) throw new Error('not enough live vectors');

        let totalRecall = 0;
        for (let qi = 0; qi < queryCount; qi++) {
            const qIdx = alive[Math.floor(Math.random() * alive.length)];
            const qVec = this.getVec(qIdx);

            // Brute-force ground truth over alive vectors only
            const scored = [];
            for (const i of alive) {
                const v = this.getVec(i);
                let dot = 0, na = 0, nb = 0;
                for (let d = 0; d < DIMS; d++) {
                    dot += qVec[d] * v[d];
                    na += qVec[d] * qVec[d];
                    nb += v[d] * v[d];
                }
                scored.push({ id: i, dist: 1 - dot / (Math.sqrt(na) * Math.sqrt(nb)) });
            }
            scored.sort((a, b) => a.dist - b.dist);
            const trueTopK = new Set(scored.slice(0, topK).map(s => s.id));

            // Worker search — request extra results to account for synthetic inserts
            const res = await this.search(qVec, topK * 5);
            let hits = 0, checked = 0;
            for (const id of res.neighbors) {
                if (id >= this.totalVectors) continue; // skip synthetic inserts
                if (excludeIds && excludeIds.has(id)) continue; // skip deleted
                if (checked >= topK) break;
                if (trueTopK.has(id)) hits++;
                checked++;
            }
            if (checked > 0) totalRecall += hits / Math.min(topK, checked);
        }

        return totalRecall / queryCount;
    }

    async proveRecallAtK() {
        const topK = 10;
        this.log(`Proof: recall@${topK} vs brute-force (20 queries)`, 'info');

        const avgRecall = await this.measureRecallAtK(null, 20);
        if (avgRecall < 0.95) throw new Error(`recall@${topK}=${(avgRecall * 100).toFixed(1)}%`);
        this.markProof('recall_at_k');
        this.log(`Recall@${topK}: ${(avgRecall * 100).toFixed(1)}%`, 'success');
    }

    async proveExportImport() {
        this.log('Proof: export/import round-trip', 'info');
        const healthBefore = await this.checkHealth();
        const countBefore = healthBefore.count;
        if (countBefore === 0) throw new Error('index is empty');

        // Export
        const binary = await this.exportIndex();
        this.log(`  Exported ${formatBytes(binary.length)}`, 'info');

        // Import into fresh index
        const importRes = await this.importIndex(binary);
        const countAfter = importRes.count;

        // Verify search works
        const searchRes = await this.search(this.randomLoadedVec());

        if (countAfter !== countBefore) {
            throw new Error(`round-trip count mismatch (before=${countBefore}, after=${countAfter})`);
        }
        this.log(`Round-trip: ${countAfter.toLocaleString()} vectors, search latency ${searchRes.client_latency.toFixed(1)}ms`, 'success');
    }

    async proveThroughput() {
        this.log('Proof: sustained insert throughput', 'info');
        const count = 200;
        const t0 = performance.now();
        for (let i = 0; i < count; i++) {
            await this.addVector(this.randomSyntheticVec());
        }
        const elapsed = (performance.now() - t0) / 1000;
        const rate = count / elapsed;
        if (rate < 50) throw new Error(`throughput too low: ${rate.toFixed(0)} vec/s`);
        this.markProof('throughput');
        this.log(`Insert throughput: ${rate.toFixed(0)} vec/s (${count} vectors in ${elapsed.toFixed(1)}s)`, 'success');
    }

    async proveStableLatency() {
        this.log('Proof: latency stability under sustained search load', 'info');
        const count = 200;
        const lats = [];
        for (let i = 0; i < count; i++) {
            const res = await this.search(this.randomLoadedVec());
            lats.push(res.client_latency);
        }
        const sorted = [...lats].sort((a, b) => a - b);
        const p50 = percentile(sorted, 0.5);
        const p99 = percentile(sorted, 0.99);
        if (p99 > 50) throw new Error(`p99 too high: ${p99.toFixed(1)}ms`);
        this.markProof('stable');
        this.log(`Stable: ${count} queries, p50=${p50.toFixed(1)}ms, p99=${p99.toFixed(1)}ms (includes HTTP)`, 'success');
    }

    // ---- Adversarial proofs ----

    async proveDeletionExclusion() {
        this.log('Proof: deleted vectors excluded from results', 'info');
        // Insert a vector, verify it's found, delete it, verify it's gone
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

    async proveGhostsAndCompaction() {
        this.log('Proof: ghost accumulation + compaction + memory decrease', 'info');

        // Delete a bunch of vectors to create ghosts
        const stats0 = await this.getStats();
        const count = stats0.count;
        const toDelete = Math.min(200, Math.floor(count * 0.3));

        this.log(`  Deleting ${toDelete} vectors...`, 'info');
        for (let i = 0; i < toDelete; i++) {
            await this.deleteVector(Math.floor(Math.random() * count));
        }

        const stats1 = await this.getStats();
        if (stats1.ghost_count > 0) this.markProof('ghosts');
        this.log(`  Ghosts: ${stats1.ghost_count} (${(stats1.ghost_ratio * 100).toFixed(1)}%), Memory: ${formatBytes(stats1.memory_bytes)}`, 'info');

        // Compact
        const memBefore = stats1.memory_bytes;
        const compactRes = await this.compactIndex();
        const stats2 = await this.getStats();

        if (stats2.memory_bytes < memBefore) this.markProof('mem_shrink');
        this.log(`  After compaction: ghosts=${stats2.ghost_count}, memory ${formatBytes(memBefore)} -> ${formatBytes(stats2.memory_bytes)} (${compactRes.elapsed_ms.toFixed(1)}ms)`, 'success');

        // Verify search still works
        const searchRes = await this.search(this.randomLoadedVec());
        if (searchRes.neighbors.length > 0) this.markProof('no_block');
        this.log(`  Post-compaction search: ${searchRes.client_latency.toFixed(1)}ms, ${searchRes.neighbors.length} results`, 'success');
    }

    async runStress(mode, seconds = 15) {
        const cfg = ADV_MODES[mode];
        if (!cfg) throw new Error(`unknown stress mode "${mode}"`);

        // Fresh index for each stress run so results are independent
        await this.buildIndex(5000);

        this.log(`Stress: ${cfg.name} for ${seconds}s — ${cfg.desc}`, 'info');

        const stats0 = await this.getStats();
        if (stats0.count < 100) throw new Error('need at least 100 vectors for stress test');

        const durationMs = seconds * 1000;
        const start = performance.now();
        let inserts = 0, deletes = 0, searches = 0, compactions = 0;
        const latencies = [];
        let lastReport = start;
        let lastCompact = start;
        const deletedIds = new Set();

        while (performance.now() - start < durationMs) {
            const now = performance.now();

            // Interleave operations sequentially to avoid overwhelming single-threaded worker
            const opsPerRound = 10;
            for (let op = 0; op < opsPerRound; op++) {
                const roll = Math.random() * (cfg.insert + cfg.delete + cfg.search);

                if (roll < cfg.insert) {
                    // Insert
                    const r = await this.addVector(this.randomSyntheticVec(), false);
                    if (r) inserts++;
                } else if (roll < cfg.insert + cfg.delete) {
                    // Delete
                    const st = await this.getStats();
                    if (st.count > 100) {
                        const delId = Math.floor(Math.random() * (st.next_id || st.count));
                        try {
                            await this.deleteVector(delId);
                            if (delId < this.totalVectors) deletedIds.add(delId);
                        } catch {}
                        deletes++;
                    }
                } else {
                    // Search
                    const res = await this.search(this.randomLoadedVec());
                    latencies.push(res.client_latency);
                    if (latencies.length > 500) latencies.shift();
                    searches++;
                }
            }

            // Periodic compaction + recall probe
            if (now - lastCompact > 3000) {
                const st = await this.getStats();
                if (st.ghost_ratio > 0.15) {
                    await this.compactIndex();
                    compactions++;

                    // Recall@10 probe over surviving loaded vectors
                    try {
                        const recall = await this.measureRecallAtK(deletedIds, 10);
                        this.log(`  post-compact recall@10: ${(recall * 100).toFixed(1)}%`, 'info');
                    } catch (e) {
                        this.log(`  post-compact recall probe: ${e.message}`, 'info');
                    }
                }
                lastCompact = now;
            }

            // Report
            if (now - lastReport > 2000) {
                const elapsed = (now - start) / 1000;
                const sorted = [...latencies].sort((a, b) => a - b);
                const p50 = percentile(sorted, 0.5);
                const p99 = percentile(sorted, 0.99);
                const st = await this.getStats();
                this.log(
                    `  ${elapsed.toFixed(0)}s | ins=${inserts} del=${deletes} qry=${searches} compact=${compactions} | ` +
                    `p50=${p50 ? p50.toFixed(1) : '—'}ms p99=${p99 ? p99.toFixed(1) : '—'}ms | ` +
                    `ghosts=${(st.ghost_ratio * 100).toFixed(1)}% mem=${formatBytes(st.memory_bytes)}`,
                    'info'
                );
                lastReport = now;
            }
        }

        const sorted = [...latencies].sort((a, b) => a - b);
        const p50 = percentile(sorted, 0.5) || 0;
        const p99 = percentile(sorted, 0.99) || 0;
        const max = sorted[sorted.length - 1] || 0;

        if (p99 < 100 && max < 300) this.markProof('stress');
        this.log(
            `Stress complete: ins=${inserts} del=${deletes} qry=${searches} compact=${compactions} | ` +
            `p50=${p50.toFixed(1)}ms p99=${p99.toFixed(1)}ms max=${max.toFixed(1)}ms`,
            'success'
        );
    }

    // ---- Full sequence ----

    async runFullProofSequence() {
        this.log('Starting full proof sequence against worker API...', 'info');
        this.log(`Target: ${BASE_URL}`, 'info');
        await sleep(100);

        // Verify worker is up
        const health = await this.checkHealth();
        this.log(`Worker healthy: ${JSON.stringify(health)}`, 'info');

        await this.buildIndex(5000);
        await sleep(200);

        await this.search(this.randomLoadedVec());
        await sleep(100);

        await this.proveRecallAtK();
        await sleep(100);

        await this.proveDeterministic();
        await sleep(100);

        await this.proveSelfRecall();
        await sleep(100);

        await this.proveThroughput();
        await sleep(100);

        await this.proveStableLatency();
        await sleep(100);

        await this.proveDeletionExclusion();
        await sleep(100);

        await this.proveGhostsAndCompaction();
        await sleep(100);

        await this.proveExportImport();
        await sleep(100);

        // Rebuild fresh index for stress test so external IDs match loaded vectors
        await this.buildIndex(5000);
        await this.runStress('worst', 15);

        this.log('Full proof sequence complete', 'success');
        this.printChecklist();
    }

    // ---- Interactive ----

    printHelp() {
        process.stdout.write('\nCommands\n');
        process.stdout.write('  help                     Show commands\n');
        process.stdout.write('  status                   Show worker health + latency stats\n');
        process.stdout.write('  checklist                Show proof checklist\n');
        process.stdout.write('  build <count>            Init + insert N vectors from vectors.bin\n');
        process.stdout.write('  search <count>           Run N random searches\n');
        process.stdout.write('  insert <count>           Insert N synthetic vectors\n');
        process.stdout.write('  delete <count>           Delete N random vectors\n');
        process.stdout.write('  compact                  Run compaction\n');
        process.stdout.write('  stats                    Show detailed index stats\n');
        process.stdout.write('  export <path>            Export index to file\n');
        process.stdout.write('  import <path>            Import index from file\n');
        process.stdout.write('  proof all                Run full proof sequence\n');
        process.stdout.write('  stress <mode> [seconds]  ghost|churn|read|write|worst (default: 15s)\n');
        process.stdout.write('  quit                     Exit\n\n');
    }

    async handleCommand(input) {
        const parts = input.trim().split(/\s+/).filter(Boolean);
        if (parts.length === 0) return true;
        const [command, ...args] = parts;

        switch (command) {
        case 'help':
            this.printHelp();
            return true;
        case 'status': {
            const h = await this.checkHealth();
            this.log(`Count: ${h.count}, Memory: ${formatBytes(h.memory_bytes)}, Initialized: ${h.initialized}`, 'info');
            this.printStatus();
            return true;
        }
        case 'checklist':
            this.printChecklist();
            return true;
        case 'build':
            await this.buildIndex(parseInt(args[0] || '5000', 10));
            return true;
        case 'search': {
            const n = parseInt(args[0] || '1', 10);
            for (let i = 0; i < n; i++) {
                const res = await this.search(this.randomLoadedVec());
                if (n === 1) {
                    this.log(`Found ${res.neighbors.length} neighbors in ${res.client_latency.toFixed(1)}ms (worker: ${res.latency_ms}ms)`, 'info');
                }
            }
            if (n > 1) this.printStatus();
            return true;
        }
        case 'insert': {
            const n = parseInt(args[0] || '1', 10);
            const t0 = performance.now();
            for (let i = 0; i < n; i++) await this.addVector(this.randomSyntheticVec());
            const elapsed = (performance.now() - t0) / 1000;
            this.log(`Inserted ${n} vectors in ${elapsed.toFixed(1)}s (${(n / elapsed).toFixed(0)} vec/s)`, 'success');
            return true;
        }
        case 'delete': {
            const n = parseInt(args[0] || '1', 10);
            const stats = await this.getStats();
            for (let i = 0; i < n; i++) {
                await this.deleteVector(Math.floor(Math.random() * (stats.next_id || stats.count)));
            }
            const after = await this.getStats();
            this.log(`Deleted ${n}, ghosts: ${after.ghost_count} (${(after.ghost_ratio * 100).toFixed(1)}%)`, 'success');
            return true;
        }
        case 'compact': {
            const res = await this.compactIndex();
            this.log(`Compacted in ${res.elapsed_ms.toFixed(1)}ms, memory: ${formatBytes(res.memory_bytes)}`, 'success');
            return true;
        }
        case 'stats': {
            const s = await this.getStats();
            this.log(`Count: ${s.count}, Ghosts: ${s.ghost_count} (${(s.ghost_ratio * 100).toFixed(1)}%), Memory: ${formatBytes(s.memory_bytes)}`, 'info');
            return true;
        }
        case 'stress': {
            await this.runStress(args[0] || 'worst', parseInt(args[1] || '15', 10));
            return true;
        }
        case 'export': {
            const binary = await this.exportIndex();
            const outPath = path.resolve(args[0] || 'pancake-index.bin');
            fs.writeFileSync(outPath, binary);
            this.log(`Exported ${formatBytes(binary.length)} to ${outPath}`, 'success');
            return true;
        }
        case 'import': {
            if (!args[0]) throw new Error('import requires a file path');
            const data = fs.readFileSync(path.resolve(args[0]));
            await this.importIndex(data);
            return true;
        }
        case 'proof':
            if (args[0] === 'all') await this.runFullProofSequence();
            else throw new Error(`unknown proof target "${args[0]}"`);
            return true;
        case 'quit':
        case 'exit':
            return false;
        default:
            throw new Error(`unknown command "${command}"`);
        }
    }

    async runInteractive() {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
            prompt: 'pancake-worker> '
        });

        rl.prompt();
        rl.on('line', async line => {
            rl.pause();
            try {
                const keepRunning = await this.handleCommand(line);
                if (!keepRunning) { rl.close(); return; }
            } catch (error) {
                this.log(error.message, 'error');
            }
            rl.resume();
            rl.prompt();
        });
        rl.on('close', () => { process.stdout.write('\n'); process.exit(0); });
    }
}

async function main() {
    const demo = new TechnicalDemoWorker();
    const args = process.argv.slice(2);

    await demo.loadEmbeddings();

    // Check worker is reachable
    try {
        await demo.checkHealth();
    } catch {
        demo.log(`Cannot reach worker at ${BASE_URL}. Start it with: npx wrangler dev --port 8787`, 'error');
        process.exit(1);
    }
    demo.log(`Connected to worker at ${BASE_URL}`, 'success');

    if (args.includes('--auto') || args.includes('proof all')) {
        try {
            await demo.runFullProofSequence();
            process.exit(0);
        } catch (error) {
            demo.log(error.message, 'error');
            demo.printChecklist();
            process.exit(1);
        }
    } else if (args.length > 0) {
        try {
            await demo.handleCommand(args.join(' '));
            demo.printChecklist();
            process.exit(0);
        } catch (error) {
            demo.log(error.message, 'error');
            process.exit(1);
        }
    } else {
        demo.printHelp();
        await demo.runInteractive();
    }
}

main().catch(error => {
    process.stderr.write(`${error.stack || error.message}\n`);
    process.exit(1);
});
