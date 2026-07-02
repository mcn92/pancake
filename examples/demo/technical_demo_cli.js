#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { pathToFileURL } = require('url');

const DIMS = 384;
const K = 10;
const MAX_ELEM = 110_000;
const LATENCY_CHECK_QUERIES = 50;
const AVG_LATENCY_THRESHOLD_MS = 2;
const P99_LATENCY_THRESHOLD_MS = 5;
const VECTORS_PATH = path.join(__dirname, '..', '..', 'dist', 'vectors.bin');
const DEFAULT_EXPORT_PATH = path.join(__dirname, 'pancake-index.pnck');

const PROOFS = [
    { id: 'load', text: 'Real embeddings loaded from vectors.bin' },
    { id: 'init', text: 'Index built through the public Pancake API' },
    { id: 'search', text: 'Successful search execution' },
    { id: 'avg_latency', text: `Average search latency under ${AVG_LATENCY_THRESHOLD_MS}ms over ${LATENCY_CHECK_QUERIES} queries` },
    { id: 'p99_latency', text: `P99 search latency under ${P99_LATENCY_THRESHOLD_MS}ms over ${LATENCY_CHECK_QUERIES} queries` },
    { id: 'insert', text: 'Live vector insertion post-build' },
    { id: 'delete', text: 'Live vector deletion' },
    { id: 'ghosts', text: 'Ghost node accumulation visible' },
    { id: 'compact', text: 'Compaction executed successfully' },
    { id: 'deterministic', text: 'Search is deterministic for a fixed query' },
    { id: 'excl_deleted', text: 'Deleted vectors are excluded from results' },
    { id: 'export', text: 'Index serialized to a Pancake snapshot' },
    { id: 'import', text: 'Index restored from a Pancake snapshot' },
    { id: 'self_recall', text: 'Self-recall: inserted vectors found at rank 1' },
    { id: 'recall_at_k', text: 'Recall@10 checked against brute force' },
    { id: 'stable', text: 'Sustained-mutation latency thresholds met' },
    { id: 'stress', text: 'Completed mixed-workload stress run' }
];

const STRESS_MODES = {
    ghost: { insert: 100, delete: 500, search: 100 },
    churn: { insert: 500, delete: 500, search: 100 },
    read: { insert: 10, delete: 10, search: 1000 },
    write: { insert: 1000, delete: 800, search: 50 },
    worst: { insert: 500, delete: 490, search: 1000 }
};

async function loadPancake() {
    const mod = await import(pathToFileURL(path.join(__dirname, '..', '..', 'pancake.node.mjs')).href);
    return mod.default;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function percentile(values, p) {
    if (values.length === 0) return null;
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
}

function formatBytes(bytes) {
    if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    return `${(bytes / 1024).toFixed(1)} KB`;
}

function normalize(vec) {
    let norm = 0;
    for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm) || 1;
    const out = new Float32Array(vec.length);
    for (let i = 0; i < vec.length; i++) out[i] = vec[i] / norm;
    return out;
}

function cosineDistance(a, b) {
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return 1 - dot;
}

class TechnicalDemoCLI {
    constructor(Pancake) {
        this.Pancake = Pancake;
        this.index = null;
        this.vectors = null;
        this.totalVectors = 0;
        this.latencyHistory = [];
        this.proofState = Object.fromEntries(PROOFS.map(proof => [proof.id, false]));
        this.liveVectors = new Map();
        this.deletedIds = new Set();
    }

    async init(showHelp = true) {
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
        process.stdout.write(`${time} ${type.toUpperCase().padEnd(7)} ${message}\n`);
    }

    markProof(id) {
        if (Object.prototype.hasOwnProperty.call(this.proofState, id)) {
            this.proofState[id] = true;
        }
    }

    getVec(idx) {
        const offset = idx * DIMS;
        return this.vectors.subarray(offset, offset + DIMS);
    }

    randomLoadedVec() {
        return this.getVec(Math.floor(Math.random() * this.totalVectors));
    }

    randomSyntheticVec() {
        const vec = new Float32Array(DIMS);
        for (let i = 0; i < DIMS; i++) vec[i] = Math.random() * 2 - 1;
        return normalize(vec);
    }

    async createIndex(maxElements = MAX_ELEM) {
        if (this.index) this.index.dispose();
        this.index = await this.Pancake.create({
            dim: DIMS,
            maxElements,
            metric: 'cosine',
            quantized: true,
            M: 12,
            efConstruction: 150,
            efSearch: 250
        });
        this.latencyHistory = [];
        this.liveVectors.clear();
        this.deletedIds.clear();
    }

    requireIndex(action) {
        if (!this.index || this.index.count === 0) {
            throw new Error(`${action} requires a built or imported index`);
        }
    }

    printStatus() {
        if (!this.index) {
            process.stdout.write('\nNo index loaded.\n\n');
            return;
        }
        const p50 = percentile(this.latencyHistory, 0.5);
        const p99 = percentile(this.latencyHistory, 0.99);
        process.stdout.write('\nStatus\n');
        process.stdout.write(`  Count:       ${this.index.count.toLocaleString()}\n`);
        process.stdout.write(`  Ghosts:      ${this.index.ghostCount.toLocaleString()} (${(this.index.ghostRatio * 100).toFixed(1)}%)\n`);
        process.stdout.write(`  Memory:      ${formatBytes(this.index.memory)}\n`);
        process.stdout.write(`  Tracked IDs: ${this.liveVectors.size.toLocaleString()}\n`);
        process.stdout.write(`  Search p50:  ${p50 === null ? '-' : `${p50.toFixed(3)} ms`}\n`);
        process.stdout.write(`  Search p99:  ${p99 === null ? '-' : `${p99.toFixed(3)} ms`}\n\n`);
    }

    printChecklist() {
        process.stdout.write('\nValidation Checklist\n');
        for (const proof of PROOFS) {
            process.stdout.write(`  ${this.proofState[proof.id] ? '[PASS]' : '[    ]'} ${proof.text}\n`);
        }
        process.stdout.write('\n');
    }

    printHelp() {
        process.stdout.write('\nCommands\n');
        process.stdout.write('  help                                      Show commands\n');
        process.stdout.write('  status                                    Show index metrics\n');
        process.stdout.write('  checklist                                 Show validation checklist\n');
        process.stdout.write('  build [count]                             Build index (default 100000)\n');
        process.stdout.write('  reset                                     Reset the index\n');
        process.stdout.write('  search [count]                            Run random searches\n');
        process.stdout.write('  insert [count]                            Insert synthetic vectors\n');
        process.stdout.write('  delete [count]                            Delete random tracked IDs\n');
        process.stdout.write('  compact                                   Compact deleted entries\n');
        process.stdout.write(`  export [path]                             Export snapshot (default: ${path.basename(DEFAULT_EXPORT_PATH)})\n`);
        process.stdout.write('  import [path]                             Import snapshot\n');
        process.stdout.write('  validate <all|deterministic|deletion|compaction|stability|export|self-recall|recall>\n');
        process.stdout.write('  stress <ghost|churn|read|write|worst> [seconds]\n');
        process.stdout.write('  quit                                      Exit\n\n');
    }

    async build(count = Math.min(this.totalVectors, 100_000)) {
        count = Math.max(1, Math.min(count, this.totalVectors, MAX_ELEM));
        await this.createIndex(Math.max(MAX_ELEM, count + 10_000));
        this.log(`Building ${count.toLocaleString()} vectors via index.addBatch()...`, 'info');
        const batch = new Array(count);
        for (let i = 0; i < count; i++) batch[i] = this.getVec(i);

        const t0 = performance.now();
        const ids = this.index.addBatch(batch);
        const elapsed = performance.now() - t0;
        for (let i = 0; i < ids.length; i++) {
            this.liveVectors.set(ids[i], new Float32Array(batch[i]));
        }
        this.markProof('init');
        this.log(`Built ${ids.length.toLocaleString()} vectors in ${elapsed.toFixed(1)}ms (${(ids.length / (elapsed / 1000)).toFixed(0)} vec/s)`, 'success');
        this.log(`Memory: ${formatBytes(this.index.memory)}`, 'info');
    }

    doSearch(query, k = K) {
        const t0 = performance.now();
        const results = this.index.search(query, k);
        const latency = performance.now() - t0;
        this.latencyHistory.push(latency);
        if (this.latencyHistory.length > 1000) this.latencyHistory.shift();
        this.markProof('search');
        return { results, latency };
    }

    async search(count = LATENCY_CHECK_QUERIES) {
        this.requireIndex('search');
        count = Math.max(1, count);
        this.log(`Running ${count} random searches...`, 'info');
        const latencies = [];
        for (let i = 0; i < count; i++) {
            const { latency } = this.doSearch(this.randomLoadedVec());
            latencies.push(latency);
        }
        const avg = latencies.reduce((sum, value) => sum + value, 0) / latencies.length;
        const p99 = percentile(latencies, 0.99);
        if (count >= LATENCY_CHECK_QUERIES && avg < AVG_LATENCY_THRESHOLD_MS) this.markProof('avg_latency');
        if (count >= LATENCY_CHECK_QUERIES && p99 < P99_LATENCY_THRESHOLD_MS) this.markProof('p99_latency');
        this.log(`Average latency: ${avg.toFixed(3)}ms`, 'success');
        this.log(`P99 latency: ${p99.toFixed(3)}ms`, 'info');
    }

    insert(count = 1, verbose = true) {
        this.requireIndex('insert');
        count = Math.max(1, count);
        const t0 = performance.now();
        for (let i = 0; i < count; i++) {
            const vec = this.randomSyntheticVec();
            const id = this.index.add(vec);
            this.liveVectors.set(id, vec);
            this.deletedIds.delete(id);
        }
        const elapsed = performance.now() - t0;
        this.markProof('insert');
        if (verbose) this.log(`Inserted ${count.toLocaleString()} vectors in ${elapsed.toFixed(1)}ms`, 'success');
    }

    delete(count = 1, verbose = true) {
        this.requireIndex('delete');
        const liveIds = [...this.liveVectors.keys()].filter(id => !this.deletedIds.has(id));
        count = Math.max(1, Math.min(count, liveIds.length));
        for (let i = 0; i < count; i++) {
            const pick = i + Math.floor(Math.random() * (liveIds.length - i));
            [liveIds[i], liveIds[pick]] = [liveIds[pick], liveIds[i]];
            this.index.delete(liveIds[i]);
            this.deletedIds.add(liveIds[i]);
        }
        this.markProof('delete');
        if (this.index.ghostCount > 0) this.markProof('ghosts');
        if (verbose) this.log(`Deleted ${count.toLocaleString()} vectors; ghosts=${this.index.ghostCount.toLocaleString()}`, 'success');
    }

    compact() {
        this.requireIndex('compact');
        const beforeGhosts = this.index.ghostCount;
        const beforeLive = [...this.liveVectors.entries()].filter(([id]) => !this.deletedIds.has(id));
        this.index.compact();
        this.liveVectors = new Map(beforeLive);
        this.deletedIds.clear();
        this.markProof('compact');
        this.log(`Compacted ${beforeGhosts.toLocaleString()} ghosts; count=${this.index.count.toLocaleString()}`, 'success');
    }

    export(filePath = DEFAULT_EXPORT_PATH) {
        this.requireIndex('export');
        if (this.index.ghostCount > 0) this.compact();
        const snapshot = this.index.export();
        fs.writeFileSync(filePath, snapshot);
        this.markProof('export');
        this.log(`Exported ${formatBytes(snapshot.byteLength)} to ${filePath}`, 'success');
    }

    async import(filePath = DEFAULT_EXPORT_PATH) {
        const snapshot = fs.readFileSync(filePath);
        await this.createIndex(MAX_ELEM);
        this.index.import(snapshot);
        this.liveVectors.clear();
        this.deletedIds.clear();
        for (let id = 0; id < this.index.count; id++) {
            if (id < this.totalVectors) this.liveVectors.set(id, new Float32Array(this.getVec(id)));
        }
        this.markProof('import');
        this.log(`Imported ${this.index.count.toLocaleString()} vectors from ${filePath}`, 'success');
    }

    validateDeterministic() {
        this.requireIndex('determinism validation');
        const query = this.randomLoadedVec();
        const a = this.index.search(query, K);
        const b = this.index.search(query, K);
        const same = JSON.stringify(a) === JSON.stringify(b);
        if (!same) throw new Error('same query returned different result lists');
        this.markProof('deterministic');
        this.log('Deterministic search verified.', 'success');
    }

    validateDeletion() {
        this.requireIndex('deletion validation');
        const vec = this.randomSyntheticVec();
        const id = this.index.add(vec);
        this.liveVectors.set(id, vec);
        let before = this.index.search(vec, K).map(result => result.id);
        if (!before.includes(id)) throw new Error('inserted vector was not searchable before delete');
        this.index.delete(id);
        this.deletedIds.add(id);
        let after = this.index.search(vec, K).map(result => result.id);
        if (after.includes(id)) throw new Error('deleted vector remained searchable');
        this.markProof('insert');
        this.markProof('delete');
        this.markProof('ghosts');
        this.markProof('excl_deleted');
        this.log('Deletion exclusion verified.', 'success');
    }

    validateCompaction() {
        this.requireIndex('compaction validation');
        if (this.index.ghostCount === 0) this.delete(Math.min(10, Math.max(1, this.index.count)), false);
        const before = this.index.count - this.index.ghostCount;
        this.compact();
        if (this.index.count !== before || this.index.ghostCount !== 0) {
            throw new Error('compaction did not remove ghosts cleanly');
        }
    }

    async validateExport() {
        this.requireIndex('export validation');
        const query = this.randomLoadedVec();
        const before = this.index.search(query, K);
        const snapshot = this.index.export();
        const restored = await this.Pancake.create({
            dim: DIMS,
            maxElements: MAX_ELEM,
            metric: 'cosine',
            quantized: true,
            M: 12,
            efConstruction: 150,
            efSearch: 250
        });
        try {
            restored.import(snapshot);
            const after = restored.search(query, K);
            if (restored.count !== this.index.count || JSON.stringify(before) !== JSON.stringify(after)) {
                throw new Error('export/import round-trip changed count or query results');
            }
        } finally {
            restored.dispose();
        }
        this.markProof('export');
        this.markProof('import');
        this.log('Export/import round-trip verified.', 'success');
    }

    validateSelfRecall(trials = 50) {
        this.requireIndex('self-recall validation');
        trials = Math.max(1, trials);
        let hits = 0;
        for (let i = 0; i < trials; i++) {
            const vec = this.randomSyntheticVec();
            const id = this.index.add(vec);
            this.liveVectors.set(id, vec);
            const top = this.index.search(vec, 1)[0];
            if (top && top.id === id) hits++;
        }
        if (hits !== trials) throw new Error(`self-recall failed: ${hits}/${trials}`);
        this.markProof('insert');
        this.markProof('self_recall');
        this.log(`Self-recall verified: ${hits}/${trials} rank-1 hits.`, 'success');
    }

    validateRecall(queryCount = 50, topK = 10) {
        this.requireIndex('recall validation');
        const candidates = [...this.liveVectors.entries()].filter(([id]) => !this.deletedIds.has(id));
        if (candidates.length < topK) {
            throw new Error(`not enough tracked vectors (${candidates.length}) for recall validation`);
        }
        queryCount = Math.min(queryCount, candidates.length);
        let totalRecall = 0;
        for (let i = 0; i < queryCount; i++) {
            const query = candidates[Math.floor(Math.random() * candidates.length)][1];
            const exact = candidates
                .map(([id, vec]) => ({ id, distance: cosineDistance(query, vec) }))
                .sort((a, b) => a.distance - b.distance)
                .slice(0, topK);
            const truth = new Set(exact.map(result => result.id));
            const approx = this.index.search(query, topK);
            let hits = 0;
            for (const result of approx) if (truth.has(result.id)) hits++;
            totalRecall += hits / truth.size;
        }
        const recall = totalRecall / queryCount;
        if (recall < 0.9) throw new Error(`recall@${topK}=${(recall * 100).toFixed(1)}%`);
        this.markProof('recall_at_k');
        this.log(`Recall@${topK}: ${(recall * 100).toFixed(1)}% vs brute force.`, 'success');
    }

    async validateStability() {
        this.requireIndex('stability validation');
        const latencies = [];
        for (let i = 0; i < 200; i++) {
            if (i % 4 === 0) this.insert(1, false);
            if (i % 5 === 0 && this.liveVectors.size > 20) this.delete(1, false);
            const { latency } = this.doSearch(this.randomLoadedVec());
            latencies.push(latency);
            if (i % 25 === 0) await sleep(0);
        }
        const p50 = percentile(latencies, 0.5);
        const p99 = percentile(latencies, 0.99);
        if (p50 > 2 || p99 > 10) throw new Error(`latency thresholds exceeded (p50=${p50.toFixed(3)}ms, p99=${p99.toFixed(3)}ms)`);
        this.markProof('stable');
        this.log(`Stability p50=${p50.toFixed(3)}ms p99=${p99.toFixed(3)}ms`, 'success');
    }

    async validate(target) {
        switch (target) {
        case 'all':
            await this.search(LATENCY_CHECK_QUERIES);
            this.validateDeterministic();
            this.validateDeletion();
            this.validateCompaction();
            await this.validateExport();
            this.validateSelfRecall();
            this.validateRecall();
            await this.validateStability();
            break;
        case 'deterministic':
            this.validateDeterministic();
            break;
        case 'deletion':
            this.validateDeletion();
            break;
        case 'compaction':
            this.validateCompaction();
            break;
        case 'export':
            await this.validateExport();
            break;
        case 'self-recall':
            this.validateSelfRecall();
            break;
        case 'recall':
            this.validateRecall();
            break;
        case 'stability':
            await this.validateStability();
            break;
        default:
            throw new Error(`unknown validation target "${target}"`);
        }
    }

    async stress(mode = 'read', seconds = 5) {
        this.requireIndex('stress');
        const config = STRESS_MODES[mode];
        if (!config) throw new Error(`unknown stress mode "${mode}"`);
        const end = performance.now() + Math.max(1, seconds) * 1000;
        const latencies = [];
        let inserts = 0;
        let deletes = 0;
        let searches = 0;
        while (performance.now() < end) {
            for (let i = 0; i < config.insert / 10; i++) {
                this.insert(1, false);
                inserts++;
            }
            for (let i = 0; i < config.delete / 10 && this.liveVectors.size > 20; i++) {
                this.delete(1, false);
                deletes++;
            }
            for (let i = 0; i < config.search / 10; i++) {
                const { latency } = this.doSearch(this.randomLoadedVec());
                latencies.push(latency);
                searches++;
            }
            await sleep(0);
        }
        const p50 = percentile(latencies, 0.5);
        const p99 = percentile(latencies, 0.99);
        this.markProof('stress');
        this.log(`stress ${mode}: inserts=${inserts} deletes=${deletes} searches=${searches} p50=${p50.toFixed(3)}ms p99=${p99.toFixed(3)}ms`, 'success');
    }

    async handleCommand(input) {
        const parts = input.trim().split(/\s+/).filter(Boolean);
        if (parts.length === 0) return true;
        const [command, ...args] = parts;
        switch (command) {
        case 'help':
            this.printHelp();
            break;
        case 'status':
            this.printStatus();
            break;
        case 'checklist':
            this.printChecklist();
            break;
        case 'build':
            await this.build(args[0] ? Number.parseInt(args[0], 10) : undefined);
            break;
        case 'reset':
            await this.createIndex();
            this.log('Index reset.', 'success');
            break;
        case 'search':
            await this.search(args[0] ? Number.parseInt(args[0], 10) : LATENCY_CHECK_QUERIES);
            break;
        case 'insert':
            this.insert(args[0] ? Number.parseInt(args[0], 10) : 1);
            break;
        case 'delete':
            this.delete(args[0] ? Number.parseInt(args[0], 10) : 1);
            break;
        case 'compact':
            this.compact();
            break;
        case 'export':
            this.export(args[0] || DEFAULT_EXPORT_PATH);
            break;
        case 'import':
            await this.import(args[0] || DEFAULT_EXPORT_PATH);
            break;
        case 'validate':
        case 'proof':
            await this.validate(args[0] || 'all');
            break;
        case 'stress':
            await this.stress(args[0] || 'read', args[1] ? Number.parseInt(args[1], 10) : 5);
            break;
        case 'quit':
        case 'exit':
            return false;
        default:
            throw new Error(`unknown command "${command}"`);
        }
        return true;
    }

    async runInteractive() {
        const rl = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: 'pancake> ' });
        let shouldExit = false;
        rl.on('SIGINT', () => {
            shouldExit = true;
            rl.close();
        });

        try {
            rl.prompt();
            for await (const line of rl) {
                try {
                    if (!await this.handleCommand(line)) {
                        shouldExit = true;
                        rl.close();
                        break;
                    }
                } catch (error) {
                    this.log(error && error.message ? error.message : String(error), 'error');
                }
                if (shouldExit) break;
                rl.prompt();
            }
        } finally {
            rl.close();
        }
    }

    dispose() {
        if (this.index) this.index.dispose();
    }
}

(async () => {
    const Pancake = await loadPancake();
    const cli = new TechnicalDemoCLI(Pancake);
    const args = process.argv.slice(2);
    await cli.init(args.length === 0);
    try {
        if (args.length > 0) {
            await cli.handleCommand(args.join(' '));
        } else {
            await cli.runInteractive();
        }
    } finally {
        cli.dispose();
    }
})().catch(error => {
    console.error(error && error.stack ? error.stack : error);
    process.exit(1);
});
