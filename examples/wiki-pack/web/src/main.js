// Wiki knowledge-pack demo skeleton: the real browser pipeline with timing
// instrumentation, minimal UI. Everything loads from this origin — the pack
// via range reads, the fp16 MiniLM encoder, and the ONNX runtime wasm.
import Pancake from '../../../../pancake.web.mjs';
import { pipeline, env } from '@huggingface/transformers';
import { createAbstentionScorer } from './abstention.js';

const PACK_BASE = '/pack';
const K = 10;
const params = new URLSearchParams(location.search);
const FETCH_PARALLELISM = Number(params.get('p') || 32);
const FETCH_GAP = Number(params.get('gap') || 16384);
const RERANK = params.get('C') ? Number(params.get('C')) : undefined;

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = '/models/';
env.backends.onnx.wasm.wasmPaths = '/ort/';

const els = Object.fromEntries(['status', 'meter', 'query', 'go', 'timing', 'results']
    .map((id) => [id, document.getElementById(id)]));

const meter = { requests: 0, bytes: 0, note: {} };
function meterAdd(kind, bytes, requests = 1) {
    meter.requests += requests;
    meter.bytes += bytes;
    meter.note[kind] = (meter.note[kind] || 0) + bytes;
    const parts = Object.entries(meter.note).map(([k, v]) => `${k} ${(v / 1e6).toFixed(1)} MB`);
    els.meter.textContent = `network: ${meter.requests} requests, ${(meter.bytes / 1e6).toFixed(1)} MB total — ${parts.join(' · ')}`;
}

function createHttpRangeSource(url, kind) {
    let fullBuffer = null;
    return {
        async read(offset, length) {
            if (fullBuffer) return fullBuffer.subarray(offset, offset + length);
            const response = await fetch(url, { headers: { Range: `bytes=${offset}-${offset + length - 1}` } });
            if (response.status !== 206 && response.status !== 200) {
                throw new Error(`Range read failed: ${response.status}`);
            }
            const bytes = new Uint8Array(await response.arrayBuffer());
            meterAdd(kind, bytes.byteLength);
            if (response.status === 200 && bytes.byteLength > length) {
                fullBuffer = bytes; // host ignored Range: degrade to one full download
                return fullBuffer.subarray(offset, offset + length);
            }
            if (bytes.byteLength !== length) throw new Error(`short range read: ${bytes.byteLength}/${length}`);
            return bytes;
        },
    };
}

const state = {};
const timings = { boot: {} };

async function timed(name, fn) {
    const t0 = performance.now();
    const out = await fn();
    timings.boot[name] = Math.round(performance.now() - t0);
    return out;
}

async function boot() {
    const status = (s) => { els.status.textContent = s; };

    status('loading encoder (fp16 MiniLM, self-hosted)…');
    state.embedder = await timed('modelLoad', () =>
        pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'fp16' }));
    // meter the model+wasm downloads coarsely via resource timing
    for (const entry of performance.getEntriesByType('resource')) {
        if (entry.name.includes('/models/') || entry.name.includes('/ort/')) {
            meterAdd('encoder', entry.transferSize || entry.encodedBodySize || 0);
        }
    }

    status('warming up encoder…');
    await timed('encoderWarmup', () => embed('warmup query'));

    status('opening pack (verifying resident tier)…');
    state.artifact = await timed('packOpen', () =>
        Pancake.SketchArtifact.open(createHttpRangeSource(`${PACK_BASE}/wiki.pancake-sketch`, 'pack'), {}));

    status('building WASM scanner…');
    state.scanner = await timed('scannerBuild', () => Pancake.createSketchScanner(state.artifact));

    status('loading corpus offsets…');
    state.offsets = await timed('offsetsLoad', async () => {
        const r = await fetch(`${PACK_BASE}/corpus-offsets.u32`);
        const b = await r.arrayBuffer();
        meterAdd('offsets', b.byteLength);
        return new Uint32Array(b);
    });
    state.corpusSource = createHttpRangeSource(`${PACK_BASE}/corpus.bin`, 'hydration');

    status('loading abstention calibration…');
    const [abstentionAsset, bloomBuf] = await Promise.all([
        fetch(`${PACK_BASE}/wiki-abstention.json`).then((r) => r.json()),
        fetch(`${PACK_BASE}/wiki-vocab.bloom`).then((r) => r.arrayBuffer()),
    ]);
    meterAdd('abstention', bloomBuf.byteLength);
    state.abstention = createAbstentionScorer(abstentionAsset, bloomBuf);

    const manifest = await (await fetch(`${PACK_BASE}/pack-manifest.json`)).json();
    state.manifest = manifest;
    status(`ready — ${state.artifact.count.toLocaleString()} chunks (${manifest.articles.toLocaleString()} articles), `
        + `${(state.artifact.residentBytes / 1e6).toFixed(1)} MB resident`
        + (state.artifact.residentVerified ? ', hash verified' : '')
        + `\nboot: ${Object.entries(timings.boot).map(([k, v]) => `${k} ${v}ms`).join(', ')}`);
    els.query.disabled = false;
    els.go.disabled = false;
    window.__wikiPackReady = true;
}

async function embed(text) {
    const out = await state.embedder(text, { pooling: 'mean', normalize: true });
    return Float32Array.from(out.data);
}

async function hydrate(ids) {
    return Promise.all(ids.map(async (id) => {
        const start = state.offsets[id];
        const bytes = await state.corpusSource.read(start, state.offsets[id + 1] - start);
        return JSON.parse(new TextDecoder().decode(bytes));
    }));
}

async function runQuery(text) {
    const t = {};
    const m0 = { requests: meter.requests, bytes: meter.bytes };
    let t0 = performance.now();
    const qv = await embed(text);
    t.embed = performance.now() - t0;

    t0 = performance.now();
    const { results } = await state.artifact.search(qv, K, { scanner: state.scanner, parallelism: FETCH_PARALLELISM, gap: FETCH_GAP, rerank: RERANK });
    t.search = performance.now() - t0;

    const abstention = state.abstention ? state.abstention.score(text, results) : null;

    t0 = performance.now();
    const rows = abstention && abstention.verdict === 'abstain'
        ? [] : await hydrate(results.map((r) => r.id));
    t.hydrate = performance.now() - t0;
    t.total = t.embed + t.search + t.hydrate;
    t.queryRequests = meter.requests - m0.requests;
    t.queryBytes = meter.bytes - m0.bytes;
    return { results, rows, t, abstention };
}

function renderQuery(text, { results, rows, t, abstention }) {
    els.timing.textContent = `embed ${t.embed.toFixed(0)} ms · search ${t.search.toFixed(0)} ms · `
        + `hydrate ${t.hydrate.toFixed(0)} ms · total ${t.total.toFixed(0)} ms — `
        + `${t.queryRequests} requests, ${(t.queryBytes / 1024).toFixed(0)} KiB for this query`;
    els.results.innerHTML = '';
    if (abstention && abstention.verdict === 'abstain') {
        const div = document.createElement('div');
        div.className = 'hit';
        div.textContent = 'This pack has nothing useful for that query — abstaining rather than '
            + `showing noise. (confidence ${abstention.p.toFixed(2)}, best distance ${abstention.signals.d0.toFixed(2)})`;
        els.results.appendChild(div);
        return;
    }
    if (abstention && abstention.verdict === 'weak') {
        const div = document.createElement('div');
        div.className = 'hit d';
        div.textContent = 'Weak match — the closest content is distant from the question; showing it anyway.';
        els.results.appendChild(div);
    }
    results.forEach((r, i) => {
        const row = rows[i];
        const div = document.createElement('div');
        div.className = 'hit';
        const title = document.createElement('div');
        title.className = 't';
        title.textContent = `${i + 1}. ${row.title}  `;
        const dist = document.createElement('span');
        dist.className = 'd';
        dist.textContent = `distance ${r.distance.toFixed(3)}`;
        title.appendChild(dist);
        const body = document.createElement('div');
        body.textContent = row.text.slice(row.title.length + 2, row.title.length + 300);
        div.append(title, body);
        els.results.appendChild(div);
    });
}

els.go.addEventListener('click', async () => {
    const q = els.query.value.trim();
    if (!q) return;
    renderQuery(q, await runQuery(q));
});
els.query.addEventListener('keydown', (e) => { if (e.key === 'Enter') els.go.click(); });

// Headless benchmark hook: window.__bench(queries) -> per-query timings.
window.__bench = async (queries) => {
    const out = { boot: timings.boot, queries: [] };
    for (const q of queries) {
        const { results, t, abstention } = await runQuery(q);
        out.queries.push({ q, t, topDistance: results[0]?.distance, verdict: abstention?.verdict });
    }
    out.meter = { requests: meter.requests, bytes: meter.bytes, note: meter.note };
    return out;
};

// Golden-probe hook: verify the shipped calibration in the real browser.
window.__probes = async () => {
    const probes = await (await fetch(`${PACK_BASE}/wiki-abstention-probes.json`)).json();
    const failures = [];
    for (const probe of probes) {
        const { abstention } = await runQuery(probe.text);
        const expected = Array.isArray(probe.expect) ? probe.expect : [probe.expect];
        if (!expected.includes(abstention.verdict)) {
            failures.push({ text: probe.text, expected, got: abstention.verdict, p: abstention.p });
        }
    }
    return { total: probes.length, failures };
};

boot().catch((e) => { els.status.textContent = `boot failed: ${e.message}`; console.error(e); window.__wikiPackError = String(e); });
