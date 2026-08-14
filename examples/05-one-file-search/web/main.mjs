// Browser host for the one-file reader: HTTP range source + a thin UI.
// The reader, encoder, sketch scan, and calibration are the same modules
// Node runs — no WASM, no server-side search.

import { openPancakeFile } from '../pancake-file-reader.mjs';

const FILE_URL = '/pancake-docs.pancake';

function httpRangeSource(url) {
    const stats = { requests: 0, bytes: 0 };
    return {
        stats,
        size: undefined,
        async init() {
            const head = await fetch(url, { method: 'HEAD' });
            if (head.ok) {
                const len = Number(head.headers.get('content-length'));
                if (Number.isFinite(len) && len > 0) this.size = len;
            }
        },
        async read(offset, length) {
            stats.requests += 1;
            stats.bytes += length;
            const response = await fetch(url, {
                headers: { Range: `bytes=${offset}-${offset + length - 1}` },
            });
            if (response.status === 206) {
                return new Uint8Array(await response.arrayBuffer());
            }
            if (response.status === 200) {
                // Host without range support: slice the full body (works, but
                // defeats laziness — the demo server supports ranges).
                const all = new Uint8Array(await response.arrayBuffer());
                return all.subarray(offset, offset + length);
            }
            throw new Error(`range read failed: HTTP ${response.status}`);
        },
    };
}

const statusEl = document.getElementById('status');
const inputEl = document.getElementById('q');
const verdictEl = document.getElementById('verdict');
const resultsEl = document.getElementById('results');

const source = httpRangeSource(FILE_URL);
await source.init();
const openStart = performance.now();
const search = await openPancakeFile(source);
const openMs = performance.now() - openStart;
const info = search.info();

statusEl.textContent = `${(info.fileBytes / 1048576).toFixed(2)} MiB file · `
    + `open ${openMs.toFixed(0)} ms, ${source.stats.requests} requests / ${(source.stats.bytes / 1024).toFixed(0)} KiB · `
    + `${info.records} chunks · identity ${info.identity.slice(0, 12)}… · hash verified: ${info.residentVerified}`;
inputEl.disabled = false;
inputEl.placeholder = info.sampleQueries[0] || 'ask the docs anything…';

let running = false;
async function run() {
    const text = inputEl.value.trim();
    if (!text || running) return;
    running = true;
    const before = { ...source.stats };
    const t0 = performance.now();
    try {
        const out = await search.query(text, { k: 5 });
        const ms = performance.now() - t0;
        const requests = source.stats.requests - before.requests;
        const kib = (source.stats.bytes - before.bytes) / 1024;
        const badge = `<span class="badge ${out.matchQuality}">${out.matchQuality}</span>`;
        const confidence = out.confidence !== undefined ? ` confidence ${out.confidence.toFixed(3)} ·` : '';
        verdictEl.innerHTML = `${badge}${confidence} ${ms.toFixed(0)} ms · ${requests} range requests · ${kib.toFixed(1)} KiB fetched`;
        resultsEl.innerHTML = out.results.length === 0
            ? '<p class="p">No results — the file knows this corpus cannot answer that.</p>'
            : out.results.map((r) => `
                <div class="hit">
                  <div class="t">${escapeHtml(r.title)}</div>
                  <div class="p">${escapeHtml(r.sourcePath)}${r.anchor ? '#' + escapeHtml(r.anchor) : ''} · distance ${r.distance.toFixed(3)}</div>
                  <div class="x">${escapeHtml((r.preview || r.text).slice(0, 220))}…</div>
                </div>`).join('');
    } catch (err) {
        verdictEl.textContent = String(err && err.message ? err.message : err);
    } finally {
        running = false;
    }
}

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

inputEl.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') run();
});
