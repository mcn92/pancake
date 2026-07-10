import Pancake from '../../pancake.workerd.mjs';
import { DEMO_DIM, embedText } from './embedder.mjs';

const INDEX_KEY = 'docs-index.bin';
const CORPUS_KEY = 'docs-corpus.json';
const MANIFEST_KEY = 'docs-manifest.json';
const MAX_RESULTS = 8;
const DEFAULT_MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024;
const DEFAULT_MAX_JSON_BYTES = 256 * 1024;
let index = null;
let manifest = null;
let corpus = [];
let corpusById = new Map();
let sourceFilters = new Map();
let loadPromise = null;
let state = {
  restoreCount: 0,
  restoredAt: null,
  lastRestoreMs: null,
};
const ADMIN_ROUTES = new Set(['/readiness', '/reset_cache']);

function formatMs(value) {
  if (!Number.isFinite(value) || value <= 0) return 0.01;
  return Math.max(0.01, Math.round(value * 100) / 100);
}

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: corsHeaders({ 'content-type': 'application/json; charset=utf-8' })
  });
}

function isReadOnly(env) {
  const value = String(env.READ_ONLY || '').trim().toLowerCase();
  return value === '1' || value === 'true' || value === 'yes' || value === 'on';
}

function requireAdminAuth(request, env) {
  if (!env.API_KEY && !allowInsecureAdmin(env)) {
    return jsonResponse({ error: 'Admin routes require API_KEY or explicit ALLOW_INSECURE_ADMIN=1' }, 403);
  }
  if (!env.API_KEY) return null;
  const authHeader = request.headers.get('Authorization') || '';
  const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : '';
  if (token !== env.API_KEY) {
    return jsonResponse({ error: 'Unauthorized' }, 401);
  }
  return null;
}

function getMaxSnapshotBytes(env) {
  const raw = env?.MAX_SNAPSHOT_BYTES;
  if (raw === undefined || raw === null || raw === '') return DEFAULT_MAX_SNAPSHOT_BYTES;
  const parsed = parseInt(raw, 10);
  if (!Number.isInteger(parsed) || parsed < 1) return DEFAULT_MAX_SNAPSHOT_BYTES;
  return parsed;
}

function getMaxJsonBytes(env) {
  const raw = env?.MAX_JSON_BYTES;
  if (raw === undefined || raw === null || raw === '') return DEFAULT_MAX_JSON_BYTES;
  const parsed = parseInt(raw, 10);
  if (!Number.isInteger(parsed) || parsed < 1) return DEFAULT_MAX_JSON_BYTES;
  return parsed;
}

function getCorsOrigin() {
  const value = String(globalThis.__PANCAKE_ALLOWED_ORIGIN__ || '').trim();
  return value || null;
}

function corsHeaders(extraHeaders = {}) {
  const origin = getCorsOrigin();
  return origin
    ? { 'access-control-allow-origin': origin, ...extraHeaders }
    : { ...extraHeaders };
}

function allowInsecureAdmin(env) {
  const value = String(env.ALLOW_INSECURE_ADMIN || '').trim().toLowerCase();
  return value === '1' || value === 'true' || value === 'yes' || value === 'on';
}

async function parseJsonBody(request, env) {
  const maxJsonBytes = getMaxJsonBytes(env);
  const contentLength = parseInt(request.headers.get('content-length') || '', 10);
  if (Number.isInteger(contentLength) && contentLength > maxJsonBytes) {
    return { error: `JSON body exceeds MAX_JSON_BYTES (${contentLength} > ${maxJsonBytes})`, status: 413 };
  }
  try {
    const text = await request.text();
    const size = new TextEncoder().encode(text).byteLength;
    if (size > maxJsonBytes) {
      return { error: `JSON body exceeds MAX_JSON_BYTES (${size} > ${maxJsonBytes})`, status: 413 };
    }
    return { body: JSON.parse(text) };
  } catch {
    return { error: 'Invalid JSON', status: 400 };
  }
}

function htmlResponse(body) {
  return new Response(body, {
    headers: {
      'content-type': 'text/html; charset=utf-8'
    }
  });
}

function sourceLabelFromPath(sourcePath) {
  if (sourcePath === 'README.md') return 'README';
  if (sourcePath === 'QUICKSTART.md') return 'Quickstart';
  if (sourcePath === 'docs/SYSTEM_DESIGN.md') return 'System design';
  if (sourcePath === 'examples/worker/README.md') return 'Worker docs';
  return sourcePath;
}

function rebuildSourceFilters() {
  sourceFilters = new Map();
  for (const chunk of corpus) {
    if (!sourceFilters.has(chunk.sourcePath)) {
      sourceFilters.set(chunk.sourcePath, new Set());
    }
    sourceFilters.get(chunk.sourcePath).add(chunk.id);
  }
}

function renderPage() {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pancake Worker Semantic Search</title>
  <style>
    :root {
      --bg: #efe6d7;
      --paper: rgba(255, 252, 247, 0.78);
      --paper-strong: rgba(255, 252, 247, 0.92);
      --ink: #1d2421;
      --muted: #5f665e;
      --line: rgba(29, 36, 33, 0.14);
      --accent: #0f7b63;
      --accent-soft: rgba(15, 123, 99, 0.12);
      --signal: #b4572f;
      --mono: "SFMono-Regular", "Menlo", "Consolas", monospace;
      --serif: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      --sans: "Avenir Next", "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: var(--sans);
      background:
        radial-gradient(circle at 12% 0%, rgba(180, 87, 47, 0.12), transparent 26%),
        radial-gradient(circle at 92% 8%, rgba(15, 123, 99, 0.18), transparent 31%),
        linear-gradient(180deg, #f7f2e8 0%, var(--bg) 100%);
    }
    main {
      max-width: 1120px;
      margin: 0 auto;
      padding: 34px 20px 80px;
    }
    .frame {
      position: relative;
      border: 1px solid var(--line);
      border-radius: 30px;
      background: linear-gradient(180deg, var(--paper-strong), var(--paper));
      box-shadow: 0 20px 80px rgba(31, 38, 33, 0.08);
      overflow: hidden;
    }
    .frame::before {
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 10px;
      background: linear-gradient(180deg, var(--signal), var(--accent));
    }
    .frame::after {
      content: "";
      position: absolute;
      inset: auto 0 0 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(29, 36, 33, 0.16), transparent);
    }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(260px, 0.7fr);
      gap: 26px;
      padding: 34px 34px 28px 42px;
      align-items: start;
    }
    .hero-copy {
      display: grid;
      gap: 18px;
    }
    .hero-side {
      display: grid;
      gap: 12px;
      align-self: stretch;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.48);
    }
    .eyebrow {
      margin: 0;
      font-family: var(--mono);
      font-size: 0.78rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--signal);
    }
    h1 {
      margin: 0;
      font-size: clamp(2.5rem, 6vw, 5.4rem);
      line-height: 0.9;
      letter-spacing: -0.05em;
      font-family: var(--serif);
      max-width: 9ch;
    }
    .lede {
      max-width: 58ch;
      color: var(--muted);
      font-size: 1.08rem;
      line-height: 1.7;
    }
    .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .pill, button {
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.7);
      color: var(--ink);
      padding: 10px 14px;
      font: inherit;
    }
    .pill {
      border-radius: 999px;
      font-size: 0.9rem;
    }
    .side-kicker {
      margin: 0;
      font-family: var(--mono);
      font-size: 0.77rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .side-value {
      margin: 0;
      font-family: var(--serif);
      font-size: 1.2rem;
      line-height: 1.25;
    }
    .side-note {
      margin: 0;
      color: var(--muted);
      font-size: 0.95rem;
      line-height: 1.55;
    }
    .search {
      padding: 0 34px 34px 42px;
      display: grid;
      gap: 14px;
    }
    .controls {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(280px, 0.8fr);
      gap: 18px;
      align-items: start;
    }
    .control-stack {
      display: grid;
      gap: 14px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.42);
    }
    .control-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .control {
      display: grid;
      gap: 6px;
    }
    .control label {
      font-family: var(--mono);
      font-size: 0.76rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .search-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
    }
    input, select {
      width: 100%;
      border-radius: 16px;
      border: 1px solid var(--line);
      padding: 17px 18px;
      font: inherit;
      background: rgba(255, 255, 255, 0.88);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
    }
    input[type="number"] {
      padding-right: 8px;
    }
    .submit {
      border-radius: 16px;
      background: linear-gradient(135deg, var(--accent), #095848);
      color: white;
      border: none;
      padding-inline: 22px;
      font-weight: 600;
      box-shadow: 0 10px 24px rgba(15, 123, 99, 0.18);
    }
    .ghost {
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.72);
      color: var(--ink);
      padding-inline: 16px;
      cursor: pointer;
      transition: transform 140ms ease, background 140ms ease;
    }
    .ghost:hover, .submit:hover {
      transform: translateY(-1px);
    }
    .ghost:hover {
      background: rgba(255, 255, 255, 0.92);
    }
    .samples {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .sample {
      cursor: pointer;
      border-radius: 999px;
      background: white;
      transition: transform 140ms ease, border-color 140ms ease;
    }
    .sample:hover {
      transform: translateY(-1px);
      border-color: rgba(15, 123, 99, 0.35);
    }
    .meta, .empty {
      color: var(--muted);
      font-size: 0.95rem;
    }
    .meta {
      font-family: var(--mono);
      letter-spacing: 0.01em;
    }
    .results {
      margin-top: 22px;
      display: grid;
      gap: 0;
    }
    .results-head {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      padding: 0 4px 14px;
      border-bottom: 1px solid var(--line);
    }
    .results-title {
      margin: 0;
      font-family: var(--mono);
      font-size: 0.8rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .results-subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 0.95rem;
    }
    .results-list {
      display: grid;
      position: relative;
    }
    .stats-panel {
      display: grid;
      gap: 10px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.45);
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .stat {
      padding: 12px 12px 10px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.68);
      border: 1px solid rgba(29, 36, 33, 0.08);
    }
    .stat-k {
      display: block;
      margin-bottom: 4px;
      font-family: var(--mono);
      font-size: 0.72rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .stat-v {
      display: block;
      font-family: var(--serif);
      font-size: 1.08rem;
      line-height: 1.2;
    }
    .card {
      padding: 22px 6px 22px 0;
      border-bottom: 1px solid var(--line);
      background: transparent;
      display: grid;
      grid-template-columns: 92px minmax(0, 1fr);
      gap: 18px;
      transform: translateY(10px);
      opacity: 0;
      animation: rise 260ms ease forwards;
    }
    .rank {
      display: grid;
      align-content: start;
      justify-items: start;
      gap: 6px;
      padding-top: 2px;
    }
    .rank-num {
      display: inline-flex;
      min-width: 44px;
      justify-content: center;
      padding: 8px 10px;
      border-radius: 999px;
      background:
        linear-gradient(135deg, rgba(15, 123, 99, 0.16), rgba(180, 87, 47, 0.12));
      color: var(--accent);
      font-family: var(--mono);
      font-size: 0.82rem;
      font-weight: 700;
    }
    .score {
      font-family: var(--mono);
      color: var(--muted);
      font-size: 0.8rem;
    }
    .card-body {
      display: grid;
      gap: 8px;
      min-width: 0;
    }
    .card-topline {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    .card h2 {
      margin: 0;
      font-size: 1.18rem;
      line-height: 1.25;
      font-family: var(--serif);
    }
    .path {
      font-family: var(--mono);
      color: var(--signal);
      font-size: 0.78rem;
      letter-spacing: 0.02em;
    }
    .card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.62;
      max-width: 70ch;
    }
    .anchor {
      font-family: var(--mono);
      color: var(--muted);
      font-size: 0.78rem;
    }
    .empty {
      padding: 28px 0 0;
      max-width: 54ch;
      line-height: 1.6;
    }
    @keyframes rise {
      to { transform: translateY(0); opacity: 1; }
    }
    @media (max-width: 720px) {
      .hero {
        grid-template-columns: 1fr;
        padding: 26px 22px 22px 30px;
      }
      .search {
        padding: 0 22px 26px 30px;
      }
      .controls,
      .control-grid,
      .stats-grid {
        grid-template-columns: 1fr;
      }
      .search-row {
        grid-template-columns: 1fr;
      }
      .submit {
        width: 100%;
      }
      .card {
        grid-template-columns: 1fr;
        gap: 12px;
      }
      .rank {
        grid-auto-flow: column;
        justify-content: start;
        align-items: center;
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="frame">
      <div class="hero">
        <div class="hero-copy">
          <p class="eyebrow">Pancake + Cloudflare Workers</p>
          <div class="stats">
            <span class="pill">Snapshot in R2</span>
            <span class="pill">Cold restore</span>
            <span class="pill">Hot in-memory search</span>
          </div>
          <h1>Docs search with a visible cold-start story.</h1>
          <p class="lede">
            The point of this demo is not “edge database.” It is a query-optimized copy:
            the index is built offline, restored from object storage when an isolate wakes up,
            and then served fast while that isolate stays warm.
          </p>
        </div>
        <aside class="hero-side">
          <p class="side-kicker">What to look for</p>
          <p class="side-value">First query pays for restore. The next ones should feel cheap.</p>
          <p class="side-note">
            Ask about snapshot restore, compaction after deletes, filtered search, or quantized
            memory tradeoffs. The response panel calls out whether you hit a cold restore or warm cache.
          </p>
        </aside>
      </div>
      <form class="search" id="search-form">
        <div class="controls">
          <div class="control-stack">
            <div class="search-row">
              <input id="query" name="query" autocomplete="off" value="How do Cloudflare Workers restore snapshots from R2?">
              <button class="submit" type="submit">Search</button>
            </div>
            <div class="control-grid">
              <div class="control">
                <label for="k">Top K</label>
                <input id="k" name="k" type="number" min="1" max="8" value="5">
              </div>
              <div class="control">
                <label for="ef-search">EF Search</label>
                <input id="ef-search" name="ef-search" type="number" min="10" max="400" step="10" value="120">
              </div>
              <div class="control">
                <label for="source-filter">Source Filter</label>
                <select id="source-filter" name="source-filter">
                  <option value="">All docs</option>
                  <option value="README.md">README</option>
                  <option value="QUICKSTART.md">Quickstart</option>
                  <option value="docs/SYSTEM_DESIGN.md">System design</option>
                  <option value="examples/worker/README.md">Worker docs</option>
                </select>
              </div>
            </div>
          </div>
          <aside class="stats-panel">
            <p class="side-kicker">Live stats</p>
            <div class="stats-grid" id="stats-grid">
              <div class="stat"><span class="stat-k">Cache</span><span class="stat-v">Not loaded</span></div>
              <div class="stat"><span class="stat-k">Chunks</span><span class="stat-v">-</span></div>
              <div class="stat"><span class="stat-k">Dim</span><span class="stat-v">256</span></div>
              <div class="stat"><span class="stat-k">Quantized</span><span class="stat-v">Yes</span></div>
              <div class="stat"><span class="stat-k">Mode</span><span class="stat-v">Public search</span></div>
              <div class="stat"><span class="stat-k">Restores</span><span class="stat-v">0</span></div>
              <div class="stat"><span class="stat-k">Last restore</span><span class="stat-v">-</span></div>
            </div>
            <button class="ghost" id="reset-cache" type="button">Force cold reload</button>
          </aside>
        </div>
        <div class="samples">
          <button class="sample" type="button" data-query="How do Cloudflare Workers restore snapshots from R2?">Cold restore</button>
          <button class="sample" type="button" data-query="Why do I need compact before export after deletes?">Ghost cleanup</button>
          <button class="sample" type="button" data-query="How does filtered search work in Pancake?">Filtered search</button>
          <button class="sample" type="button" data-query="What are the memory tradeoffs for quantized indexes?">Memory tradeoffs</button>
        </div>
        <div class="meta" id="meta">Ready.</div>
      </form>
    </section>
    <section class="results" id="results">
      <div class="results-head">
        <div>
          <p class="results-title">Top Matches</p>
          <p class="results-subtitle">Run a query to inspect the chunks the Worker is retrieving from the snapshot-backed index.</p>
        </div>
        <p class="side-kicker">Query-time controls on the left, runtime state on the right.</p>
      </div>
      <div class="results-list">
        <div class="empty">Run a query to load the index and inspect the top matching doc chunks.</div>
      </div>
    </section>
  </main>
  <script>
    const form = document.getElementById('search-form');
    const input = document.getElementById('query');
    const kInput = document.getElementById('k');
    const efInput = document.getElementById('ef-search');
    const sourceFilter = document.getElementById('source-filter');
    const meta = document.getElementById('meta');
    const results = document.getElementById('results');
    const resultsList = document.querySelector('.results-list');
    const statsGrid = document.getElementById('stats-grid');
    const resetButton = document.getElementById('reset-cache');

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function renderStats(payload) {
      const rows = [
        ['Cache', payload.loaded ? 'Warm' : 'Not loaded'],
        ['Chunks', payload.corpus_chunks ?? '-'],
        ['Dim', payload.dim ?? '256'],
        ['Quantized', payload.quantized ? 'Yes' : 'No'],
        ['Mode', payload.read_only ? 'Read-only' : 'Admin-capable'],
        ['Restores', payload.restore_count ?? 0],
        ['Last restore', payload.last_restore_ms ? payload.last_restore_ms.toFixed(2) + 'ms' : '-'],
      ];
      statsGrid.innerHTML = rows.map(([k, v]) =>
        '<div class="stat"><span class="stat-k">' + escapeHtml(k) + '</span><span class="stat-v">' + escapeHtml(v) + '</span></div>'
      ).join('');
      resetButton.disabled = !!payload.read_only;
      resetButton.title = payload.read_only
        ? 'Read-only mode blocks cache reset.'
        : 'Drop the warm in-memory cache so the next query restores from R2.';
    }

    async function refreshStats() {
      const res = await fetch('/health');
      const payload = await res.json();
      if (res.ok) renderStats(payload);
    }

    function renderResults(payload) {
      meta.textContent =
        'Search ' + payload.search_ms.toFixed(2) + 'ms' +
        (payload.cache_state === 'cold-restored'
          ? ' • cold restore ' + payload.restore_ms.toFixed(2) + 'ms'
          : ' • warm cache') +
        ' • ' + payload.result_count + ' results' +
        (payload.filter_label ? ' • filter ' + payload.filter_label : '') +
        ' • ef ' + payload.ef_search;

      if (!payload.results.length) {
        resultsList.innerHTML = '<div class="empty">No results.</div>';
        return;
      }

      resultsList.innerHTML = payload.results.map((item, i) => {
        const score = Math.max(0, 1 - item.distance).toFixed(3);
        return '<article class="card" style="animation-delay:' + (i * 45) + 'ms">' +
          '<div class="rank">' +
            '<span class="rank-num">#' + (i + 1) + '</span>' +
            '<span class="score">score ' + score + '</span>' +
          '</div>' +
          '<div class="card-body">' +
            '<div class="card-topline">' +
              '<span class="path">' + escapeHtml(item.source_path) + '</span>' +
              '<span class="anchor">#' + escapeHtml(item.anchor) + '</span>' +
            '</div>' +
            '<h2>' + escapeHtml(item.title) + '</h2>' +
            '<p>' + escapeHtml(item.preview) + '</p>' +
          '</div>' +
          '</article>';
      }).join('');
    }

    async function runSearch(query) {
      meta.textContent = 'Searching...';
      const params = new URLSearchParams({
        q: query,
        k: kInput.value || '5',
        ef: efInput.value || '120',
      });
      if (sourceFilter.value) params.set('source', sourceFilter.value);
      const res = await fetch('/search?' + params.toString());
      const payload = await res.json();
      if (!res.ok) {
        meta.textContent = payload.error || 'Search failed';
        return;
      }
      renderResults(payload);
      renderStats({
        loaded: true,
        corpus_chunks: payload.corpus_chunks,
        dim: payload.dim,
        quantized: payload.quantized,
        restore_count: payload.restore_count,
        last_restore_ms: payload.cache_state === 'cold-restored' ? payload.restore_ms : payload.last_restore_ms
      });
    }

    form.addEventListener('submit', (event) => {
      event.preventDefault();
      runSearch(input.value.trim());
    });

    document.querySelectorAll('.sample').forEach((button) => {
      button.addEventListener('click', () => {
        input.value = button.dataset.query;
        runSearch(input.value);
      });
    });

    resetButton.addEventListener('click', async () => {
      meta.textContent = 'Clearing in-memory cache...';
      const res = await fetch('/reset_cache', { method: 'POST' });
      const payload = await res.json();
      if (!res.ok) {
        meta.textContent = payload.error || 'Reset failed';
        return;
      }
      meta.textContent = 'Cache cleared. Next query will restore from R2.';
      await refreshStats();
    });

    refreshStats();
  </script>
</body>
</html>`;
}

async function loadJson(obj) {
  return JSON.parse(await obj.text());
}

async function restoreFromR2(env) {
  if (!env.DOCS_BUCKET) {
    throw new Error('DOCS_BUCKET binding is missing');
  }

  const t0 = performance.now();
  const [manifestObj, corpusObj, indexObj] = await Promise.all([
    env.DOCS_BUCKET.get(MANIFEST_KEY),
    env.DOCS_BUCKET.get(CORPUS_KEY),
    env.DOCS_BUCKET.get(INDEX_KEY)
  ]);

  if (!manifestObj || !corpusObj || !indexObj) {
    throw new Error(`Expected ${MANIFEST_KEY}, ${CORPUS_KEY}, and ${INDEX_KEY} in R2`);
  }
  const maxSnapshotBytes = getMaxSnapshotBytes(env);
  if (Number.isInteger(indexObj.size) && indexObj.size > maxSnapshotBytes) {
    throw new Error(`Index snapshot exceeds MAX_SNAPSHOT_BYTES (${indexObj.size} > ${maxSnapshotBytes})`);
  }

  const loadedManifest = await loadJson(manifestObj);
  if (loadedManifest.dim !== DEMO_DIM) {
    throw new Error(`Manifest dim ${loadedManifest.dim} does not match demo dim ${DEMO_DIM}`);
  }

  const [loadedCorpus, snapshotBuffer] = await Promise.all([
    loadJson(corpusObj),
    indexObj.arrayBuffer()
  ]);
  if (snapshotBuffer.byteLength > maxSnapshotBytes) {
    throw new Error(`Index snapshot exceeds MAX_SNAPSHOT_BYTES (${snapshotBuffer.byteLength} > ${maxSnapshotBytes})`);
  }

  const restored = await Pancake.create({
    dim: loadedManifest.dim,
    maxElements: loadedManifest.maxElements,
    metric: loadedManifest.metric,
    quantized: loadedManifest.quantized,
    M: loadedManifest.M,
    efConstruction: loadedManifest.efConstruction,
    efSearch: loadedManifest.efSearch
  });
  restored.import(new Uint8Array(snapshotBuffer));

  corpus = loadedCorpus;
  corpusById = new Map(corpus.map((entry) => [entry.id, entry]));
  rebuildSourceFilters();
  manifest = loadedManifest;
  index = restored;
  state.restoreCount += 1;
  state.lastRestoreMs = performance.now() - t0;
  state.restoredAt = new Date().toISOString();
}

async function ensureLoaded(env) {
  if (index) {
    return { loadedFromR2: false, restoreMs: 0 };
  }

  const callerTriggeredRestore = !loadPromise;
  if (!loadPromise) {
    loadPromise = restoreFromR2(env).finally(() => {
      loadPromise = null;
    });
  }

  await loadPromise;
  return {
    loadedFromR2: callerTriggeredRestore,
    restoreMs: callerTriggeredRestore ? state.lastRestoreMs : 0
  };
}

async function getSnapshotAvailability(env) {
  if (!env.DOCS_BUCKET) {
    return { available: false, missing: ['DOCS_BUCKET binding'] };
  }
  const [manifestObj, corpusObj, indexObj] = await Promise.all([
    env.DOCS_BUCKET.head(MANIFEST_KEY),
    env.DOCS_BUCKET.head(CORPUS_KEY),
    env.DOCS_BUCKET.head(INDEX_KEY)
  ]);
  const missing = [];
  if (!manifestObj) missing.push(MANIFEST_KEY);
  if (!corpusObj) missing.push(CORPUS_KEY);
  if (!indexObj) missing.push(INDEX_KEY);
  return { available: missing.length === 0, missing };
}

function buildResult(hit) {
  const chunk = corpusById.get(hit.id);
  return {
    id: hit.id,
    distance: hit.distance,
    title: chunk?.title || `Chunk ${hit.id}`,
    preview: chunk?.preview || '',
    source_path: chunk?.sourcePath || '',
    anchor: chunk?.anchor || ''
  };
}

async function handleSearch(request, env) {
  const url = new URL(request.url);
  let query = url.searchParams.get('q') || '';
  let k = parseInt(url.searchParams.get('k') || '5', 10);
  let ef = parseInt(url.searchParams.get('ef') || String(manifest?.efSearch || 120), 10);
  let source = url.searchParams.get('source') || '';

  if (request.method === 'POST') {
    const parsed = await parseJsonBody(request, env);
    if (parsed.error) {
      return jsonResponse({ error: parsed.error }, parsed.status);
    }
    const body = parsed.body;
    if (!body || typeof body.query !== 'string') {
      return jsonResponse({ error: 'POST /search requires JSON body { query: string, k?: number }' }, 400);
    }
    query = body.query;
    k = body.k ?? k;
    ef = body.ef ?? ef;
    source = body.source ?? source;
  }

  query = query.trim();
  if (!query) return jsonResponse({ error: 'query is required' }, 400);
  if (!Number.isInteger(k) || k < 1) k = 5;
  if (!Number.isInteger(ef) || ef < 10) ef = manifest?.efSearch || 120;
  k = Math.min(k, MAX_RESULTS);
  ef = Math.min(ef, 400);

  const loadInfo = await ensureLoaded(env);
  const t0 = performance.now();
  index.setEfSearch(ef);
  let hits;
  let filterLabel = null;
  if (source) {
    const allowedIds = sourceFilters.get(source);
    filterLabel = sourceLabelFromPath(source);
    hits = allowedIds ? index.searchFiltered(embedText(query), k, allowedIds) : [];
  } else {
    hits = index.search(embedText(query), k);
  }
  const searchMs = performance.now() - t0;

  return jsonResponse({
    query,
    result_count: hits.length,
    loaded_from_r2: loadInfo.loadedFromR2,
    cache_state: loadInfo.loadedFromR2 ? 'cold-restored' : 'warm-cache',
    restore_ms: formatMs(loadInfo.restoreMs),
    search_ms: formatMs(searchMs),
    corpus_chunks: manifest?.chunkCount || corpus.length,
    dim: manifest?.dim || DEMO_DIM,
    quantized: manifest?.quantized ?? true,
    ef_search: ef,
    restore_count: state.restoreCount,
    last_restore_ms: formatMs(state.lastRestoreMs || 0),
    filter_label: filterLabel,
    results: hits.map(buildResult)
  });
}

export default {
  async fetch(request, env) {
    globalThis.__PANCAKE_ALLOWED_ORIGIN__ = String(env.ALLOWED_ORIGIN || '').trim();
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      if (!getCorsOrigin()) {
        return new Response(null, { status: 403 });
      }
      return new Response(null, {
        headers: {
          ...corsHeaders(),
          'access-control-allow-methods': 'GET, POST, OPTIONS',
          'access-control-allow-headers': 'content-type'
        }
      });
    }

    if (url.pathname === '/') {
      return htmlResponse(renderPage());
    }

    if (ADMIN_ROUTES.has(url.pathname)) {
      const authError = requireAdminAuth(request, env);
      if (authError) return authError;
    }

    if (url.pathname === '/health') {
      return jsonResponse({
        loaded: !!index,
        manifest_loaded: !!manifest,
        corpus_chunks: corpus.length,
        dim: manifest?.dim || DEMO_DIM,
        quantized: manifest?.quantized ?? true,
        default_ef_search: manifest?.efSearch || 120,
        restore_count: state.restoreCount,
        restored_at: state.restoredAt,
        last_restore_ms: state.lastRestoreMs ? formatMs(state.lastRestoreMs) : null,
        read_only: isReadOnly(env),
        sources: Array.from(sourceFilters.keys()).map((source) => ({
          value: source,
          label: sourceLabelFromPath(source),
          count: sourceFilters.get(source)?.size || 0
        }))
      });
    }

    if (url.pathname === '/readiness') {
      const snapshot = await getSnapshotAvailability(env);
      return jsonResponse({
        ready: !!index || snapshot.available,
        loaded: !!index,
        snapshot_available: snapshot.available,
        missing_assets: snapshot.missing,
        restore_count: state.restoreCount,
        restored_at: state.restoredAt,
        last_restore_ms: state.lastRestoreMs ? formatMs(state.lastRestoreMs) : null,
        read_only: isReadOnly(env)
      });
    }

    if (url.pathname === '/reset_cache' && request.method === 'POST') {
      if (isReadOnly(env)) {
        return jsonResponse({ error: 'Worker is in read-only mode.' }, 403);
      }
      if (index) {
        index.dispose();
      }
      index = null;
      manifest = null;
      corpus = [];
      corpusById = new Map();
      sourceFilters = new Map();
      return jsonResponse({ cleared: true });
    }

    if (url.pathname === '/search' && (request.method === 'GET' || request.method === 'POST')) {
      try {
        return await handleSearch(request, env);
      } catch (err) {
        const message = err && typeof err.message === 'string' ? err.message : String(err);
        console.error(`semantic-search /search failed: ${message}`);
        return jsonResponse({ error: 'Internal server error' }, 500);
      }
    }

    return jsonResponse({
      name: 'Pancake Worker Semantic Search Demo',
      endpoints: {
        'GET /': 'Minimal docs-search UI',
        'GET /health': 'Restore state and cache status',
        'GET /readiness': 'Authenticated snapshot visibility and warm-load state',
        'GET /search?q=...': 'Search repo docs using the prebuilt snapshot',
        'POST /search': '{ query: string, k?: number }',
        'POST /reset_cache': 'Authenticated admin cache reset'
      }
    }, 404);
  }
};
