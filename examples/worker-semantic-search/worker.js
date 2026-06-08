import loadEngine from '../../dist/engine.js';
import wasmModule from '../../dist/engine.wasm';
import createPancakeApi from '../../pancake-core.js';
import { DEMO_DIM, embedText } from './embedder.mjs';

const INDEX_KEY = 'docs-index.bin';
const CORPUS_KEY = 'docs-corpus.json';
const MANIFEST_KEY = 'docs-manifest.json';
const MAX_RESULTS = 8;
const Pancake = createPancakeApi(() =>
  loadEngine({
    instantiateWasm(imports, successCallback) {
      WebAssembly.instantiate(wasmModule, imports)
        .then((instance) => successCallback(instance))
        .catch((err) => {
          throw err;
        });
      return {};
    }
  })
);

let index = null;
let manifest = null;
let corpus = [];
let corpusById = new Map();
let loadPromise = null;
let state = {
  restoreCount: 0,
  restoredAt: null,
  lastRestoreMs: null,
};

function formatMs(value) {
  if (!Number.isFinite(value) || value <= 0) return 0.01;
  return Math.max(0.01, Math.round(value * 100) / 100);
}

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      'access-control-allow-origin': '*'
    }
  });
}

function htmlResponse(body) {
  return new Response(body, {
    headers: {
      'content-type': 'text/html; charset=utf-8'
    }
  });
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
      --bg: #f2ede2;
      --ink: #1d2b2a;
      --muted: #5d6b67;
      --line: rgba(29, 43, 42, 0.15);
      --panel: rgba(255, 251, 245, 0.84);
      --accent: #0d7c66;
      --accent-2: #cc6d3d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(13, 124, 102, 0.16), transparent 34%),
        radial-gradient(circle at top right, rgba(204, 109, 61, 0.16), transparent 28%),
        linear-gradient(180deg, #f9f5ec 0%, var(--bg) 100%);
    }
    main {
      max-width: 980px;
      margin: 0 auto;
      padding: 48px 20px 72px;
    }
    .hero {
      display: grid;
      gap: 18px;
      padding: 28px;
      border: 1px solid var(--line);
      border-radius: 28px;
      background: var(--panel);
      box-shadow: 0 18px 60px rgba(29, 43, 42, 0.07);
    }
    h1 {
      margin: 0;
      font-size: clamp(2.2rem, 6vw, 4.8rem);
      line-height: 0.92;
      letter-spacing: -0.05em;
      font-family: Georgia, "Iowan Old Style", serif;
      max-width: 10ch;
    }
    .lede {
      max-width: 60ch;
      color: var(--muted);
      font-size: 1.05rem;
      line-height: 1.6;
    }
    .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .pill, button {
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.7);
      color: var(--ink);
      padding: 10px 14px;
      font: inherit;
    }
    .search {
      margin-top: 24px;
      display: grid;
      gap: 14px;
    }
    .search-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
    }
    input {
      width: 100%;
      border-radius: 18px;
      border: 1px solid var(--line);
      padding: 16px 18px;
      font: inherit;
      background: rgba(255, 255, 255, 0.85);
    }
    .submit {
      background: linear-gradient(135deg, var(--accent), #095848);
      color: white;
      border: none;
      padding-inline: 20px;
    }
    .samples {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .sample {
      cursor: pointer;
    }
    .meta, .empty {
      color: var(--muted);
      font-size: 0.95rem;
    }
    .results {
      margin-top: 28px;
      display: grid;
      gap: 14px;
    }
    .card {
      padding: 18px 18px 16px;
      border-radius: 22px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.72);
      display: grid;
      gap: 8px;
      transform: translateY(10px);
      opacity: 0;
      animation: rise 260ms ease forwards;
    }
    .card h2 {
      margin: 0;
      font-size: 1.1rem;
    }
    .card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }
    .card small {
      color: var(--muted);
    }
    @keyframes rise {
      to { transform: translateY(0); opacity: 1; }
    }
    @media (max-width: 720px) {
      .search-row {
        grid-template-columns: 1fr;
      }
      .submit {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="stats">
        <span class="pill">Cloudflare Worker</span>
        <span class="pill">R2 snapshot restore</span>
        <span class="pill">Pancake ANN index</span>
      </div>
      <h1>Semantic Docs Search at the Edge</h1>
      <p class="lede">
        This demo prebuilds a Pancake index from the repo docs, stores the snapshot in R2,
        restores it on cold start, and then serves text queries from hot Worker memory.
      </p>
      <form class="search" id="search-form">
        <div class="search-row">
          <input id="query" name="query" autocomplete="off" value="How do Cloudflare Workers restore snapshots from R2?">
          <button class="submit" type="submit">Search</button>
        </div>
        <div class="samples">
          <button class="sample" type="button" data-query="How do Cloudflare Workers restore snapshots from R2?">Cold restore</button>
          <button class="sample" type="button" data-query="Why do I need compact before export after deletes?">Ghost cleanup</button>
          <button class="sample" type="button" data-query="How does filtered search work in Pancake?">Filtered search</button>
          <button class="sample" type="button" data-query="What are the memory tradeoffs for quantized indexes?">Memory tradeoffs</button>
        </div>
      </form>
      <div class="meta" id="meta">Ready.</div>
    </section>
    <section class="results" id="results">
      <div class="empty">Run a query to load the index and inspect the top matching doc chunks.</div>
    </section>
  </main>
  <script>
    const form = document.getElementById('search-form');
    const input = document.getElementById('query');
    const meta = document.getElementById('meta');
    const results = document.getElementById('results');

    function renderResults(payload) {
      meta.textContent =
        'Search ' + payload.search_ms.toFixed(2) + 'ms' +
        (payload.cache_state === 'cold-restored'
          ? ' • cold restore ' + payload.restore_ms.toFixed(2) + 'ms'
          : ' • warm cache') +
        ' • ' + payload.result_count + ' results';

      if (!payload.results.length) {
        results.innerHTML = '<div class="empty">No results.</div>';
        return;
      }

      results.innerHTML = payload.results.map((item, i) => {
        const score = Math.max(0, 1 - item.distance).toFixed(3);
        return '<article class="card" style="animation-delay:' + (i * 45) + 'ms">' +
          '<small>' + item.source_path + ' • ' + score + '</small>' +
          '<h2>' + item.title + '</h2>' +
          '<p>' + item.preview + '</p>' +
          '</article>';
      }).join('');
    }

    async function runSearch(query) {
      meta.textContent = 'Searching...';
      const res = await fetch('/search?q=' + encodeURIComponent(query));
      const payload = await res.json();
      if (!res.ok) {
        meta.textContent = payload.error || 'Search failed';
        return;
      }
      renderResults(payload);
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

  const loadedManifest = await loadJson(manifestObj);
  if (loadedManifest.dim !== DEMO_DIM) {
    throw new Error(`Manifest dim ${loadedManifest.dim} does not match demo dim ${DEMO_DIM}`);
  }

  const [loadedCorpus, snapshotBuffer] = await Promise.all([
    loadJson(corpusObj),
    indexObj.arrayBuffer()
  ]);

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

  if (request.method === 'POST') {
    const body = await request.json().catch(() => null);
    if (!body || typeof body.query !== 'string') {
      return jsonResponse({ error: 'POST /search requires JSON body { query: string, k?: number }' }, 400);
    }
    query = body.query;
    k = body.k ?? k;
  }

  query = query.trim();
  if (!query) return jsonResponse({ error: 'query is required' }, 400);
  if (!Number.isInteger(k) || k < 1) k = 5;
  k = Math.min(k, MAX_RESULTS);

  const loadInfo = await ensureLoaded(env);
  const t0 = performance.now();
  const hits = index.search(embedText(query), k);
  const searchMs = performance.now() - t0;

  return jsonResponse({
    query,
    result_count: hits.length,
    loaded_from_r2: loadInfo.loadedFromR2,
    cache_state: loadInfo.loadedFromR2 ? 'cold-restored' : 'warm-cache',
    restore_ms: formatMs(loadInfo.restoreMs),
    search_ms: formatMs(searchMs),
    corpus_chunks: manifest?.chunkCount || corpus.length,
    results: hits.map(buildResult)
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'access-control-allow-origin': '*',
          'access-control-allow-methods': 'GET, POST, OPTIONS',
          'access-control-allow-headers': 'content-type'
        }
      });
    }

    if (url.pathname === '/') {
      return htmlResponse(renderPage());
    }

    if (url.pathname === '/health') {
      return jsonResponse({
        loaded: !!index,
        manifest_loaded: !!manifest,
        corpus_chunks: corpus.length,
        restore_count: state.restoreCount,
        restored_at: state.restoredAt,
        last_restore_ms: state.lastRestoreMs
      });
    }

    if (url.pathname === '/search' && (request.method === 'GET' || request.method === 'POST')) {
      try {
        return await handleSearch(request, env);
      } catch (err) {
        return jsonResponse({ error: err.message || String(err) }, 500);
      }
    }

    return jsonResponse({
      name: 'Pancake Worker Semantic Search Demo',
      endpoints: {
        'GET /': 'Minimal docs-search UI',
        'GET /health': 'Restore state and cache status',
        'GET /search?q=...': 'Search repo docs using the prebuilt snapshot',
        'POST /search': '{ query: string, k?: number }'
      }
    }, 404);
  }
};
