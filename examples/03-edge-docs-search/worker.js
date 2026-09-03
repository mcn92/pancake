import Pikelet from '../../pikelet.workerd.mjs';
import { embedTextWithStudent, loadStudentModel, scoreQuery } from './student-embedder.mjs';
import SNAPSHOT_ASSET from './assets/docs-index.bin';
import STUDENT_ASSET from './assets/docs-student.bin';
import ABSTENTION_ASSET from './assets/docs-abstention.json';
import CORPUS_ASSET from './assets/docs-corpus.json';
import MANIFEST_ASSET from './assets/docs-manifest.json';

const MAX_RESULTS = 8;
const DEFAULT_MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024;
const DEFAULT_MAX_STUDENT_BYTES = 4 * 1024 * 1024;
const DEFAULT_MAX_ABSTENTION_BYTES = 256 * 1024;
const DEFAULT_MAX_JSON_BYTES = 256 * 1024;
let index = null;
let manifest = null;
let corpus = [];
let corpusById = new Map();
let sourceFilters = new Map();
let studentModel = null;
let abstentionModel = null;
let loadPromise = null;
let state = {
  restoreCount: 0,
  restoredAt: null,
  lastRestoreMs: null,
  abstention: {
    present: false,
    error: null,
    bytes: null,
    sha256: null
  },
};
const ADMIN_ROUTES = new Set(['/readiness', '/reset_cache']);

function formatMs(value) {
  if (!Number.isFinite(value) || value <= 0) return 0.01;
  return Math.max(0.01, Math.round(value * 100) / 100);
}

function formatMicroseconds(valueMs) {
  if (!Number.isFinite(valueMs) || valueMs <= 0) return 1;
  return Math.max(1, Math.round(valueMs * 1000));
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

function getMaxStudentBytes(env) {
  const raw = env?.MAX_STUDENT_BYTES;
  if (raw === undefined || raw === null || raw === '') return DEFAULT_MAX_STUDENT_BYTES;
  const parsed = parseInt(raw, 10);
  if (!Number.isInteger(parsed) || parsed < 1) return DEFAULT_MAX_STUDENT_BYTES;
  return parsed;
}

function getMaxAbstentionBytes(env) {
  const raw = env?.MAX_ABSTENTION_BYTES;
  if (raw === undefined || raw === null || raw === '') return DEFAULT_MAX_ABSTENTION_BYTES;
  const parsed = parseInt(raw, 10);
  if (!Number.isInteger(parsed) || parsed < 1) return DEFAULT_MAX_ABSTENTION_BYTES;
  return parsed;
}

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

function validateAbstentionModel(candidate) {
  if (!candidate || typeof candidate !== 'object') {
    throw new Error('Abstention asset is not an object');
  }
  if (candidate.version !== 1) {
    throw new Error(`Unsupported abstention version: ${candidate.version}`);
  }
  if (!Array.isArray(candidate.features) || candidate.features.length < 5) {
    throw new Error('Abstention features must contain signal names');
  }
  if (!Array.isArray(candidate.weights) || candidate.weights.length !== candidate.features.length) {
    throw new Error('Abstention weights must match features');
  }
  for (const value of [...candidate.weights, candidate.bias]) {
    if (!Number.isFinite(value)) throw new Error('Abstention model contains non-finite weights');
  }
  const thresholds = candidate.thresholds || {};
  for (const key of ['weak', 'hard', 'preNormFloor']) {
    if (!Number.isFinite(thresholds[key])) throw new Error(`Abstention threshold ${key} is missing or non-finite`);
  }
  if (!Number.isInteger(thresholds.minFeatures) || thresholds.minFeatures < 0) {
    throw new Error('Abstention minFeatures must be a non-negative integer');
  }
  if (!Array.isArray(candidate.wordBuckets) || !Array.isArray(candidate.charBuckets)) {
    throw new Error('Abstention bucket tables are missing');
  }
  if (candidate.hiddenProbe) {
    if (!Array.isArray(candidate.hiddenProbe.weights) || !Number.isFinite(candidate.hiddenProbe.bias)) {
      throw new Error('Abstention hidden probe is malformed');
    }
  }
  return candidate;
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

function searchDisabled(env) {
  const value = String(env.DISABLE_SEARCH || '').trim().toLowerCase();
  return value === '1' || value === 'true' || value === 'yes' || value === 'on';
}

function debugTelemetryEnabled(env) {
  const value = String(env.DEBUG_TELEMETRY || '').trim().toLowerCase();
  return value === '1' || value === 'true' || value === 'yes' || value === 'on';
}

function privateSearchEnabled(env) {
  const expected = String(env.DEMO_SEARCH_KEY || '').trim();
  if (!expected) return false;
  const value = String(env.PRIVATE_SEARCH || '').trim().toLowerCase();
  return value !== '0' && value !== 'false' && value !== 'no' && value !== 'off';
}

function getSearchAccessKey(request, url) {
  const headerKey = request.headers.get('X-Pikelet-Demo-Key') || '';
  if (headerKey) return headerKey.trim();
  const authHeader = request.headers.get('Authorization') || '';
  if (authHeader.startsWith('Bearer ')) return authHeader.slice(7).trim();
  return '';
}

function requireSearchAccess(request, env, url) {
  if (!privateSearchEnabled(env)) return null;
  const expected = String(env.DEMO_SEARCH_KEY || '').trim();
  const actual = getSearchAccessKey(request, url);
  if (actual !== expected) {
    return jsonResponse({ error: 'Search requires a demo access key' }, 401);
  }
  return null;
}

function getHeader(request, name) {
  return request.headers.get(name) || null;
}

function getRequestLogContext(request, url) {
  return {
    request_id: getHeader(request, 'cf-ray'),
    method: request.method,
    pathname: url.pathname
  };
}

function logSearchEvent(event, request, url, extra = {}) {
  console.log(JSON.stringify({
    event,
    at: new Date().toISOString(),
    ...getRequestLogContext(request, url),
    ...extra
  }));
}

async function enforceRateLimit(request, env, url) {
  const checks = [
    {
      binding: env.GLOBAL_RATE_LIMITER,
      name: 'global',
      key: 'global'
    },
    {
      binding: env.RATE_LIMITER,
      name: 'ip',
      key: getHeader(request, 'cf-connecting-ip') || 'unknown'
    }
  ];

  for (const check of checks) {
    if (!check.binding || typeof check.binding.limit !== 'function') continue;
    const result = await check.binding.limit({ key: check.key });
    if (result.success) continue;

    console.log(JSON.stringify({
      event: 'semantic-search.request.rate_limited',
      at: new Date().toISOString(),
      ...getRequestLogContext(request, url),
      rate_limit_name: check.name,
      status: 429
    }));

    return new Response(JSON.stringify({ error: 'Too many requests' }, null, 2), {
      status: 429,
      headers: corsHeaders({
        'content-type': 'application/json; charset=utf-8',
        'retry-after': '60'
      })
    });
  }

  return null;
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
  if (sourcePath === 'examples/reference-worker/README.md') return 'Worker docs';
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
  <title>Pikelet Worker Semantic Search</title>
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
    .quality-banner {
      margin: 16px 0 0;
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.62);
      color: var(--muted);
      line-height: 1.45;
    }
    .quality-banner.weak {
      border-color: rgba(180, 87, 47, 0.28);
      background: rgba(180, 87, 47, 0.08);
      color: #6c4938;
    }
    .quality-banner.none {
      border-color: rgba(29, 36, 33, 0.18);
      background: rgba(29, 36, 33, 0.05);
      color: var(--ink);
    }
    .results-list.weak .card {
      opacity: 0.82;
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
    .card h2 a {
      color: inherit;
      text-decoration-color: rgba(15, 123, 99, 0.35);
      text-underline-offset: 0.18em;
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
          <p class="eyebrow">Pikelet + Cloudflare Workers</p>
          <div class="stats">
            <span class="pill">Zero runtime dependencies</span>
            <span class="pill">No model API</span>
            <span class="pill">Pikelet WASM in a Worker</span>
          </div>
          <h1>Distilled docs search with nothing to call.</h1>
          <p class="lede">
            A 1.08 MB domain-specific student turns text into 384D vectors locally. Pikelet
            restores its bundled quantized index and searches it in memory. The Worker makes
            no outbound requests at any point in the query path.
          </p>
        </div>
        <aside class="hero-side">
          <p class="side-kicker">What to look for</p>
          <p class="side-value">Teacher quality offline. Tiny student and vector index at runtime.</p>
          <p class="side-note">
            The teacher never ships. The response separates student embedding time, Pikelet
            search time, and any cold snapshot restore so the boundary stays visible.
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
                  <option value="examples/reference-worker/README.md">Worker docs</option>
                </select>
              </div>
            </div>
          </div>
          <aside class="stats-panel">
            <p class="side-kicker">Live stats</p>
            <div class="stats-grid" id="stats-grid">
              <div class="stat"><span class="stat-k">Cache</span><span class="stat-v">Not loaded</span></div>
              <div class="stat"><span class="stat-k">Chunks</span><span class="stat-v">-</span></div>
              <div class="stat"><span class="stat-k">Dim</span><span class="stat-v">384</span></div>
              <div class="stat"><span class="stat-k">Quantized</span><span class="stat-v">Yes</span></div>
              <div class="stat"><span class="stat-k">Student</span><span class="stat-v">1.08 MB int8</span></div>
              <div class="stat"><span class="stat-k">Outbound</span><span class="stat-v">None</span></div>
              <div class="stat"><span class="stat-k">Restores</span><span class="stat-v">0</span></div>
              <div class="stat"><span class="stat-k">Last restore</span><span class="stat-v">-</span></div>
            </div>
          </aside>
        </div>
        <div class="samples">
          <button class="sample" type="button" data-query="How do Cloudflare Workers restore snapshots from R2?">Cold restore</button>
          <button class="sample" type="button" data-query="Why do I need compact before export after deletes?">Ghost cleanup</button>
          <button class="sample" type="button" data-query="How does filtered search work in Pikelet?">Filtered search</button>
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
        ['Dim', payload.dim ?? '384'],
        ['Quantized', payload.quantized ? 'Yes' : 'No'],
        ['Student', payload.encoder?.modelBytes ? (payload.encoder.modelBytes / 1048576).toFixed(2) + ' MB int8' : '1.08 MB int8'],
        ['Outbound', 'None'],
        ['Restores', payload.restore_count ?? 0],
        ['Last restore', payload.last_restore_ms ? payload.last_restore_ms.toFixed(2) + 'ms' : '-'],
      ];
      statsGrid.innerHTML = rows.map(([k, v]) =>
        '<div class="stat"><span class="stat-k">' + escapeHtml(k) + '</span><span class="stat-v">' + escapeHtml(v) + '</span></div>'
      ).join('');
    }

    async function refreshStats() {
      const res = await fetch('/health');
      const payload = await res.json();
      if (res.ok) renderStats(payload);
    }

    function renderResults(payload) {
      const qualityText = payload.match_quality && payload.match_quality !== 'unscored'
        ? ' • ' + payload.match_quality + ' match' + (payload.confidence !== undefined ? ' ' + payload.confidence.toFixed(3) : '')
        : '';
      meta.textContent =
        'Embed ' + payload.embedding_ms.toFixed(2) + 'ms' +
        ' • Pikelet ' + payload.search_ms.toFixed(2) + 'ms' +
        ' • restore ' + payload.restore_ms.toFixed(2) + 'ms' +
        ' • ' + payload.result_count + ' results' +
        (payload.filter_label ? ' • filter ' + payload.filter_label : '') +
        ' • ef ' + payload.ef_search +
        qualityText;

      resultsList.classList.toggle('weak', payload.match_quality === 'weak');
      let banner = '';
      if (payload.match_quality === 'weak') {
        banner = '<div class="quality-banner weak">No strong match — showing the closest sections.</div>';
      } else if (payload.match_quality === 'none') {
        banner = '<div class="quality-banner none">No reliable match. Try a query about Pikelet APIs, Workers, snapshots, filtering, or compaction.</div>';
      }
      if (!payload.results.length) {
        resultsList.innerHTML = banner || '<div class="empty">No results.</div>';
        return;
      }

      resultsList.innerHTML = banner + payload.results.map((item, i) => {
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
            '<h2><a href="' + escapeHtml(item.source_url) + '" target="_blank" rel="noreferrer">' + escapeHtml(item.title) + '</a></h2>' +
            '<p>' + escapeHtml(item.preview) + '</p>' +
          '</div>' +
          '</article>';
      }).join('');
    }

    async function runSearch(query) {
      meta.textContent = 'Searching...';
      const res = await fetch('/search', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          query,
          k: kInput.value || '5',
          ef: efInput.value || '120',
          source: sourceFilter.value || undefined
        })
      });
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
        encoder: payload.encoder,
        restore_count: payload.restore_count ?? '-',
        last_restore_ms: payload.last_restore_ms ?? payload.restore_ms
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

    refreshStats();
  </script>
</body>
</html>`;
}

function assetBytes(asset, label) {
  if (asset instanceof ArrayBuffer) return new Uint8Array(asset);
  if (ArrayBuffer.isView(asset)) {
    return new Uint8Array(asset.buffer, asset.byteOffset, asset.byteLength);
  }
  throw new Error(`${label} was not bundled as binary data`);
}

async function restoreBundledAssets(env) {
  const t0 = performance.now();
  const loadedManifest = MANIFEST_ASSET;
  const loadedCorpus = CORPUS_ASSET;
  const snapshotBytes = assetBytes(SNAPSHOT_ASSET, 'Snapshot');
  const studentBytes = assetBytes(STUDENT_ASSET, 'Student model');
  const maxSnapshotBytes = getMaxSnapshotBytes(env);
  const maxStudentBytes = getMaxStudentBytes(env);
  if (snapshotBytes.byteLength > maxSnapshotBytes) {
    throw new Error(`Index snapshot exceeds MAX_SNAPSHOT_BYTES (${snapshotBytes.byteLength} > ${maxSnapshotBytes})`);
  }
  if (studentBytes.byteLength > maxStudentBytes) {
    throw new Error(`Student model exceeds MAX_STUDENT_BYTES (${studentBytes.byteLength} > ${maxStudentBytes})`);
  }

  const loadedStudent = loadStudentModel(studentBytes);
  if (loadedStudent.outputDim !== loadedManifest.dim) {
    throw new Error(`Student dim ${loadedStudent.outputDim} does not match index dim ${loadedManifest.dim}`);
  }
  if (loadedManifest.encoder?.modelBytes !== undefined
      && loadedManifest.encoder.modelBytes !== studentBytes.byteLength) {
    throw new Error('Student model byte length does not match the manifest');
  }
  if (loadedManifest.encoder?.modelSha256) {
    const actualHash = await sha256Hex(studentBytes);
    if (actualHash !== loadedManifest.encoder.modelSha256) {
      throw new Error('Student model SHA-256 does not match the manifest');
    }
  }

  abstentionModel = null;
  state.abstention = {
    present: false,
    error: null,
    bytes: null,
    sha256: null
  };
  const abstentionManifest = loadedManifest.abstention || loadedManifest.encoder?.abstention || null;
  const abstentionAsset = ABSTENTION_ASSET || null;
  if (abstentionManifest && abstentionAsset) {
    try {
      const serialized = JSON.stringify(abstentionAsset);
      const abstentionBytes = new TextEncoder().encode(serialized);
      const maxAbstentionBytes = getMaxAbstentionBytes(env);
      if (abstentionBytes.byteLength > maxAbstentionBytes) {
        throw new Error(`Abstention asset exceeds MAX_ABSTENTION_BYTES (${abstentionBytes.byteLength} > ${maxAbstentionBytes})`);
      }
      if (abstentionManifest?.bytes !== undefined && abstentionManifest.bytes !== abstentionBytes.byteLength) {
        throw new Error('Abstention asset byte length does not match the manifest');
      }
      const actualHash = await sha256Hex(abstentionBytes);
      if (abstentionManifest?.sha256 && actualHash !== abstentionManifest.sha256) {
        throw new Error('Abstention asset SHA-256 does not match the manifest');
      }
      abstentionModel = validateAbstentionModel(abstentionAsset);
      if (loadedManifest.generatedAt && abstentionModel.calibratedAt) {
        const generatedAt = Date.parse(loadedManifest.generatedAt);
        const calibratedAt = Date.parse(abstentionModel.calibratedAt);
        if (Number.isFinite(generatedAt) && Number.isFinite(calibratedAt) && calibratedAt < generatedAt) {
          throw new Error('Abstention calibration predates the corpus manifest');
        }
      }
      state.abstention = {
        present: true,
        error: null,
        bytes: abstentionBytes.byteLength,
        sha256: actualHash
      };
    } catch (error) {
      state.abstention = {
        present: false,
        error: error && error.message ? error.message : String(error),
        bytes: null,
        sha256: null
      };
      console.warn(`Abstention disabled: ${state.abstention.error}`);
    }
  } else if (abstentionManifest) {
    state.abstention.error = `Manifest declares ${abstentionManifest.assetKey || 'docs-abstention.json'} but the asset was not bundled`;
    console.warn(`Abstention disabled: ${state.abstention.error}`);
  }

  const restored = await Pikelet.restore(snapshotBytes, {
    maxElements: loadedManifest.maxElements,
    efSearch: loadedManifest.efSearch,
  });

  corpus = loadedCorpus;
  corpusById = new Map(corpus.map((entry) => [entry.id, entry]));
  rebuildSourceFilters();
  manifest = loadedManifest;
  studentModel = loadedStudent;
  index = restored;
  state.restoreCount += 1;
  state.lastRestoreMs = performance.now() - t0;
  state.restoredAt = new Date().toISOString();
}

async function ensureLoaded(env) {
  if (index && studentModel) {
    return { loadedFromBundle: false, restoreMs: 0 };
  }

  const callerTriggeredRestore = !loadPromise;
  if (!loadPromise) {
    loadPromise = restoreBundledAssets(env).finally(() => {
      loadPromise = null;
    });
  }

  await loadPromise;
  return {
    loadedFromBundle: callerTriggeredRestore,
    restoreMs: callerTriggeredRestore ? state.lastRestoreMs : 0
  };
}

function getBundledAssetAvailability() {
  return {
    available: true,
    missing: [],
    snapshotBytes: assetBytes(SNAPSHOT_ASSET, 'Snapshot').byteLength,
    studentBytes: assetBytes(STUDENT_ASSET, 'Student model').byteLength,
    corpusChunks: CORPUS_ASSET.length,
  };
}

function buildResult(hit) {
  const chunk = corpusById.get(hit.id);
  const sourcePath = chunk?.sourcePath || '';
  const anchor = chunk?.anchor || '';
  return {
    id: hit.id,
    distance: hit.distance,
    title: chunk?.title || `Chunk ${hit.id}`,
    preview: chunk?.preview || '',
    source_path: sourcePath,
    anchor,
    source_url: `https://github.com/mcn92/pancake/blob/main/${sourcePath}${anchor ? `#${anchor}` : ''}`
  };
}

function buildKnownBucketTables(model) {
  const word = new Map();
  let maxIdf = 0;
  for (const row of model?.wordBuckets || []) {
    if (!Array.isArray(row) || row.length < 2) continue;
    const bucket = Number(row[0]);
    const idf = Number(row[1]);
    if (Number.isInteger(bucket) && bucket >= 0 && Number.isFinite(idf) && idf >= 0) {
      word.set(bucket, idf);
      maxIdf = Math.max(maxIdf, idf);
    }
  }
  const char = new Set();
  for (const value of model?.charBuckets || []) {
    const bucket = Number(value);
    if (Number.isInteger(bucket) && bucket >= 0) char.add(bucket);
  }
  return { word, char, maxIdf: maxIdf || 255 };
}

function computeKnownFractions(features, model) {
  if (!model) return { known_word: 0, known_char: 0, n_feats: features.length };
  const tables = model._knownBucketTables || (model._knownBucketTables = buildKnownBucketTables(model));
  let wordKnown = 0;
  let wordTotal = 0;
  let charKnown = 0;
  let charTotal = 0;

  for (const feature of features) {
    if (feature.family === 'word') {
      wordTotal += 1;
      if (tables.word.has(feature.bucket)) wordKnown += 1;
    } else if (feature.family === 'char') {
      charTotal += 1;
      if (tables.char.has(feature.bucket)) charKnown += 1;
    }
  }

  return {
    known_word: wordTotal > 0 ? wordKnown / wordTotal : 0,
    known_char: charTotal > 0 ? charKnown / charTotal : 0,
    n_feats: features.length
  };
}

function computeHiddenProbe(embedded, model) {
  const probe = model?.hiddenProbe;
  if (!probe || !Array.isArray(probe.weights) || !embedded.hidden) return 0;
  let logit = Number(probe.bias) || 0;
  const limit = Math.min(probe.weights.length, embedded.hidden.length);
  for (let index = 0; index < limit; index++) {
    logit += (Number(probe.weights[index]) || 0) * embedded.hidden[index];
  }
  if (logit >= 0) {
    const z = Math.exp(-logit);
    return 1 / (1 + z);
  }
  const z = Math.exp(logit);
  return z / (1 + z);
}

function computeMatchQuality(hits, embedded) {
  if (!abstentionModel) return { match_quality: 'unscored' };
  const d0 = hits.length > 0 ? hits[0].distance : 1;
  const marginIndex = Math.min(4, hits.length - 1);
  const margin = marginIndex > 0 ? hits[marginIndex].distance - d0 : 0;
  const known = computeKnownFractions(embedded.features, abstentionModel);
  const signals = {
    d0,
    margin,
    pre_norm: embedded.preNorm,
    known_word: known.known_word,
    known_char: known.known_char,
    hidden_probe: computeHiddenProbe(embedded, abstentionModel),
    n_feats: known.n_feats
  };
  return scoreQuery(signals, abstentionModel);
}

function computePreSearchAbstention(embedded) {
  if (!abstentionModel) return null;
  const thresholds = abstentionModel.thresholds || {};
  const minFeatures = Number.isInteger(thresholds.minFeatures) ? thresholds.minFeatures : 3;
  const preNormFloor = Number.isFinite(thresholds.preNormFloor) ? thresholds.preNormFloor : 0.4;
  if (embedded.features.length >= minFeatures && embedded.preNorm >= preNormFloor) return null;

  const known = computeKnownFractions(embedded.features, abstentionModel);
  return scoreQuery({
    d0: 1,
    margin: 0,
    pre_norm: embedded.preNorm,
    known_word: known.known_word,
    known_char: known.known_char,
    hidden_probe: computeHiddenProbe(embedded, abstentionModel),
    n_feats: known.n_feats
  }, abstentionModel);
}

async function handleSearch(request, env) {
  const totalStart = performance.now();
  const url = new URL(request.url);
  let query = url.searchParams.get('q') || '';
  let k = parseInt(url.searchParams.get('k') || '5', 10);
  let ef = parseInt(url.searchParams.get('ef') || String(manifest?.efSearch || 120), 10);
  let source = url.searchParams.get('source') || '';

  if (request.method === 'POST') {
    const parsed = await parseJsonBody(request, env);
    if (parsed.error) {
      logSearchEvent('semantic-search.search.invalid_body', request, url, {
        error: parsed.error,
        status: parsed.status
      });
      return jsonResponse({ error: parsed.error }, parsed.status);
    }
    const body = parsed.body;
    if (!body || typeof body.query !== 'string') {
      logSearchEvent('semantic-search.search.invalid_body', request, url, {
        error: 'POST /search requires JSON body { query: string, k?: number }',
        status: 400
      });
      return jsonResponse({ error: 'POST /search requires JSON body { query: string, k?: number }' }, 400);
    }
    query = body.query;
    k = body.k ?? k;
    ef = body.ef ?? ef;
    source = body.source ?? source;
  }

  query = query.trim();
  if (!query) {
    logSearchEvent('semantic-search.search.invalid_query', request, url, {
      error: 'query is required',
      status: 400
    });
    return jsonResponse({ error: 'query is required' }, 400);
  }
  if (!Number.isInteger(k) || k < 1) k = 5;
  if (!Number.isInteger(ef) || ef < 10) ef = manifest?.efSearch || 120;
  k = Math.min(k, MAX_RESULTS);
  ef = Math.min(ef, 400);

  const loadInfo = await ensureLoaded(env);
  const embeddingStart = performance.now();
  const embedded = embedTextWithStudent(query, studentModel);
  const queryVector = embedded.vector;
  const embeddingMs = performance.now() - embeddingStart;
  const preSearchMatchQuality = computePreSearchAbstention(embedded);
  let searchMs = 0;
  let hits = [];
  let filterLabel = null;
  if (source) {
    filterLabel = sourceLabelFromPath(source);
  }
  if (!preSearchMatchQuality) {
    const searchStart = performance.now();
    if (source) {
      const allowedIds = sourceFilters.get(source);
      hits = allowedIds
        ? index.searchFiltered(queryVector, k, allowedIds, { efSearch: ef })
        : [];
    } else {
      hits = index.search(queryVector, k, { efSearch: ef });
    }
    searchMs = performance.now() - searchStart;
  }
  const matchQuality = preSearchMatchQuality || computeMatchQuality(hits, embedded);
  const returnedHits = matchQuality.match_quality === 'none' ? [] : hits;

  const responseBody = {
    match_quality: matchQuality.match_quality,
    result_count: returnedHits.length,
    restore_ms: formatMs(loadInfo.restoreMs),
    embedding_ms: formatMs(embeddingMs),
    search_ms: formatMs(searchMs),
    corpus_chunks: manifest?.chunkCount || corpus.length,
    dim: manifest?.dim || null,
    quantized: manifest?.quantized ?? true,
    ef_search: ef,
    filter_label: filterLabel,
    results: returnedHits.map(buildResult)
  };
  if (matchQuality.confidence !== undefined) responseBody.confidence = matchQuality.confidence;
  if (debugTelemetryEnabled(env)) {
    responseBody.loaded_from_bundle = loadInfo.loadedFromBundle;
    responseBody.cache_state = loadInfo.loadedFromBundle ? 'cold-restored' : 'warm-cache';
    responseBody.restore_count = state.restoreCount;
    responseBody.last_restore_ms = formatMs(state.lastRestoreMs || 0);
    responseBody.encoder = manifest?.encoder || null;
    responseBody.timings_us = {
      total: formatMicroseconds(performance.now() - totalStart),
      restore: formatMicroseconds(loadInfo.restoreMs),
      embedding: formatMicroseconds(embeddingMs),
      search: formatMicroseconds(searchMs)
    };
  }
  logSearchEvent('semantic-search.search.completed', request, url, {
    match_quality: responseBody.match_quality,
    result_count: responseBody.result_count,
    status: 200
  });
  return jsonResponse(responseBody);
}

export default {
  async fetch(request, env) {
    globalThis.__PANCAKE_ALLOWED_ORIGIN__ = String(env.ALLOWED_ORIGIN || '').trim();
    const url = new URL(request.url);
    const rateLimitResponse = await enforceRateLimit(request, env, url);
    if (rateLimitResponse) return rateLimitResponse;

    if (request.method === 'OPTIONS') {
      if (!getCorsOrigin()) {
        return new Response(null, { status: 403 });
      }
      return new Response(null, {
        headers: {
          ...corsHeaders(),
          'access-control-allow-methods': 'GET, POST, OPTIONS',
          'access-control-allow-headers': 'authorization, content-type, x-pancake-demo-key'
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
      const body = {
        ok: true,
        loaded: !!index,
        read_only: isReadOnly(env),
        search_disabled: searchDisabled(env),
        private_search: privateSearchEnabled(env)
      };
      if (debugTelemetryEnabled(env)) {
        body.manifest_loaded = !!manifest;
        body.corpus_chunks = corpus.length;
        body.dim = manifest?.dim || null;
        body.quantized = manifest?.quantized ?? true;
        body.default_ef_search = manifest?.efSearch || 120;
        body.restore_count = state.restoreCount;
        body.restored_at = state.restoredAt;
        body.last_restore_ms = state.lastRestoreMs ? formatMs(state.lastRestoreMs) : null;
        body.encoder = manifest?.encoder || null;
        body.abstention = {
          present: !!abstentionModel,
          error: state.abstention.error
        };
        body.sources = Array.from(sourceFilters.keys()).map((source) => ({
          value: source,
          label: sourceLabelFromPath(source),
          count: sourceFilters.get(source)?.size || 0
        }));
      }
      return jsonResponse(body);
    }

    if (url.pathname === '/readiness') {
      const snapshot = getBundledAssetAvailability();
      return jsonResponse({
        ready: !!index || snapshot.available,
        loaded: !!index,
        snapshot_available: snapshot.available,
        missing_assets: snapshot.missing,
        snapshot_bytes: snapshot.snapshotBytes,
        student_bytes: snapshot.studentBytes,
        abstention_bytes: state.abstention.bytes,
        abstention_sha256: state.abstention.sha256,
        abstention_error: state.abstention.error,
        bundled_corpus_chunks: snapshot.corpusChunks,
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
      studentModel = null;
      abstentionModel = null;
      manifest = null;
      corpus = [];
      corpusById = new Map();
      sourceFilters = new Map();
      state.abstention = {
        present: false,
        error: null,
        bytes: null,
        sha256: null
      };
      return jsonResponse({ cleared: true });
    }

    if (url.pathname === '/search' && request.method !== 'POST') {
      return jsonResponse({ error: 'Use POST /search with a JSON body' }, 405);
    }

    if (url.pathname === '/search' && request.method === 'POST') {
      if (searchDisabled(env)) {
        logSearchEvent('semantic-search.search.disabled', request, url, {
          status: 503
        });
        return jsonResponse({ error: 'Search is temporarily disabled' }, 503);
      }
      const searchAccessError = requireSearchAccess(request, env, url);
      if (searchAccessError) {
        logSearchEvent('semantic-search.search.unauthorized', request, url, {
          status: 401
        });
        return searchAccessError;
      }
      try {
        return await handleSearch(request, env);
      } catch (err) {
        const message = err && typeof err.message === 'string' ? err.message : String(err);
        logSearchEvent('semantic-search.search.error', request, url, {
          error: message,
          status: 500
        });
        return jsonResponse({ error: 'Internal server error' }, 500);
      }
    }

    return jsonResponse({
      name: 'Pikelet Worker Semantic Search Demo',
      endpoints: {
        'GET /': 'Minimal docs-search UI',
        'GET /health': 'Liveness and coarse mode status',
        'GET /readiness': 'Authenticated snapshot visibility and warm-load state',
        'POST /search': '{ query: string, k?: number, ef?: number, source?: string }',
        'POST /reset_cache': 'Authenticated admin cache reset'
      }
    }, 404);
  }
};
