import './search.css';
import { computeMatchQuality, embedTextWithStudent, loadStudentModel } from '../../src/student-embedder.mjs';

const DEFAULT_K = 8;
const DEFAULT_EF_SEARCH = 120;
const DEFAULT_ASSET_BASE = '/pancake-search';

function normalizeBase(value) {
  return String(value || DEFAULT_ASSET_BASE).replace(/\/+$/g, '');
}

function createRangeSource(url) {
  let fullBuffer = null;
  return {
    async read(offset, length) {
      if (fullBuffer) return fullBuffer.subarray(offset, offset + length);
      const response = await fetch(`${url}?r=${offset}-${offset + length - 1}`, {
        headers: { Range: `bytes=${offset}-${offset + length - 1}` },
      });
      if (response.status !== 206 && response.status !== 200) {
        throw new Error(`Range read failed: ${response.status} ${response.statusText}`.trim());
      }
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (response.status === 200 && bytes.byteLength > length) {
        fullBuffer = bytes;
        return fullBuffer.subarray(offset, offset + length);
      }
      if (bytes.byteLength !== length) {
        throw new Error(`Range read returned ${bytes.byteLength} bytes; expected ${length}`);
      }
      return bytes;
    },
  };
}

function resultUrl(chunk) {
  if (!chunk) return '#';
  const url = chunk.url || chunk.sourcePath || '#';
  if (!chunk.anchor) return url;
  return `${url.replace(/#.*$/, '')}#${chunk.anchor}`;
}

function htmlEscape(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderRows(results, corpusById) {
  if (!results.length) return '<p class="pancake-search-empty">No results</p>';
  return `<ol class="pancake-search-results">${results.map((hit) => {
    const chunk = corpusById.get(hit.id);
    return `<li>
      <a href="${htmlEscape(resultUrl(chunk))}">${htmlEscape(chunk?.title || `Chunk ${hit.id}`)}</a>
      <p>${htmlEscape(chunk?.preview || chunk?.text || '')}</p>
      <span>${hit.distance.toFixed(4)}</span>
    </li>`;
  }).join('')}</ol>`;
}

async function loadBrowserRuntime() {
  if (typeof window === 'undefined') throw new Error('Pancake search can only load in a browser');
  const pancakeModule = await Promise.resolve().then(() => require('pancake-wasm/web'));
  return {
    Pancake: pancakeModule.default || pancakeModule,
  };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function clampPanel(root) {
  const rect = root.getBoundingClientRect();
  const margin = 12;
  const left = clamp(rect.left, margin, Math.max(margin, window.innerWidth - rect.width - margin));
  const top = clamp(rect.top, margin, Math.max(margin, window.innerHeight - rect.height - margin));
  root.style.left = `${left}px`;
  root.style.top = `${top}px`;
  root.style.right = 'auto';
  root.style.bottom = 'auto';
}

function enableDrag(root, handle) {
  let drag = null;
  handle.addEventListener('pointerdown', (event) => {
    if (event.button !== undefined && event.button !== 0) return;
    const rect = root.getBoundingClientRect();
    drag = {
      pointerId: event.pointerId,
      dx: event.clientX - rect.left,
      dy: event.clientY - rect.top,
    };
    root.classList.add('pancake-search-dragging');
    handle.setPointerCapture(event.pointerId);
  });
  handle.addEventListener('pointermove', (event) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    const rect = root.getBoundingClientRect();
    const margin = 12;
    const left = clamp(event.clientX - drag.dx, margin, Math.max(margin, window.innerWidth - rect.width - margin));
    const top = clamp(event.clientY - drag.dy, margin, Math.max(margin, window.innerHeight - rect.height - margin));
    root.style.left = `${left}px`;
    root.style.top = `${top}px`;
    root.style.right = 'auto';
    root.style.bottom = 'auto';
  });
  function endDrag(event) {
    if (!drag || event.pointerId !== drag.pointerId) return;
    root.classList.remove('pancake-search-dragging');
    drag = null;
  }
  handle.addEventListener('pointerup', endDrag);
  handle.addEventListener('pointercancel', endDrag);
  window.addEventListener('resize', () => clampPanel(root));
}

class PancakeDocusaurusSearch {
  constructor(options = {}) {
    this.assetBase = normalizeBase(options.assetBase);
    this.k = Number(options.k || DEFAULT_K);
    this.efSearch = Number(options.efSearch || DEFAULT_EF_SEARCH);
    this.state = null;
    this.ready = null;
  }

  async load() {
    if (this.ready) return this.ready;
    this.ready = this._load();
    return this.ready;
  }

  async _load() {
    const manifestUrl = `${this.assetBase}/manifest.json`;
    const manifest = await fetch(manifestUrl).then((r) => {
      if (!r.ok) throw new Error(`Manifest fetch failed: ${r.status}`);
      return r.json();
    });
    const urls = manifest.docusaurus || {
      artifactUrl: `${this.assetBase}/index.pancake-range`,
      corpusUrl: `${this.assetBase}/corpus.json`,
      studentModelUrl: `${this.assetBase}/student-model.bin`,
      abstentionUrl: `${this.assetBase}/student-abstention.json`,
    };
    const { Pancake } = await loadBrowserRuntime();
    const [corpus, studentBytes, abstention, artifact] = await Promise.all([
      fetch(urls.corpusUrl).then((r) => {
        if (!r.ok) throw new Error(`Corpus fetch failed: ${r.status}`);
        return r.json();
      }),
      fetch(urls.studentModelUrl).then((r) => {
        if (!r.ok) throw new Error(`Student model fetch failed: ${r.status}`);
        return r.arrayBuffer();
      }),
      urls.abstentionUrl
        ? fetch(urls.abstentionUrl).then((r) => {
            if (r.status === 404) return null;
            if (!r.ok) throw new Error(`Abstention fetch failed: ${r.status}`);
            return r.json();
          })
        : null,
      Pancake.RangeArtifact.open(createRangeSource(urls.artifactUrl)),
    ]);
    const studentModel = loadStudentModel(studentBytes);
    this.state = {
      manifest,
      artifact,
      studentModel,
      abstention,
      corpusById: new Map(corpus.map((chunk) => [chunk.id, chunk])),
    };
    return this.state;
  }

  async search(query, options = {}) {
    const state = await this.load();
    const prefixed = `${state.manifest.prefixPolicy?.query || ''}${query}`;
    const embedded = embedTextWithStudent(prefixed, state.studentModel);
    const embedding = embedded.vector;
    const result = await state.artifact.search(embedding, Number(options.k || this.k), {
      efSearch: Number(options.efSearch || state.manifest.efSearch || this.efSearch),
    });
    const quality = computeMatchQuality(result.results, embedded, state.abstention);
    const results = quality.match_quality === 'none' ? [] : result.results;
    return {
      ...result,
      ...quality,
      results,
      rows: results.map((hit) => state.corpusById.get(hit.id)),
    };
  }
}

function mountSearch() {
  const root = document.querySelector('[data-pancake-search]');
  if (!root || root.dataset.pancakeMounted === '1') return;
  root.dataset.pancakeMounted = '1';
  const assetBase = root.dataset.pancakeAssetBase || window.__PANCAKE_SEARCH__?.assetBase;
  const search = new PancakeDocusaurusSearch({ assetBase });
  root.innerHTML = `
    <button class="pancake-search-launcher" type="button" aria-label="Open search">Search</button>
    <section class="pancake-search-card" aria-label="Search documentation" hidden>
      <div class="pancake-search-titlebar">
        <button class="pancake-search-drag-handle" type="button" aria-label="Move search panel">
          <span>Search docs</span>
        </button>
        <button class="pancake-search-close" type="button" aria-label="Close search">×</button>
      </div>
      <form class="pancake-search-form">
        <input class="pancake-search-input" type="search" autocomplete="off" placeholder="Search docs" />
        <button class="pancake-search-button" type="submit">Search</button>
      </form>
      <div class="pancake-search-status">Loading search...</div>
      <div class="pancake-search-output"></div>
    </section>
  `;
  const launcher = root.querySelector('.pancake-search-launcher');
  const card = root.querySelector('.pancake-search-card');
  const close = root.querySelector('.pancake-search-close');
  const handle = root.querySelector('.pancake-search-drag-handle');
  const form = root.querySelector('form');
  const input = root.querySelector('input');
  const button = root.querySelector('.pancake-search-button');
  const status = root.querySelector('.pancake-search-status');
  const output = root.querySelector('.pancake-search-output');

  enableDrag(root, handle);
  launcher.addEventListener('click', () => {
    card.hidden = false;
    launcher.hidden = true;
    clampPanel(root);
    input.focus();
  });
  close.addEventListener('click', () => {
    card.hidden = true;
    launcher.hidden = false;
  });

  search.load()
    .then((state) => { status.textContent = `Ready - ${state.artifact.count.toLocaleString()} chunks`; })
    .catch((error) => { status.textContent = error.message || String(error); });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const query = input.value.trim();
    if (!query) return;
    button.disabled = true;
    status.textContent = 'Searching...';
    try {
      const t0 = performance.now();
      const result = await search.search(query);
      const quality = result.match_quality && result.match_quality !== 'unscored'
        ? ` - ${result.match_quality}`
        : '';
      status.textContent = `${result.results.length} results${quality} in ${(performance.now() - t0).toFixed(0)} ms`;
      output.innerHTML = renderRows(result.results, search.state.corpusById);
    } catch (error) {
      status.textContent = error.message || String(error);
    } finally {
      button.disabled = false;
    }
  });
}

if (typeof window !== 'undefined') {
  window.PancakeDocusaurusSearch = PancakeDocusaurusSearch;
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mountSearch);
  else mountSearch();
}

export { PancakeDocusaurusSearch, mountSearch };
