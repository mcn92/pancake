const config = window.PANCAKE_DEMO_CONFIG || {};
const form = document.getElementById('search-form');
const queryInput = document.getElementById('query');
const kInput = document.getElementById('k');
const efInput = document.getElementById('ef');
const apiInput = document.getElementById('api-base');
const accessKeyInput = document.getElementById('access-key');
const saveApiButton = document.getElementById('save-api');
const statusEl = document.getElementById('status');
const metricsEl = document.getElementById('metrics');
const resultsEl = document.getElementById('results');

const storedApiBase = localStorage.getItem('pancakeWorkerApiBase');
const storedAccessKey = localStorage.getItem('pancakeDemoAccessKey');
apiInput.value = storedApiBase || config.apiBase || '';
accessKeyInput.value = storedAccessKey || '';

function getApiBase() {
  return apiInput.value.trim().replace(/\/+$/, '');
}

function setStatus(message, tone = '') {
  statusEl.textContent = message;
  statusEl.dataset.tone = tone;
}

function renderMetrics(payload, roundTripMs = null) {
  const parts = [
    ['Quality', payload.match_quality],
    Number.isFinite(roundTripMs) ? ['Round trip', `${roundTripMs.toFixed(1)}ms`] : null,
    ['Restore', `${payload.restore_ms}ms`],
    ['Embed', `${payload.embedding_ms}ms`],
    ['Search', `${payload.search_ms}ms`],
    ['EF', payload.ef_search]
  ].filter(Boolean);
  metricsEl.innerHTML = parts.map(([label, value]) => (
    `<span><b>${escapeHtml(label)}</b>${escapeHtml(value)}</span>`
  )).join('');
}

function renderResults(payload) {
  if (!payload.results || payload.results.length === 0) {
    resultsEl.innerHTML = '<article class="empty">No results returned.</article>';
    return;
  }

  resultsEl.innerHTML = payload.results.map((item, index) => `
    <article class="result">
      <div class="rank">${index + 1}</div>
      <div>
        <h2><a href="${item.source_url}" target="_blank" rel="noreferrer">${escapeHtml(item.title)}</a></h2>
        <p>${escapeHtml(item.preview)}</p>
        <div class="meta">
          <span>${escapeHtml(item.source_path)}</span>
          <span>distance ${Number(item.distance).toFixed(3)}</span>
        </div>
      </div>
    </article>
  `).join('');
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[char]));
}

async function runSearch() {
  const apiBase = getApiBase();
  if (!apiBase) {
    setStatus('Set the deployed Worker API URL first.', 'error');
    resultsEl.innerHTML = '';
    metricsEl.textContent = '';
    return;
  }

  const params = new URLSearchParams({
    q: queryInput.value.trim(),
    k: kInput.value,
    ef: efInput.value
  });

  setStatus('Searching...');
  metricsEl.textContent = '';

  const headers = {};
  const accessKey = accessKeyInput.value.trim();
  if (accessKey) headers['X-Pancake-Demo-Key'] = accessKey;

  const requestStart = performance.now();
  const response = await fetch(`${apiBase}/search?${params.toString()}`, { headers });
  const roundTripMs = performance.now() - requestStart;
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Search failed with HTTP ${response.status}`);
  }

  setStatus(`${payload.result_count} result${payload.result_count === 1 ? '' : 's'}`, payload.match_quality);
  renderMetrics(payload, roundTripMs);
  renderResults(payload);
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    await runSearch();
  } catch (error) {
    setStatus(error.message, 'error');
    resultsEl.innerHTML = '';
    metricsEl.textContent = '';
  }
});

saveApiButton.addEventListener('click', () => {
  const apiBase = getApiBase();
  const accessKey = accessKeyInput.value.trim();
  if (apiBase) {
    localStorage.setItem('pancakeWorkerApiBase', apiBase);
    apiInput.value = apiBase;
  }
  if (accessKey) {
    localStorage.setItem('pancakeDemoAccessKey', accessKey);
  } else {
    localStorage.removeItem('pancakeDemoAccessKey');
  }
  setStatus('Connection settings saved.');
});

document.querySelectorAll('[data-query]').forEach((button) => {
  button.addEventListener('click', () => {
    queryInput.value = button.dataset.query;
    form.requestSubmit();
  });
});
