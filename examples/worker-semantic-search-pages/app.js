const config = window.PANCAKE_DEMO_CONFIG || {};
const form = document.getElementById('search-form');
const queryInput = document.getElementById('query');
const kInput = document.getElementById('k');
const efInput = document.getElementById('ef');
const apiInput = document.getElementById('api-base');
const saveApiButton = document.getElementById('save-api');
const statusEl = document.getElementById('status');
const metricsEl = document.getElementById('metrics');
const resultsEl = document.getElementById('results');

const storedApiBase = localStorage.getItem('pancakeWorkerApiBase');
apiInput.value = storedApiBase || config.apiBase || '';

function getApiBase() {
  return apiInput.value.trim().replace(/\/+$/, '');
}

function setStatus(message, tone = '') {
  statusEl.textContent = message;
  statusEl.dataset.tone = tone;
}

function renderMetrics(payload) {
  metricsEl.textContent = [
    `quality ${payload.match_quality}`,
    `restore ${payload.restore_ms}ms`,
    `embed ${payload.embedding_ms}ms`,
    `search ${payload.search_ms}ms`,
    `ef ${payload.ef_search}`
  ].join(' / ');
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

  const response = await fetch(`${apiBase}/search?${params.toString()}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Search failed with HTTP ${response.status}`);
  }

  setStatus(`${payload.result_count} result${payload.result_count === 1 ? '' : 's'}`, payload.match_quality);
  renderMetrics(payload);
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
  if (apiBase) {
    localStorage.setItem('pancakeWorkerApiBase', apiBase);
    apiInput.value = apiBase;
    setStatus('Worker API saved.');
  }
});

document.querySelectorAll('[data-query]').forEach((button) => {
  button.addEventListener('click', () => {
    queryInput.value = button.dataset.query;
    form.requestSubmit();
  });
});
