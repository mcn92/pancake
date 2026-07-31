import Pancake from 'pancake-wasm/web';
import './style.css';

const ARTIFACT_URL = '/artifacts/pancake-smoke-split.pancake-range';
const QUERY_DIM = 128;

const queries = [
  makeQuery(11),
  makeQuery(29),
  makeQuery(47),
  makeQuery(83),
  makeQuery(131),
];

const els = {
  artifactLabel: document.querySelector('#artifactLabel'),
  querySelect: document.querySelector('#querySelect'),
  kInput: document.querySelector('#kInput'),
  efInput: document.querySelector('#efInput'),
  gapInput: document.querySelector('#gapInput'),
  searchButton: document.querySelector('#searchButton'),
  wallMetric: document.querySelector('#wallMetric'),
  searchMetric: document.querySelector('#searchMetric'),
  requestsMetric: document.querySelector('#requestsMetric'),
  bytesMetric: document.querySelector('#bytesMetric'),
  missMetric: document.querySelector('#missMetric'),
  cacheMetric: document.querySelector('#cacheMetric'),
  resultsList: document.querySelector('#resultsList'),
  roundsList: document.querySelector('#roundsList'),
};

let artifact;

function makeQuery(seed) {
  let state = seed >>> 0;
  const vector = new Float32Array(QUERY_DIM);
  for (let i = 0; i < vector.length; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    vector[i] = (state / 0xffffffff) * 255;
  }
  return vector;
}

function createHttpRangeSource(url) {
  return {
    async read(offset, length) {
      const response = await fetch(url, {
        headers: { Range: `bytes=${offset}-${offset + length - 1}` },
      });
      if (response.status !== 206 && response.status !== 200) {
        throw new Error(`Range read failed: ${response.status} ${response.statusText}`.trim());
      }
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (bytes.byteLength !== length) {
        throw new Error(`Range read returned ${bytes.byteLength} bytes; expected ${length}`);
      }
      return bytes;
    },
  };
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1048576).toFixed(2)} MiB`;
}

function formatMs(ms) {
  return `${ms.toFixed(ms < 10 ? 2 : 1)} ms`;
}

function setBusy(busy) {
  els.searchButton.disabled = busy;
  els.searchButton.textContent = busy ? 'Searching' : 'Search';
}

function renderResults(rows) {
  els.resultsList.replaceChildren(...rows.map((row) => {
    const li = document.createElement('li');
    const id = document.createElement('strong');
    id.textContent = `#${row.id}`;
    const dist = document.createElement('span');
    dist.textContent = row.distance.toFixed(4);
    li.append(id, dist);
    return li;
  }));
}

function renderRounds(rounds) {
  const visible = rounds.filter((round) => round.requests > 0);
  els.roundsList.replaceChildren(...visible.map((round, index) => {
    const div = document.createElement('div');
    div.className = 'round';
    const label = document.createElement('span');
    label.textContent = `${index + 1}. ${round.phase || 'round'}`;
    const details = document.createElement('strong');
    details.textContent = `${round.requests} req / ${formatBytes(round.bytes)}`;
    div.append(label, details);
    return div;
  }));
}

async function runSearch() {
  if (!artifact) return;
  setBusy(true);
  try {
    const query = queries[Number(els.querySelect.value)];
    const k = Number(els.kInput.value);
    const efSearch = Number(els.efInput.value);
    const gap = Number(els.gapInput.value);
    const before = artifact.stats();
    const t0 = performance.now();
    const result = await artifact.search(query, k, {
      efSearch,
      gap,
      expansionBatch: 8,
      rangeParallelism: 6,
    });
    const wallMs = performance.now() - t0;
    const after = artifact.stats();
    const requests = after.rangeRequests - before.rangeRequests;
    const bytes = after.rangeBytes - before.rangeBytes;
    const missRounds = result.rounds.filter((round) => round.requests > 0).length;

    els.wallMetric.textContent = formatMs(wallMs);
    els.searchMetric.textContent = formatMs(result.searchMs ?? wallMs);
    els.requestsMetric.textContent = requests.toLocaleString();
    els.bytesMetric.textContent = formatBytes(bytes);
    els.missMetric.textContent = missRounds.toLocaleString();
    els.cacheMetric.textContent = after.cachedNodes.toLocaleString();
    renderResults(result.results);
    renderRounds(result.rounds);
  } catch (error) {
    els.resultsList.replaceChildren();
    els.roundsList.textContent = error && error.message ? error.message : String(error);
  } finally {
    setBusy(false);
  }
}

async function boot() {
  queries.forEach((_, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = `Synthetic ${index + 1}`;
    els.querySelect.append(option);
  });

  setBusy(true);
  const source = createHttpRangeSource(ARTIFACT_URL);
  artifact = await Pancake.RangeArtifact.open(source);
  els.artifactLabel.textContent = [
    `${artifact.count.toLocaleString()} vectors`,
    `${artifact.dim}D`,
    `router ${artifact.routerResident.records.toLocaleString()} records`,
    `${formatBytes(artifact.routerResident.bytes)} resident`,
  ].join(' / ');
  await runSearch();
}

els.searchButton.addEventListener('click', runSearch);
els.querySelect.addEventListener('change', runSearch);

boot().catch((error) => {
  setBusy(false);
  els.artifactLabel.textContent = error && error.message ? error.message : String(error);
});
