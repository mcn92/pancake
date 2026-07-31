import Pancake from 'pancake-wasm/web';
import { loadStudentModel, embedTextWithStudent } from '../../../worker-semantic-search/student-embedder.mjs';
import './style.css';

const ARTIFACT_URL = '/artifacts/pancake-docs.pancake-range';
const MODEL_URL = '/models/docs-student.bin';
const CORPUS_URL = '/corpus/docs-corpus.json';

const SAMPLE_QUERIES = [
  'how does compaction work',
  'How do Cloudflare Workers restore snapshots from R2?',
  'What are the memory tradeoffs for quantized indexes?',
  'how does filtered search work',
];

const els = {
  artifactLabel: document.querySelector('#artifactLabel'),
  queryInput: document.querySelector('#queryInput'),
  sampleQueries: document.querySelector('#sampleQueries'),
  kInput: document.querySelector('#kInput'),
  efInput: document.querySelector('#efInput'),
  gapInput: document.querySelector('#gapInput'),
  searchButton: document.querySelector('#searchButton'),
  wallMetric: document.querySelector('#wallMetric'),
  embedMetric: document.querySelector('#embedMetric'),
  requestsMetric: document.querySelector('#requestsMetric'),
  bytesMetric: document.querySelector('#bytesMetric'),
  missMetric: document.querySelector('#missMetric'),
  cacheMetric: document.querySelector('#cacheMetric'),
  resultsList: document.querySelector('#resultsList'),
  roundsList: document.querySelector('#roundsList'),
};

let artifact;
let model;
let corpusById;

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
    const chunk = corpusById.get(row.id);
    const li = document.createElement('li');
    li.className = 'result';

    const header = document.createElement('div');
    header.className = 'resultHeader';
    const title = document.createElement('strong');
    title.textContent = chunk?.title || `Chunk ${row.id}`;
    const dist = document.createElement('span');
    dist.textContent = row.distance.toFixed(4);
    header.append(title, dist);

    const source = document.createElement('div');
    source.className = 'resultSource';
    source.textContent = chunk?.anchor
      ? `${chunk.sourcePath} #${chunk.anchor}`
      : chunk?.sourcePath || '';

    const preview = document.createElement('p');
    preview.className = 'resultPreview';
    const text = chunk?.preview || chunk?.text || '';
    preview.textContent = text.length > 240 ? `${text.slice(0, 240)}…` : text;

    li.append(header, source, preview);
    return li;
  }));
}

function renderRounds(rounds) {
  const visible = rounds.filter((round) => round.requests > 0);
  if (!visible.length) {
    const div = document.createElement('div');
    div.className = 'round';
    const label = document.createElement('span');
    label.textContent = 'All records served from the local cache';
    div.append(label);
    els.roundsList.replaceChildren(div);
    return;
  }
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
  if (!artifact || !model) return;
  const text = els.queryInput.value.trim();
  if (!text) return;
  setBusy(true);
  try {
    const k = Number(els.kInput.value);
    const efSearch = Number(els.efInput.value);
    const gap = Number(els.gapInput.value);
    const before = artifact.stats();
    const t0 = performance.now();
    const { vector } = embedTextWithStudent(text, model);
    const embedMs = performance.now() - t0;
    const result = await artifact.search(vector, k, {
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
    els.embedMetric.textContent = formatMs(embedMs);
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
  SAMPLE_QUERIES.forEach((sample) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    chip.textContent = sample;
    chip.addEventListener('click', () => {
      els.queryInput.value = sample;
      runSearch();
    });
    els.sampleQueries.append(chip);
  });

  setBusy(true);
  const [modelBytes, corpus] = await Promise.all([
    fetch(MODEL_URL).then((r) => {
      if (!r.ok) throw new Error(`Model fetch failed: ${r.status}`);
      return r.arrayBuffer();
    }),
    fetch(CORPUS_URL).then((r) => {
      if (!r.ok) throw new Error(`Corpus fetch failed: ${r.status}`);
      return r.json();
    }),
  ]);
  model = loadStudentModel(new Uint8Array(modelBytes));
  corpusById = new Map(corpus.map((chunk) => [chunk.id, chunk]));

  const source = createHttpRangeSource(ARTIFACT_URL);
  artifact = await Pancake.RangeArtifact.open(source);
  els.artifactLabel.textContent = [
    `${artifact.count.toLocaleString()} doc chunks`,
    `${artifact.dim}D`,
    `router ${artifact.routerResident.records.toLocaleString()} records`,
    `${formatBytes(artifact.routerResident.bytes)} resident`,
    `encoder ${formatBytes(modelBytes.byteLength)} local`,
  ].join(' / ');
  setBusy(false);
  els.queryInput.focus();
}

els.searchButton.addEventListener('click', runSearch);
els.queryInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') runSearch();
});

boot().catch((error) => {
  setBusy(false);
  els.artifactLabel.textContent = error && error.message ? error.message : String(error);
});
