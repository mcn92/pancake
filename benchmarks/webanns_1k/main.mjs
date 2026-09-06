import Pikelet from '../../pikelet.web.mjs';

const DATASET_URL = '/webanns_1k/arxiv_1k.jsonl';
const TOP_K = 10;
const RANDOM_QUERY_COUNT = 100;
const SELF_QUERY_COUNT = 100;
const SEED = 0x5eed1234;

function status(message) {
  document.getElementById('status').textContent = message;
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return function next() {
    state += 0x6D2B79F5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function percentile(values, p) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[index];
}

function summarizeLatencies(latencies) {
  const total = latencies.reduce((sum, value) => sum + value, 0);
  return {
    count: latencies.length,
    mean_ms: total / latencies.length,
    min_ms: Math.min(...latencies),
    p50_ms: percentile(latencies, 50),
    p95_ms: percentile(latencies, 95),
    p99_ms: percentile(latencies, 99),
    max_ms: Math.max(...latencies),
  };
}

function l2sq(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

function exactTopK(vectors, query, k) {
  return vectors
    .map((vector, id) => ({ id, distance: l2sq(vector, query) }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, k)
    .map((item) => item.id);
}

function recallAtK(actual, expected) {
  const actualIds = new Set(actual.map((item) => item.id));
  let hit = 0;
  for (const id of expected) {
    if (actualIds.has(id)) hit++;
  }
  return hit / expected.length;
}

async function loadDataset() {
  const response = await fetch(DATASET_URL);
  if (!response.ok) {
    throw new Error(`failed to fetch ${DATASET_URL}: ${response.status}`);
  }
  const text = await response.text();
  const rows = text
    .split('\n')
    .filter((line) => line.trim().length > 0)
    .map((line, id) => {
      const parsed = JSON.parse(line);
      return {
        id,
        key: parsed.key,
        layer: parsed.layer,
        vector: new Float32Array(parsed.vector),
      };
    });
  if (rows.length === 0) throw new Error('dataset is empty');
  return rows;
}

function randomQuery(dim, rng) {
  const query = new Float32Array(dim);
  for (let i = 0; i < dim; i++) query[i] = rng();
  return query;
}

async function runOne(config, rows) {
  const vectors = rows.map((row) => row.vector);
  const dim = vectors[0].length;
  const index = await Pikelet.create({
    dim,
    maxElements: vectors.length,
    metric: 'l2',
    quantized: config.quantized,
    M: config.M,
    efConstruction: config.efConstruction,
    efSearch: config.efSearch,
  });

  const buildStart = performance.now();
  index.addBatch(vectors);
  const buildMs = performance.now() - buildStart;

  const randomRng = mulberry32(SEED);
  index.search(randomQuery(dim, randomRng), TOP_K, { efSearch: config.efSearch });

  const randomLatencies = [];
  for (let i = 0; i < RANDOM_QUERY_COUNT; i++) {
    const query = randomQuery(dim, randomRng);
    const start = performance.now();
    index.search(query, TOP_K, { efSearch: config.efSearch });
    randomLatencies.push(performance.now() - start);
  }

  const selfRng = mulberry32(SEED ^ 0xa11ce);
  const selfLatencies = [];
  let recallSum = 0;
  for (let i = 0; i < SELF_QUERY_COUNT; i++) {
    const queryId = Math.floor(selfRng() * vectors.length);
    const query = vectors[queryId];
    const expected = exactTopK(vectors, query, TOP_K);
    const start = performance.now();
    const actual = index.search(query, TOP_K, { efSearch: config.efSearch });
    selfLatencies.push(performance.now() - start);
    recallSum += recallAtK(actual, expected);
  }

  const snapshotStart = performance.now();
  const snapshot = index.export();
  const snapshotMs = performance.now() - snapshotStart;

  const result = {
    name: config.name,
    dataset: 'webanns-arxiv-1k',
    vectors: vectors.length,
    dim,
    metric: 'l2',
    top_k: TOP_K,
    random_queries: RANDOM_QUERY_COUNT,
    self_queries: SELF_QUERY_COUNT,
    config,
    build_ms: buildMs,
    snapshot_bytes: snapshot.byteLength,
    snapshot_ms: snapshotMs,
    random_query_latency: summarizeLatencies(randomLatencies),
    self_query_latency: summarizeLatencies(selfLatencies),
    self_query_recall_at_10: recallSum / SELF_QUERY_COUNT,
  };

  index.dispose();
  return result;
}

async function main() {
  const startedAt = new Date().toISOString();
  status('loading dataset');
  const rows = await loadDataset();
  const configs = [
    { name: 'pikelet-int8-M16-ef1000', quantized: true, M: 16, efConstruction: 1000, efSearch: 1000 },
    { name: 'pikelet-f32-M16-ef1000', quantized: false, M: 16, efConstruction: 1000, efSearch: 1000 },
  ];

  const results = [];
  for (const config of configs) {
    status(`running ${config.name}`);
    results.push(await runOne(config, rows));
  }

  window.__PANCAKE_WEBANNS_1K__ = {
    ok: true,
    started_at: startedAt,
    finished_at: new Date().toISOString(),
    user_agent: navigator.userAgent,
    results,
  };
  status(JSON.stringify(window.__PANCAKE_WEBANNS_1K__, null, 2));
}

main().catch((error) => {
  window.__PANCAKE_WEBANNS_1K__ = {
    ok: false,
    error: error && error.stack ? error.stack : String(error),
  };
  status(window.__PANCAKE_WEBANNS_1K__.error);
});
