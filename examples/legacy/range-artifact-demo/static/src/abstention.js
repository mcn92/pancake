// Client-side abstention scoring for the artifact demo.
// Mirrors the signal computation in examples/03-edge-docs-search/worker.js
// (buildKnownBucketTables / computeKnownFractions / computeHiddenProbe) so the
// static demo applies the same shipped calibration as the Worker. The learned
// scoring itself comes from the canonical student-embedder module.

import { scoreQuery } from '../../../../03-edge-docs-search/student-embedder.mjs';

function buildKnownBucketTables(model) {
  const word = new Map();
  for (const row of model?.wordBuckets || []) {
    if (!Array.isArray(row) || row.length < 2) continue;
    const bucket = Number(row[0]);
    const idf = Number(row[1]);
    if (Number.isInteger(bucket) && bucket >= 0 && Number.isFinite(idf) && idf >= 0) {
      word.set(bucket, idf);
    }
  }
  const char = new Set();
  for (const value of model?.charBuckets || []) {
    const bucket = Number(value);
    if (Number.isInteger(bucket) && bucket >= 0) char.add(bucket);
  }
  return { word, char };
}

function computeKnownFractions(features, tables) {
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
    n_feats: features.length,
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

export function createAbstentionScorer(asset) {
  if (!asset || !Array.isArray(asset.weights) || !asset.thresholds) return null;
  const tables = buildKnownBucketTables(asset);
  const thresholds = asset.thresholds;
  const minFeatures = Number.isInteger(thresholds.minFeatures) ? thresholds.minFeatures : 3;
  const preNormFloor = Number.isFinite(thresholds.preNormFloor) ? thresholds.preNormFloor : 0.4;

  const signalsFor = (embedded, d0, margin) => {
    const known = computeKnownFractions(embedded.features, tables);
    return {
      d0,
      margin,
      pre_norm: embedded.preNorm,
      known_word: known.known_word,
      known_char: known.known_char,
      hidden_probe: computeHiddenProbe(embedded, asset),
      n_feats: known.n_feats,
    };
  };

  return {
    // Degenerate queries (too few features, tiny pre-norm) are scored before
    // any search runs, exactly as the Worker does.
    scorePreSearch(embedded) {
      if (embedded.features.length >= minFeatures && embedded.preNorm >= preNormFloor) return null;
      return scoreQuery(signalsFor(embedded, 1, 0), asset);
    },
    score(hits, embedded) {
      const d0 = hits.length > 0 ? hits[0].distance : 1;
      const marginIndex = Math.min(4, hits.length - 1);
      const margin = marginIndex > 0 ? hits[marginIndex].distance - d0 : 0;
      return scoreQuery(signalsFor(embedded, d0, margin), asset);
    },
  };
}
