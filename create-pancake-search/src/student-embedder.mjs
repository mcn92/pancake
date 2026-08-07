const MAGIC = 'PSTU';
const VERSION = 1;
const HEADER_BYTES = 32;
const TOKEN_RE = /[a-z0-9]+/g;

function align4(value) {
  return (value + 3) & ~3;
}

function hash32(text, seed) {
  let hash = seed >>> 0;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export function extractStudentFeatureRecords(text, model) {
  const tokens = String(text || '').toLowerCase().match(TOKEN_RE) || [];
  const features = [];
  const limit = model.maxFeatures;

  const push = (feature, family) => {
    if (features.length < limit) {
      features.push({
        bucket: hash32(feature, model.hashSeed) % model.bucketCount,
        family,
      });
    }
  };

  for (let tokenIndex = 0; tokenIndex < tokens.length && features.length < limit; tokenIndex++) {
    const token = tokens[tokenIndex].slice(0, 48);
    push(`w:${token}`, 'word');

    const padded = `^${token}$`;
    for (let width = 3; width <= 5 && features.length < limit; width++) {
      for (let start = 0; start + width <= padded.length && features.length < limit; start++) {
        push(`c${width}:${padded.slice(start, start + width)}`, 'char');
      }
    }

    if (tokenIndex > 0) {
      push(`b:${tokens[tokenIndex - 1].slice(0, 48)}:${token}`, 'word');
    }
  }

  return features;
}

export function loadStudentModel(input) {
  const bytes = input instanceof Uint8Array
    ? input
    : input instanceof ArrayBuffer
      ? new Uint8Array(input)
      : new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  if (bytes.byteLength < HEADER_BYTES) throw new Error('Student model is truncated');

  const magic = String.fromCharCode(...bytes.subarray(0, 4));
  if (magic !== MAGIC) throw new Error(`Invalid student model magic: ${magic}`);

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint32(4, true);
  if (version !== VERSION) throw new Error(`Unsupported student model version: ${version}`);

  const bucketCount = view.getUint32(8, true);
  const hiddenDim = view.getUint32(12, true);
  const outputDim = view.getUint32(16, true);
  const hashSeed = view.getUint32(20, true);
  const maxFeatures = view.getUint32(24, true);
  if (!bucketCount || !hiddenDim || !outputDim || !maxFeatures) {
    throw new Error('Student model has invalid dimensions');
  }

  let offset = HEADER_BYTES;
  const embeddingScalesBytes = bucketCount * 4;
  const embeddingBytes = bucketCount * hiddenDim;
  const projectionScalesBytes = outputDim * 4;
  const projectionBytes = outputDim * hiddenDim;
  const biasBytes = outputDim * 4;
  const required = align4(offset + embeddingScalesBytes + embeddingBytes)
    + projectionScalesBytes + projectionBytes + biasBytes;
  if (required !== bytes.byteLength) {
    throw new Error(`Student model size mismatch: expected ${required}, received ${bytes.byteLength}`);
  }

  const embeddingScales = new Float32Array(
    bytes.buffer,
    bytes.byteOffset + offset,
    bucketCount
  );
  offset += embeddingScalesBytes;
  const embeddingWeights = new Int8Array(
    bytes.buffer,
    bytes.byteOffset + offset,
    embeddingBytes
  );
  offset = align4(offset + embeddingBytes);
  const projectionScales = new Float32Array(
    bytes.buffer,
    bytes.byteOffset + offset,
    outputDim
  );
  offset += projectionScalesBytes;
  const projectionWeights = new Int8Array(
    bytes.buffer,
    bytes.byteOffset + offset,
    projectionBytes
  );
  offset += projectionBytes;
  const projectionBias = new Float32Array(
    bytes.buffer,
    bytes.byteOffset + offset,
    outputDim
  );

  return {
    version,
    bucketCount,
    hiddenDim,
    outputDim,
    hashSeed,
    maxFeatures,
    embeddingScales,
    embeddingWeights,
    projectionScales,
    projectionWeights,
    projectionBias,
    byteLength: bytes.byteLength,
  };
}

export function embedTextWithStudent(text, model) {
  const features = extractStudentFeatureRecords(text, model);
  const hidden = new Float32Array(model.hiddenDim);
  const output = new Float32Array(model.outputDim);
  if (features.length === 0) {
    return { vector: output, preNorm: 0, features, hidden };
  }

  for (const { bucket } of features) {
    const scale = model.embeddingScales[bucket];
    const rowOffset = bucket * model.hiddenDim;
    for (let column = 0; column < model.hiddenDim; column++) {
      hidden[column] += model.embeddingWeights[rowOffset + column] * scale;
    }
  }

  const inverseCount = 1 / features.length;
  for (let column = 0; column < model.hiddenDim; column++) {
    hidden[column] = Math.tanh(hidden[column] * inverseCount);
  }
  const hiddenOutput = new Float32Array(hidden);

  let normSquared = 0;
  for (let row = 0; row < model.outputDim; row++) {
    const scale = model.projectionScales[row];
    const rowOffset = row * model.hiddenDim;
    let value = model.projectionBias[row];
    for (let column = 0; column < model.hiddenDim; column++) {
      value += model.projectionWeights[rowOffset + column] * scale * hidden[column];
    }
    output[row] = value;
    normSquared += value * value;
  }

  const norm = Math.sqrt(normSquared);
  if (!(norm > 0) || !Number.isFinite(norm)) {
    return { vector: new Float32Array(model.outputDim), preNorm: 0, features, hidden: hiddenOutput };
  }
  const inverseNorm = 1 / norm;
  for (let row = 0; row < model.outputDim; row++) output[row] *= inverseNorm;
  return { vector: output, preNorm: norm, features, hidden: hiddenOutput };
}

function sigmoid(value) {
  if (value >= 0) {
    const z = Math.exp(-value);
    return 1 / (1 + z);
  }
  const z = Math.exp(value);
  return z / (1 + z);
}

function buildKnownBucketTables(abstention) {
  const word = new Map();
  for (const row of abstention?.wordBuckets || []) {
    if (!Array.isArray(row) || row.length < 2) continue;
    const bucket = Number(row[0]);
    const idf = Number(row[1]);
    if (Number.isInteger(bucket) && bucket >= 0 && Number.isFinite(idf) && idf >= 0) {
      word.set(bucket, idf);
    }
  }
  const char = new Set();
  for (const value of abstention?.charBuckets || []) {
    const bucket = Number(value);
    if (Number.isInteger(bucket) && bucket >= 0) char.add(bucket);
  }
  return { word, char };
}

export function computeKnownFractions(features, abstention) {
  if (!abstention) return { known_word: 0, known_char: 0, n_feats: features.length };
  const tables = abstention._knownBucketTables || (abstention._knownBucketTables = buildKnownBucketTables(abstention));
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

export function computeHiddenProbe(embedded, abstention) {
  const probe = abstention?.hiddenProbe;
  if (!probe || !Array.isArray(probe.weights) || !embedded.hidden) return 0;
  let logit = Number(probe.bias) || 0;
  const limit = Math.min(probe.weights.length, embedded.hidden.length);
  for (let index = 0; index < limit; index++) {
    logit += (Number(probe.weights[index]) || 0) * embedded.hidden[index];
  }
  return sigmoid(logit);
}

export function scoreQuery(signals, abstention) {
  if (!abstention) return { match_quality: 'unscored' };
  const featureNames = abstention.features || ['d0', 'margin', 'pre_norm', 'known_word', 'known_char'];
  const weights = abstention.weights || [];
  let logit = Number(abstention.bias) || 0;
  for (let i = 0; i < featureNames.length; i++) {
    logit += (Number(weights[i]) || 0) * (Number(signals[featureNames[i]]) || 0);
  }

  const confidence = sigmoid(logit);
  const thresholds = abstention.thresholds || {};
  const hard = Number.isFinite(thresholds.hard) ? thresholds.hard : 0.05;
  const weak = Number.isFinite(thresholds.weak) ? thresholds.weak : 0;
  const preNormFloor = Number.isFinite(thresholds.preNormFloor) ? thresholds.preNormFloor : 0.4;
  const minFeatures = Number.isInteger(thresholds.minFeatures) ? thresholds.minFeatures : 3;
  const epsilon = 1e-6;
  const floorTriggered = confidence < hard
    || (Number(signals.n_feats) || 0) < minFeatures
    || (Number(signals.pre_norm) || 0) < preNormFloor;
  const match_quality = floorTriggered
    ? 'none'
    : confidence + epsilon >= weak
      ? 'strong'
      : 'weak';
  return {
    match_quality,
    score: confidence,
    confidence: Math.round(confidence * 1000) / 1000,
    signals: {
      d0: signals.d0,
      margin: signals.margin,
      pre_norm: signals.pre_norm,
      known_word: signals.known_word,
      known_char: signals.known_char,
      hidden_probe: signals.hidden_probe,
      n_feats: signals.n_feats,
    },
  };
}

export function computeMatchQuality(hits, embedded, abstention) {
  if (!abstention) return { match_quality: 'unscored' };
  const d0 = hits.length > 0 ? hits[0].distance : 1;
  const marginIndex = Math.min(4, hits.length - 1);
  const margin = marginIndex > 0 ? hits[marginIndex].distance - d0 : 0;
  const known = computeKnownFractions(embedded.features, abstention);
  return scoreQuery({
    d0,
    margin,
    pre_norm: embedded.preNorm,
    known_word: known.known_word,
    known_char: known.known_char,
    hidden_probe: computeHiddenProbe(embedded, abstention),
    n_feats: known.n_feats,
  }, abstention);
}
