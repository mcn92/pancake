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

export function extractStudentFeatures(text, model) {
  const tokens = String(text || '').toLowerCase().match(TOKEN_RE) || [];
  const features = [];
  const limit = model.maxFeatures;

  const push = (feature) => {
    if (features.length < limit) {
      features.push(hash32(feature, model.hashSeed) % model.bucketCount);
    }
  };

  for (let tokenIndex = 0; tokenIndex < tokens.length && features.length < limit; tokenIndex++) {
    const token = tokens[tokenIndex].slice(0, 48);
    push(`w:${token}`);

    const padded = `^${token}$`;
    for (let width = 3; width <= 5 && features.length < limit; width++) {
      for (let start = 0; start + width <= padded.length && features.length < limit; start++) {
        push(`c${width}:${padded.slice(start, start + width)}`);
      }
    }

    if (tokenIndex > 0) {
      push(`b:${tokens[tokenIndex - 1].slice(0, 48)}:${token}`);
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
    _bytes: bytes,
  };
}

export function embedTextWithStudent(text, model) {
  const features = extractStudentFeatures(text, model);
  const hidden = new Float32Array(model.hiddenDim);
  const output = new Float32Array(model.outputDim);
  if (features.length === 0) return output;

  for (const bucket of features) {
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
  if (!(norm > 0) || !Number.isFinite(norm)) return new Float32Array(model.outputDim);
  const inverseNorm = 1 / norm;
  for (let row = 0; row < model.outputDim; row++) output[row] *= inverseNorm;
  return output;
}
