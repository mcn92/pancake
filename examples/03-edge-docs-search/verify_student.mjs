#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { embedTextWithStudent, loadStudentModel, scoreQuery } from './student-embedder.mjs';

function argument(name) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : null;
}

const studentDir = argument('student-dir');
if (!studentDir) throw new Error('--student-dir is required');
const resolved = path.resolve(studentDir);
const modelBytes = fs.readFileSync(path.join(resolved, 'student-model.bin'));
const manifest = JSON.parse(fs.readFileSync(path.join(resolved, 'student-manifest.json'), 'utf8'));
const evaluation = JSON.parse(fs.readFileSync(path.join(resolved, 'student-evaluation.json'), 'utf8'));
const abstentionPath = path.join(resolved, 'student-abstention.json');
const abstentionBytes = fs.existsSync(abstentionPath) ? fs.readFileSync(abstentionPath) : null;
const abstention = abstentionBytes ? JSON.parse(abstentionBytes.toString('utf8')) : null;
const vectorsBytes = fs.readFileSync(path.join(resolved, 'docs-vectors.f32'));
const model = loadStudentModel(modelBytes);

const sha256 = crypto.createHash('sha256').update(modelBytes).digest('hex');
if (sha256 !== manifest.modelSha256) throw new Error('student-model.bin SHA-256 does not match manifest');
if (model.outputDim !== manifest.outputDim) throw new Error('student output dimension does not match manifest');
if (vectorsBytes.byteLength % (model.outputDim * 4) !== 0) {
  throw new Error('docs-vectors.f32 length is not divisible by the student dimension');
}

const documents = new Float32Array(
  vectorsBytes.buffer,
  vectorsBytes.byteOffset,
  vectorsBytes.byteLength / 4
);
const documentCount = documents.length / model.outputDim;
let matchingTop1 = 0;
let minimumNorm = Infinity;
let maximumNorm = 0;

function buildKnownBucketTables(candidate) {
  const word = new Map();
  let maxIdf = 0;
  for (const row of candidate?.wordBuckets || []) {
    const bucket = Number(row[0]);
    const idf = Number(row[1]);
    if (Number.isInteger(bucket) && Number.isFinite(idf)) {
      word.set(bucket, idf);
      maxIdf = Math.max(maxIdf, idf);
    }
  }
  return {
    word,
    maxIdf: maxIdf || 255,
    char: new Set(candidate?.charBuckets || []),
    abstention: candidate,
  };
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

function computeHiddenProbe(embedded, abstention) {
  const probe = abstention?.hiddenProbe;
  if (!probe || !embedded.hidden) return 0;
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

function computeSignals(embedded, tables) {
  const distances = [];
  for (let document = 0; document < documentCount; document++) {
    let similarity = 0;
    const offset = document * model.outputDim;
    for (let dimension = 0; dimension < model.outputDim; dimension++) {
      similarity += embedded.vector[dimension] * documents[offset + dimension];
    }
    distances.push(1 - similarity);
  }
  distances.sort((left, right) => left - right);
  const d0 = distances.length ? distances[0] : 1;
  const marginIndex = Math.min(4, distances.length - 1);
  const margin = marginIndex > 0 ? distances[marginIndex] - d0 : 0;
  const known = computeKnownFractions(embedded.features, tables);
  return {
    d0,
    margin,
    pre_norm: embedded.preNorm,
    known_word: known.known_word,
    known_char: known.known_char,
    hidden_probe: computeHiddenProbe(embedded, tables.abstention),
    n_feats: known.n_feats,
  };
}

for (const row of evaluation.queries) {
  const embedded = embedTextWithStudent(row.text, model);
  const query = embedded.vector;
  let normSquared = 0;
  for (const value of query) normSquared += value * value;
  const norm = Math.sqrt(normSquared);
  minimumNorm = Math.min(minimumNorm, norm);
  maximumNorm = Math.max(maximumNorm, norm);

  let bestId = -1;
  let bestScore = -Infinity;
  for (let document = 0; document < documentCount; document++) {
    let score = 0;
    const offset = document * model.outputDim;
    for (let dimension = 0; dimension < model.outputDim; dimension++) {
      score += query[dimension] * documents[offset + dimension];
    }
    if (score > bestScore) {
      bestScore = score;
      bestId = document;
    }
  }
  if (bestId === row.studentTop) matchingTop1++;
}

const parity = matchingTop1 / evaluation.queries.length;
if (parity < 0.995) {
  throw new Error(`JavaScript/Python top-1 parity ${(parity * 100).toFixed(2)}% is below 99.5%`);
}
if (Math.abs(minimumNorm - 1) > 1e-4 || Math.abs(maximumNorm - 1) > 1e-4) {
  throw new Error(`Student vectors are not normalized: min=${minimumNorm} max=${maximumNorm}`);
}

let abstentionParity = null;
if (abstention) {
  const abstentionSha = crypto.createHash('sha256').update(abstentionBytes).digest('hex');
  if (manifest.abstention?.sha256 && abstentionSha !== manifest.abstention.sha256) {
    throw new Error('student-abstention.json SHA-256 does not match manifest');
  }
  const rows = evaluation.abstentionQueries || [];
  const tables = buildKnownBucketTables(abstention);
  let matchingLabels = 0;
  let maxScoreDelta = 0;
  for (const row of rows) {
    const embedded = embedTextWithStudent(row.text, model);
    const scored = scoreQuery(computeSignals(embedded, tables), abstention);
    const expectedScore = Number(row.score);
    const scoreDelta = Math.abs(scored.score - expectedScore);
    maxScoreDelta = Math.max(maxScoreDelta, scoreDelta);
    if (scoreDelta > 1e-4) {
      throw new Error(`Abstention score parity failed for "${row.text}": JS=${scored.score} Python=${expectedScore}`);
    }
    if (scored.match_quality !== row.label) {
      throw new Error(`Abstention label parity failed for "${row.text}": JS=${scored.match_quality} Python=${row.label}`);
    }
    matchingLabels++;
  }
  abstentionParity = { rows: rows.length, matchingLabels, maxScoreDelta };
}

console.log(`Student artifact verified: ${(model.byteLength / 1024).toFixed(1)} KiB`);
console.log(`JavaScript/Python top-1 parity: ${(parity * 100).toFixed(2)}% (${matchingTop1}/${evaluation.queries.length})`);
console.log(`Output norms: ${minimumNorm.toFixed(6)}..${maximumNorm.toFixed(6)}`);
if (abstentionParity) {
  console.log(
    `Abstention parity: 100.00% (${abstentionParity.matchingLabels}/${abstentionParity.rows}), ` +
    `max score delta ${abstentionParity.maxScoreDelta.toFixed(6)}`
  );
}
