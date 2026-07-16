#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { embedTextWithStudent, loadStudentModel } from './student-embedder.mjs';

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

for (const row of evaluation.queries) {
  const query = embedTextWithStudent(row.text, model);
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

console.log(`Student artifact verified: ${(model.byteLength / 1024).toFixed(1)} KiB`);
console.log(`JavaScript/Python top-1 parity: ${(parity * 100).toFixed(2)}% (${matchingTop1}/${evaluation.queries.length})`);
console.log(`Output norms: ${minimumNorm.toFixed(6)}..${maximumNorm.toFixed(6)}`);
