// Build-time embeddings: local transformers.js, the student trainer, the inline
// transformer, precomputed vectors, the deterministic stub, and self-recall.

import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { sha256, loadCompleteModules, STUDENT_TRAINER, MODEL_MAP, CliError } from './common.mjs';
import { tokenize, publicChunk } from './ingest.mjs';
import { inlineEncoderDeclaration, resolveInlineEncoderInputs } from './complete-build.mjs';

async function embedChunks(chunks, config, log, projectDir) {
  const embeddingConfig = config.embedding;
  if (process.env.PANCAKE_SEARCH_STUB_EMBEDDINGS === '1') {
    log('Embedding with deterministic local stub (PANCAKE_SEARCH_STUB_EMBEDDINGS=1)');
    return chunks.map((chunk) => hashEmbedding(chunk.text, embeddingConfig.dims));
  }
  if (embeddingConfig.mode === 'student') {
    if (embeddingConfig.trainStudent?.enabled) {
      return trainStudentVectors(chunks, embeddingConfig, log, projectDir);
    }
    if (embeddingConfig.teacherVectorsPath) {
      const vectorPath = path.resolve(projectDir, embeddingConfig.teacherVectorsPath);
      const vectors = await readF32Vectors(vectorPath, chunks.length, embeddingConfig.dims);
      log(`Loaded ${vectors.length} teacher document vectors from ${embeddingConfig.teacherVectorsPath}`);
      return vectors;
    }
    throw new CliError('Student mode requires trainStudent.enabled or teacherVectorsPath; refusing to embed passages with the student query encoder.', 2);
  }
  if (embeddingConfig.mode === 'inline-transformer') {
    if (embeddingConfig.teacherVectorsPath) {
      const vectorPath = path.resolve(projectDir, embeddingConfig.teacherVectorsPath);
      const vectors = await readF32Vectors(vectorPath, chunks.length, embeddingConfig.dims);
      log(`Loaded ${vectors.length} inline-transformer document vectors from ${embeddingConfig.teacherVectorsPath}`);
      return vectors;
    }
    return embedChunksWithInlineTransformer(chunks, config, log, projectDir);
  }
  const model = MODEL_MAP[embeddingConfig.buildModel];
  let transformers;
  try {
    transformers = await import('@xenova/transformers');
  } catch (error) {
    throw new CliError(`Failed to load @xenova/transformers. Next: run npm install in the create-pancake-search package or generated project. ${error.message}`, 2);
  }
  const extractor = await transformers.pipeline('feature-extraction', model.hfModel, { quantized: true });
  const vectors = [];
  for (let i = 0; i < chunks.length; i += 16) {
    const batch = chunks.slice(i, i + 16).map((chunk) => `${embeddingConfig.prefixPolicy?.passage || ''}${chunk.text}`);
    const output = await extractor(batch, { pooling: embeddingConfig.pooling || 'mean', normalize: embeddingConfig.normalize !== false });
    vectors.push(...tensorToVectors(output, embeddingConfig.dims));
    log(`Embedded ${Math.min(i + 16, chunks.length)}/${chunks.length} chunks`);
  }
  return vectors;
}

async function trainStudentVectors(chunks, embeddingConfig, log, projectDir) {
  const trainConfig = embeddingConfig.trainStudent || {};
  const outDir = path.resolve(projectDir, trainConfig.outDir || 'student');
  const corpusPath = path.join(outDir, 'student-corpus.json');
  await fs.rm(outDir, { recursive: true, force: true });
  await fs.mkdir(outDir, { recursive: true });
  await fs.writeFile(corpusPath, `${JSON.stringify(chunks.map(publicChunk), null, 2)}\n`);

  const python = trainConfig.python || process.env.PANCAKE_SEARCH_PYTHON || 'python3';
  const args = [
    STUDENT_TRAINER,
    '--corpus',
    corpusPath,
    '--out',
    outDir,
  ];
  if (trainConfig.teacher) args.push('--teacher', String(trainConfig.teacher));
  if (trainConfig.teacherRevision) args.push('--teacher-revision', String(trainConfig.teacherRevision));
  if (trainConfig.epochs) args.push('--epochs', String(trainConfig.epochs));
  if (trainConfig.buckets) args.push('--buckets', String(trainConfig.buckets));
  if (trainConfig.hidden) args.push('--hidden', String(trainConfig.hidden));
  if (trainConfig.batchSize) args.push('--batch-size', String(trainConfig.batchSize));
  if (trainConfig.learningRate) args.push('--learning-rate', String(trainConfig.learningRate));
  if (trainConfig.maxFeatures) args.push('--max-features', String(trainConfig.maxFeatures));
  if (trainConfig.seed) args.push('--seed', String(trainConfig.seed));
  if (trainConfig.skipAbstention) args.push('--skip-abstention');

  log(`Training corpus-specific student encoder with ${python}`);
  const result = spawnSync(python, args, { cwd: projectDir, stdio: 'inherit' });
  if (result.error) {
    throw new CliError(`Failed to start student trainer: ${result.error.message}`, 2);
  }
  if (result.status !== 0) {
    throw new CliError(`Student trainer failed with exit code ${result.status}. Next: if Python lacks torch/transformers install them; if abstention acceptance failed, rerun with --skip-abstention or improve the source corpus; or provide a corpus-specific --student-model.`, 2);
  }

  const manifestPath = path.join(outDir, 'student-manifest.json');
  const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  if (manifest.outputDim !== embeddingConfig.dims) {
    throw new CliError(`Student trainer dimension mismatch (${manifest.outputDim} !== ${embeddingConfig.dims})`, 2);
  }
  const vectorPath = path.join(outDir, 'docs-vectors.f32');
  const vectors = await readF32Vectors(vectorPath, chunks.length, embeddingConfig.dims);
  log(`Loaded ${vectors.length} teacher document vectors from student trainer`);
  return vectors;
}

async function embedChunksWithInlineTransformer(chunks, config, log, projectDir) {
  const { encoder, vocabPath, weightsPath } = await resolveInlineEncoderInputs(config, projectDir);
  const { builder, reader } = await loadCompleteModules();
  // loadInlineEncoderKernel uses the web glue with an explicit wasmBinary,
  // not encoder.node.mjs: when the Docusaurus plugin runs this module it is
  // loaded through jiti's CJS transform, which breaks the node glue's
  // createRequire bootstrap ("require is not a function").
  const declaration = inlineEncoderDeclaration(config, encoder);
  const embedder = await reader.createInlineTransformerEmbedder({
    declaration,
    vocabText: await fs.readFile(vocabPath, 'utf8'),
    blob: await fs.readFile(weightsPath),
    createEncoder: await builder.loadInlineEncoderKernel(),
  });
  const vectors = [];
  let windowed = 0;
  try {
    for (let i = 0; i < chunks.length; i++) {
      const text = `${declaration.prefixPolicy?.passage || ''}${chunks[i].text}`;
      const embedded = await embedder.embed(text);
      if (embedded.windows > 1) windowed++;
      vectors.push(embedded.vector);
      if ((i + 1) % 16 === 0 || i + 1 === chunks.length) {
        log(`Embedded ${i + 1}/${chunks.length} chunks with inline transformer`);
      }
    }
  } finally {
    embedder.dispose();
  }
  if (windowed > 0) {
    log(`${windowed}/${chunks.length} chunks exceeded the ${embedder.maxSeq}-token encoder window and were mean-pooled across windows`);
  }
  return vectors;
}

async function readF32Vectors(file, count, dims) {
  const bytes = await fs.readFile(file);
  const expected = count * dims * 4;
  if (bytes.byteLength !== expected) {
    throw new CliError(`docs-vectors.f32 size mismatch: expected ${expected}, received ${bytes.byteLength}`, 2);
  }
  const view = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  const vectors = [];
  for (let row = 0; row < count; row++) {
    const start = row * dims;
    vectors.push(Float32Array.from(view.subarray(start, start + dims)));
  }
  return vectors;
}

function tensorToVectors(output, dims) {
  const data = output.data || output;
  const vectors = [];
  for (let offset = 0; offset < data.length; offset += dims) {
    vectors.push(Float32Array.from(data.slice(offset, offset + dims)));
  }
  return vectors;
}

function hashEmbedding(text, dims) {
  const v = new Float32Array(dims);
  const words = tokenize(text.toLowerCase());
  for (const word of words) {
    const h = crypto.createHash('sha256').update(word).digest();
    const idx = h.readUInt32LE(0) % dims;
    v[idx] += (h[4] / 255) * 2 - 1;
  }
  let norm = 0;
  for (let i = 0; i < dims; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < dims; i++) v[i] /= norm;
  return v;
}
function validateSelfRecall(index, chunks, vectors, log) {
  const sampleCount = Math.min(100, chunks.length);
  for (let i = 0; i < sampleCount; i++) {
    const idx = Math.floor(i * chunks.length / sampleCount);
    const hit = index.search(vectors[idx], 1, { efSearch: Math.max(120, index.config.efSearch) })[0];
    if (!hit || chunks[hit.id]?.text !== chunks[idx].text) {
      throw new CliError(`Self-recall validation failed at chunk ${idx}. Next: reduce quantization risk or inspect duplicated content.`, 2);
    }
  }
  log(`Validated self-recall@1 on ${sampleCount} sampled chunks`);
}

export {
  embedChunks,
  trainStudentVectors,
  embedChunksWithInlineTransformer,
  readF32Vectors,
  tensorToVectors,
  hashEmbedding,
  validateSelfRecall,
};
