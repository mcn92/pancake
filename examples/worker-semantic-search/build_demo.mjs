#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import Pancake from '../../pancake.node.mjs';
import { embedTextWithStudent, loadStudentModel } from './student-embedder.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..');

const DEFAULT_OUT_DIR = path.join(os.tmpdir(), 'pancake-docs-demo');
const TARGET_CHARS = 700;

const DOC_SOURCES = [
  'README.md',
  'QUICKSTART.md',
  'docs/SYSTEM_DESIGN.md',
  'examples/worker/README.md'
];

const SAMPLE_QUERIES = [
  'How do Cloudflare Workers restore snapshots from R2?',
  'Why do I need compact before export after deletes?',
  'How does filtered search work in Pancake?',
  'What are the memory tradeoffs for quantized indexes?'
];

function parseArgs(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith('--')) continue;
    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) {
      out[arg.slice(2)] = true;
    } else {
      out[arg.slice(2)] = next;
      i++;
    }
  }
  return out;
}

function slugify(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80) || 'section';
}

function stripMarkdown(text) {
  return String(text || '')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`+/g, ' ')
    .replace(/!\[[^\]]*\]\([^)]+\)/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/^\s*[-*+]\s+/gm, ' ')
    .replace(/^\s*\d+\.\s+/gm, ' ')
    .replace(/[>#*_~|]/g, ' ')
    .replace(/\bghostCount\s*>\s*0\b/g, 'ghostCount > 0')
    .replace(/\bpancake\s+wasm\b/gi, 'pancake-wasm')
    .replace(/\s+/g, ' ')
    .trim();
}

function createPreview(text, maxChars = 220) {
  let preview = stripMarkdown(text)
    .replace(/\s*:\s*/g, ': ')
    .replace(/\s+([.,;:!?])/g, '$1')
    .trim();

  if (preview.length <= maxChars) return preview;
  let cut = preview.lastIndexOf(' ', maxChars);
  if (cut < maxChars * 0.6) cut = maxChars;
  preview = preview.slice(0, cut).trim();
  return `${preview}...`;
}

function splitLongText(text, maxChars) {
  const parts = [];
  let remaining = text.trim();
  while (remaining.length > maxChars) {
    let cut = remaining.lastIndexOf('. ', maxChars);
    if (cut < maxChars * 0.5) cut = remaining.lastIndexOf(' ', maxChars);
    if (cut < maxChars * 0.4) cut = maxChars;
    parts.push(remaining.slice(0, cut).trim());
    remaining = remaining.slice(cut).trim();
  }
  if (remaining) parts.push(remaining);
  return parts;
}

function createChunk(doc, headingTrail, anchor, text) {
  const clean = stripMarkdown(text);
  if (!clean) return [];
  const title = headingTrail.length ? headingTrail.join(' / ') : doc.title;
  return splitLongText(clean, TARGET_CHARS).map((part) => ({
    sourcePath: doc.path,
    docTitle: doc.title,
    title,
    anchor,
    text: part,
    preview: createPreview(part)
  }));
}

function chunkMarkdownDocument(doc) {
  const lines = doc.content.split(/\r?\n/);
  const headingTrail = [doc.title];
  let currentAnchor = slugify(doc.title);
  let buffer = [];
  const chunks = [];

  function flush() {
    if (buffer.length === 0) return;
    chunks.push(...createChunk(doc, headingTrail, currentAnchor, buffer.join('\n')));
    buffer = [];
  }

  for (const line of lines) {
    const headingMatch = /^(#{1,6})\s+(.*)$/.exec(line);
    if (headingMatch) {
      flush();
      const level = headingMatch[1].length;
      const heading = stripMarkdown(headingMatch[2]);
      headingTrail.length = Math.max(1, level);
      headingTrail[level - 1] = heading;
      currentAnchor = slugify(heading);
      continue;
    }

    buffer.push(line);
    if (stripMarkdown(buffer.join('\n')).length > TARGET_CHARS) {
      flush();
    }
  }

  flush();
  return chunks;
}

function loadDocuments() {
  return DOC_SOURCES.map((relativePath) => {
    const absPath = path.join(REPO_ROOT, relativePath);
    const content = fs.readFileSync(absPath, 'utf8');
    const firstHeading = /^#\s+(.+)$/m.exec(content)?.[1]?.trim();
    return {
      path: relativePath,
      absPath,
      title: firstHeading || path.basename(relativePath, path.extname(relativePath)),
      content
    };
  });
}

function buildCorpus() {
  const docs = loadDocuments();
  const chunks = [];
  for (const doc of docs) {
    for (const chunk of chunkMarkdownDocument(doc)) {
      chunk.id = chunks.length;
      chunks.push(chunk);
    }
  }
  return chunks;
}

async function main() {
  const args = parseArgs(process.argv);
  const outDir = path.resolve(args.out || DEFAULT_OUT_DIR);
  fs.mkdirSync(outDir, { recursive: true });

  const corpus = buildCorpus();
  if (corpus.length === 0) {
    throw new Error('No chunks were generated from the docs corpus');
  }

  console.log(`[build] generated ${corpus.length} chunks from ${DOC_SOURCES.length} markdown files`);
  const corpusPath = path.join(outDir, 'docs-corpus.json');
  fs.writeFileSync(corpusPath, JSON.stringify(corpus, null, 2));
  console.log(`[write] ${corpusPath}`);
  if (args['corpus-only']) return;

  if (!args['student-dir']) {
    throw new Error('--student-dir is required unless --corpus-only is used');
  }
  const studentDir = path.resolve(args['student-dir']);
  const studentBytes = fs.readFileSync(path.join(studentDir, 'student-model.bin'));
  const studentManifest = JSON.parse(
    fs.readFileSync(path.join(studentDir, 'student-manifest.json'), 'utf8')
  );
  const evaluation = JSON.parse(
    fs.readFileSync(path.join(studentDir, 'student-evaluation.json'), 'utf8')
  );
  const abstentionPath = path.join(studentDir, 'student-abstention.json');
  const abstention = fs.existsSync(abstentionPath)
    ? JSON.parse(fs.readFileSync(abstentionPath, 'utf8'))
    : null;
  const vectorBytes = fs.readFileSync(path.join(studentDir, 'docs-vectors.f32'));
  const expectedVectorBytes = corpus.length * studentManifest.outputDim * 4;
  if (vectorBytes.byteLength !== expectedVectorBytes) {
    throw new Error(
      `docs-vectors.f32 size mismatch: expected ${expectedVectorBytes}, received ${vectorBytes.byteLength}`
    );
  }
  const vectorsView = new Float32Array(
    vectorBytes.buffer,
    vectorBytes.byteOffset,
    vectorBytes.byteLength / 4
  );
  const vectors = new Array(corpus.length);
  for (let row = 0; row < corpus.length; row++) {
    const start = row * studentManifest.outputDim;
    vectors[row] = vectorsView.subarray(start, start + studentManifest.outputDim);
  }

  const index = await Pancake.create({
    dim: studentManifest.outputDim,
    maxElements: corpus.length + 16,
    metric: 'cosine',
    quantized: true,
    M: 12,
    efConstruction: 150,
    efSearch: 120
  });
  index.addBatch(vectors);
  const abstentionSerialized = abstention ? JSON.stringify(abstention) : null;
  const abstentionSha256 = abstentionSerialized
    ? crypto.createHash('sha256').update(abstentionSerialized).digest('hex')
    : null;

  const manifest = {
    generatedAt: abstention?.calibratedAt || new Date().toISOString(),
    dim: studentManifest.outputDim,
    metric: 'cosine',
    quantized: true,
    maxElements: corpus.length + 16,
    M: 12,
    efConstruction: 150,
    efSearch: 120,
    chunkCount: corpus.length,
    docsCount: DOC_SOURCES.length,
    sampleQueries: SAMPLE_QUERIES,
    corpusKey: 'docs-corpus.json',
    indexKey: 'docs-index.bin',
    studentKey: 'docs-student.bin',
    encoder: {
      format: studentManifest.format,
      architecture: studentManifest.architecture,
      teacher: studentManifest.teacher,
      teacherRevision: studentManifest.teacherRevision,
      modelBytes: studentManifest.modelBytes,
      modelSha256: studentManifest.modelSha256,
      runtimeDependencies: 0,
      outboundRequests: 0,
      evaluation: studentManifest.evaluation,
    },
  };
  if (abstention) {
    manifest.abstention = {
      assetKey: 'docs-abstention.json',
      bytes: new TextEncoder().encode(abstentionSerialized).byteLength,
      sha256: abstentionSha256,
      evaluation: evaluation.abstention || abstention.evaluation?.test || null,
      calibratedAt: abstention.calibratedAt,
      farTarget: abstention.farTarget,
    };
  }

  const snapshot = index.export();
  fs.writeFileSync(path.join(outDir, 'docs-index.bin'), Buffer.from(snapshot));
  fs.writeFileSync(path.join(outDir, 'docs-manifest.json'), JSON.stringify(manifest, null, 2));
  fs.copyFileSync(path.join(studentDir, 'student-model.bin'), path.join(outDir, 'docs-student.bin'));
  if (abstention) {
    fs.writeFileSync(path.join(outDir, 'docs-abstention.json'), abstentionSerialized);
  }
  fs.writeFileSync(
    path.join(outDir, 'docs-student-evaluation.json'),
    `${JSON.stringify(evaluation, null, 2)}\n`
  );

  console.log(`[write] ${path.join(outDir, 'docs-index.bin')}`);
  console.log(`[write] ${path.join(outDir, 'docs-manifest.json')}`);
  console.log(`[write] ${path.join(outDir, 'docs-student.bin')}`);
  if (abstention) console.log(`[write] ${path.join(outDir, 'docs-abstention.json')}`);

  console.log('\n[preview]');
  const student = loadStudentModel(studentBytes);
  for (const query of SAMPLE_QUERIES) {
    const results = index.search(embedTextWithStudent(query, student).vector, 3);
    console.log(`  ${query}`);
    for (const hit of results) {
      const chunk = corpus.find((entry) => entry.id === hit.id);
      console.log(`    - ${chunk.title} (${chunk.sourcePath})`);
    }
  }

  index.dispose();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
