#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import Pancake from '../../pancake.node.mjs';
import { DEMO_DIM, embedText } from './embedder.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..');

const DEFAULT_OUT_DIR = '/tmp/pancake-docs-demo';
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
    out[arg.slice(2)] = argv[i + 1];
    i++;
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
  const index = await Pancake.create({
    dim: DEMO_DIM,
    maxElements: corpus.length + 16,
    metric: 'cosine',
    quantized: true,
    M: 12,
    efConstruction: 150,
    efSearch: 120
  });

  const vectors = corpus.map((chunk) =>
    embedText(`${chunk.docTitle}\n${chunk.title}\n${chunk.title}\n${chunk.text}`, DEMO_DIM)
  );
  index.addBatch(vectors);

  const manifest = {
    generatedAt: new Date().toISOString(),
    dim: DEMO_DIM,
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
    indexKey: 'docs-index.bin'
  };

  const snapshot = index.export();
  fs.writeFileSync(path.join(outDir, 'docs-index.bin'), Buffer.from(snapshot));
  fs.writeFileSync(path.join(outDir, 'docs-corpus.json'), JSON.stringify(corpus, null, 2));
  fs.writeFileSync(path.join(outDir, 'docs-manifest.json'), JSON.stringify(manifest, null, 2));

  console.log(`[write] ${path.join(outDir, 'docs-index.bin')}`);
  console.log(`[write] ${path.join(outDir, 'docs-corpus.json')}`);
  console.log(`[write] ${path.join(outDir, 'docs-manifest.json')}`);

  console.log('\n[preview]');
  for (const query of SAMPLE_QUERIES) {
    const results = index.search(embedText(query, DEMO_DIM), 3);
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
