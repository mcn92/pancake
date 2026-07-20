import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import fssync from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import zlib from 'node:zlib';
import { spawnSync } from 'node:child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PACKAGE_ROOT = path.resolve(__dirname, '..');
const REPO_ROOT = path.resolve(PACKAGE_ROOT, '..');
const DEFAULT_PREFIX = 'Represent this sentence for searching relevant passages: ';
const DEFAULT_CONFIG = Object.freeze({
  chunking: { targetTokens: 256, overlapPercent: 15 },
  embedding: {
    mode: 'workers-ai',
    buildModel: 'bge-small-en-v1.5',
    dims: 384,
    prefixPolicy: { passage: '', query: DEFAULT_PREFIX },
    pooling: 'mean',
    normalize: true,
  },
  index: { metric: 'cosine', quantized: true, M: 16, efConstruction: 200, efSearch: 120 },
  validation: { minRecallAt10: 0.98 },
});
const MODEL_MAP = Object.freeze({
  'bge-small-en-v1.5': {
    dims: 384,
    workersAiModel: '@cf/baai/bge-small-en-v1.5',
    hfModel: 'Xenova/bge-small-en-v1.5',
    maxInputTokens: 512,
  },
});

class CliError extends Error {
  constructor(message, exitCode = 1) {
    super(message);
    this.exitCode = exitCode;
  }
}

export async function main(argv) {
  const parsed = parseArgs(argv.slice(2));
  const command = parsed.positionals[0] === 'rebuild' ? 'rebuild' : 'create';
  if (parsed.flags.help || parsed.flags.h) {
    printHelp();
    return;
  }
  if (command === 'rebuild') {
    await rebuildProject(process.cwd(), parsed.flags);
    return;
  }
  await createProject(parsed.flags);
}

function printHelp() {
  console.log(`create-pancake-search

Usage:
  npm create pancake-search -- --name my-docs-search --source ./docs --no-deploy --yes
  create-pancake-search rebuild --yes

Flags:
  --name <dir>          Generated project directory
  --source <path|url>   Local folder or website URL
  --mode workers-ai     Query embedding mode (student is stubbed)
  --model <id>          bge-small-en-v1.5
  --max-pages <n>       URL crawl cap
  --include <glob>      Folder include glob, repeatable
  --exclude <glob>      Folder exclude glob, repeatable
  --deploy / --no-deploy
  --yes
  --force               Replace an existing target directory
`);
}

function parseArgs(args) {
  const flags = {};
  const repeated = new Set(['include', 'exclude']);
  const positionals = [];
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (!arg.startsWith('--')) {
      positionals.push(arg);
      continue;
    }
    const name = arg.slice(2);
    if (name === 'deploy' || name === 'yes' || name === 'force') {
      flags[name] = true;
      continue;
    }
    if (name === 'no-deploy') {
      flags.deploy = false;
      continue;
    }
    if (i + 1 >= args.length) throw new CliError(`Missing value for --${name}`);
    const value = args[++i];
    if (repeated.has(name)) {
      if (!flags[name]) flags[name] = [];
      flags[name].push(value);
    } else {
      flags[name] = value;
    }
  }
  return { flags, positionals };
}

async function createProject(flags) {
  if (flags.mode && flags.mode !== 'workers-ai') {
    throw new CliError(`--mode ${flags.mode} is coming soon. v0.1.0 supports --mode workers-ai only.`);
  }
  const answers = await resolveCreateOptions(flags);
  const targetDir = path.resolve(process.cwd(), answers.name);
  if (fssync.existsSync(targetDir)) {
    if (!flags.force) throw new CliError(`Target directory already exists: ${targetDir}\nNext: rerun with --force or choose --name`);
    await fs.rm(targetDir, { recursive: true, force: true });
  }

  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'create-pancake-search-'));
  try {
    const config = makeConfig(answers);
    await writeProject(tmpDir, config);
    await buildAssets(tmpDir, config);
    await fs.rename(tmpDir, targetDir);
    console.log(`Scaffolded ${targetDir}`);
    console.log(`Rebuild: cd ${answers.name} && npm run reindex`);
    if (answers.deploy) await deployProject(targetDir);
  } catch (error) {
    await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
    throw error;
  }
}

async function resolveCreateOptions(flags) {
  if (flags.yes && (!flags.name || !flags.source)) {
    throw new CliError('--yes requires --name and --source');
  }
  const rl = flags.yes ? null : readline.createInterface({ input, output });
  try {
    const name = flags.name || await rl.question('Project name: ');
    const source = flags.source || await rl.question('Path / URL: ');
    const deploy = flags.deploy !== undefined
      ? !!flags.deploy
      : flags.yes ? false : /^y/i.test(await rl.question('Deploy to Cloudflare when done? (y/N) '));
    if (!name || !source) throw new CliError('Project name and source are required');
    return {
      name,
      source,
      deploy,
      model: flags.model || 'bge-small-en-v1.5',
      maxPages: parsePositiveInt(flags['max-pages'] || '500', '--max-pages'),
      include: flags.include || ['**/*.{md,mdx,html,txt}'],
      exclude: flags.exclude || ['**/node_modules/**', '**/.git/**', '**/dist/**', '**/build/**'],
    };
  } finally {
    rl?.close();
  }
}

function makeConfig(options) {
  const model = MODEL_MAP[options.model];
  if (!model) throw new CliError(`Unsupported --model ${options.model}. Supported: ${Object.keys(MODEL_MAP).join(', ')}`);
  const isUrl = /^https?:\/\//i.test(options.source);
  return {
    $schema: 'https://raw.githubusercontent.com/mcn92/pancake/main/schemas/v1/pancake.config.schema.json',
    version: 1,
    name: path.basename(options.name),
    source: isUrl
      ? { type: 'url', url: options.source, maxPages: options.maxPages }
      : { type: 'folder', path: path.relative(path.resolve(process.cwd(), options.name), path.resolve(process.cwd(), options.source)) || '.', include: options.include, exclude: options.exclude },
    chunking: { ...DEFAULT_CONFIG.chunking },
    embedding: { ...DEFAULT_CONFIG.embedding, buildModel: options.model, dims: model.dims },
    index: { ...DEFAULT_CONFIG.index },
    validation: { ...DEFAULT_CONFIG.validation },
  };
}

async function rebuildProject(projectDir, flags = {}) {
  const configPath = path.join(projectDir, 'pancake.config.json');
  const config = JSON.parse(await fs.readFile(configPath, 'utf8'));
  validateConfig(config);
  await buildAssets(projectDir, config, { verbose: !!flags.verbose });
  console.log('Rebuilt Pancake search assets.');
}

async function writeProject(projectDir, config) {
  await fs.mkdir(path.join(projectDir, 'assets'), { recursive: true });
  await fs.mkdir(path.join(projectDir, '.github', 'workflows'), { recursive: true });
  await fs.writeFile(path.join(projectDir, 'pancake.config.json'), `${JSON.stringify(config, null, 2)}\n`);
  await copyTemplate('worker.js', path.join(projectDir, 'worker.js'));
  await copyTemplate('ui.html', path.join(projectDir, 'ui.html'));
  await copyTemplate('README.generated.md', path.join(projectDir, 'README.md'));
  await copyTemplate('reindex.yml', path.join(projectDir, '.github', 'workflows', 'reindex.yml'));
  await fs.writeFile(path.join(projectDir, 'wrangler.toml'), wranglerToml(config));
  await fs.writeFile(path.join(projectDir, 'package.json'), generatedPackageJson(config));
  await fs.writeFile(path.join(projectDir, '.gitignore'), 'node_modules\n.wrangler\n.pancake/last-build.log\n');
}

async function copyTemplate(name, dest) {
  await fs.copyFile(path.join(PACKAGE_ROOT, 'templates', name), dest);
}

function wranglerToml(config) {
  return `name = "${tomlString(config.name)}"
main = "worker.js"
compatibility_date = "2025-04-09"
compatibility_flags = ["nodejs_compat"]

[ai]
binding = "AI"

[vars]
READ_ONLY = "1"
RATE_LIMIT_RPM = "120"
MAX_JSON_BYTES = "262144"
MAX_QUERY_CHARS = "4096"

[[rules]]
type = "Data"
globs = ["**/*.pnck"]
fallthrough = true

[[rules]]
type = "Text"
globs = ["ui.html"]
fallthrough = true
`;
}

function generatedPackageJson(config) {
  return `${JSON.stringify({
    name: config.name,
    version: '0.1.0',
    private: true,
    type: 'module',
    scripts: {
      dev: 'wrangler dev',
      deploy: 'wrangler deploy',
      reindex: 'create-pancake-search rebuild --yes',
    },
    dependencies: {
      'pancake-wasm': '^0.2.0',
    },
    devDependencies: {
      'create-pancake-search': '0.1.0',
      wrangler: '^4.81.1',
    },
  }, null, 2)}\n`;
}

function tomlString(value) {
  return String(value).replace(/[^a-zA-Z0-9_-]/g, '-').replace(/^-+|-+$/g, '') || 'pancake-search';
}

async function buildAssets(projectDir, config) {
  validateConfig(config);
  const logLines = [];
  const log = (msg) => {
    logLines.push(msg);
    console.log(msg);
  };
  const sourceRoot = config.source.type === 'folder'
    ? path.resolve(projectDir, config.source.path)
    : config.source.url;
  const docs = config.source.type === 'folder'
    ? await ingestFolder(sourceRoot, config.source, log)
    : await ingestUrl(config.source, log);
  if (docs.length === 0) throw new CliError('Ingest produced 0 documents. Next: check --source/include/exclude.', 1);
  const chunks = dedupeChunks(chunkDocs(docs, config.chunking));
  if (chunks.length === 0) throw new CliError('Chunking produced 0 chunks. Next: use longer content or adjust source filters.', 1);
  log(`Ingested ${docs.length} docs -> ${chunks.length} chunks`);

  const vectors = await embedChunks(chunks, config.embedding, log);
  const Pancake = await loadPancake();
  const maxElements = Math.max(chunks.length, Math.ceil(chunks.length * 1.25));
  const index = await Pancake.create({ ...config.index, dim: config.embedding.dims, maxElements });
  let snapshot;
  try {
    index.addBatch(vectors);
    validateSelfRecall(index, chunks, vectors, log);
    snapshot = index.export();
  } finally {
    index.dispose();
  }

  const assetsDir = path.join(projectDir, 'assets');
  await fs.mkdir(assetsDir, { recursive: true });
  const manifest = makeManifest(config, chunks, snapshot, vectors);
  await fs.writeFile(path.join(assetsDir, 'snapshot.pnck'), snapshot);
  await fs.writeFile(path.join(assetsDir, 'corpus.json'), `${JSON.stringify(chunks.map(publicChunk), null, 2)}\n`);
  await fs.writeFile(path.join(assetsDir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  await fs.mkdir(path.join(projectDir, '.pancake'), { recursive: true });
  await fs.writeFile(path.join(projectDir, '.pancake', 'last-build.log'), `${logLines.join('\n')}\n`);
  const gzipBytes = await projectedGzipBytes(projectDir);
  log(`Built index: ${(snapshot.byteLength / 1024 / 1024).toFixed(2)} MB snapshot`);
  log(`Projected bundled gzip size: ${(gzipBytes / 1024 / 1024).toFixed(2)} MB`);
  if (gzipBytes > 3 * 1024 * 1024) {
    throw new CliError('Projected Worker bundle exceeds the free-plan 3 MB compressed limit. Next: reduce source scope or wait for the R2-backed tier.', 2);
  }
}

function validateConfig(config) {
  if (!config || config.version !== 1) throw new CliError('pancake.config.json version must be 1', 1);
  if (!config.name) throw new CliError('pancake.config.json name is required', 1);
  if (!config.source || !['folder', 'url'].includes(config.source.type)) throw new CliError('source.type must be folder or url', 1);
  if (config.embedding?.mode !== 'workers-ai') throw new CliError('embedding.mode must be workers-ai', 1);
  const model = MODEL_MAP[config.embedding?.buildModel];
  if (!model) throw new CliError(`Unsupported embedding.buildModel ${config.embedding?.buildModel}`, 1);
  if (config.embedding.dims !== model.dims) throw new CliError(`embedding.dims must be ${model.dims}`, 1);
}

async function ingestFolder(root, source, log) {
  const docs = [];
  const files = await walk(root);
  for (const file of files) {
    const rel = path.relative(root, file).split(path.sep).join('/');
    if (!matchesSource(rel, source)) continue;
    try {
      const text = await fs.readFile(file, 'utf8');
      const extracted = extractByExtension(text, file);
      if (extracted.text.trim()) docs.push({ id: docs.length, sourcePath: rel, title: extracted.title || path.basename(file), text: extracted.text });
    } catch (error) {
      log(`warn: skipped unreadable file ${rel}: ${error.message}`);
    }
  }
  return docs;
}

async function walk(root) {
  const out = [];
  const entries = await fs.readdir(root, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(root, entry.name);
    if (entry.isDirectory()) out.push(...await walk(full));
    else if (entry.isFile()) out.push(full);
  }
  return out;
}

function matchesSource(rel, source) {
  const includes = source.include || ['**/*.{md,mdx,html,txt}'];
  const excludes = source.exclude || [];
  return includes.some((glob) => globMatch(rel, glob)) && !excludes.some((glob) => globMatch(rel, glob));
}

function globMatch(file, glob) {
  const normalized = glob.replace(/\\/g, '/');
  if (normalized.includes('{md,mdx,html,txt}')) return /\.(md|mdx|html|txt)$/i.test(file);
  if (normalized.startsWith('**/') && normalized.endsWith('/**')) return file.includes(normalized.slice(3, -3));
  if (normalized.startsWith('**/')) return file.endsWith(normalized.slice(3)) || file.includes(normalized.slice(3).replace('/**', ''));
  return file === normalized || file.startsWith(`${normalized}/`);
}

async function ingestUrl(source, log) {
  const seed = new URL(source.url);
  const seen = new Set();
  const queue = [seed.href];
  const docs = [];
  while (queue.length && docs.length < (source.maxPages || 500)) {
    const href = queue.shift();
    if (seen.has(href)) continue;
    seen.add(href);
    let response;
    try {
      response = await fetch(href, { headers: { 'User-Agent': 'create-pancake-search/0.1.0' }, signal: AbortSignal.timeout(15000) });
    } catch (error) {
      log(`warn: skipped ${href}: ${error.message}`);
      continue;
    }
    const type = response.headers.get('content-type') || '';
    if (!response.ok || !type.includes('text/html')) continue;
    const html = await response.text();
    const extracted = extractHtml(html);
    docs.push({ id: docs.length, url: href, title: extracted.title || href, text: extracted.text });
    for (const link of extractLinks(html, href, seed.origin)) {
      if (!seen.has(link) && queue.length + docs.length < (source.maxPages || 500)) queue.push(link);
    }
  }
  return docs;
}

function extractByExtension(text, file) {
  if (/\.html?$/i.test(file)) return extractHtml(text);
  const title = (text.match(/^#\s+(.+)$/m) || [])[1];
  return { title, text: stripMarkdown(text) };
}

function extractHtml(html) {
  const title = decodeEntities((html.match(/<title[^>]*>([\s\S]*?)<\/title>/i) || [])[1] || '');
  let body = (html.match(/<(main|article)[^>]*>([\s\S]*?)<\/\1>/i) || [])[2]
    || (html.match(/<body[^>]*>([\s\S]*?)<\/body>/i) || [])[1]
    || html;
  body = body
    .replace(/<(script|style|nav|header|footer|aside)[^>]*>[\s\S]*?<\/\1>/gi, ' ')
    .replace(/<h([1-3])[^>]*>/gi, '\n### ')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/p>/gi, '\n\n')
    .replace(/<[^>]+>/g, ' ');
  return { title, text: normalizeText(decodeEntities(body)) };
}

function extractLinks(html, baseHref, origin) {
  const links = [];
  for (const match of html.matchAll(/href=["']([^"']+)["']/gi)) {
    try {
      const url = new URL(match[1], baseHref);
      url.hash = '';
      if (url.origin === origin && /^https?:$/.test(url.protocol)) links.push(url.href);
    } catch {}
  }
  return links;
}

function stripMarkdown(text) {
  return normalizeText(text
    .replace(/```[\s\S]*?```/g, (block) => `\n${block}\n`)
    .replace(/!\[[^\]]*]\([^)]*\)/g, ' ')
    .replace(/\[([^\]]+)]\([^)]*\)/g, '$1')
    .replace(/[#>*_`~-]/g, ' '));
}

function decodeEntities(text) {
  return String(text || '')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

function normalizeText(text) {
  return String(text || '').replace(/\r/g, '').replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
}

function chunkDocs(docs, options) {
  const chunks = [];
  for (const doc of docs) {
    const tokens = tokenize(doc.text);
    if (tokens.length < 25) continue;
    const target = options.targetTokens || 256;
    const overlap = Math.floor(target * ((options.overlapPercent || 15) / 100));
    for (let start = 0; start < tokens.length; start += Math.max(1, target - overlap)) {
      const slice = tokens.slice(start, start + target);
      if (slice.length < 25 && chunks.length) {
        chunks[chunks.length - 1].text += ` ${slice.join(' ')}`;
        continue;
      }
      chunks.push({
        id: chunks.length,
        docId: doc.id,
        title: doc.title,
        headingPath: [],
        url: doc.url || doc.sourcePath,
        anchor: '',
        sourcePath: doc.sourcePath || doc.url,
        text: slice.join(' '),
      });
      if (start + target >= tokens.length) break;
    }
  }
  return chunks;
}

function tokenize(text) {
  return normalizeText(text).split(/\s+/).filter(Boolean);
}

function dedupeChunks(chunks) {
  const seen = new Set();
  const out = [];
  for (const chunk of chunks) {
    const key = chunk.text;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ ...chunk, id: out.length });
  }
  return out;
}

async function embedChunks(chunks, embeddingConfig, log) {
  if (process.env.PANCAKE_SEARCH_STUB_EMBEDDINGS === '1') {
    log('Embedding with deterministic local stub (PANCAKE_SEARCH_STUB_EMBEDDINGS=1)');
    return chunks.map((chunk) => hashEmbedding(chunk.text, embeddingConfig.dims));
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

async function loadPancake() {
  try {
    const mod = await import('pancake-wasm');
    return mod.default;
  } catch {
    const mod = await import(pathToFileURL(path.join(REPO_ROOT, 'pancake.node.mjs')).href);
    return mod.default;
  }
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

function publicChunk(chunk) {
  return {
    id: chunk.id,
    docId: chunk.docId,
    title: chunk.title,
    headingPath: chunk.headingPath,
    url: chunk.url,
    anchor: chunk.anchor,
    sourcePath: chunk.sourcePath,
    text: chunk.text,
    preview: chunk.text.slice(0, 220),
  };
}

function makeManifest(config, chunks, snapshot, vectors) {
  const model = MODEL_MAP[config.embedding.buildModel];
  const firstVectorHash = vectors[0]
    ? crypto.createHash('sha256').update(Buffer.from(vectors[0].buffer, vectors[0].byteOffset, vectors[0].byteLength)).digest('hex')
    : null;
  return {
    version: 1,
    generatedAt: new Date().toISOString(),
    cliVersion: '0.1.0',
    name: config.name,
    model: config.embedding.buildModel,
    workersAiModel: model.workersAiModel,
    dims: config.embedding.dims,
    dim: config.embedding.dims,
    metric: config.index.metric,
    quantized: config.index.quantized,
    M: config.index.M,
    efConstruction: config.index.efConstruction,
    efSearch: config.index.efSearch,
    maxElements: Math.max(chunks.length, Math.ceil(chunks.length * 1.25)),
    chunkCount: chunks.length,
    prefixPolicy: config.embedding.prefixPolicy || DEFAULT_CONFIG.embedding.prefixPolicy,
    pooling: config.embedding.pooling || 'mean',
    normalize: config.embedding.normalize !== false,
    maxInputTokens: model.maxInputTokens,
    maxQueryChars: 4096,
    firstVectorSha256: firstVectorHash,
    snapshotSha256: crypto.createHash('sha256').update(snapshot).digest('hex'),
    configHash: crypto.createHash('sha256').update(JSON.stringify(config)).digest('hex'),
  };
}

async function projectedGzipBytes(projectDir) {
  const files = [
    'worker.js',
    'ui.html',
    'assets/snapshot.pnck',
    'assets/corpus.json',
    'assets/manifest.json',
  ];
  const buffers = [];
  for (const file of files) buffers.push(await fs.readFile(path.join(projectDir, file)));
  return zlib.gzipSync(Buffer.concat(buffers)).byteLength;
}

async function deployProject(projectDir) {
  const who = spawnSync('npx', ['wrangler', 'whoami'], { cwd: projectDir, encoding: 'utf8' });
  if (who.status !== 0) {
    throw new CliError(`Cloudflare auth required. Next: cd ${projectDir} && npx wrangler login && npm run deploy`, 3);
  }
  const deploy = spawnSync('npx', ['wrangler', 'deploy'], { cwd: projectDir, encoding: 'utf8' });
  if (deploy.status !== 0) {
    throw new CliError(`Deploy failed.\n${deploy.stderr || deploy.stdout}\nNext: cd ${projectDir} && npm run deploy`, 3);
  }
  process.stdout.write(deploy.stdout);
}

function parsePositiveInt(value, name) {
  const parsed = parseInt(String(value), 10);
  if (!Number.isInteger(parsed) || parsed < 1) throw new CliError(`${name} must be a positive integer`);
  return parsed;
}
