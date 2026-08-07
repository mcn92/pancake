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
const STUDENT_TRAINER = path.join(PACKAGE_ROOT, 'tools', 'train_student.py');
const DEFAULT_PREFIX = 'Represent this sentence for searching relevant passages: ';
const CONFIG_SCHEMA_URL = 'https://raw.githubusercontent.com/mcn92/pancake/main/create-pancake-search/schemas/v1/pancake.config.schema.json';
const MAX_CRAWL_BODY_BYTES = 2 * 1024 * 1024;
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
  runtime: { mode: 'snapshot', storage: 'bundled' },
  validation: { minRecallAt10: 0.98 },
});
const RANGE_ARTIFACT_MAGIC = 0x31415250;
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
  --mode workers-ai     Query embedding mode
  --model <id>          bge-small-en-v1.5
  --max-pages <n>       URL crawl cap
  --include <glob>      Folder include glob, repeatable
  --exclude <glob>      Folder exclude glob, repeatable
  --runtime snapshot|artifact
  --artifact <file>     Optional prebuilt .pancake-range artifact for --runtime artifact
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
    await buildAssets(tmpDir, config, { sourceBaseDir: targetDir });
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
      runtime: flags.runtime || (flags.artifact ? 'artifact' : 'snapshot'),
      artifact: flags.artifact || null,
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
  if (!['snapshot', 'artifact'].includes(options.runtime)) {
    throw new CliError(`--runtime must be snapshot or artifact, got ${options.runtime}`);
  }
  const isUrl = /^https?:\/\//i.test(options.source);
  const targetDir = path.resolve(process.cwd(), options.name);
  const artifactPath = options.artifact
    ? path.relative(targetDir, path.resolve(process.cwd(), options.artifact)) || '.'
    : null;
  return {
    $schema: CONFIG_SCHEMA_URL,
    version: 1,
    name: path.basename(options.name),
    source: isUrl
      ? { type: 'url', url: options.source, maxPages: options.maxPages }
      : { type: 'folder', path: path.relative(path.resolve(process.cwd(), options.name), path.resolve(process.cwd(), options.source)) || '.', include: options.include, exclude: options.exclude },
    chunking: { ...DEFAULT_CONFIG.chunking },
    embedding: { ...DEFAULT_CONFIG.embedding, buildModel: options.model, dims: model.dims },
    index: { ...DEFAULT_CONFIG.index },
    runtime: options.runtime === 'artifact'
      ? { mode: 'artifact', storage: 'bundled', ...(artifactPath ? { artifactPath } : {}) }
      : { ...DEFAULT_CONFIG.runtime },
    validation: { ...DEFAULT_CONFIG.validation },
  };
}

async function rebuildProject(projectDir, flags = {}) {
  const configPath = path.join(projectDir, 'pancake.config.json');
  const config = JSON.parse(await fs.readFile(configPath, 'utf8'));
  applyRuntimeOverrides(config, projectDir, flags);
  validateConfig(config);
  await copyTemplate(config.runtime?.mode === 'artifact' ? 'worker.artifact.js' : 'worker.js', path.join(projectDir, 'worker.js'));
  await fs.writeFile(path.join(projectDir, 'wrangler.toml'), wranglerToml(config));
  await buildAssets(projectDir, config, { verbose: !!flags.verbose });
  await fs.writeFile(configPath, `${JSON.stringify(config, null, 2)}\n`);
  console.log('Rebuilt Pancake search assets.');
}

export async function buildSearchAssets(projectDir, config, options = {}) {
  await buildAssets(projectDir, config, options);
}

function applyRuntimeOverrides(config, projectDir, flags) {
  if (!flags.runtime && !flags.artifact) return;
  const runtime = flags.runtime || (flags.artifact ? 'artifact' : config.runtime?.mode || 'snapshot');
  if (!['snapshot', 'artifact'].includes(runtime)) {
    throw new CliError(`--runtime must be snapshot or artifact, got ${runtime}`);
  }
  if (runtime === 'artifact') {
    const artifactPath = flags.artifact
      ? path.relative(projectDir, path.resolve(process.cwd(), flags.artifact)) || '.'
      : config.runtime?.artifactPath;
    config.runtime = { mode: 'artifact', storage: 'bundled', ...(artifactPath ? { artifactPath } : {}) };
  } else {
    config.runtime = { ...DEFAULT_CONFIG.runtime };
  }
}

async function writeProject(projectDir, config) {
  await fs.mkdir(path.join(projectDir, 'assets'), { recursive: true });
  await fs.mkdir(path.join(projectDir, '.github', 'workflows'), { recursive: true });
  await fs.writeFile(path.join(projectDir, 'pancake.config.json'), `${JSON.stringify(config, null, 2)}\n`);
  await copyTemplate(config.runtime?.mode === 'artifact' ? 'worker.artifact.js' : 'worker.js', path.join(projectDir, 'worker.js'));
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
LOCAL_STUB_AI = "0"

[[rules]]
type = "Data"
globs = ["**/*.pnck", "**/*.pancake-range"]
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

async function buildAssets(projectDir, config, options = {}) {
  validateConfig(config);
  const logLines = [];
  const log = (msg) => {
    logLines.push(msg);
    console.log(msg);
  };
  const sourceRoot = config.source.type === 'folder'
    ? path.resolve(options.sourceBaseDir || projectDir, config.source.path)
    : config.source.url;
  const docs = config.source.type === 'folder'
    ? await ingestFolder(sourceRoot, config.source, log)
    : await ingestUrl(config.source, log);
  if (docs.length === 0) {
    const detail = config.source.type === 'folder'
      ? `resolved path ${sourceRoot}; include ${JSON.stringify(config.source.include || ['**/*.{md,mdx,html,txt}'])}; exclude ${JSON.stringify(config.source.exclude || [])}`
      : `seed URL ${sourceRoot}; maxPages ${config.source.maxPages || 500}`;
    throw new CliError(`Ingest produced 0 documents from ${detail}. Next: check --source/include/exclude and ensure files contain enough text.`, 1);
  }
  const chunks = dedupeChunks(chunkDocs(docs, config.chunking));
  if (chunks.length === 0) throw new CliError('Chunking produced 0 chunks. Next: use longer content or adjust source filters.', 1);
  log(`Ingested ${docs.length} docs -> ${chunks.length} chunks`);

  const vectors = await embedChunks(chunks, config.embedding, log, projectDir);
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

  const assetsDir = options.assetsDir
    ? path.resolve(options.assetsDir)
    : path.join(projectDir, 'assets');
  await fs.mkdir(assetsDir, { recursive: true });
  const artifactPath = config.runtime?.mode === 'artifact' && config.runtime.artifactPath
    ? path.resolve(projectDir, config.runtime.artifactPath)
    : null;
  let artifact = null;
  let artifactInfo = null;
  if (config.runtime?.mode === 'artifact') {
    if (artifactPath && !fssync.existsSync(artifactPath)) {
      throw new CliError(`Configured Search Artifact not found: ${artifactPath}\nNext: update runtime.artifactPath in pancake.config.json or rebuild with --artifact <file>.`, 2);
    }
    if (artifactPath) {
      artifact = await fs.readFile(artifactPath);
      artifactInfo = inspectRangeArtifactHeader(artifact, artifactPath);
      if (artifactInfo.dim !== config.embedding.dims) {
        throw new CliError(`Search Artifact dimension mismatch (${artifactInfo.dim} !== ${config.embedding.dims}). Next: supply an artifact built from the same embedding model.`, 2);
      }
      if (artifactInfo.count !== chunks.length) {
        throw new CliError(`Search Artifact count mismatch (${artifactInfo.count} !== ${chunks.length}). Next: rebuild the artifact from this generated corpus.`, 2);
      }
    }
  }
  const manifest = makeManifest(config, chunks, snapshot, vectors, artifact, artifactInfo);
  if (config.runtime?.mode === 'artifact') {
    const outPath = path.join(assetsDir, 'index.pancake-range');
    if (artifact) {
      await fs.writeFile(outPath, artifact);
    } else {
      const buildManifest = Pancake.buildRangeArtifact(snapshot, outPath, { layout: 'rcm' });
      artifactInfo = {
        version: buildManifest.formatVersion,
        kind: 1,
        dim: buildManifest.graph.dim,
        count: buildManifest.graph.count,
        entryPoint: buildManifest.graph.entryPoint,
        maxLevel: buildManifest.graph.maxLevel,
        M: buildManifest.graph.M,
        M0: buildManifest.graph.M0,
        metric: config.index.metric === 'l2' ? 0 : 1,
        recordBytes: buildManifest.addressing.recordBytes,
        parts: 0,
        routerCount: buildManifest.addressing.routerCount,
        baseCount: buildManifest.addressing.baseCount,
      };
      artifact = await fs.readFile(outPath);
    }
    manifest.artifactSha256 = crypto.createHash('sha256').update(artifact).digest('hex');
    manifest.artifact = artifactInfo;
  } else {
    await fs.writeFile(path.join(assetsDir, 'snapshot.pnck'), snapshot);
  }
  await fs.writeFile(path.join(assetsDir, 'corpus.json'), `${JSON.stringify(chunks.map(publicChunk), null, 2)}\n`);
  await fs.writeFile(path.join(assetsDir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  await fs.mkdir(path.join(projectDir, '.pancake'), { recursive: true });
  await fs.writeFile(path.join(projectDir, '.pancake', 'last-build.log'), `${logLines.join('\n')}\n`);
  const gzipBytes = options.skipBundleSizeCheck ? null : await projectedGzipBytes(projectDir);
  if (config.runtime?.mode === 'artifact') {
    log(`Built corpus assets with ${(artifact.byteLength / 1024 / 1024).toFixed(2)} MB Search Artifact`);
  } else {
    log(`Built index: ${(snapshot.byteLength / 1024 / 1024).toFixed(2)} MB snapshot`);
  }
  if (gzipBytes !== null) log(`Projected bundled gzip size: ${(gzipBytes / 1024 / 1024).toFixed(2)} MB`);
  if (gzipBytes !== null && gzipBytes > 3 * 1024 * 1024) {
    throw new CliError('Projected Worker bundle exceeds the free-plan 3 MB compressed limit. Next: reduce source scope or wait for the R2-backed tier.', 2);
  }
}

function validateConfig(config) {
  if (!config || config.version !== 1) throw new CliError('pancake.config.json version must be 1', 1);
  if (!config.name) throw new CliError('pancake.config.json name is required', 1);
  if (!config.source || !['folder', 'url'].includes(config.source.type)) throw new CliError('source.type must be folder or url', 1);
  const embeddingMode = config.embedding?.mode || 'workers-ai';
  if (!['workers-ai', 'student'].includes(embeddingMode)) throw new CliError('embedding.mode must be workers-ai or student', 1);
  if (embeddingMode === 'workers-ai') {
    const model = MODEL_MAP[config.embedding?.buildModel];
    if (!model) throw new CliError(`Unsupported embedding.buildModel ${config.embedding?.buildModel}`, 1);
    if (config.embedding.dims !== model.dims) throw new CliError(`embedding.dims must be ${model.dims}`, 1);
  } else if (!config.embedding.studentModelPath) {
    throw new CliError('embedding.studentModelPath is required when embedding.mode is student', 1);
  } else if (!config.embedding.trainStudent?.enabled && !config.embedding.teacherVectorsPath) {
    throw new CliError('embedding.teacherVectorsPath is required when embedding.mode is student and trainStudent is disabled', 1);
  }
  const runtimeMode = config.runtime?.mode || 'snapshot';
  if (!['snapshot', 'artifact'].includes(runtimeMode)) throw new CliError('runtime.mode must be snapshot or artifact', 1);
  if (runtimeMode === 'artifact' && config.runtime?.storage !== 'bundled') throw new CliError('runtime.storage must be bundled for artifact mode in this release', 1);
}

async function ingestFolder(root, source, log) {
  const docs = [];
  if (!fssync.existsSync(root)) {
    throw new CliError(`Source folder not found: ${root}\nNext: check --source or source.path in pancake.config.json.`, 1);
  }
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
  let entries;
  try {
    entries = await fs.readdir(root, { withFileTypes: true });
  } catch (error) {
    throw new CliError(`Unable to read source folder ${root}: ${error.message}`, 1);
  }
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
  if (normalized.startsWith('**/*.')) return file.toLowerCase().endsWith(normalized.slice(4).toLowerCase());
  if (normalized.startsWith('*.')) return !file.includes('/') && file.toLowerCase().endsWith(normalized.slice(1).toLowerCase());
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
      response = await fetch(href, {
        headers: { 'User-Agent': 'create-pancake-search/0.1.0' },
        redirect: 'manual',
        signal: AbortSignal.timeout(15000),
      });
    } catch (error) {
      log(`warn: skipped ${href}: ${error.message}`);
      continue;
    }
    if (response.status >= 300 && response.status < 400) {
      log(`warn: skipped redirect from ${href}`);
      continue;
    }
    const type = response.headers.get('content-type') || '';
    if (!response.ok || !type.includes('text/html')) continue;
    let html;
    try {
      html = await readLimitedText(response, MAX_CRAWL_BODY_BYTES);
    } catch (error) {
      log(`warn: skipped ${href}: ${error.message}`);
      continue;
    }
    const extracted = extractHtml(html);
    docs.push({ id: docs.length, url: href, title: extracted.title || href, text: extracted.text });
    for (const link of extractLinks(html, href, seed.origin)) {
      if (!seen.has(link) && queue.length + docs.length < (source.maxPages || 500)) queue.push(link);
    }
  }
  return docs;
}

async function readLimitedText(response, maxBytes) {
  const declaredLength = Number(response.headers.get('content-length') || '0');
  if (Number.isFinite(declaredLength) && declaredLength > maxBytes) {
    throw new Error(`response body exceeds ${maxBytes} bytes`);
  }

  const reader = response.body?.getReader?.();
  if (!reader) {
    const text = await response.text();
    if (Buffer.byteLength(text, 'utf8') > maxBytes) throw new Error(`response body exceeds ${maxBytes} bytes`);
    return text;
  }

  const chunks = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > maxBytes) {
      await reader.cancel().catch(() => {});
      throw new Error(`response body exceeds ${maxBytes} bytes`);
    }
    chunks.push(value);
  }
  return Buffer.concat(chunks.map((chunk) => Buffer.from(chunk))).toString('utf8');
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

async function embedChunks(chunks, embeddingConfig, log, projectDir) {
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

  log(`Training corpus-specific student encoder with ${python}`);
  const result = spawnSync(python, args, { cwd: projectDir, stdio: 'inherit' });
  if (result.error) {
    throw new CliError(`Failed to start student trainer: ${result.error.message}`, 2);
  }
  if (result.status !== 0) {
    throw new CliError(`Student trainer failed with exit code ${result.status}. Next: install torch/transformers or provide a corpus-specific studentModel.`, 2);
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

function inspectRangeArtifactHeader(bytes, label = 'Search Artifact') {
  if (!bytes || bytes.byteLength < 56) {
    throw new CliError(`${label} is too small to be a Pancake Search Artifact`, 2);
  }
  const magic = bytes.readUInt32LE(0);
  if (magic !== RANGE_ARTIFACT_MAGIC) {
    throw new CliError(`${label} is not a Pancake Search Artifact`, 2);
  }
  const version = bytes.readUInt32LE(4);
  const kind = bytes.readUInt32LE(8);
  const dim = bytes.readUInt32LE(12);
  const count = bytes.readUInt32LE(16);
  const entryPoint = bytes.readUInt32LE(20);
  const maxLevel = bytes.readUInt32LE(24);
  const M = bytes.readUInt32LE(28);
  const M0 = bytes.readUInt32LE(32);
  const metric = bytes.readUInt32LE(36);
  const recordBytes = bytes.readUInt32LE(40);
  const parts = bytes.readUInt32LE(52);
  const routerCount = version >= 2 && bytes.byteLength >= 64 ? bytes.readUInt32LE(56) : 0;
  const baseCount = version >= 2 && bytes.byteLength >= 68 ? bytes.readUInt32LE(60) : count;
  return { version, kind, dim, count, entryPoint, maxLevel, M, M0, metric, recordBytes, parts, routerCount, baseCount };
}

function makeManifest(config, chunks, snapshot, vectors, artifact = null, artifactInfo = null) {
  const model = config.embedding.mode === 'student'
    ? null
    : MODEL_MAP[config.embedding.buildModel];
  const firstVectorHash = vectors[0]
    ? crypto.createHash('sha256').update(Buffer.from(vectors[0].buffer, vectors[0].byteOffset, vectors[0].byteLength)).digest('hex')
    : null;
  return {
    version: 1,
    generatedAt: new Date().toISOString(),
    cliVersion: '0.1.0',
    name: config.name,
    model: config.embedding.mode === 'student'
      ? 'pancake-distilled-student'
      : config.embedding.buildModel,
    workersAiModel: model?.workersAiModel || null,
    encoder: config.embedding.mode === 'student'
      ? {
          mode: 'student',
          format: 'pstu',
          studentModelPath: config.embedding.studentModelPath,
          teacherVectorsPath: config.embedding.teacherVectorsPath || null,
          trainedDuringBuild: config.embedding.trainStudent?.enabled === true,
        }
      : {
          mode: 'workers-ai',
          hfModel: model.hfModel,
          workersAiModel: model.workersAiModel,
        },
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
    maxInputTokens: model?.maxInputTokens || null,
    maxQueryChars: 4096,
    runtime: config.runtime || DEFAULT_CONFIG.runtime,
    firstVectorSha256: firstVectorHash,
    snapshotSha256: crypto.createHash('sha256').update(snapshot).digest('hex'),
    artifactSha256: artifact ? crypto.createHash('sha256').update(artifact).digest('hex') : null,
    artifact: artifactInfo,
    configHash: crypto.createHash('sha256').update(JSON.stringify(config)).digest('hex'),
  };
}

async function projectedGzipBytes(projectDir) {
  const files = [
    'worker.js',
    'ui.html',
    'assets/corpus.json',
    'assets/manifest.json',
  ];
  const artifactPath = path.join(projectDir, 'assets', 'index.pancake-range');
  files.push(fssync.existsSync(artifactPath) ? 'assets/index.pancake-range' : 'assets/snapshot.pnck');
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
