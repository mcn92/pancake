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
import {
  assemblePancakeFile,
  buildCorpusSegmentFromBuffers,
  buildInlineTransformerEncoderSegment,
  buildQueryInterpSegment,
  sha256,
} from './complete-profile.mjs';
import { createInlineTransformerEmbedder } from './inline-transformer.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PACKAGE_ROOT = path.resolve(__dirname, '..');
const REPO_ROOT = path.resolve(PACKAGE_ROOT, '..');
const STUDENT_TRAINER = path.join(PACKAGE_ROOT, 'tools', 'train_student.py');
const CLI_VERSION = JSON.parse(fssync.readFileSync(path.join(PACKAGE_ROOT, 'package.json'), 'utf8')).version;
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
  --mode <mode>         Query embedding mode: workers-ai (default) or student.
                        student distills a corpus-specific encoder at build
                        time (needs Python with torch/transformers) and embeds
                        queries inside the Worker with no AI binding.
  --student-model <f>   Pre-trained PSTU model; skips training (--mode student)
  --student-vectors <f> Teacher document vectors (.f32) matching --student-model
  --student-abstention <f>  Calibrated abstention scorer for --student-model
  --epochs <n>          Student training epochs (--mode student)
  --skip-abstention     Train without the abstention scorer (--mode student)
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
    if (name === 'deploy' || name === 'yes' || name === 'force' || name === 'skip-abstention') {
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
    await stageStudentInputs(tmpDir, answers);
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
    const mode = flags.mode || 'workers-ai';
    if (!['workers-ai', 'student'].includes(mode)) {
      throw new CliError(`--mode must be workers-ai or student, got ${mode}`);
    }
    if (mode !== 'student' && (flags['student-model'] || flags['student-vectors'] || flags['student-abstention'] || flags.epochs || flags['skip-abstention'])) {
      throw new CliError('--student-model, --student-vectors, --student-abstention, --epochs and --skip-abstention require --mode student');
    }
    if (flags['student-model'] && !flags['student-vectors']) {
      throw new CliError('--student-model requires --student-vectors so passage indexing uses matching teacher document vectors');
    }
    if (flags['student-vectors'] && !flags['student-model']) {
      throw new CliError('--student-vectors requires --student-model');
    }
    return {
      name,
      source,
      deploy,
      mode,
      studentModel: flags['student-model'] || null,
      studentVectors: flags['student-vectors'] || null,
      studentAbstention: flags['student-abstention'] || null,
      epochs: flags.epochs ? parsePositiveInt(flags.epochs, '--epochs') : null,
      skipAbstention: !!flags['skip-abstention'],
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
  const student = options.mode === 'student';
  const model = student ? null : MODEL_MAP[options.model];
  if (!student && !model) throw new CliError(`Unsupported --model ${options.model}. Supported: ${Object.keys(MODEL_MAP).join(', ')}`);
  if (!['snapshot', 'artifact', 'complete'].includes(options.runtime)) {
    throw new CliError(`--runtime must be snapshot, artifact, or complete, got ${options.runtime}`);
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
    embedding: student
      ? {
          mode: 'student',
          studentModelPath: options.studentModel ? 'student-model.bin' : 'student/student-model.bin',
          ...(options.studentVectors ? { teacherVectorsPath: 'docs-vectors.f32' } : {}),
          ...(options.studentModel
            ? {}
            : {
                trainStudent: {
                  enabled: true,
                  outDir: 'student',
                  ...(options.epochs ? { epochs: options.epochs } : {}),
                  ...(options.skipAbstention ? { skipAbstention: true } : {}),
                },
              }),
          dims: 384,
          prefixPolicy: { passage: '', query: '' },
          pooling: 'mean',
          normalize: true,
        }
      : { ...DEFAULT_CONFIG.embedding, buildModel: options.model, dims: model.dims },
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
  await writeRuntimeModules(projectDir, config);
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
  if (!['snapshot', 'artifact', 'complete'].includes(runtime)) {
    throw new CliError(`--runtime must be snapshot, artifact, or complete, got ${runtime}`);
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
  await writeRuntimeModules(projectDir, config);
  await copyTemplate('ui.html', path.join(projectDir, 'ui.html'));
  await copyTemplate('README.generated.md', path.join(projectDir, 'README.md'));
  await copyTemplate('reindex.yml', path.join(projectDir, '.github', 'workflows', 'reindex.yml'));
  await fs.writeFile(path.join(projectDir, 'wrangler.toml'), wranglerToml(config));
  await fs.writeFile(path.join(projectDir, 'wrangler.local.toml'), wranglerToml(config, { localStubAi: true }));
  await fs.writeFile(path.join(projectDir, 'package.json'), generatedPackageJson(config));
  await fs.writeFile(path.join(projectDir, '.gitignore'), 'node_modules\n.wrangler\n.pancake/last-build.log\n');
}

async function copyTemplate(name, dest) {
  await fs.copyFile(path.join(PACKAGE_ROOT, 'templates', name), dest);
}

async function writeRuntimeModules(projectDir, config) {
  const student = config.embedding?.mode === 'student';
  await copyTemplate(config.runtime?.mode === 'artifact' ? 'worker.artifact.js' : 'worker.js', path.join(projectDir, 'worker.js'));
  await copyTemplate(student ? 'encoder.student.js' : 'encoder.workers-ai.js', path.join(projectDir, 'encoder.js'));
  if (student) {
    await fs.copyFile(path.join(PACKAGE_ROOT, 'src', 'student-embedder.mjs'), path.join(projectDir, 'student-embedder.mjs'));
  }
}

async function stageStudentInputs(projectDir, options) {
  if (options.mode !== 'student' || !options.studentModel) return;
  const modelPath = path.resolve(process.cwd(), options.studentModel);
  if (!fssync.existsSync(modelPath)) throw new CliError(`Student model not found: ${modelPath}`);
  await fs.copyFile(modelPath, path.join(projectDir, 'student-model.bin'));
  const vectorsPath = path.resolve(process.cwd(), options.studentVectors);
  if (!fssync.existsSync(vectorsPath)) throw new CliError(`Teacher vectors not found: ${vectorsPath}`);
  await fs.copyFile(vectorsPath, path.join(projectDir, 'docs-vectors.f32'));
  if (options.studentAbstention) {
    const abstentionPath = path.resolve(process.cwd(), options.studentAbstention);
    if (!fssync.existsSync(abstentionPath)) throw new CliError(`Abstention model not found: ${abstentionPath}`);
    await fs.copyFile(abstentionPath, path.join(projectDir, 'student-abstention.json'));
  }
}

function wranglerToml(config, options = {}) {
  const student = config.embedding?.mode === 'student';
  const localStubAi = options.localStubAi === true;
  return `name = "${tomlString(config.name)}"
main = "worker.js"
compatibility_date = "2025-04-09"
compatibility_flags = ["nodejs_compat"]
${student || localStubAi ? '' : `
[ai]
binding = "AI"
`}
[vars]
READ_ONLY = "1"
RATE_LIMIT_RPM = "120"
MAX_JSON_BYTES = "262144"
MAX_QUERY_CHARS = "4096"
${student ? '' : `LOCAL_STUB_AI = "${localStubAi ? '1' : '0'}"\n`}
[[rules]]
type = "Data"
globs = ["**/*.pnck", "**/*.pancake-range", "**/*.bin"]
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
      dev: 'wrangler dev --config wrangler.local.toml',
      deploy: 'wrangler deploy',
      reindex: 'create-pancake-search rebuild --yes',
    },
    dependencies: {
      'pancake-wasm': '^0.2.0',
    },
    devDependencies: {
      'create-pancake-search': CLI_VERSION,
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
  const chunked = chunkDocs(docs, config.chunking);
  const chunks = dedupeChunks(chunked);
  if (chunks.length === 0) {
    const dropped = (chunked.dropped || [])
      .slice(0, 8)
      .map((doc) => `${doc.sourcePath || doc.url || doc.title || doc.id} (${doc.tokens} tokens)`)
      .join(', ');
    const suffix = dropped ? ` Dropped as too short: ${dropped}${chunked.dropped.length > 8 ? ', ...' : ''}.` : '';
    throw new CliError(`Chunking produced 0 chunks. Chunking counts whitespace-separated tokens and drops documents under 25 of them.${suffix} Next: use longer whitespace-delimited content or adjust source filters.`, 1);
  }
  log(`Ingested ${docs.length} docs -> ${chunks.length} chunks`);

  const vectors = await embedChunks(chunks, config, log, projectDir);
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
  if (config.runtime?.mode === 'complete') {
    artifactInfo = await buildCompleteArtifact({
      Pancake,
      projectDir,
      assetsDir,
      config,
      chunks,
      snapshot,
    });
    manifest.artifactSha256 = artifactInfo.sha256;
    manifest.artifact = artifactInfo;
    manifest.runtime = { ...manifest.runtime, mode: 'complete', artifactUrl: 'search.pancake' };
  } else if (config.runtime?.mode === 'artifact') {
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
  if (config.embedding.mode === 'student') {
    await publishStudentAssets(projectDir, config, assetsDir, manifest, log);
  }
  await fs.writeFile(path.join(assetsDir, 'corpus.json'), `${JSON.stringify(chunks.map(publicChunk), null, 2)}\n`);
  await fs.writeFile(path.join(assetsDir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  await fs.mkdir(path.join(projectDir, '.pancake'), { recursive: true });
  await fs.writeFile(path.join(projectDir, '.pancake', 'last-build.log'), `${logLines.join('\n')}\n`);
  const gzipBytes = options.skipBundleSizeCheck ? null : await projectedGzipBytes(projectDir);
  if (config.runtime?.mode === 'complete') {
    log(`Built complete .pancake artifact with ${(artifactInfo.bytes / 1024 / 1024).toFixed(2)} MB`);
  } else if (config.runtime?.mode === 'artifact') {
    log(`Built corpus assets with ${(artifact.byteLength / 1024 / 1024).toFixed(2)} MB Search Artifact`);
  } else {
    log(`Built index: ${(snapshot.byteLength / 1024 / 1024).toFixed(2)} MB snapshot`);
  }
  if (gzipBytes !== null) log(`Projected bundled gzip size: ${(gzipBytes / 1024 / 1024).toFixed(2)} MB`);
  if (gzipBytes !== null && gzipBytes > 3 * 1024 * 1024) {
    throw new CliError('Projected Worker bundle exceeds the free-plan 3 MB compressed limit. Next: reduce source scope or wait for the R2-backed tier.', 2);
  }
}

async function buildCompleteArtifact({ Pancake, projectDir, assetsDir, config, chunks, snapshot }) {
  const artifactContract = await loadArtifactContract();
  const runtime = config.runtime || {};
  const { encoder, vocabPath, weightsPath } = resolveInlineEncoderInputs(config, projectDir);

  const declaration = inlineEncoderDeclaration(config, encoder);
  const inlineEncoderBytes = buildInlineTransformerEncoderSegment({
    declaration,
    vocabBytes: await fs.readFile(vocabPath),
    weightBytes: await fs.readFile(weightsPath),
  });
  const calibrationBytes = encoder.calibrationPath
    ? await fs.readFile(path.resolve(projectDir, encoder.calibrationPath))
    : Buffer.from(JSON.stringify({ kind: 'retrieval-signals-v1', asset: null, vocabBloomBase64: '' }), 'utf8');
  const sketch = artifactContract.buildSketchArtifactBytes(snapshot, {
    sketchDims: runtime.sketchDims,
    sketchBits: runtime.sketchBits,
    recommendedRerank: config.index.efSearch || 120,
  });
  const records = chunks.map((chunk) => Buffer.from(JSON.stringify(publicChunk(chunk)), 'utf8'));
  const evaluation = Buffer.from(JSON.stringify({
    kind: 'docs-site-build-v1',
    generatedAt: new Date().toISOString(),
    querySet: runtime.evaluation?.queries || [],
  }), 'utf8');
  const outPath = path.join(assetsDir, runtime.fileName || 'search.pancake');
  const result = assemblePancakeFile({
    profile: 'pancake-complete-v1',
    corpus: { records: chunks.length, provenance: { source: config.source.type, name: config.name } },
    dim: config.embedding.dims,
    metric: config.index.metric,
    encoder: {
      kind: 'inline-transformer-v1',
      model: declaration.model,
      pooling: declaration.pooling,
      normalized: declaration.normalized,
      maxTokens: declaration.maxTokens,
    },
    recommendedRerank: config.index.efSearch || 120,
    sampleQueries: runtime.sampleQueries || [],
  }, [
    { kind: 'index', bytes: Buffer.from(sketch.bytes) },
    { kind: 'corpus', bytes: buildCorpusSegmentFromBuffers(records) },
    { kind: 'query-interp', bytes: buildQueryInterpSegment(3, inlineEncoderBytes, calibrationBytes) },
    { kind: 'evaluation', bytes: evaluation },
  ], outPath);
  return {
    profile: 'pancake-complete-v1',
    kind: 'inline-transformer-v1',
    path: path.basename(outPath),
    bytes: result.fileBytes,
    sha256: sha256(fssync.readFileSync(outPath)).toString('hex'),
    identity: result.identity,
    segments: result.manifest.segments,
  };
}

function inlineEncoderDeclaration(config, encoder = {}) {
  return encoder.declaration || {
    kind: 'inline-transformer-v1',
    model: encoder.model || 'sentence-transformers/all-MiniLM-L6-v2',
    license: encoder.license || 'apache-2.0',
    attribution: encoder.attribution || 'sentence-transformers/all-MiniLM-L6-v2 (quantized derivative)',
    dim: config.embedding.dims,
    pooling: config.embedding.pooling || 'mean',
    normalized: config.embedding.normalize !== false,
    maxTokens: encoder.maxTokens || 128,
    prefixPolicy: {
      passage: config.embedding.prefixPolicy?.passage || '',
      query: config.embedding.prefixPolicy?.query || '',
    },
    layout: encoder.layout || { V: 30522, P: 512, T: 2, D: 384, F: 1536, L: 6, B: 64, H: 12 },
  };
}

function resolveInlineEncoderInputs(config, projectDir) {
  const encoder = config.runtime?.inlineEncoder || {};
  const resolveInput = (value, label) => {
    if (!value) throw new CliError(`runtime.inlineEncoder.${label} is required for complete kind-3 artifacts`, 1);
    return path.resolve(projectDir, value);
  };
  const vocabPath = resolveInput(encoder.vocabPath, 'vocabPath');
  const weightsPath = resolveInput(encoder.weightsPath, 'weightsPath');
  if (!fssync.existsSync(vocabPath)) throw new CliError(`Inline encoder vocab not found: ${vocabPath}`, 1);
  if (!fssync.existsSync(weightsPath)) throw new CliError(`Inline encoder weights not found: ${weightsPath}`, 1);
  return { encoder, vocabPath, weightsPath };
}

async function loadArtifactContract() {
  try {
    const mod = await import('pancake-wasm/artifact');
    const contract = mod.default || mod;
    if (typeof contract.buildSketchArtifactBytes === 'function') return contract;
  } catch {}
  const mod = await import(pathToFileURL(path.join(REPO_ROOT, 'pancake-artifact.js')).href);
  const contract = mod.default || mod;
  if (typeof contract.buildSketchArtifactBytes !== 'function') {
    throw new CliError('pancake-wasm/artifact does not expose buildSketchArtifactBytes; update pancake-wasm or use the monorepo root fallback.', 2);
  }
  return contract;
}

function validateConfig(config) {
  if (!config || config.version !== 1) throw new CliError('pancake.config.json version must be 1', 1);
  if (!config.name) throw new CliError('pancake.config.json name is required', 1);
  if (!config.source || !['folder', 'url'].includes(config.source.type)) throw new CliError('source.type must be folder or url', 1);
  const embeddingMode = config.embedding?.mode || 'workers-ai';
  if (!['workers-ai', 'student', 'inline-transformer'].includes(embeddingMode)) throw new CliError('embedding.mode must be workers-ai, student, or inline-transformer', 1);
  if (embeddingMode === 'workers-ai') {
    const model = MODEL_MAP[config.embedding?.buildModel];
    if (!model) throw new CliError(`Unsupported embedding.buildModel ${config.embedding?.buildModel}`, 1);
    if (config.embedding.dims !== model.dims) throw new CliError(`embedding.dims must be ${model.dims}`, 1);
  } else if (embeddingMode === 'student' && !config.embedding.studentModelPath) {
    throw new CliError('embedding.studentModelPath is required when embedding.mode is student', 1);
  } else if (embeddingMode === 'student' && !config.embedding.trainStudent?.enabled && !config.embedding.teacherVectorsPath) {
    throw new CliError('embedding.teacherVectorsPath is required when embedding.mode is student and trainStudent is disabled', 1);
  }
  const runtimeMode = config.runtime?.mode || 'snapshot';
  if (!['snapshot', 'artifact', 'complete'].includes(runtimeMode)) throw new CliError('runtime.mode must be snapshot, artifact, or complete', 1);
  if (embeddingMode === 'inline-transformer' && runtimeMode !== 'complete') {
    throw new CliError('embedding.mode inline-transformer requires runtime.mode complete: snapshot/artifact runtimes deploy the Workers AI query encoder, which cannot serve inline-transformer builds', 1);
  }
  if (runtimeMode === 'artifact' && config.runtime?.storage !== 'bundled') throw new CliError('runtime.storage must be bundled for artifact mode in this release', 1);
  if (runtimeMode === 'complete' && config.runtime?.profile !== 'kind3') throw new CliError('runtime.profile must be kind3 for complete mode', 1);
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
  chunks.dropped = [];
  for (const doc of docs) {
    const tokens = tokenize(doc.text);
    if (tokens.length < 25) {
      chunks.dropped.push({
        id: doc.id,
        title: doc.title,
        sourcePath: doc.sourcePath,
        url: doc.url,
        tokens: tokens.length,
      });
      continue;
    }
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
  const { encoder, vocabPath, weightsPath } = resolveInlineEncoderInputs(config, projectDir);
  const { default: createEncoder } = await import('./encoder-kernels/encoder.node.mjs');
  const declaration = inlineEncoderDeclaration(config, encoder);
  const embedder = await createInlineTransformerEmbedder({
    declaration,
    vocabText: await fs.readFile(vocabPath, 'utf8'),
    blob: await fs.readFile(weightsPath),
    createEncoder,
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

async function publishStudentAssets(projectDir, config, assetsDir, manifest, log) {
  const modelPath = path.resolve(projectDir, config.embedding.studentModelPath);
  if (!fssync.existsSync(modelPath)) {
    throw new CliError(`Student model was not produced: ${modelPath}. Next: rerun training or pass --student-model.`, 2);
  }
  const modelBytes = await fs.readFile(modelPath);
  await fs.writeFile(path.join(assetsDir, 'student-model.bin'), modelBytes);
  // The abstention scorer travels beside the model both in the trainer's out
  // dir and when staged externally. A `null` placeholder keeps the Worker's
  // static asset import valid when no calibrated scorer exists.
  const abstentionPath = path.join(path.dirname(modelPath), 'student-abstention.json');
  let abstention = null;
  if (fssync.existsSync(abstentionPath)) {
    const abstentionBytes = await fs.readFile(abstentionPath);
    await fs.writeFile(path.join(assetsDir, 'student-abstention.json'), abstentionBytes);
    abstention = {
      bytes: abstentionBytes.byteLength,
      sha256: crypto.createHash('sha256').update(abstentionBytes).digest('hex'),
    };
  } else {
    await fs.writeFile(path.join(assetsDir, 'student-abstention.json'), 'null\n');
  }
  manifest.encoder = {
    ...manifest.encoder,
    studentModelBytes: modelBytes.byteLength,
    studentModelSha256: crypto.createHash('sha256').update(modelBytes).digest('hex'),
    abstentionBytes: abstention?.bytes || null,
    abstentionSha256: abstention?.sha256 || null,
  };
  log(`Bundled student encoder (${(modelBytes.byteLength / 1024).toFixed(1)} KiB${abstention ? ', calibrated abstention' : ', no abstention scorer'})`);
}

function makeManifest(config, chunks, snapshot, vectors, artifact = null, artifactInfo = null) {
  const model = config.embedding.mode === 'student' || config.embedding.mode === 'inline-transformer'
    ? null
    : MODEL_MAP[config.embedding.buildModel];
  const firstVectorHash = vectors[0]
    ? crypto.createHash('sha256').update(Buffer.from(vectors[0].buffer, vectors[0].byteOffset, vectors[0].byteLength)).digest('hex')
    : null;
  return {
    version: 1,
    generatedAt: new Date().toISOString(),
    cliVersion: CLI_VERSION,
    name: config.name,
    model: config.embedding.mode === 'student'
      ? 'pancake-distilled-student'
      : config.embedding.mode === 'inline-transformer'
        ? config.runtime?.inlineEncoder?.model || 'sentence-transformers/all-MiniLM-L6-v2'
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
      : config.embedding.mode === 'inline-transformer'
        ? {
            mode: 'inline-transformer',
            kind: 'inline-transformer-v1',
            teacherVectorsPath: config.embedding.teacherVectorsPath,
            runtimeWeightsPath: config.runtime?.inlineEncoder?.weightsPath || null,
            runtimeVocabPath: config.runtime?.inlineEncoder?.vocabPath || null,
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
    'encoder.js',
    'ui.html',
    'assets/corpus.json',
    'assets/manifest.json',
  ];
  const completePath = path.join(projectDir, 'assets', 'search.pancake');
  const artifactPath = path.join(projectDir, 'assets', 'index.pancake-range');
  files.push(fssync.existsSync(completePath)
    ? 'assets/search.pancake'
    : fssync.existsSync(artifactPath) ? 'assets/index.pancake-range' : 'assets/snapshot.pnck');
  for (const studentFile of ['student-embedder.mjs', 'assets/student-model.bin', 'assets/student-abstention.json']) {
    if (fssync.existsSync(path.join(projectDir, studentFile))) files.push(studentFile);
  }
  const buffers = [];
  for (const file of files) buffers.push(await fs.readFile(path.join(projectDir, file)));
  return zlib.gzipSync(Buffer.concat(buffers)).byteLength;
}

async function deployProject(projectDir) {
  const who = runNpmSync(['exec', '--', 'wrangler', 'whoami'], { cwd: projectDir, encoding: 'utf8' });
  if (who.status !== 0) {
    throw new CliError(`Cloudflare auth required. Next: cd ${projectDir} && npx wrangler login && npm run deploy`, 3);
  }
  const deploy = runNpmSync(['exec', '--', 'wrangler', 'deploy'], { cwd: projectDir, encoding: 'utf8' });
  if (deploy.status !== 0) {
    throw new CliError(`Deploy failed.\n${deploy.stderr || deploy.stdout}\nNext: cd ${projectDir} && npm run deploy`, 3);
  }
  process.stdout.write(deploy.stdout);
}

function npmCliPath() {
  if (process.env.npm_execpath && process.env.npm_execpath.endsWith('.js')) {
    return { command: process.execPath, args: [process.env.npm_execpath], shell: false };
  }
  if (process.platform === 'win32') return { command: 'cmd.exe', args: ['/d', '/s', '/c', 'npm'], shell: false };
  return { command: 'npm', args: [], shell: false };
}

function runNpmSync(args, options) {
  const npm = npmCliPath();
  return spawnSync(npm.command, [...npm.args, ...args], { ...options, shell: npm.shell });
}

function parsePositiveInt(value, name) {
  const parsed = parseInt(String(value), 10);
  if (!Number.isInteger(parsed) || parsed < 1) throw new CliError(`${name} must be a positive integer`);
  return parsed;
}
