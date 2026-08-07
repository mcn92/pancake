import fs from 'node:fs/promises';
import fssync from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildSearchAssets } from '../src/cli.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));

function slugifyName(value) {
  return String(value || 'docusaurus')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'docusaurus';
}

function trimSlashes(value) {
  return String(value || '').replace(/^\/+|\/+$/g, '');
}

function joinSitePath(baseUrl, ...parts) {
  const base = String(baseUrl || '/');
  const prefix = base.endsWith('/') ? base : `${base}/`;
  const suffix = parts.map(trimSlashes).filter(Boolean).join('/');
  return `${prefix}${suffix}`;
}

function normalizeOptions(options = {}) {
  return {
    enabled: options.enabled !== false,
    assetBase: trimSlashes(options.assetBase || 'pancake-search'),
    mount: options.mount !== false,
    workDir: options.workDir || path.join('.docusaurus', 'pancake-search'),
    name: options.name,
    include: options.include || ['**/*.html'],
    exclude: options.exclude || [],
    stubEmbeddings: options.stubEmbeddings === true,
    studentModel: options.studentModel || null,
    studentVectors: options.studentVectors || null,
    studentAbstention: options.studentAbstention || null,
    trainStudent: {
      enabled: options.trainStudent !== false && !options.studentModel,
      python: options.trainStudent?.python,
      teacher: options.trainStudent?.teacher,
      teacherRevision: options.trainStudent?.teacherRevision,
      epochs: options.trainStudent?.epochs || 100,
      buckets: options.trainStudent?.buckets,
      hidden: options.trainStudent?.hidden,
      batchSize: options.trainStudent?.batchSize,
      learningRate: options.trainStudent?.learningRate,
      maxFeatures: options.trainStudent?.maxFeatures,
      seed: options.trainStudent?.seed,
    },
    chunking: {
      targetTokens: options.chunking?.targetTokens || 256,
      overlapPercent: options.chunking?.overlapPercent ?? 15,
    },
    index: {
      metric: 'cosine',
      quantized: options.index?.quantized !== false,
      M: options.index?.M || 16,
      efConstruction: options.index?.efConstruction || 200,
      efSearch: options.index?.efSearch || 120,
    },
  };
}

function makeConfig(context, options, workDir, outDir) {
  return {
    version: 1,
    name: options.name || `${slugifyName(context.siteConfig?.title || path.basename(context.siteDir))}-search`,
    source: {
      type: 'folder',
      path: path.relative(workDir, outDir) || '.',
      include: options.include,
      exclude: options.exclude,
    },
    chunking: options.chunking,
    embedding: {
      mode: 'student',
      studentModelPath: options.studentModel ? 'student-model.bin' : 'student/student-model.bin',
      ...(options.studentVectors ? { teacherVectorsPath: 'docs-vectors.f32' } : {}),
      ...(options.trainStudent.enabled ? { trainStudent: { ...options.trainStudent, outDir: 'student' } } : {}),
      dims: 384,
      prefixPolicy: { passage: '', query: '' },
      pooling: 'mean',
      normalize: true,
    },
    index: options.index,
    runtime: { mode: 'artifact', storage: 'bundled' },
    validation: { minRecallAt10: 0.98 },
  };
}

async function withOptionalStubEmbeddings(enabled, fn) {
  if (!enabled) return fn();
  const previous = process.env.PANCAKE_SEARCH_STUB_EMBEDDINGS;
  process.env.PANCAKE_SEARCH_STUB_EMBEDDINGS = '1';
  try {
    return await fn();
  } finally {
    if (previous === undefined) delete process.env.PANCAKE_SEARCH_STUB_EMBEDDINGS;
    else process.env.PANCAKE_SEARCH_STUB_EMBEDDINGS = previous;
  }
}

async function copyRuntimeManifest(assetDir, context, options, studentModelInfo) {
  const manifestPath = path.join(assetDir, 'manifest.json');
  const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  const assetUrlBase = joinSitePath(context.siteConfig?.baseUrl || '/', options.assetBase);
  manifest.docusaurus = {
    assetBase: options.assetBase,
    artifactUrl: `${assetUrlBase}/index.pancake-range`,
    corpusUrl: `${assetUrlBase}/corpus.json`,
    manifestUrl: `${assetUrlBase}/manifest.json`,
    studentModelUrl: `${assetUrlBase}/student-model.bin`,
    abstentionUrl: studentModelInfo.abstention ? `${assetUrlBase}/student-abstention.json` : null,
  };
  manifest.encoder = {
    ...(manifest.encoder || {}),
    studentModelUrl: manifest.docusaurus.studentModelUrl,
    studentModelSha256: studentModelInfo.sha256,
    studentModelBytes: studentModelInfo.bytes,
    abstentionUrl: manifest.docusaurus.abstentionUrl,
    abstentionSha256: studentModelInfo.abstention?.sha256 || null,
    abstentionBytes: studentModelInfo.abstention?.bytes || null,
  };
  await fs.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
}

async function stageExternalStudentModel(options, context, workDir) {
  if (!options.studentModel) return null;
  const sourcePath = path.resolve(context.siteDir, options.studentModel);
  if (!fssync.existsSync(sourcePath)) {
    throw new Error(`Pancake student model not found: ${sourcePath}`);
  }
  await fs.copyFile(sourcePath, path.join(workDir, 'student-model.bin'));
  const vectorsPath = path.resolve(context.siteDir, options.studentVectors);
  if (!fssync.existsSync(vectorsPath)) {
    throw new Error(`Pancake teacher vectors not found: ${vectorsPath}`);
  }
  await fs.copyFile(vectorsPath, path.join(workDir, 'docs-vectors.f32'));
  if (options.studentAbstention) {
    const abstentionPath = path.resolve(context.siteDir, options.studentAbstention);
    if (!fssync.existsSync(abstentionPath)) {
      throw new Error(`Pancake abstention model not found: ${abstentionPath}`);
    }
    await fs.copyFile(abstentionPath, path.join(workDir, 'student-abstention.json'));
  }
}

async function publishStudentModel(options, context, workDir, assetDir) {
  const sourcePath = options.studentModel
    ? path.resolve(context.siteDir, options.studentModel)
    : path.join(workDir, 'student', 'student-model.bin');
  const abstentionPath = options.studentAbstention
    ? path.resolve(context.siteDir, options.studentAbstention)
    : path.join(workDir, 'student', 'student-abstention.json');
  if (options.studentModel && !fssync.existsSync(sourcePath)) {
    throw new Error(`Pancake student model not found: ${sourcePath}`);
  }
  const assetPath = path.join(assetDir, 'student-model.bin');
  if (!fssync.existsSync(sourcePath)) {
    throw new Error(`Pancake student model was not produced: ${sourcePath}`);
  }
  await fs.copyFile(sourcePath, assetPath);
  const bytes = await fs.readFile(sourcePath);
  const { createHash } = await import('node:crypto');
  let abstention = null;
  if (fssync.existsSync(abstentionPath)) {
    const abstentionBytes = await fs.readFile(abstentionPath);
    await fs.writeFile(path.join(assetDir, 'student-abstention.json'), abstentionBytes);
    abstention = {
      bytes: abstentionBytes.byteLength,
      sha256: createHash('sha256').update(abstentionBytes).digest('hex'),
    };
  }
  return {
    bytes: bytes.byteLength,
    sha256: createHash('sha256').update(bytes).digest('hex'),
    abstention,
  };
}

export default function pancakeDocusaurusPlugin(context, rawOptions = {}) {
  const options = normalizeOptions(rawOptions);
  const workDir = path.resolve(context.siteDir, options.workDir);

  return {
    name: 'docusaurus-plugin-pancake-search',

    async postBuild({ outDir }) {
      if (!options.enabled) return;
      const resolvedOutDir = path.resolve(outDir);
      if (!fssync.existsSync(resolvedOutDir)) {
        throw new Error(`Docusaurus output directory not found: ${resolvedOutDir}`);
      }

      const assetDir = path.join(resolvedOutDir, options.assetBase);
      await fs.rm(assetDir, { recursive: true, force: true });
      await fs.mkdir(workDir, { recursive: true });
      await fs.mkdir(assetDir, { recursive: true });

      const config = makeConfig(context, options, workDir, resolvedOutDir);
      await fs.writeFile(path.join(workDir, 'pancake.config.json'), `${JSON.stringify(config, null, 2)}\n`);
      await stageExternalStudentModel(options, context, workDir);

      await withOptionalStubEmbeddings(options.stubEmbeddings, () =>
        buildSearchAssets(workDir, config, {
          sourceBaseDir: workDir,
          assetsDir: assetDir,
          skipBundleSizeCheck: true,
        })
      );
      const studentModelInfo = await publishStudentModel(options, context, workDir, assetDir);
      await copyRuntimeManifest(assetDir, context, options, studentModelInfo);
    },

    injectHtmlTags() {
      if (!options.enabled || !options.mount) return {};
      const assetUrlBase = joinSitePath(context.siteConfig?.baseUrl || '/', options.assetBase);
      return {
        headTags: [
          {
            tagName: 'script',
            innerHTML: `window.__PANCAKE_SEARCH__ = ${JSON.stringify({ assetBase: assetUrlBase })};`,
          },
        ],
        preBodyTags: [
          {
            tagName: 'div',
            attributes: {
              class: 'pancake-search',
              'data-pancake-search': '',
              'data-pancake-asset-base': assetUrlBase,
            },
          },
        ],
      };
    },

    getClientModules() {
      return options.enabled && options.mount
        ? [path.join(here, 'client', 'search.js')]
        : [];
    },

    getPathsToWatch() {
      return [
        path.join(context.siteDir, 'docs'),
        path.join(context.siteDir, 'blog'),
        path.join(context.siteDir, 'src', 'pages'),
      ];
    },
  };
}

export function validateOptions({ options }) {
  const normalized = normalizeOptions(options);
  if (!normalized.assetBase) throw new Error('assetBase must not be empty');
  if (!['boolean', 'undefined'].includes(typeof options?.mount)) throw new Error('mount must be a boolean');
  if (options?.stubEmbeddings === true) {
    throw new Error('stubEmbeddings is not supported by the Docusaurus plugin; provide a studentModel instead');
  }
  if (options?.studentModel !== undefined && typeof options.studentModel !== 'string') {
    throw new Error('studentModel must be a string path');
  }
  if (options?.studentVectors !== undefined && typeof options.studentVectors !== 'string') {
    throw new Error('studentVectors must be a string path');
  }
  if (options?.studentAbstention !== undefined && typeof options.studentAbstention !== 'string') {
    throw new Error('studentAbstention must be a string path');
  }
  if (options?.studentModel && !options?.studentVectors) {
    throw new Error('studentModel requires studentVectors so passage indexing uses matching teacher document vectors');
  }
  if (options?.trainStudent !== undefined && typeof options.trainStudent !== 'object' && options.trainStudent !== false) {
    throw new Error('trainStudent must be an object or false');
  }
  if (options?.trainStudent === false && !options?.studentModel) {
    throw new Error('trainStudent: false requires studentModel to point at a corpus-specific PSTU model');
  }
  return options || {};
}
