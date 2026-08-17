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
  const completeProfile = options.completeProfile || {};
  return {
    enabled: options.enabled !== false,
    assetBase: trimSlashes(options.assetBase || 'pancake-search'),
    mount: options.mount !== false,
    workDir: options.workDir || path.join('.docusaurus', 'pancake-search'),
    name: options.name,
    include: options.include || ['**/*.html'],
    // The build-output 404 page would otherwise be indexed as a document.
    exclude: options.exclude || ['404.html'],
    stubEmbeddings: options.stubEmbeddings === true,
    studentModel: options.studentModel || null,
    studentVectors: options.studentVectors || null,
    studentAbstention: options.studentAbstention || null,
    completeProfile: {
      enabled: completeProfile.enabled === true,
      vectors: completeProfile.vectors || null,
      vocab: completeProfile.vocab || null,
      weights: completeProfile.weights || null,
      calibration: completeProfile.calibration || null,
      model: completeProfile.model || 'sentence-transformers/all-MiniLM-L6-v2',
      maxTokens: completeProfile.maxTokens || 128,
    },
    trainStudent: {
      enabled: completeProfile.enabled === true ? false : options.trainStudent !== false && !options.studentModel,
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
      ...(options.trainStudent?.skipAbstention === true ? { skipAbstention: true } : {}),
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
  const complete = options.completeProfile.enabled;
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
      mode: complete ? 'inline-transformer' : 'student',
      ...(complete ? (options.completeProfile.vectors ? { teacherVectorsPath: 'docs-vectors.f32' } : {}) : {
        studentModelPath: options.studentModel ? 'student-model.bin' : 'student/student-model.bin',
        ...(options.studentVectors ? { teacherVectorsPath: 'docs-vectors.f32' } : {}),
        ...(options.trainStudent.enabled ? { trainStudent: { ...options.trainStudent, outDir: 'student' } } : {}),
      }),
      dims: 384,
      prefixPolicy: { passage: '', query: '' },
      pooling: 'mean',
      normalize: true,
    },
    index: options.index,
    runtime: complete
      ? {
          mode: 'complete',
          profile: 'kind3',
          storage: 'bundled',
          fileName: 'search.pancake',
          inlineEncoder: {
            vocabPath: 'inline-encoder/vocab.txt',
            weightsPath: 'inline-encoder/encoder-weights.bin',
            ...(options.completeProfile.calibration ? { calibrationPath: 'inline-encoder/calibration.json' } : {}),
            model: options.completeProfile.model,
            maxTokens: options.completeProfile.maxTokens,
          },
        }
      : { mode: 'artifact', storage: 'bundled' },
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

// Ingest sees build-output file paths (docs/intro/index.html); the search UI
// needs site routes. Map file path -> route and prefix the site baseUrl so
// result links resolve from any page, at any nesting depth.
function routeForBuildPath(buildPath, baseUrl) {
  const raw = String(buildPath || '');
  if (/^[a-z][a-z0-9+.-]*:/i.test(raw) || raw.startsWith('/')) return raw;
  let route = raw.replace(/\\/g, '/').replace(/^\.\//, '');
  if (route.endsWith('index.html')) route = route.slice(0, -'index.html'.length);
  else if (route.endsWith('.html')) route = route.slice(0, -'.html'.length);
  return joinSitePath(baseUrl, route) || baseUrl;
}

async function rewriteCorpusUrls(assetDir, context) {
  const corpusPath = path.join(assetDir, 'corpus.json');
  const corpus = JSON.parse(await fs.readFile(corpusPath, 'utf8'));
  const baseUrl = context.siteConfig?.baseUrl || '/';
  for (const chunk of corpus) {
    chunk.url = routeForBuildPath(chunk.url || chunk.sourcePath, baseUrl);
  }
  await fs.writeFile(corpusPath, JSON.stringify(corpus));
}

async function copyRuntimeManifest(assetDir, context, options, studentModelInfo) {
  const manifestPath = path.join(assetDir, 'manifest.json');
  const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  const assetUrlBase = joinSitePath(context.siteConfig?.baseUrl || '/', options.assetBase);
  manifest.docusaurus = {
    assetBase: options.assetBase,
    artifactUrl: options.completeProfile.enabled ? `${assetUrlBase}/search.pancake` : `${assetUrlBase}/index.pancake-range`,
    completeArtifactUrl: options.completeProfile.enabled ? `${assetUrlBase}/search.pancake` : null,
    corpusUrl: `${assetUrlBase}/corpus.json`,
    manifestUrl: `${assetUrlBase}/manifest.json`,
    studentModelUrl: options.completeProfile.enabled ? null : `${assetUrlBase}/student-model.bin`,
    abstentionUrl: studentModelInfo.abstention ? `${assetUrlBase}/student-abstention.json` : null,
  };
  manifest.encoder = {
    ...(manifest.encoder || {}),
    ...(options.completeProfile.enabled ? {} : {
      studentModelUrl: manifest.docusaurus.studentModelUrl,
      studentModelSha256: studentModelInfo.sha256,
      studentModelBytes: studentModelInfo.bytes,
    }),
    abstentionUrl: manifest.docusaurus.abstentionUrl,
    abstentionSha256: studentModelInfo.abstention?.sha256 || null,
    abstentionBytes: studentModelInfo.abstention?.bytes || null,
  };
  await fs.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
}

async function stageExternalStudentModel(options, context, workDir) {
  if (options.completeProfile.enabled) return null;
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

async function stageCompleteProfileInputs(options, context, workDir) {
  if (!options.completeProfile.enabled) return;
  const copyRequired = async (source, targetRel, label) => {
    if (!source) throw new Error(`Pancake completeProfile.${label} is required`);
    const sourcePath = path.resolve(context.siteDir, source);
    if (!fssync.existsSync(sourcePath)) throw new Error(`Pancake completeProfile.${label} not found: ${sourcePath}`);
    const targetPath = path.join(workDir, targetRel);
    await fs.mkdir(path.dirname(targetPath), { recursive: true });
    await fs.copyFile(sourcePath, targetPath);
  };
  if (options.completeProfile.vectors) await copyRequired(options.completeProfile.vectors, 'docs-vectors.f32', 'vectors');
  await copyRequired(options.completeProfile.vocab, path.join('inline-encoder', 'vocab.txt'), 'vocab');
  await copyRequired(options.completeProfile.weights, path.join('inline-encoder', 'encoder-weights.bin'), 'weights');
  if (options.completeProfile.calibration) {
    await copyRequired(options.completeProfile.calibration, path.join('inline-encoder', 'calibration.json'), 'calibration');
  }
}

async function publishStudentModel(options, context, workDir, assetDir) {
  if (options.completeProfile.enabled) return { bytes: null, sha256: null, abstention: null };
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
      await stageCompleteProfileInputs(options, context, workDir);

      await withOptionalStubEmbeddings(options.stubEmbeddings, () =>
        buildSearchAssets(workDir, config, {
          sourceBaseDir: workDir,
          assetsDir: assetDir,
          skipBundleSizeCheck: true,
        })
      );
      await rewriteCorpusUrls(assetDir, context);
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

    configureWebpack(config, isServer) {
      if (isServer) return {};
      // pancake-wasm/artifact is pure JS in the browser, but its Node-only
      // file helpers statically reference fs/crypto, which webpack must stub.
      return { resolve: { fallback: { fs: false, crypto: false } } };
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
  if (options?.completeProfile?.enabled === true) {
    for (const key of ['vocab', 'weights']) {
      if (typeof options.completeProfile[key] !== 'string' || !options.completeProfile[key]) {
        throw new Error(`completeProfile.${key} must be a string path when completeProfile.enabled is true`);
      }
    }
    if (options.completeProfile.vectors != null && typeof options.completeProfile.vectors !== 'string') {
      throw new Error('completeProfile.vectors must be a string path when provided');
    }
    if (options.completeProfile.calibration != null && typeof options.completeProfile.calibration !== 'string') {
      throw new Error('completeProfile.calibration must be a string path when provided');
    }
  }
  return options || {};
}
