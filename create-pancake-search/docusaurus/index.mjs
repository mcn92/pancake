import fs from 'node:fs/promises';
import fssync from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildSearchAssets } from '../src/cli.mjs';

const DEFAULT_PREFIX = 'Represent this sentence for searching relevant passages: ';
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
      mode: 'workers-ai',
      buildModel: 'bge-small-en-v1.5',
      dims: 384,
      prefixPolicy: { passage: '', query: DEFAULT_PREFIX },
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

async function copyRuntimeManifest(assetDir, context, options) {
  const manifestPath = path.join(assetDir, 'manifest.json');
  const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  const assetUrlBase = joinSitePath(context.siteConfig?.baseUrl || '/', options.assetBase);
  manifest.docusaurus = {
    assetBase: options.assetBase,
    artifactUrl: `${assetUrlBase}/index.pancake-range`,
    corpusUrl: `${assetUrlBase}/corpus.json`,
    manifestUrl: `${assetUrlBase}/manifest.json`,
  };
  await fs.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
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

      const config = makeConfig(context, options, workDir, resolvedOutDir);
      await fs.writeFile(path.join(workDir, 'pancake.config.json'), `${JSON.stringify(config, null, 2)}\n`);

      await withOptionalStubEmbeddings(options.stubEmbeddings, () =>
        buildSearchAssets(workDir, config, {
          sourceBaseDir: workDir,
          assetsDir: assetDir,
          skipBundleSizeCheck: true,
        })
      );
      await copyRuntimeManifest(assetDir, context, options);
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
  return options || {};
}
