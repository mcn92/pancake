// Shared by the create-pancake-search modules: package paths and version, config
// defaults, the model table, CliError, and the loaders that resolve pancake-wasm
// (engine, artifact layer, complete profile) from npm or the monorepo root.

import crypto from 'node:crypto';
import fssync from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest();
// The complete-profile builder and reader live in pancake-wasm
// (pancake-wasm/complete[-builder]) — resolved like loadArtifactContract:
// bare specifier for npm consumers, repo root when developing in the
// monorepo (where create-pancake-search's installed pancake-wasm may lag).
let completeModulesPromise = null;
function loadCompleteModules() {
  completeModulesPromise ??= (async () => {
    const attempts = [];
    try {
      return {
        builder: await import('pancake-wasm/complete/builder'),
        reader: await import('pancake-wasm/complete'),
      };
    } catch (error) {
      attempts.push(error.message.split('\n')[0]);
    }
    try {
      return {
        builder: await import(pathToFileURL(path.join(REPO_ROOT, 'complete', 'builder.mjs')).href),
        reader: await import(pathToFileURL(path.join(REPO_ROOT, 'complete', 'index.mjs')).href),
      };
    } catch (error) {
      attempts.push(error.message.split('\n')[0]);
    }
    throw new CliError(`could not resolve pancake-wasm/complete; install pancake-wasm >= 0.3 alongside create-pancake-search. Tried: ${attempts.join(' | ')}`, 2);
  })();
  return completeModulesPromise;
}
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PACKAGE_ROOT = path.resolve(__dirname, '..');
const REPO_ROOT = path.resolve(PACKAGE_ROOT, '..');
const STUDENT_TRAINER = path.join(PACKAGE_ROOT, 'tools', 'train_student.py');
const OWN_PACKAGE = JSON.parse(fssync.readFileSync(path.join(PACKAGE_ROOT, 'package.json'), 'utf8'));
const CLI_VERSION = OWN_PACKAGE.version;
// The generated project must run its artifacts on a pancake-wasm at least as
// new as the one that built them (range artifact format revisions are not
// readable by older readers), so it inherits this package's own range rather
// than carrying a second hardcoded one that drifts.
const PANCAKE_WASM_RANGE = OWN_PACKAGE.dependencies['pancake-wasm'];
const DEFAULT_PREFIX = 'Represent this sentence for searching relevant passages: ';
const CONFIG_SCHEMA_URL = 'https://raw.githubusercontent.com/mcn92/pancake/main/create-pancake-search/schemas/v1/pancake.config.schema.json';
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
async function loadPancake() {
  try {
    const mod = await import('pancake-wasm');
    return mod.default;
  } catch {
    const mod = await import(pathToFileURL(path.join(REPO_ROOT, 'pancake.node.mjs')).href);
    return mod.default;
  }
}

export {
  sha256,
  completeModulesPromise,
  loadCompleteModules,
  __filename,
  __dirname,
  PACKAGE_ROOT,
  REPO_ROOT,
  STUDENT_TRAINER,
  OWN_PACKAGE,
  CLI_VERSION,
  PANCAKE_WASM_RANGE,
  DEFAULT_PREFIX,
  CONFIG_SCHEMA_URL,
  DEFAULT_CONFIG,
  RANGE_ARTIFACT_MAGIC,
  MODEL_MAP,
  CliError,
  loadArtifactContract,
  loadPancake,
};
