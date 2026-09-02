// Shared by the pikelet modules: package paths and version, config
// defaults, the model table, CliError, and the loaders that resolve pancake-wasm
// (engine, artifact layer, complete profile) from npm or the monorepo root.

import crypto from 'node:crypto';
import fssync from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest();
// The complete-profile builder and reader live in pancake-wasm
// (pikelet-wasm/complete[-builder]) — resolved like loadArtifactContract:
// bare specifier for npm consumers, repo root when developing in the
// monorepo (where pikelet's installed pikelet-wasm may lag).
let completeModulesPromise = null;
function loadCompleteModules() {
  completeModulesPromise ??= (async () => {
    const attempts = [];
    const fromRepo = async () => ({
      builder: await import(pathToFileURL(path.join(REPO_ROOT, 'complete', 'builder.mjs')).href),
      reader: await import(pathToFileURL(path.join(REPO_ROOT, 'complete', 'index.mjs')).href),
    });
    const fromPackage = async () => ({
      builder: await import('pikelet-wasm/complete/builder'),
      reader: await import('pikelet-wasm/complete'),
    });
    // Inside the monorepo the sibling checkout is the source of truth (a
    // stale node_modules copy up the tree must not shadow it); npm consumers
    // have no REPO_ROOT and resolve the installed package.
    for (const attempt of inMonorepo() ? [fromRepo, fromPackage] : [fromPackage, fromRepo]) {
      try {
        return await attempt();
      } catch (error) {
        attempts.push(error.message.split('\n')[0]);
      }
    }
    throw new CliError(`could not resolve pikelet-wasm/complete; install pikelet-wasm >= 0.3 alongside pikelet. Tried: ${attempts.join(' | ')}`, 2);
  })();
  return completeModulesPromise;
}

// True when this package runs from its checkout in the pancake monorepo
// (REPO_ROOT holds the engine entry), where the in-tree pikelet-wasm must be
// preferred over any installed copy.
function inMonorepo() {
  return fssync.existsSync(path.join(REPO_ROOT, 'pancake.node.mjs')) && fssync.existsSync(path.join(REPO_ROOT, 'complete', 'index.mjs'));
}
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PACKAGE_ROOT = path.resolve(__dirname, '..');
const REPO_ROOT = path.resolve(PACKAGE_ROOT, '..');
const STUDENT_TRAINER = path.join(PACKAGE_ROOT, 'tools', 'train_student.py');
const OWN_PACKAGE = JSON.parse(fssync.readFileSync(path.join(PACKAGE_ROOT, 'package.json'), 'utf8'));
const CLI_VERSION = OWN_PACKAGE.version;
// The generated project must run its artifacts on a pikelet-wasm at least as
// new as the one that built them (range artifact format revisions are not
// readable by older readers), so it inherits this package's own range rather
// than carrying a second hardcoded one that drifts.
const PANCAKE_WASM_RANGE = OWN_PACKAGE.dependencies['pikelet-wasm'];
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
  const fromRepo = () => import(pathToFileURL(path.join(REPO_ROOT, 'pancake-artifact.js')).href);
  const fromPackage = () => import('pikelet-wasm/artifact');
  for (const attempt of inMonorepo() ? [fromRepo, fromPackage] : [fromPackage, fromRepo]) {
    try {
      const mod = await attempt();
      const contract = mod.default || mod;
      if (typeof contract.buildSketchArtifactBytes === 'function') return contract;
    } catch {}
  }
  throw new CliError('pikelet-wasm/artifact does not expose buildSketchArtifactBytes; install pikelet-wasm >= 0.3 alongside pikelet.', 2);
}
async function loadPancake() {
  const fromRepo = () => import(pathToFileURL(path.join(REPO_ROOT, 'pancake.node.mjs')).href);
  const fromPackage = () => import('pikelet-wasm');
  const [first, second] = inMonorepo() ? [fromRepo, fromPackage] : [fromPackage, fromRepo];
  try {
    return (await first()).default;
  } catch {
    return (await second()).default;
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
