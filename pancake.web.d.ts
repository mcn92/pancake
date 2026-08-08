// Types for the browser / Cloudflare Workers entrypoints (`pancake-wasm/web`).
// These runtimes expose the portable API, including range-readable Search
// Artifacts. Only PancakeError and PANCAKE_ERROR_CODES exist as named runtime
// exports; every other re-export below is type-only. Node-only helpers
// (NodeFileRangeSource, the build*/open*File functions, loadJsonFile /
// loadSnapshotFile) are absent from this surface so using them is a compile
// error rather than a runtime throw.
export { PancakeError, PANCAKE_ERROR_CODES } from './pancake.js';
export type {
  Metric,
  PancakeErrorCode,
  VectorInput,
  VectorRecord,
  CreateOptions,
  SearchOptions,
  FromVectorsResult,
  RestoreOptions,
  SnapshotFormat,
  SnapshotInspection,
  SearchResult,
  MemoryUsage,
  ResolvedConfig,
  PancakeIndex,
  RangeReadSource,
  RangeArtifactSearchOptions,
  RangeArtifactNode,
  RangeArtifactStats,
  RangeArtifactRound,
  RangeArtifactSearchResult,
  RangeArtifactOpenOptions,
  PancakeRangeArtifact,
  PancakeRangeArtifactConstructor,
  SketchTier,
  SketchStageEvent,
  SketchArtifactOpenOptions,
  SketchScanner,
  SketchScannerOptions,
  SketchArtifactSearchOptions,
  SketchArtifactSearchResult,
  SketchArtifactStats,
  PancakeSketchArtifact,
  PancakeSketchArtifactConstructor,
  PancakeApi,
} from './pancake.js';

import type { PancakeApi } from './pancake.js';

declare const Pancake: PancakeApi;

export default Pancake;
