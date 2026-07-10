export type Metric = 'cosine' | 'l2';

export const PANCAKE_ERROR_CODES: Readonly<{
  INVALID_ARGUMENT: 'INVALID_ARGUMENT';
  DIMENSION_MISMATCH: 'DIMENSION_MISMATCH';
  INVALID_VECTOR: 'INVALID_VECTOR';
  INDEX_FULL: 'INDEX_FULL';
  INDEX_DISPOSED: 'INDEX_DISPOSED';
  COMPACTION_REQUIRED: 'COMPACTION_REQUIRED';
  SNAPSHOT_INVALID: 'SNAPSHOT_INVALID';
  SNAPSHOT_CONFIG_MISMATCH: 'SNAPSHOT_CONFIG_MISMATCH';
  SNAPSHOT_CAPACITY_EXCEEDED: 'SNAPSHOT_CAPACITY_EXCEEDED';
  WASM_LOAD_FAILED: 'WASM_LOAD_FAILED';
  WASM_ALLOCATION_FAILED: 'WASM_ALLOCATION_FAILED';
  FILE_IO_FAILED: 'FILE_IO_FAILED';
  PARSE_FAILED: 'PARSE_FAILED';
  INTERNAL_INVARIANT: 'INTERNAL_INVARIANT';
  INDEX_LIMIT: 'INDEX_LIMIT';
}>;

export type PancakeErrorCode = typeof PANCAKE_ERROR_CODES[keyof typeof PANCAKE_ERROR_CODES];

export class PancakeError extends Error {
  readonly code: PancakeErrorCode;
  readonly details?: Readonly<Record<string, unknown>>;
  readonly cause?: unknown;
  constructor(code: PancakeErrorCode, message: string, details?: Readonly<Record<string, unknown>>, cause?: unknown);
}

export type VectorInput = Float32Array | readonly number[];

export interface VectorRecord<Id = unknown> {
  id?: Id;
  vector: VectorInput;
  [key: string]: unknown;
}

export interface CreateOptions {
  /** Input vector dimension (required). */
  dim: number;
  /** Maximum index capacity (default: 100000). */
  maxElements?: number;
  /** Distance metric (default: 'cosine'). */
  metric?: Metric;
  /** Use int8 quantized storage (default: true). */
  quantized?: boolean;
  /** HNSW graph connectivity (default: 16). Valid range: 2-128. */
  M?: number;
  /** Build-time search breadth (default: 50). Valid range: 1-4096. */
  efConstruction?: number;
  /** Query-time search breadth (default: 100). Valid range: 1-4096. */
  efSearch?: number;
}

export interface SearchOptions {
  /** Override query-time search breadth for this call. Valid range: 1-4096. */
  efSearch?: number;
}

export interface FromVectorsResult<Id = unknown> {
  /** Created and populated index. Caller owns dispose(). */
  index: PancakeIndex;
  /** Stable Pancake IDs returned in insertion order. */
  ids: number[];
  /** Mapping from Pancake IDs back to caller-provided source IDs. */
  idMap: Map<number, Id>;
}

export interface JsonFileOptions extends Omit<CreateOptions, 'dim'> {
  /** Input vector dimension. Default: inferred from the first vector in the file. */
  dim?: number;
  /** Force JSON or JSONL parsing. Default: inferred from file extension. */
  format?: 'json' | 'jsonl';
  /** Record field holding the vector. Default: 'vector'. */
  vectorKey?: string;
  /** Record field holding the caller's source ID. Default: 'id'. */
  idKey?: string;
  /** Maximum JSON/JSONL file size in bytes. Default: 64 MiB. */
  maxFileBytes?: number;
}

export interface RestoreOptions extends Partial<CreateOptions> {}

export type SnapshotFormat = 'pancake' | 'raw';

export interface SnapshotInspection {
  readonly format: SnapshotFormat;
  readonly version: number;
  readonly dim: number;
  readonly count: number;
  readonly metric: Metric;
  readonly quantized: boolean;
  readonly M: number;
  readonly efConstruction: number;
  readonly nextId: number;
}

export interface SnapshotFileOptions extends RestoreOptions {
  /** Maximum snapshot file size in bytes. Default: 512 MiB. */
  maxFileBytes?: number;
}

export interface SearchResult {
  /** Vector ID as returned by add(). */
  id: number;
  /** Distance from query (lower is more similar). */
  distance: number;
}

export interface MemoryUsage {
  /** Backend-owned graph and vector storage estimate. */
  readonly logicalIndexBytes: number;
  /** Current byte length of this index's isolated WebAssembly heap. */
  readonly wasmHeapBytes: number;
  /** Raw backend snapshot buffer retained by the most recent export(). */
  readonly snapshotBufferBytes: number;
}

export interface ResolvedConfig {
  readonly dim: number;
  readonly maxElements: number;
  readonly metric: Metric;
  readonly quantized: boolean;
  readonly M: number;
  readonly efConstruction: number;
  readonly efSearch: number;
}

export interface PancakeIndex {
  /** Insert a single vector. Returns its stable external ID. */
  add(vector: VectorInput): number;
  /** Insert multiple vectors. Returns stable external IDs in insertion order. */
  addBatch(vectors: readonly VectorInput[]): number[];
  /** Find the k nearest neighbors of a query vector. */
  search(query: VectorInput, k: number, options?: SearchOptions): SearchResult[];
  /** Find the k nearest neighbors restricted to IDs in allowedIds. */
  searchFiltered(query: VectorInput, k: number, allowedIds: Set<number>, options?: SearchOptions): SearchResult[];
  /** Change the default query-time search breadth for future searches. */
  setEfSearch(efSearch: number): void;
  /** Soft-delete a live vector. Returns false for unknown or already-deleted IDs. */
  delete(id: number): boolean;
  /** Whether an ID currently refers to a live vector. */
  has(id: number): boolean;
  /** Whether an ID is awaiting reclamation by compact(). */
  isDeleted(id: number): boolean;
  /** Rebuild the graph without soft-deleted entries. */
  compact(): void;
  /** Serialize index state. Throws if ghostCount > 0; call compact() first after deletions. */
  export(): Uint8Array;
  /**
   * Restore index state from a previous export().
   * Validation failures leave the target index unchanged.
   */
  import(data: Uint8Array | ArrayBufferLike): void;
  /** Free WASM buffers. The instance is unusable after this. */
  dispose(): void;
  /** Dispose support for JavaScript `using` declarations. */
  [Symbol.dispose](): void;
  /** Vectors stored by the backend, including soft-deleted entries until compact(). */
  readonly count: number;
  /** Live vectors available to search. */
  readonly liveCount: number;
  /** Preferred name for soft-deleted vectors awaiting compaction. */
  readonly deletedCount: number;
  /** Preferred name for the soft-deleted fraction. */
  readonly deletedRatio: number;
  /** Fixed insertion capacity. */
  readonly capacity: number;
  /** Remaining insertion slots; soft deletion does not increase this value. */
  readonly remainingCapacity: number;
  /** Soft-deleted vectors awaiting compaction. */
  readonly ghostCount: number;
  /** Ratio reported by the backend; after soft deletes this behaves as ghostCount / count. */
  readonly ghostRatio: number;
  /** Estimated index memory in bytes. */
  readonly memory: number;
  /** Logical index, isolated WASM heap, and retained snapshot-buffer memory. */
  readonly memoryUsage: MemoryUsage;
  /** Fully resolved construction and current query-policy values. */
  readonly config: ResolvedConfig;
  /** Input vector dimension. */
  readonly dim: number;
}

/**
 * The portable API exposed by every entrypoint (Node, browser, Workers).
 * The browser/Workers entrypoints expose exactly this surface; the Node
 * entrypoints add file helpers (see {@link NodePancakeApi}).
 */
export interface PancakeApi {
  readonly PancakeError: typeof PancakeError;
  readonly PANCAKE_ERROR_CODES: typeof PANCAKE_ERROR_CODES;
  /** Create a new Pancake index using the runtime-specific packaged entrypoint. */
  create(opts: CreateOptions): Promise<PancakeIndex>;
  /** Restore an envelope snapshot, inferring its construction config. */
  restore(snapshot: Uint8Array | ArrayBufferLike, overrides?: RestoreOptions): Promise<PancakeIndex>;
  /** Validate and inspect snapshot headers without creating a WASM index. */
  inspectSnapshot(snapshot: Uint8Array | ArrayBufferLike): SnapshotInspection;
  /** Create an index, run a callback, and always dispose the index afterward. */
  withIndex<T>(opts: CreateOptions, fn: (index: PancakeIndex) => T | Promise<T>): Promise<T>;
  /** Create and populate an index from raw vectors. Infers dim and maxElements by default. */
  fromVectors(vectors: readonly VectorInput[], opts?: Omit<CreateOptions, 'dim'> & Partial<Pick<CreateOptions, 'dim'>>): Promise<FromVectorsResult<never>>;
  /** Create and populate an index from { id, vector } records. */
  fromVectors<Id = unknown>(records: readonly VectorRecord<Id>[], opts?: Omit<CreateOptions, 'dim'> & Partial<Pick<CreateOptions, 'dim'>>): Promise<FromVectorsResult<Id>>;
}

/**
 * The Node.js API: the portable surface plus filesystem helpers. These helpers
 * exist only on the Node entrypoints (`pancake-wasm`, `pancake-wasm/node`); the
 * browser/Workers entrypoints do not expose them, and importing those returns
 * the narrower {@link PancakeApi}.
 */
export interface NodePancakeApi extends PancakeApi {
  /** Load vectors from a JSON/JSONL file and build an index. */
  loadJsonFile<Id = unknown>(filePath: string, opts?: JsonFileOptions): Promise<FromVectorsResult<Id>>;
  /** Load a previously exported Pancake snapshot from disk. */
  loadSnapshotFile(filePath: string, opts: SnapshotFileOptions): Promise<PancakeIndex>;
}

declare const Pancake: NodePancakeApi;

export default Pancake;
