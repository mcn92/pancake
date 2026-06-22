export type Metric = 'cosine' | 'l2';

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
  /** Query-time search breadth (default: 100). Must be a positive integer. */
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

export interface JsonFileOptions extends CreateOptions {
  /** Force JSON or JSONL parsing. Default: inferred from file extension. */
  format?: 'json' | 'jsonl';
  /** Record field holding the vector. Default: 'vector'. */
  vectorKey?: string;
  /** Record field holding the caller's source ID. Default: 'id'. */
  idKey?: string;
  /** Maximum JSON/JSONL file size in bytes. Default: 64 MiB. */
  maxFileBytes?: number;
}

export interface SnapshotFileOptions extends CreateOptions {
  /** Maximum snapshot file size in bytes. Default: 512 MiB. */
  maxFileBytes?: number;
}

export interface SearchResult {
  /** Vector ID as returned by add(). */
  id: number;
  /** Distance from query (lower is more similar). */
  distance: number;
}

export interface PancakeIndex {
  /** Insert a single vector. Returns its stable external ID. */
  add(vector: VectorInput): number;
  /** Insert multiple vectors. Returns stable external IDs in insertion order. */
  addBatch(vectors: readonly VectorInput[]): number[];
  /** Find the k nearest neighbors of a query vector. */
  search(query: VectorInput, k: number): SearchResult[];
  /** Find the k nearest neighbors restricted to IDs in allowedIds. */
  searchFiltered(query: VectorInput, k: number, allowedIds: Set<number>): SearchResult[];
  /** Soft-delete a vector by ID. Unknown IDs are ignored. */
  delete(id: number): void;
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
  /** Vectors stored by the backend, including soft-deleted entries until compact(). */
  readonly count: number;
  /** Soft-deleted vectors awaiting compaction. */
  readonly ghostCount: number;
  /** Ratio reported by the backend; after soft deletes this behaves as ghostCount / count. */
  readonly ghostRatio: number;
  /** Estimated index memory in bytes. */
  readonly memory: number;
  /** Input vector dimension. */
  readonly dim: number;
}

export interface PancakeApi {
  /** Create a new Pancake index using the runtime-specific packaged entrypoint. */
  create(opts: CreateOptions): Promise<PancakeIndex>;
  /** Create and populate an index from raw vectors. Infers dim and maxElements by default. */
  fromVectors(vectors: readonly VectorInput[], opts?: Omit<CreateOptions, 'dim'> & Partial<Pick<CreateOptions, 'dim'>>): Promise<FromVectorsResult<never>>;
  /** Create and populate an index from { id, vector } records. */
  fromVectors<Id = unknown>(records: readonly VectorRecord<Id>[], opts?: Omit<CreateOptions, 'dim'> & Partial<Pick<CreateOptions, 'dim'>>): Promise<FromVectorsResult<Id>>;
  /** Load vectors from a JSON/JSONL file and build an index. Node.js entrypoints only. */
  loadJsonFile<Id = unknown>(filePath: string, opts?: JsonFileOptions): Promise<FromVectorsResult<Id>>;
  /** Load a previously exported Pancake snapshot from disk. Node.js entrypoints only. */
  loadSnapshotFile(filePath: string, opts: SnapshotFileOptions): Promise<PancakeIndex>;
}

declare const Pancake: PancakeApi;

export default Pancake;
