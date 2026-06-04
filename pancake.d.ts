export interface CreateOptions {
  /** Input vector dimension (required). */
  dim: number;
  /** Maximum index capacity (default: 100000). */
  maxElements?: number;
  /** Distance metric (default: 'cosine'). */
  metric?: 'cosine' | 'l2';
  /** Use int8 quantized storage — ~1 byte/dim vs 4 bytes for float32 (default: true). */
  quantized?: boolean;
  /** HNSW graph connectivity (default: 16). Higher improves recall but uses more memory. */
  M?: number;
  /** Build-time search breadth (default: 200). Higher improves index quality. */
  efConstruction?: number;
  /** Query-time search breadth (default: 100). Tune for speed vs recall tradeoff. */
  efSearch?: number;
}

export interface SearchResult {
  /** Vector ID as returned by add(). */
  id: number;
  /** Distance from query (lower is more similar). */
  distance: number;
}

export interface PancakeIndex {
  /** Insert a single vector. Returns its ID. */
  add(vector: Float32Array | number[]): number;
  /** Insert multiple vectors. Returns array of IDs. */
  addBatch(vectors: (Float32Array | number[])[]): number[];
  /** Find the k nearest neighbors of a query vector. k must be a non-negative integer. */
  search(query: Float32Array | number[], k: number): SearchResult[];
  /** Find the k nearest neighbors, restricted to IDs in allowedIds. Uses in-graph filtering with iterative deepening. */
  searchFiltered(query: Float32Array | number[], k: number, allowedIds: Set<number>): SearchResult[];
  /** Soft-delete a vector by ID. O(1), excluded from future searches. No-op for unknown IDs. */
  delete(id: number): void;
  /** Reclaim space from soft-deleted vectors via graph rewiring. */
  compact(): void;
  /** Serialize the index to a Uint8Array for persistence. Throws if ghostCount > 0; call compact() first. */
  export(): Uint8Array;
  /** Restore index state from a previous export(). A failed import leaves the index unusable; back up with export() first if recovery matters. */
  import(data: Uint8Array | ArrayBuffer): void;
  /** Free WASM heap buffers. Instance is unusable after this. */
  dispose(): void;
  /** Number of vectors currently stored by the backend, including soft-deleted entries until compact(). */
  readonly count: number;
  /** Number of soft-deleted vectors awaiting compaction. */
  readonly ghostCount: number;
  /** Ratio reported by the backend; after soft deletes this behaves as ghostCount / count. */
  readonly ghostRatio: number;
  /** WASM heap bytes used by the index. */
  readonly memory: number;
  /** Input vector dimension. */
  readonly dim: number;
}

declare const Pancake: {
  /** Create a new Pancake vector search index using the runtime-specific packaged entrypoint. */
  create(opts: CreateOptions): Promise<PancakeIndex>;
};

export default Pancake;
