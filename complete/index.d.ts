/**
 * pancake-wasm/complete — reader for complete .pancake artifacts
 * (spec/COMPLETE_PROFILE.md, profile pancake-complete-v1). Opens any kind
 * (1 student-inline, 2 declared-external, 3 inline-transformer) from a file
 * path in Node or a range source anywhere.
 */

/** A byte-range source: local file, HTTP, memory, or anything else. */
export interface CompleteRangeSource {
  /** Total bytes, when known. Verified against the header when present. */
  size?: number;
  preferredParallelism?: number;
  preferredGapBytes?: number;
  read(offset: number, length: number): Promise<Uint8Array | ArrayBufferLike>;
  close?(): Promise<void>;
}

export interface HttpRangeSourceStats {
  requests: number;
  bytes: number;
  acceptRanges: string | null;
  etag: string | null;
  fullFallback: boolean;
}

export interface HttpRangeSource extends CompleteRangeSource {
  stats: HttpRangeSourceStats;
  init(): Promise<void>;
}

export declare function httpRangeSource(url: string, options?: {
  preferredParallelism?: number;
  preferredGapBytes?: number;
  maxFullFallbackBytes?: number;
}): HttpRangeSource;

export interface CompleteQueryResult {
  matchQuality: 'strong' | 'weak' | 'none' | 'unscored';
  confidence?: number;
  results: Array<{ id: number; distance: number } & Record<string, unknown>>;
}

export interface CompleteSearch {
  info(): {
    identity: string;
    records: number;
    dim: number;
    metric: string | number;
    encoder: Record<string, unknown>;
    fileBytes: number;
    residentBytes: number;
    residentVerified: boolean;
    sampleQueries: string[];
  };
  query(text: string, options?: {
    k?: number;
    rerank?: number;
    parallelism?: number;
    gap?: number;
    maxRangeBytes?: number;
  }): Promise<CompleteQueryResult>;
  evaluation(): Promise<Record<string, unknown>>;
  close(): Promise<void>;
}

export declare function openPancakeFile(
  input: string | CompleteRangeSource,
  options?: {
    /** Required for kind-2 (declared external encoder) artifacts. */
    encodeQuery?: (text: string) => Promise<Float32Array> | Float32Array;
    /** Override the kind-3 wasm kernel loader. */
    createEncoder?: () => Promise<unknown> | unknown;
    rerankParallelism?: number;
    rerankGap?: number;
    rerankMaxRangeBytes?: number;
  }
): Promise<CompleteSearch>;

/** Kernel layout compiled into the kind-3 encoder (mirrors encoder.cpp). */
export declare const KERNEL_LAYOUT: Record<'V' | 'P' | 'T' | 'D' | 'F' | 'L' | 'B' | 'H', number>;
export declare function expectedBlobBytes(layout?: Record<string, number>): number;
export declare function parseInlineTransformerEncoder(encoderBytes: Uint8Array): {
  declaration: Record<string, unknown>;
  vocabText: string;
  blob: Uint8Array;
};
export declare function createInlineTransformerEmbedder(input: {
  declaration: Record<string, unknown>;
  vocabText: string;
  blob: Uint8Array;
  createEncoder: () => Promise<unknown> | unknown;
  verify?: boolean;
}): Promise<{
  declaration: Record<string, unknown>;
  dim: number;
  maxSeq: number;
  embed(text: string): Promise<{ vector: Float32Array; text: string; windows: number }>;
  dispose(): void;
}>;
export declare const INLINE_TEST_VECTOR_TEXTS: string[];
export declare function buildInlineTestVectors(embedder: unknown, texts?: string[]): Promise<Array<{
  text: string; windows: number; embedding: number[]; tolerance: number;
}>>;
export declare function verifyInlineTestVectors(embedder: unknown): Promise<{ checked: number }>;
