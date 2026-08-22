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
    /** Container header formatVersion (1 or 2). */
    formatVersion: number;
    /** Manifest profile: 'pancake-complete-v1' | 'pancake-complete-v2'. */
    profile: string;
    records: number;
    dim: number;
    metric: string | number;
    encoder: Record<string, unknown>;
    /**
     * Kind 2: true when the host encoder passed the declaration's verification
     * vectors, false when served unverified (verifyEncoder:false or
     * allowUnverifiedEncoder), null when no host encoder was supplied.
     * Kind 3: whether the inline encoder verified against its own vectors.
     * Kind 1: null (inline student encoder, nothing external to verify).
     */
    encoderVerified: boolean | null;
    /**
     * 'per-record-sha256' on format 2 (each hydrated record verified on its
     * own range read); 'segment-sha256' on format 1 (whole-segment digests
     * only, lazy record reads not independently verifiable).
     */
    corpusIntegrity: string;
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
  /** Hydrate one corpus record by id (verified per record on format 2). */
  record(id: number): Promise<Record<string, unknown>>;
  evaluation(): Promise<Record<string, unknown>>;
  close(): Promise<void>;
}

export declare function openPancakeFile(
  input: string | CompleteRangeSource,
  options?: {
    /**
     * Required to serve kind-2 (declared external encoder) artifacts. It is
     * run against the declaration's verification vectors at open and the
     * open fails if any disagree beyond the declared tolerance.
     */
    encodeQuery?: (text: string) => Promise<Float32Array | number[]> | Float32Array | number[];
    /** Skip encoder verification (kind 2 host vectors, kind 3 inline vectors). Default true. */
    verifyEncoder?: boolean;
    /** Serve a kind-2 declaration that carries no verification vectors, marked unverified. Default false. */
    allowUnverifiedEncoder?: boolean;
    /** Verify each hydrated record against its digest on format 2. Default true. */
    verifyRecords?: boolean;
    /**
     * Budget for any single open-path read (manifest, segment table,
     * query-interp, corpus tables, evaluation, digest pages), honored
     * strictly. Default 256 MiB; Infinity defers to the 2 GiB backstop.
     */
    maxReadBytes?: number;
    /** Budget for a single corpus record read. Default 16 MiB. */
    maxRecordBytes?: number;
    /** Override the kind-3 wasm kernel loader. */
    createEncoder?: () => Promise<unknown> | unknown;
    rerankParallelism?: number;
    rerankGap?: number;
    rerankMaxRangeBytes?: number;
  }
): Promise<CompleteSearch>;

/** Header formatVersion -> manifest profile accepted by this reader. */
export declare const SUPPORTED_PROFILES: Readonly<Record<number, string>>;
export declare const CORPUS_LAYOUT_V2: 'records-v2';
/**
 * Check a host encoder against a kind-2 declaration's verification vectors
 * (contract 4.4 mode 2) without opening an artifact. Throws on dimension
 * or tolerance disagreement; resolves with the number of vectors checked.
 */
export declare function verifyHostEncoder(
  declaration: Record<string, unknown>,
  encodeQuery: (text: string) => Promise<Float32Array | number[]> | Float32Array | number[],
  dim: number
): Promise<{ checked: number }>;

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
