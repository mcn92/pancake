/**
 * pancake-wasm/complete/builder — Node-only assembly of complete .pancake
 * artifacts (spec/COMPLETE_PROFILE.md). The wire-format constants here are
 * the same objects the reader uses (complete/format.mjs).
 */

export declare const MAGIC: number;
export declare const HEADER_BYTES: number;
export declare const TABLE_ENTRY_BYTES: number;
export declare const KINDS: Record<string, number>;
export declare const KIND_NAMES: Record<number, string>;

export declare function sha256(bytes: Uint8Array | Buffer): Buffer;
export declare function canonicalJson(value: unknown): string;
export declare const align16: (n: number) => number;

export declare function buildQueryInterpSegment(kind: number, encoderBytes: Buffer | Uint8Array, calibrationBytes: Buffer | Uint8Array): Buffer;
export declare function buildCorpusSegmentFromBuffers(records: Array<Buffer | Uint8Array>): Buffer;
export declare function buildInlineTransformerEncoderSegment(input: {
  declaration: Record<string, unknown>;
  vocabBytes: Buffer | Uint8Array;
  weightBytes: Buffer | Uint8Array;
}): Buffer;
export declare function assemblePancakeFile(
  manifestFields: Record<string, unknown>,
  segments: Array<{ kind: string; bytes: Buffer | Uint8Array }>,
  outPath: string
): { outPath: string; fileBytes: number; identity: string; manifest: Record<string, unknown> };

/** Measured recall-vs-rerank operating point (SKETCH_PROFILE.md section 5). */
export declare function measureRecommendedRerank(input: {
  artifactModule: unknown;
  sketchBytes: Uint8Array | ArrayBufferLike;
  queryVectors?: Float32Array[] | null;
  snapshotBytes?: Uint8Array | ArrayBufferLike | null;
  k?: number;
  targetRecall?: number;
  maxQueries?: number;
  sweep?: number[] | null;
}): Promise<{
  recommendedRerank: number;
  recall: number;
  k: number;
  targetRecall: number;
  queries: number;
  querySource: string;
  curve: Array<{ rerank: number; recall: number }>;
}>;

/** Kind-3 kernel via the web glue with an explicit wasm binary (jiti-safe). */
export declare function loadInlineEncoderKernel(): Promise<() => unknown>;
