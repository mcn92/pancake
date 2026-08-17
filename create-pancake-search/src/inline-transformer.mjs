import { createWordPiece } from './wordpiece.mjs';

const decoder = new TextDecoder();

export function parseInlineTransformerEncoder(encoderBytes) {
  const view = new DataView(encoderBytes.buffer, encoderBytes.byteOffset, encoderBytes.byteLength);
  const declLen = view.getUint32(0, true);
  const vocabLen = view.getUint32(4, true);
  const blobLen = view.getUint32(8, true);
  if (12 + declLen + vocabLen + blobLen !== encoderBytes.length) {
    throw new Error('.pancake inline-encoder layout is inconsistent');
  }
  return {
    declaration: JSON.parse(decoder.decode(encoderBytes.subarray(12, 12 + declLen))),
    vocabText: decoder.decode(encoderBytes.subarray(12 + declLen, 12 + declLen + vocabLen)),
    blob: encoderBytes.subarray(12 + declLen + vocabLen),
  };
}

export async function createInlineTransformerEmbedder({ declaration, vocabText, blob, createEncoder }) {
  const tokenizer = createWordPiece(vocabText);
  const EM = await createEncoder();
  const dim = declaration.dim;
  const maxSeq = Math.min(declaration.maxTokens || 128, 128);
  const blobPtr = EM._malloc(blob.length);
  EM.HEAPU8.set(blob, blobPtr);
  const idsPtr = EM._malloc(maxSeq * 4);
  const hiddenPtr = EM._malloc(maxSeq * dim * 4);

  async function embed(text) {
    const allIds = tokenizer.encode(String(text || ''));
    // Inputs longer than maxSeq are mean-pooled across [CLS]…[SEP]-framed
    // windows instead of truncated, so long chunks keep their tail content.
    const interior = allIds.slice(1, -1);
    const windowLen = maxSeq - 2;
    const windows = Math.max(1, Math.ceil(interior.length / windowLen));
    const pooled = new Float32Array(dim);
    let pooledTokens = 0;
    for (let w = 0; w < windows; w++) {
      const tokenIds = [allIds[0], ...interior.slice(w * windowLen, (w + 1) * windowLen), allIds[allIds.length - 1]];
      new Int32Array(EM.HEAP32.buffer, idsPtr, tokenIds.length).set(tokenIds);
      const rc = EM._encoder_forward(blobPtr, idsPtr, tokenIds.length, hiddenPtr, 0);
      if (rc !== tokenIds.length) throw new Error(`inline encoder failed: ${rc}`);
      const hidden = new Float32Array(EM.HEAPF32.buffer, hiddenPtr, tokenIds.length * dim);
      for (let t = 0; t < tokenIds.length; t++) {
        for (let d = 0; d < dim; d++) pooled[d] += hidden[t * dim + d];
      }
      pooledTokens += tokenIds.length;
    }
    let norm = 0;
    for (let d = 0; d < dim; d++) {
      pooled[d] /= pooledTokens;
      norm += pooled[d] ** 2;
    }
    norm = Math.sqrt(norm);
    for (let d = 0; d < dim; d++) pooled[d] /= norm;
    return { vector: pooled, text, windows };
  }

  return {
    declaration,
    dim,
    maxSeq,
    embed,
    dispose() {
      EM._free(hiddenPtr);
      EM._free(idsPtr);
      EM._free(blobPtr);
    },
  };
}
