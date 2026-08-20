// Thin adapter over the packaged one-file reader (pancake-wasm/complete).
// The widget downloads the whole artifact once and hands the bytes here; we
// serve them through a memory range source. The plugin's webpack alias
// points the bare specifier at the site-resolved module (the repo root when
// building inside the pancake monorepo), exactly like pancake-wasm/artifact.
// This file used to be a second implementation of the PSF1 reader; keeping
// it as an adapter preserves the openCompletePancake(bytes) API the widget
// calls while the format logic lives in one place.
import { openPancakeFile } from 'pancake-wasm/complete';

function memorySource(bytes) {
  return {
    size: bytes.length,
    preferredParallelism: Infinity,
    preferredGapBytes: 2048,
    async read(offset, length) {
      return bytes.subarray(offset, offset + length);
    },
    async close() {},
  };
}

export async function openCompletePancake(bytes, options = {}) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const search = await openPancakeFile(memorySource(u8), {
    rerankParallelism: options.rerankParallelism,
    rerankGap: options.rerankGap,
  });
  return {
    info: () => search.info(),
    query(text, queryOptions = {}) {
      return search.query(text, { k: 8, rerank: options.rerank, ...queryOptions });
    },
    close: () => search.close(),
  };
}
