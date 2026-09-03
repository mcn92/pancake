// Worker for inline-transformer passage embedding: one embedder instance per
// worker (own WASM kernel + weight blob), fed chunk texts by index from the
// pool in embed.mjs. Vectors are copied out of the WASM heap before transfer
// — transferring a heap view's buffer would detach the kernel's memory.
import fs from 'node:fs/promises';
import { parentPort, workerData } from 'node:worker_threads';
import { loadCompleteModules } from './common.mjs';

const { declaration, vocabPath, weightsPath } = workerData;
const { builder, reader } = await loadCompleteModules();
const embedder = await reader.createInlineTransformerEmbedder({
  declaration,
  vocabText: await fs.readFile(vocabPath, 'utf8'),
  blob: await fs.readFile(weightsPath),
  createEncoder: await builder.loadInlineEncoderKernel(),
});

parentPort.on('message', async (msg) => {
  if (msg.done) {
    embedder.dispose();
    parentPort.close();
    return;
  }
  try {
    const embedded = await embedder.embed(msg.text);
    const vector = Float32Array.from(embedded.vector);
    parentPort.postMessage({ idx: msg.idx, vector, windows: embedded.windows }, [vector.buffer]);
  } catch (error) {
    parentPort.postMessage({ idx: msg.idx, error: String(error?.message || error) });
  }
});

parentPort.postMessage({ ready: true, maxSeq: embedder.maxSeq });
