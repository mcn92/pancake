import Pikelet from 'pikelet-wasm/web';

const app = document.getElementById('app');

async function run() {
  const idx = await Pikelet.create({
    dim: 4,
    maxElements: 10,
    metric: 'cosine',
    quantized: true
  });

  try {
    const idA = idx.add(new Float32Array([1, 0, 0, 0]));
    const idB = idx.add(new Float32Array([0.9, 0.1, 0, 0]));
    const idC = idx.add(new Float32Array([0, 1, 0, 0]));

    const before = idx.search(new Float32Array([1, 0, 0, 0]), 3);
    idx.delete(idC);
    idx.compact();

    const snapshot = idx.export();

    const restored = await Pikelet.restore(snapshot, { maxElements: 10 });

    try {
      const after = restored.search(new Float32Array([1, 0, 0, 0]), 2);
      const importedCount = restored.count;
      const nextId = restored.add(new Float32Array([0, 0, 1, 0]));

      const result = {
        ok: true,
        beforeTopIds: before.map(r => r.id),
        afterTopIds: after.map(r => r.id),
        importedCount,
        finalCount: restored.count,
        nextId,
        ids: { idA, idB, idC }
      };

      window.__PIKELET_BROWSER_SMOKE__ = result;
      app.textContent = JSON.stringify(result);
    } finally {
      restored.dispose();
    }
  } finally {
    idx.dispose();
  }
}

run().catch((error) => {
  const result = {
    ok: false,
    error: error && error.message ? error.message : String(error)
  };
  window.__PIKELET_BROWSER_SMOKE__ = result;
  app.textContent = JSON.stringify(result);
});
