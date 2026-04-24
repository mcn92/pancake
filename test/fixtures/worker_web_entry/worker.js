import Pancake from 'pancake-wasm/web';

let indexPromise = null;

async function getIndex() {
  if (!indexPromise) {
    indexPromise = (async () => {
      const index = await Pancake.create({
        dim: 4,
        maxElements: 16,
        metric: 'l2',
      });

      index.add([0, 0, 0, 0]);
      index.add([1, 0, 0, 0]);
      index.add([0, 1, 0, 0]);
      return index;
    })();
  }
  return indexPromise;
}

export default {
  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname === '/health') {
      return Response.json({ ok: true });
    }

    if (url.pathname === '/smoke') {
      const index = await getIndex();
      const results = index.search([1, 0, 0, 0], 2);
      return Response.json({
        ok: true,
        count: index.count,
        topId: results[0]?.id ?? null,
        topDistance: results[0]?.distance ?? null,
        resultCount: results.length,
      });
    }

    return new Response('Not found', { status: 404 });
  },
};
