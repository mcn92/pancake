# Search Artifact Demo

This is the smallest demo of Pancake's range-readable Search Artifact path.
It opens a compiled `.pancake-range` artifact, keeps the v2 router segment
resident, and lazily materializes base-layer records through byte-range reads.

From the repository root:

```bash
npm run demo:artifact
```

For full JSON output:

```bash
node examples/search-artifact-demo/demo.js \
  --artifact benchmark_results/layout/pancake-sift1m-u8-metis-split.pancake-range \
  --query-file sift/sift_query.fvecs \
  --queries 10 \
  --k 10 \
  --ef-search 10
```

Expected shape on the current SIFT1M artifact:

- artifact graph: `1,000,000` nodes, `128` dimensions
- resident router: `83,254` records, about `38.9 MiB`
- first 10-query warm stream: `2,689` cumulative lazy range requests
- first 10-query warm stream: about `1.30 MiB` cumulative lazy bytes

The demo intentionally does not load the base index into memory. It uses the
same public API a Worker or HTTP/R2 deployment would use:

```js
const artifact = await Pancake.openRangeArtifactFile('index.pancake-range');
const result = await artifact.search(query, 10, { efSearch: 10 });
```

For HTTP or R2, provide a range source instead of a local file. See
[`range_sources.js`](range_sources.js) for minimal adapters.

HTTP range source:

```js
const { PancakeRangeArtifact } = require('pancake-wasm/artifact');
const { createHttpRangeSource } = require('./range_sources.js');

const source = createHttpRangeSource('https://example.com/index.pancake-range');
const artifact = await PancakeRangeArtifact.open(source);
```

R2 range source:

```js
const source = {
  async read(offset, length) {
    const object = await env.INDEX_BUCKET.get('index.pancake-range', {
      range: { offset, length },
    });
    if (!object) throw new Error('artifact missing');
    return new Uint8Array(await object.arrayBuffer());
  },
};

const artifact = await Pancake.RangeArtifact.open(source);
```
