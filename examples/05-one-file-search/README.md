# One-file search (spike)

The composed Search Artifact reader: one call from query text to hydrated,
calibrated results.

```js
import { openDocsSearch, docsAssetPaths } from './search-reader.mjs';

const search = await openDocsSearch(docsAssetPaths());
const out = await search.query('how do workers restore snapshots');
// { matchQuality: 'strong', confidence: 0.94, results: [{ title, text, sourcePath, anchor, ... }] }
```

This is a **spike** toward the complete artifact profile
(`spec/SEARCH_ARTIFACT_CONTRACT.md` section 9.4): it chains the five
components that `examples/03-edge-docs-search` ships as separate Worker
assets — corpus, index, encoder, evaluation, calibration — through a single
facade, with the index served from a sketch tier (the lazy, range-readable
path the future one-file format will embed). The API here is a draft of the
future `open('docs.pancake').query(text)` reader, not a published surface.

Run it:

```bash
node examples/05-one-file-search/demo.mjs "how does compaction work"
node examples/05-one-file-search/demo.mjs        # manifest's sample queries
node examples/05-one-file-search/test.mjs        # abstention-golden acceptance
```

## What the spike answered

1. **Does the composition work outside the Worker?** Yes, cleanly. The
   encoder module (`student-embedder.mjs`) imports with no workerd coupling;
   the abstention helpers extracted from `worker.js` are pure functions of
   (embedded query, hits, calibration model). All 10 committed abstention
   goldens reproduce their exact labels through this reader.
2. **Does the sketch tier preserve calibration?** Yes — the goldens were
   calibrated against snapshot-restore search, and every label (including
   the borderline `weak` foreign query at 0.099 confidence) survives the
   switch to sketch-scan + rerank candidates unchanged.
3. **What must the container format provide?** See lessons below.

## Container-format lessons (input to the section 9.4 byte layout)

- **Id binding:** the sketch tier binds ids positionally, but `.pnck`
  snapshots carry an internal→external id map. This reader refuses
  non-identity mappings; the complete profile must either carry the id map
  as a segment or require identity mapping of its index segment at compile
  time (compile-time renumbering is the cleaner answer).
- **Bytes-in/bytes-out builders:** `buildSketchArtifact` is path-based, so
  the sketch tier detours through a temp file at open. The compiler needs
  byte-level segment assembly. (Better: the compiler should embed the
  *sketch artifact itself* as the index segment, so the reader opens it
  directly with range reads instead of deriving it.)
- **Corpus segment shape:** hydration needed exactly
  `{title, text, preview, sourcePath, anchor}` by record id. JSON keyed by
  id works at 208 records; the one-file format needs the id→byte-range
  layout the contract's section 4.8 already prescribes.
- **Encoder + calibration are coupled:** the abstention model consumes the
  encoder's feature stream (`embedded.features`, `embedded.hidden`,
  `preNorm`), not just its vector. The container should treat
  encoder+calibration as one query-interpretation unit with a shared
  version, or declare the feature-stream contract between them explicitly.
- **The manifest already exists in embryo:** 03's `docs-manifest.json`
  names every component and inlines the encoder config. The section 4.1
  canonical manifest is this plus content digests and segment offsets.

## Deliberately out of scope

No container file yet (that is task 3), no browser build, no UI, no changes
to `pancake-wasm` or to 03's Worker.
