# Quickstart Fixtures

These fixtures provide a tiny deterministic dataset for docs, local smoke
checks, and import/export examples.

Files:

- `quickstart_vectors.jsonl` — source vectors with stable caller IDs
- `quickstart_index.pnck` — Pancake package snapshot from the same vectors
- `quickstart_worker_export.bin` — Worker `/export`-compatible blob from the same vectors

Dataset:

- `doc-1` => `[1, 0, 0, 0]`
- `doc-2` => `[0, 1, 0, 0]`
- `doc-3` => `[0, 0, 1, 0]`

Expected nearest-neighbor behavior for cosine search:

- query `[1, 0, 0, 0]` => top hit `doc-1`
- query `[0, 1, 0, 0]` => top hit `doc-2`
- query `[0, 0, 1, 0]` => top hit `doc-3`

Index config used for both binary fixtures:

- `dim: 4`
- `maxElements: 3`
- `metric: 'cosine'`
- `quantized: true`
- `M: 16`
- `efConstruction: 50`
- `efSearch: 100`
