---
title: One-File Search
description: How the complete .pikelet profile packages index, corpus, query interpretation, and evaluation metadata.
sidebar_position: 2
---

# One-File Search

The complete `.pikelet` profile has four segment types:

- `index`: the embedded sketch artifact used for candidate generation and reranking.
- `corpus`: length-prefixed JSON rows used to hydrate result records.
- `query-interp`: the query encoder contract, inline encoder data, and abstention calibration.
- `evaluation`: acceptance metadata and query sets that let readers prove the artifact they opened.

The browser reader verifies the header identity, recomputes the manifest hash, validates the segment table, checks segment hashes, opens the embedded sketch artifact, embeds the query, searches candidates, hydrates rows, and returns final results.

Kind 3 is the self-contained query path. The artifact carries the tokenizer vocabulary and quantized MiniLM weight blob. The reader supplies the small WASM kernels that run the transformer forward pass.
