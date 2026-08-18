---
title: Quickstart
description: Build a small Pancake index, add vectors, and query it from JavaScript.
sidebar_position: 1
---

# Quickstart

Install the package and create an engine index:

```js
import Pancake from 'pancake-wasm';

const index = await Pancake.create({
  dim: 384,
  metric: 'cosine',
  M: 12,
  efConstruction: 75,
});

index.add(new Float32Array(384).fill(0.1));
const results = index.search(new Float32Array(384).fill(0.1), 1);
```

Cosine vectors must have non-zero norm. The example uses filled vectors for that reason; a zero vector is rejected by validation before it reaches the index.

For deployable search, prefer building a search artifact rather than shipping a raw engine snapshot. The complete `.pancake` profile is the path that packages index data, corpus rows, query encoder data, and verification metadata together.
