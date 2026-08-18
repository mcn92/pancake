---
title: Kind 3 Inline Transformer
description: How the inline-transformer-v1 query interpretation segment carries MiniLM as verified artifact data.
sidebar_position: 2
---

# Kind 3 Inline Transformer

Kind 3 is the complete-profile query interpretation mode named `inline-transformer-v1`. The artifact carries three pieces of encoder data: a JSON declaration, a WordPiece vocabulary, and a block-affine `uint8` MiniLM weight blob.

The reader supplies the code. In this repository that code is a small WASM module compiled from the encoder kernel. The kernel performs MiniLM-L6 forward inference over the quantized weight blob and keeps dense projection work fused with dequantization.

This split is intentional. Model identity, revision, license, layout, vocabulary, and weights are artifact data. Execution kernels are reader capability. That keeps a `.pancake` file verifiable while avoiding a host dependency on Python, PyTorch, Transformers, or a remote embedding service.
