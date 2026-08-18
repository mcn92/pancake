---
title: System Design
description: A short map of the current Pancake architecture and where to find the deeper design notes.
sidebar_position: 1
---

# System Design

Pancake's design is built around one repeated choice: store dense vectors and model weights in compact affine `uint8` layouts, then fuse dequantization with the work that consumes those bytes.

At the engine layer, that means compact HNSW storage and distance kernels that avoid unnecessary float materialization. At the artifact layer, it means sketch profiles and range-readable row hydration. At the complete-profile layer, it means query interpretation can use the same philosophy: an inline MiniLM encoder is carried as artifact data and executed by reader-owned WASM kernels.

The deeper design notes currently live in the repository root under `docs/SYSTEM_DESIGN.md`. This page is a placeholder for the Docusaurus migration; it keeps the system-design topic reachable while the long-form document is moved deliberately.
