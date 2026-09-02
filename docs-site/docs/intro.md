---
title: Introduction
description: What Pikelet is, what the current project focuses on, and why complete search artifacts matter.
sidebar_position: 1
---

# Introduction

Pikelet is a vector search project built around portable search artifacts. The core engine provides HNSW search over normalized vectors with a row-wise affine `uint8` storage path. The artifact work extends that idea across the whole data lifecycle: graph construction, sketch search, range reads, row hydration, query interpretation, and reranking.

The current center of gravity is the complete `.pancake` profile. A complete artifact contains the index, corpus rows, query interpretation metadata, and evaluation metadata behind a manifest identity. A reader can verify the manifest and segment hashes before serving results.

The docs site is intended to prove that model: every production build compiles these docs into a `.pancake` file and mounts the browser search UI against that file.
