'use strict';
// pikelet-wasm/artifact — the Search Artifact layer (spec/SEARCH_ARTIFACT_CONTRACT.md).
// Public entry point; the implementation is split by profile:
//   pikelet-artifact-common.js  read budgets + range validation, range sources
//                               (NodeFileRangeSource), result heap, snapshot
//                               parsing (parseUint8Snapshot), SHA-256 helpers
//   pikelet-artifact-range.js   .pancake-range reader (PancakeRangeArtifact) + builder
//   pikelet-artifact-sketch.js  .pancake-sketch reader (PancakeSketchArtifact),
//                               builders, and createSketchScanner
// The export list below is the package's contract (pikelet-artifact.d.ts).

const { NodeFileRangeSource, parseUint8Snapshot } = require('./pikelet-artifact-common.js');
const { PancakeRangeArtifact, buildRangeArtifact, buildRangeArtifactFile } = require('./pikelet-artifact-range.js');
const {
    PancakeSketchArtifact, createSketchScanner,
    buildSketchArtifact, buildSketchArtifactBytes, buildSketchArtifactFile, exportSketchArtifact,
} = require('./pikelet-artifact-sketch.js');

module.exports = {
    PancakeRangeArtifact,
    PancakeSketchArtifact,
    createSketchScanner,
    NodeFileRangeSource,
    buildRangeArtifact,
    buildRangeArtifactFile,
    buildSketchArtifact,
    buildSketchArtifactBytes,
    buildSketchArtifactFile,
    exportSketchArtifact,
    parseUint8Snapshot,
};
