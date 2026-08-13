// Spike: the composed Search Artifact reader — one call from query text to
// hydrated, calibrated results, over the five components that
// examples/03-edge-docs-search ships as separate Worker assets:
//
//   corpus      docs-corpus.json       (records: title/text/preview/anchor/path)
//   index       docs-index.bin         (.pnck snapshot -> sketch tier at open)
//   encoder     docs-student.bin       (distilled int8 student)
//   calibration docs-abstention.json   (feature weights + thresholds)
//   manifest    docs-manifest.json     (component naming + encoder config)
//
// This file exists to answer the composition questions ahead of the complete
// artifact profile (SEARCH_ARTIFACT_CONTRACT.md section 9.4) — its API is a
// draft of the future one-file reader, not a published surface. The
// abstention helpers are extracted verbatim from 03's worker.js so the
// committed golden fixtures transfer as acceptance tests.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import {
    loadStudentModel,
    embedTextWithStudent,
    scoreQuery,
} from '../03-edge-docs-search/student-embedder.mjs';

const require = createRequire(import.meta.url);
const Pancake = require('../../pancake.js');
const { buildSketchArtifact } = require('../../pancake-artifact.js');

// --- Abstention scoring, extracted from worker.js (computeMatchQuality and
// friends), parameterized on the calibration model instead of module globals.

function buildKnownBucketTables(model) {
    const word = new Set();
    let maxIdf = 0;
    for (const entry of model?.wordBuckets || []) {
        if (!Array.isArray(entry) || entry.length < 2) continue;
        const bucket = Number(entry[0]);
        const idf = Number(entry[1]);
        if (Number.isInteger(bucket) && bucket >= 0) {
            word.add(bucket);
            if (idf > maxIdf) maxIdf = idf;
        }
    }
    const char = new Set();
    for (const value of model?.charBuckets || []) {
        const bucket = Number(value);
        if (Number.isInteger(bucket) && bucket >= 0) char.add(bucket);
    }
    return { word, char, maxIdf: maxIdf || 255 };
}

function computeKnownFractions(features, model) {
    if (!model) return { known_word: 0, known_char: 0, n_feats: features.length };
    const tables = model._knownBucketTables || (model._knownBucketTables = buildKnownBucketTables(model));
    let wordKnown = 0;
    let wordTotal = 0;
    let charKnown = 0;
    let charTotal = 0;
    for (const feature of features) {
        if (feature.family === 'word') {
            wordTotal += 1;
            if (tables.word.has(feature.bucket)) wordKnown += 1;
        } else if (feature.family === 'char') {
            charTotal += 1;
            if (tables.char.has(feature.bucket)) charKnown += 1;
        }
    }
    return {
        known_word: wordTotal > 0 ? wordKnown / wordTotal : 0,
        known_char: charTotal > 0 ? charKnown / charTotal : 0,
        n_feats: features.length,
    };
}

function computeHiddenProbe(embedded, model) {
    const probe = model?.hiddenProbe;
    if (!probe || !Array.isArray(probe.weights) || !embedded.hidden) return 0;
    let logit = Number(probe.bias) || 0;
    const limit = Math.min(probe.weights.length, embedded.hidden.length);
    for (let index = 0; index < limit; index++) {
        logit += (Number(probe.weights[index]) || 0) * embedded.hidden[index];
    }
    if (logit >= 0) {
        const z = Math.exp(-logit);
        return 1 / (1 + z);
    }
    const z = Math.exp(logit);
    return z / (1 + z);
}

function computeMatchQuality(hits, embedded, model) {
    if (!model) return { match_quality: 'unscored' };
    const d0 = hits.length > 0 ? hits[0].distance : 1;
    const marginIndex = Math.min(4, hits.length - 1);
    const margin = marginIndex > 0 ? hits[marginIndex].distance - d0 : 0;
    const known = computeKnownFractions(embedded.features, model);
    const signals = {
        d0,
        margin,
        pre_norm: embedded.preNorm,
        known_word: known.known_word,
        known_char: known.known_char,
        hidden_probe: computeHiddenProbe(embedded, model),
        n_feats: known.n_feats,
    };
    return scoreQuery(signals, model);
}

function computePreSearchAbstention(embedded, model) {
    if (!model) return null;
    const thresholds = model.thresholds || {};
    const minFeatures = Number.isInteger(thresholds.minFeatures) ? thresholds.minFeatures : 3;
    const preNormFloor = Number.isFinite(thresholds.preNormFloor) ? thresholds.preNormFloor : 0.4;
    if (embedded.features.length >= minFeatures && embedded.preNorm >= preNormFloor) return null;
    const known = computeKnownFractions(embedded.features, model);
    return scoreQuery({
        d0: 1,
        margin: 0,
        pre_norm: embedded.preNorm,
        known_word: known.known_word,
        known_char: known.known_char,
        hidden_probe: computeHiddenProbe(embedded, model),
        n_feats: known.n_feats,
    }, model);
}

// --- The v3 snapshot envelope maps internal to external ids; the sketch
// profile binds ids positionally, so this reader requires the identity
// mapping (true for any freshly built, never-compacted snapshot).
// CONTAINER LESSON: the complete profile must either carry the id map as a
// segment or require identity mapping of its index segment.
function assertIdentityMapping(snapshotBytes) {
    const view = new DataView(snapshotBytes.buffer, snapshotBytes.byteOffset, snapshotBytes.byteLength);
    if (view.getUint32(0, true) !== 0x504e434b || view.getUint32(4, true) !== 3) return;
    const mappingCount = view.getUint32(24, true);
    for (let i = 0; i < mappingCount; i++) {
        if (view.getUint32(32 + i * 8, true) !== view.getUint32(32 + i * 8 + 4, true)) {
            throw new Error('openDocsSearch: snapshot has a non-identity id mapping; '
                + 'the sketch tier binds ids positionally, so corpus hydration would be wrong');
        }
    }
}

/**
 * Open a composed docs search over the five component files.
 * Returns { query(text, {k}), close(), info() }.
 */
export async function openDocsSearch({ manifestPath, indexPath, encoderPath, calibrationPath, corpusPath }) {
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    const abstention = JSON.parse(fs.readFileSync(calibrationPath, 'utf8'));
    const corpusRaw = JSON.parse(fs.readFileSync(corpusPath, 'utf8'));
    const corpusById = new Map(Object.entries(corpusRaw).map(([id, record]) => [Number(id), record]));
    const student = loadStudentModel(fs.readFileSync(encoderPath));

    const snapshotBytes = fs.readFileSync(indexPath);
    assertIdentityMapping(snapshotBytes);

    // Derive the sketch tier from the snapshot at open. Written through a
    // temp file because the builder is currently path-based — CONTAINER
    // LESSON: buildSketchArtifact needs a bytes-in/bytes-out variant before
    // the compiler can assemble segments without touching disk.
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pancake-one-file-'));
    const sketchPath = path.join(tmpDir, 'index.pancake-sketch');
    buildSketchArtifact(snapshotBytes, sketchPath, {
        recommendedRerank: manifest.efSearch || 120,
    });
    const sketch = await Pancake.openSketchArtifactFile(sketchPath);
    if (sketch.count !== corpusById.size) {
        throw new Error(`openDocsSearch: index count ${sketch.count} != corpus records ${corpusById.size}`);
    }

    return {
        info() {
            return {
                chunks: sketch.count,
                dim: sketch.dim,
                encoder: manifest.encoder || null,
                residentBytes: sketch.residentBytes,
                residentVerified: sketch.stats().residentVerified,
                sampleQueries: manifest.sampleQueries || [],
            };
        },

        async query(text, options = {}) {
            const k = Number.isInteger(options.k) && options.k > 0 ? options.k : 5;
            const trimmed = String(text || '').trim();
            if (!trimmed) throw new Error('query text is required');

            const embedded = embedTextWithStudent(trimmed, student);
            const pre = computePreSearchAbstention(embedded, abstention);
            let hits = [];
            if (!pre) {
                hits = (await sketch.search(embedded.vector, k)).results;
            }
            const quality = pre || computeMatchQuality(hits, embedded, abstention);
            const returned = quality.match_quality === 'none' ? [] : hits;

            return {
                matchQuality: quality.match_quality,
                confidence: quality.confidence,
                results: returned.map((hit) => {
                    const chunk = corpusById.get(hit.id);
                    return {
                        id: hit.id,
                        distance: hit.distance,
                        title: chunk?.title || `Chunk ${hit.id}`,
                        preview: chunk?.preview || '',
                        text: chunk?.text || '',
                        sourcePath: chunk?.sourcePath || '',
                        anchor: chunk?.anchor || '',
                    };
                }),
            };
        },

        async close() {
            await sketch.close();
            fs.rmSync(tmpDir, { recursive: true, force: true });
        },
    };
}

/** Convenience: open directly over 03's committed assets. */
export function docsAssetPaths() {
    const assets = path.join(path.dirname(new URL(import.meta.url).pathname), '..', '03-edge-docs-search', 'assets');
    return {
        manifestPath: path.join(assets, 'docs-manifest.json'),
        indexPath: path.join(assets, 'docs-index.bin'),
        encoderPath: path.join(assets, 'docs-student.bin'),
        calibrationPath: path.join(assets, 'docs-abstention.json'),
        corpusPath: path.join(assets, 'docs-corpus.json'),
    };
}
