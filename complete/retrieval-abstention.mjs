// Client-side abstention scorer for the wiki pack. Mirrors the math in
// ../../calibrate_abstention.mjs exactly: retrieval signals (d0, margin,
// mean10) plus the corpus-vocabulary known-token fraction from the shipped
// bloom filter, standardized and passed through the fitted logistic model.
// Verdict semantics: 'answer' (strong match), 'weak' (closest match is
// distant — shown with a caveat), 'abstain' (nothing useful in the pack).
//
// Assets calibrated by create-pancake-search's self-templates-v2+ may carry
// an additional grounding term (asset.coverage): the fraction of the query's
// content words that appear in the top retrieved passage's text. Every base
// feature measures topic similarity, so "the corpus discusses this area" and
// "this passage answers this question" are indistinguishable without it. The
// term is serialized outside features[]/weights[] deliberately: a reader
// that predates it scores the topic-only model against the same thresholds
// (a conservative degradation) instead of hitting an unknown feature name
// and computing NaN. Word rules (min length, stopwords) ship in the asset so
// builder and reader cannot drift.

export function createAbstentionScorer(asset, bloomBytes) {
    if (!asset || !Array.isArray(asset.weights) || !asset.thresholds) return null;
    const coverageCfg = asset.coverage && typeof asset.coverage.weight === 'number'
        && Number.isFinite(asset.coverage.mean) && Number.isFinite(asset.coverage.std)
        ? asset.coverage : null;
    const coverageStopwords = coverageCfg ? new Set(coverageCfg.stopwords || []) : null;
    const coverageMinLen = coverageCfg ? (coverageCfg.minWordLen || 3) : 3;

    function coverageFrac(text, passageText) {
        const words = (String(text).toLowerCase().match(/[a-z0-9']+/g) || [])
            .filter((w) => w.length >= coverageMinLen && !coverageStopwords.has(w));
        if (!words.length) return 0;
        const passage = new Set(String(passageText || '').toLowerCase().match(/[a-z0-9']+/g) || []);
        const present = (w) => passage.has(w) || passage.has(`${w}s`) || passage.has(`${w}es`)
            || (w.endsWith('s') && passage.has(w.slice(0, -1)));
        return words.filter(present).length / words.length;
    }
    const bloom = new Uint8Array(bloomBytes);
    const bits = asset.vocabBloom.bits;

    function fnv1a(str, seed) {
        let h = 0x811c9dc5 ^ seed;
        for (let i = 0; i < str.length; i++) {
            h ^= str.charCodeAt(i);
            h = Math.imul(h, 0x01000193);
        }
        return (h >>> 0) % bits;
    }
    const SEEDS = [0, 0x9e3779b9];

    function knownFrac(text) {
        const words = text.toLowerCase().match(/[a-z0-9']+/g) || [];
        if (!words.length) return 0;
        let known = 0;
        for (const w of words) {
            const hit = SEEDS.every((seed) => {
                const bit = fnv1a(w, seed);
                return (bloom[bit >> 3] >> (bit & 7)) & 1;
            });
            if (hit) known++;
        }
        return known / words.length;
    }

    return {
        // The caller must hydrate the top result's text before scoring when
        // this is true; the coverage term reads it.
        usesPassage: !!coverageCfg,
        score(queryText, results, topPassageText) {
            const d0 = results.length ? results[0].distance : 1;
            const margin = results.length > 1
                ? results[Math.min(4, results.length - 1)].distance - d0 : 0;
            const mean10 = results.length
                ? results.reduce((s, r) => s + r.distance, 0) / results.length : 1;
            const signals = { d0, margin, mean10, known_frac: knownFrac(queryText) };
            let z = asset.bias;
            asset.features.forEach((f, j) => {
                z += ((signals[f] - asset.standardize.mean[f]) / asset.standardize.std[f]) * asset.weights[j];
            });
            if (coverageCfg) {
                signals.coverage1 = coverageFrac(queryText, topPassageText);
                z += ((signals.coverage1 - coverageCfg.mean) / (coverageCfg.std || 1)) * coverageCfg.weight;
            }
            // A malformed asset (unknown feature name, non-numeric term) must
            // degrade to unscored, never to a NaN that compares false against
            // both thresholds and answers everything.
            if (!Number.isFinite(z)) return { p: null, verdict: 'unscored', signals };
            const p = 1 / (1 + Math.exp(-z));
            const verdict = p < asset.thresholds.hard ? 'abstain'
                : p < asset.thresholds.weak ? 'weak' : 'answer';
            return { p, verdict, signals };
        },
    };
}
