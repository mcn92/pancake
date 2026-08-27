// Build-time abstention calibration for complete kind-3 artifacts. Mirrors
// the signal and scoring math of the wiki pack calibrator
// (examples/04-static-wiki-pack/calibrate_abstention.mjs) and of the reader's
// scorer (complete/retrieval-abstention.mjs) — retrieval signals (d0, margin,
// mean10) plus the corpus-vocabulary known-token fraction, standardized and
// passed through a fitted logistic model — but is corpus-generic: positives
// are templated questions from chunk titles verified by retrieval, negatives
// are a built-in off-domain query bank plus synthetic out-of-vocabulary
// gibberish. No held-out shard: near-domain-but-absent negatives need corpus
// structure this module cannot assume, so the fit leans on the off-domain and
// gibberish separation and reports what it measured in the asset.
//
// Returns { calibrationJson, summary } on success, or null (with a logged
// reason) when the corpus cannot support a trustworthy fit — the caller then
// ships the unscored placeholder, which is strictly safer than a bad model.

const SEED = 424242;
const FEATS = ['d0', 'margin', 'mean10', 'known_frac'];
const BLOOM_SEEDS = [0, 0x9e3779b9];
const MAX_POSITIVES = 96;
const GIBBERISH_QUERIES = 24;
const MIN_VERIFIED_POSITIVES = 4;
const MIN_NEGATIVES = 8;
const MIN_AUC = 0.85;

// Off-domain but real-English queries, spanning consumer, developer, finance,
// and household domains. Same role as the wiki calibrator's foreign title
// bank: teach the scorer what a well-formed query with no corpus support
// looks like, beyond what gibberish covers.
const FOREIGN_TITLE_BANK = [
  '1040 tax form earned income credit', 'mortgage escrow shortage statement',
  'kubernetes crashloopbackoff', 'react usestate batching',
  'postgres vacuum analyze', 'aws iam role trust policy',
  'stripe webhook signature verification', 'chase sapphire annual fee waiver',
  'tsa precheck renewal appointment', 'iphone battery health service message',
  'netgear router admin password reset', 'excel vlookup not available error',
  'docker compose port binding', 'github actions cache miss',
  'python nonetype subscriptable error', 'medicare part d formulary exception',
  'irs quarterly estimated tax payment', 'student loan income driven repayment',
  'car alternator belt squeal', 'ev charger nema 14-50 permit',
  'tenant security deposit demand letter', 'small claims filing fee',
  'hsa eligible expense receipt', 'merchant chargeback reason code',
  'oauth redirect uri mismatch', 'terraform state lock',
  'nginx reverse proxy websocket upgrade', 'redis eviction policy',
  'pandas dataframe groupby transform', 'homeowners insurance deductible claim',
  'credit card balance transfer fee', 'passport expedited renewal appointment',
  'property tax homestead exemption', 'medical prior authorization denial',
  'printer offline windows settings', 'wifi mesh backhaul channel',
  'airbnb host cancellation refund', 'uber driver cancelled ride refund',
  'shopify abandoned cart email', 'quickbooks payroll tax deposit',
  'salesforce validation rule formula', 'figma component variant property',
  'blender cycles render noise', 'unity rigidbody collision layer',
  'rust borrow checker lifetime error', 'go module replace directive',
  'java maven dependency conflict', 'android gradle signing config',
  'ios provisioning profile expired', 'linux systemd service restart loop',
  'windows bitlocker recovery key', 'router dns over https setting',
  'zelle payment pending review', 'venmo instant transfer fee',
  'mortgage refinance closing disclosure', 'california dmv real id appointment',
  'new york parking ticket dispute', 'texas franchise tax no tax due report',
  'college fafsa dependency override', 'w2 corrected form box 12 code',
  'health insurance out of network appeal', 'clinic cpt code billing modifier',
  'jira workflow transition condition', 'slack app manifest oauth scope',
  'zoom webinar registration limit', 'mailchimp dkim authentication',
  'cloudflare cname flattening', 'dns txt spf include limit',
  'elasticsearch shard relocation', 'snowflake warehouse auto suspend',
  'datadog log exclusion filter', 'prometheus alertmanager silence',
  'kafka consumer group lag', 's3 lifecycle transition rule',
  'azure managed identity role assignment', 'gcp service account key rotation',
  'kubernetes ingress tls secret', 'nextjs server action form data',
  'tailwind container query plugin', 'vite dependency prebundle cache',
  'playwright trace viewer',
];

const TEMPLATES = [
  (t) => `what is ${t}`,
  (t) => `tell me about ${t}`,
  (t) => `${t} explained`,
  (t) => `facts about ${t}`,
  (t) => `information about ${t}`,
  (t) => `overview of ${t}`,
  (t) => `history of ${t}`,
  (t) => `definition of ${t}`,
  (t) => `who is ${t}`,
  (t) => `where is ${t}`,
  (t) => `why is ${t} important`,
  (t) => `how does ${t} work`,
];

function rng(seed) {
  let state = seed >>> 0;
  return () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 0xffffffff);
}

function sample(items, n, seed) {
  const next = rng(seed);
  const picked = [];
  const used = new Set();
  while (picked.length < n && used.size < items.length) {
    const i = Math.floor(next() * items.length);
    if (used.has(i)) continue;
    used.add(i);
    picked.push(items[i]);
  }
  return picked;
}

function titleQuestions(titles, n, seed) {
  const per = Math.min(TEMPLATES.length, Math.max(1, Math.floor(n / titles.length)));
  const seen = new Set();
  return sample(titles, Math.min(n, titles.length * TEMPLATES.length), seed)
    .flatMap((t, i) => Array.from({ length: per }, (_, j) => ({
      text: TEMPLATES[(i + j) % TEMPLATES.length](t.toLowerCase()),
      sourceTitle: t,
    })))
    .filter(({ text }) => !seen.has(text) && seen.add(text))
    .slice(0, n);
}

const STOPWORDS = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'into', 'through', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'it', 'its', 'this', 'that', 'these', 'those', 'as', 'their', 'they', 'them', 'also', 'can', 'will', 'which', 'when', 'where', 'while', 'his', 'her', 'has', 'have', 'had', 'not', 'often', 'usually', 'without']);

// Keyword-style positives sampled from chunk content words. Title-templated
// questions retrieve too easily — every fit positive lands near the corpus
// and the hard threshold creeps up until honest paraphrases abstain. These
// sit closer to how a real query scores, dragging the answerable floor down
// to where it belongs.
function contentWordQuestions(chunks, n, seed) {
  const next = rng(seed);
  const picked = sample(chunks.map((chunk, pos) => ({ chunk, pos })), Math.min(n, chunks.length * 2), seed ^ 0x77aa11);
  const seen = new Set();
  const out = [];
  for (const { chunk, pos } of picked) {
    const words = [...new Set(tokenize(chunk.text).filter((w) => w.length >= 4 && !STOPWORDS.has(w)))];
    if (words.length < 3) continue;
    const count = Math.min(3 + Math.floor(next() * 3), words.length);
    const start = Math.floor(next() * Math.max(1, words.length - count));
    const text = words.slice(start, start + count).join(' ');
    if (seen.has(text)) continue;
    seen.add(text);
    out.push({ text, sourceId: pos });
    if (out.length >= n) break;
  }
  return out;
}

const tokenize = (text) => (text.toLowerCase().match(/[a-z0-9']+/g) || []);

function fnv1a(str, seed, bits) {
  let h = 0x811c9dc5 ^ seed;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0) % bits;
}

function buildVocabBloom(chunks) {
  const counts = new Map();
  for (const chunk of chunks) {
    for (const w of tokenize(chunk.text)) counts.set(w, (counts.get(w) || 0) + 1);
  }
  // Small corpora cannot afford the wiki pack's appears-3-times floor: with a
  // handful of documents most content words appear once or twice and the
  // known-token signal would read every real query as out-of-vocabulary.
  const minCount = chunks.length >= 200 ? 3 : 1;
  const kept = [...counts.entries()].filter(([, c]) => c >= minCount).map(([w]) => w);
  let bits = 1 << 14;
  while (bits < kept.length * 32 && bits < (1 << 21)) bits <<= 1;
  const bloom = new Uint8Array(bits / 8);
  for (const w of kept) {
    for (const seed of BLOOM_SEEDS) {
      const bit = fnv1a(w, seed, bits);
      bloom[bit >> 3] |= 1 << (bit & 7);
    }
  }
  return { bloom, bits, minCount, keptWords: kept.length, uniqueWords: counts.size };
}

function knownFrac(text, bloom, bits) {
  const words = tokenize(text);
  if (!words.length) return 0;
  let known = 0;
  for (const w of words) {
    const hit = BLOOM_SEEDS.every((seed) => {
      const bit = fnv1a(w, seed, bits);
      return (bloom[bit >> 3] >> (bit & 7)) & 1;
    });
    if (hit) known++;
  }
  return known / words.length;
}

function syntheticGibberish(n, seed, bloom, bits) {
  const next = rng(seed);
  const alphabet = 'abcdefghijklmnopqrstuvwxyz';
  const queries = [];
  const used = new Set();
  function token() {
    const len = 5 + Math.floor(next() * 6);
    let s = '';
    for (let i = 0; i < len; i++) s += alphabet[Math.floor(next() * alphabet.length)];
    return s;
  }
  while (queries.length < n && used.size < n * 100) {
    const words = [];
    const wordCount = 3 + Math.floor(next() * 4);
    for (let i = 0; i < wordCount; i++) words.push(token());
    const text = words.join(' ');
    used.add(text);
    if (knownFrac(text, bloom, bits) === 0) queries.push(text);
  }
  return queries;
}

function aucFor(posRows, negRows) {
  if (!posRows.length || !negRows.length) return null;
  let score = 0;
  for (const a of posRows) for (const c of negRows) score += a.p > c.p ? 1 : a.p === c.p ? 0.5 : 0;
  return score / (posRows.length * negRows.length);
}

export async function calibrateRetrievalAbstention({ Pancake, chunks, vectors, config, embedQuery, log = () => {} }) {
  const skip = (reason) => {
    log(`Abstention calibration skipped: ${reason}; the artifact will report match_quality "unscored"`);
    return null;
  };
  const titles = [...new Set(chunks.map((c) => (c.title || '').trim()).filter(Boolean))];
  if (titles.length < 2) return skip('needs at least 2 distinct chunk titles for verified positives');

  const { bloom, bits, minCount, keptWords, uniqueWords } = buildVocabBloom(chunks);
  const K = Math.min(10, chunks.length);
  // Signals must be computed the way the reader computes them from its own
  // search hits (complete/retrieval-abstention.mjs): d0, the rank-4 margin,
  // and the mean over the returned list.
  const signalsFor = (text, hits) => {
    const top = hits.slice(0, K);
    const d0 = top.length ? top[0].distance : 1;
    const margin = top.length > 1 ? top[Math.min(4, top.length - 1)].distance - d0 : 0;
    const mean10 = top.length ? top.reduce((s, r) => s + r.distance, 0) / top.length : 1;
    return { d0, margin, mean10, known_frac: knownFrac(text, bloom, bits) };
  };

  const index = await Pancake.create({
    dim: config.embedding.dims,
    maxElements: Math.max(chunks.length, Math.ceil(chunks.length * 1.25)),
    metric: config.index.metric,
    quantized: config.index.quantized !== false,
  });
  try {
    index.addBatch(vectors);
    const search = async (text) => index.search(await embedQuery(text), K);

    const rows = [];
    let droppedPositives = 0;
    const positiveTemplates = [
      ...titleQuestions(titles, Math.ceil(MAX_POSITIVES / 2), SEED ^ 0x51f15e),
      ...contentWordQuestions(chunks, Math.floor(MAX_POSITIVES / 2), SEED ^ 0xc0ffee),
    ];
    for (const { text, sourceTitle, sourceId } of positiveTemplates) {
      const hits = await search(text);
      // A positive counts only when retrieval verifiably lands on the source
      // document — otherwise the query is unanswerable in practice and would
      // drag the no-false-abstain floor toward zero. Title queries verify by
      // title (any chunk of the source document counts); content-word queries
      // verify by the chunk they were sampled from.
      const found = sourceId !== undefined
        ? hits.some((h) => h.id === sourceId)
        : hits.some((h) => (chunks[h.id]?.title || '').trim() === sourceTitle);
      if (found) rows.push({ text, label: 1, ...signalsFor(text, hits) });
      else droppedPositives++;
    }
    const positives = rows.filter((r) => r.label === 1);
    if (positives.length < MIN_VERIFIED_POSITIVES) {
      return skip(`only ${positives.length} of ${positiveTemplates.length} templated queries verified by retrieval (need ${MIN_VERIFIED_POSITIVES})`);
    }
    const positiveD0 = positives.map((r) => r.d0).sort((a, b) => a - b);
    const positiveMedianD0 = positiveD0[Math.floor(positiveD0.length / 2)];

    let droppedForeign = 0;
    for (const { text } of titleQuestions(FOREIGN_TITLE_BANK, FOREIGN_TITLE_BANK.length, SEED ^ 0xf03e16)) {
      const sig = signalsFor(text, await search(text));
      // An off-domain query that retrieves as strongly as a median positive
      // overlaps the corpus domain; keeping it would teach the scorer to
      // abstain on useful matches.
      if (sig.d0 <= positiveMedianD0) droppedForeign++;
      else rows.push({ text, label: 0, negativeKind: 'foreign-bank', ...sig });
    }
    for (const text of syntheticGibberish(GIBBERISH_QUERIES, SEED ^ 0x9166e11, bloom, bits)) {
      rows.push({ text, label: 0, negativeKind: 'synthetic-gibberish', ...signalsFor(text, await search(text)) });
    }
    const negatives = rows.filter((r) => r.label === 0);
    if (negatives.length < MIN_NEGATIVES) return skip(`only ${negatives.length} negatives survived the overlap drop (need ${MIN_NEGATIVES})`);

    // Weak band: retained-title questions whose source lands at rank 5..K on
    // corpora deep enough to have one. They keep the weak threshold honest so
    // adjacent content is shown with a caveat instead of hidden.
    if (chunks.length >= 25) {
      const weakTemplates = titleQuestions(titles, MAX_POSITIVES, SEED ^ 0x0ddba11);
      for (const { text, sourceTitle } of weakTemplates) {
        if (rows.filter((r) => r.label === -1).length >= 24) break;
        const hits = await search(text);
        const rank = hits.findIndex((h) => (chunks[h.id]?.title || '').trim() === sourceTitle) + 1;
        if (rank >= 5) rows.push({ text, label: -1, ...signalsFor(text, hits) });
      }
    }

    // Fit: logistic regression on standardized signals, gradient descent —
    // the same optimizer, weighting, and regularization as the wiki
    // calibrator so the two assets stay comparable.
    const fit = rows.filter((r) => r.label >= 0);
    const fitWeights = fit.map((r) => (r.negativeKind === 'synthetic-gibberish' ? 0.25 : 1));
    const weightSum = fitWeights.reduce((a, c) => a + c, 0);
    const mean = {}, std = {};
    for (const f of FEATS) {
      mean[f] = fit.reduce((sum, r, i) => sum + r[f] * fitWeights[i], 0) / weightSum;
      std[f] = Math.sqrt(fit.reduce((sum, r, i) => sum + ((r[f] - mean[f]) ** 2) * fitWeights[i], 0) / weightSum) || 1;
    }
    const xs = fit.map((r) => FEATS.map((f) => (r[f] - mean[f]) / std[f]));
    const ys = fit.map((r) => r.label);
    let w = FEATS.map(() => 0);
    let b = 0;
    for (let epoch = 0; epoch < 4000; epoch++) {
      const gw = FEATS.map(() => 0);
      let gb = 0;
      for (let i = 0; i < xs.length; i++) {
        const z = xs[i].reduce((s, v, j) => s + v * w[j], b);
        const p = 1 / (1 + Math.exp(-z));
        const err = (p - ys[i]) * fitWeights[i];
        xs[i].forEach((v, j) => { gw[j] += err * v; });
        gb += err;
      }
      w = w.map((wj, j) => wj - 0.1 * (gw[j] / weightSum + 1e-3 * wj));
      b -= 0.1 * (gb / weightSum);
    }
    const prob = (r) => 1 / (1 + Math.exp(-(FEATS.reduce((s, f, j) => s + ((r[f] - mean[f]) / std[f]) * w[j], b))));
    for (const r of rows) r.p = prob(r);

    const auc = aucFor(rows.filter((r) => r.label === 1), negatives);
    if (auc === null || auc < MIN_AUC) {
      return skip(`fit separates answerable from off-domain at AUC ${auc === null ? 'n/a' : auc.toFixed(3)} (< ${MIN_AUC})`);
    }

    // Threshold placement, as in the wiki calibrator: hard between the
    // negative ceiling and the answerable/weak floor, weak between the weak
    // ceiling and the answerable floor when a weak band exists.
    const pos = positives.map((r) => r.p);
    const neg = negatives.map((r) => r.p);
    const minPos = Math.min(...pos);
    const maxNeg = Math.max(...neg);
    // A weak row scoring near the negative ceiling is indistinguishable from
    // a negative — protecting it would drag the hard threshold toward zero
    // and give off-domain queries a weak verdict instead of abstaining
    // (observed on a docs corpus whose auto-generated weak queries retrieve
    // with near-zero probability: the floor collapsed and gibberish scored
    // "weak"). The fit saturates when separation is clean, so "near" needs
    // an absolute floor as well as the relative one: a weak row the model
    // scores below 5% answerable is unanswerable in all but name.
    const allWeakP = rows.filter((r) => r.label === -1).map((r) => r.p);
    const weakP = allWeakP.filter((p) => p > Math.max(maxNeg, 0.05));
    const minWeak = weakP.length ? Math.min(...weakP) : minPos;
    const maxWeak = weakP.length ? Math.max(...weakP) : 0;
    const floor = Math.min(minPos, minWeak);
    const hardOverlap = maxNeg >= floor;
    // With a weak band the floor sits low and the wiki calibrator's geometric
    // mean places hard well; without one (small corpora) the floor is the
    // weakest *verified* positive — still an easy query once the logistic
    // saturates — and the geometric mean abstains on honest paraphrases that
    // score between the negatives and it. The costs are asymmetric: a false
    // abstain hides results, a false weak shows them with a caveat. So hard
    // hugs the negative ceiling and the weak verdict owns the uncertain band.
    const hard = hardOverlap ? floor * 0.5
      : weakP.length ? Math.sqrt(maxNeg * floor)
        : maxNeg + 0.1 * (floor - maxNeg);
    const weak = weakP.length && maxWeak > 0 && maxWeak < minPos ? Math.sqrt(maxWeak * minPos) : minPos * 0.9;

    const summary = {
      method: 'self-templates-v1',
      seed: SEED,
      searchConfig: { k: K },
      verifiedPositiveQueries: positives.length,
      positivesDroppedAsUnretrievable: droppedPositives,
      foreignNegativeQueries: negatives.filter((r) => r.negativeKind === 'foreign-bank').length,
      foreignDroppedAsSemanticOverlap: droppedForeign,
      foreignDropD0Threshold: +positiveMedianD0.toFixed(6),
      syntheticGibberishQueries: negatives.filter((r) => r.negativeKind === 'synthetic-gibberish').length,
      syntheticGibberishFitWeight: 0.25,
      weakQueries: weakP.length,
      weakDroppedAsIndistinguishableFromNegatives: allWeakP.length - weakP.length,
      auc: +auc.toFixed(6),
      hardThresholdOverlap: hardOverlap,
      vocab: { uniqueWords, keptWords, minCount },
    };
    const asset = {
      version: 1,
      corpus: config.name,
      searchConfig: { k: K },
      calibration: summary,
      features: FEATS,
      standardize: { mean, std },
      weights: w,
      bias: b,
      thresholds: { hard: +hard.toFixed(6), weak: +weak.toFixed(6) },
      vocabBloom: { bits, hashes: ['fnv1a:0', 'fnv1a:0x9e3779b9'], minCount },
    };
    return {
      calibrationJson: {
        kind: 'retrieval-signals-v1',
        asset,
        vocabBloomBase64: Buffer.from(bloom).toString('base64'),
      },
      summary,
    };
  } finally {
    index.dispose();
  }
}
