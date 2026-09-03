// Markdown ingestion conformance: nested/duplicate headings, custom ids,
// fenced code, inline code/links in headings, snake_case identifiers,
// setext headings, Unicode headings and slugs, oversized-section splitting.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { check, section } from './harness.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const CPS = path.resolve(here, '..', '..', 'pikelet', 'src', 'ingest.mjs');
const { extractByExtension, chunkDocs, applySourceRoutes, tokenize } = await import(CPS);

export function pipeline(file, { routePrefix = 'docs', targetTokens = 256 } = {}) {
  const text = fs.readFileSync(path.join(here, 'fixtures', file), 'utf8');
  const extracted = extractByExtension(text, file);
  const doc = {
    id: 0,
    sourcePath: file,
    title: extracted.title || path.basename(file),
    slug: extracted.slug || null,
    text: extracted.text,
    sections: extracted.sections,
  };
  const chunks = chunkDocs([doc], { targetTokens, overlapPercent: 15 });
  applySourceRoutes(chunks, { source: { routeBaseUrl: '/', routePrefix } });
  for (const c of chunks) c.href = c.anchor ? `${c.url}#${c.anchor}` : c.url;
  return { doc, chunks };
}

const byAnchor = (chunks, anchor) => chunks.find((c) => c.anchor === anchor);

section('markdown: auth.md (nested, duplicates, custom ids, fences, inline formatting)');
{
  const { doc, chunks } = pipeline('auth.md');
  check('document title from the h1', doc.title === 'Authentication');
  check('every chunk routes to the document URL', chunks.every((c) => c.url === '/docs/auth'), JSON.stringify(chunks.map((c) => c.url)));

  const rotating = byAnchor(chunks, 'rotating-keys');
  check('nested section carries its ancestor path', JSON.stringify(rotating?.headingPath) === '["API keys","Rotating keys"]', JSON.stringify(rotating?.headingPath));
  check('nested section href', rotating?.href === '/docs/auth#rotating-keys', rotating?.href);
  check('nested section text belongs to its section', /issuing a second key first/.test(rotating?.text || ''));

  const dup = byAnchor(chunks, 'rotating-keys-1');
  check('duplicate heading gets the -1 suffix', !!dup && /slug deduplication/.test(dup.text), dup?.anchor);

  check('explicit {#custom-id} wins over the derived slug', byAnchor(chunks, 'custom-hooks')?.headingPath?.[0] === 'Webhooks');
  check('no chunk derived a webhooks slug', !byAnchor(chunks, 'webhooks'));

  const apiKeys = byAnchor(chunks, 'api-keys');
  check('fenced # comment stays inside its section, splits nothing',
    /must not become a heading/.test(apiKeys?.text || '') && !chunks.some((c) => /this comment must not/.test(c.headingPath.join(' '))));

  const config = byAnchor(chunks, 'reading-configjson-files');
  check('inline code in heading unwraps into path and slug', config?.headingPath?.[0] === 'Reading config.json files', JSON.stringify(config?.headingPath));

  const guide = byAnchor(chunks, 'see-the-upgrade-guide');
  check('link in heading slugs to its text, not its URL', guide?.headingPath?.[0] === 'See the upgrade guide', JSON.stringify(guide?.headingPath));

  const snake = byAnchor(chunks, 'action_query_params');
  check('snake_case identifier keeps its underscores', snake?.headingPath?.[0] === 'ACTION_QUERY_PARAMS', JSON.stringify(snake?.headingPath));

  const intro = chunks.find((c) => /clients prove who they are/.test(c.text));
  check('title h1 stays out of heading paths', !!intro && intro.headingPath.length === 0 && intro.anchor === 'authentication',
    JSON.stringify({ path: intro?.headingPath, anchor: intro?.anchor }));
}

section('markdown: unicode.md (setext headings, Unicode slugs)');
{
  const { doc, chunks } = pipeline('unicode.md');
  check('setext === heading becomes the document title', doc.title === 'Guía de la API', doc.title);
  const uber = byAnchor(chunks, 'über-uns');
  check('setext --- heading is depth 2 with a Unicode slug', uber?.headingPath?.[0] === 'Über uns', JSON.stringify(uber?.headingPath));
  const jp = byAnchor(chunks, '日本語の見出し');
  check('fully Japanese heading keeps every character in its anchor', !!jp && /Unicode letters and numbers/.test(jp.text));
  check('a dashed line under a list item is not promoted to a heading',
    /cannot be promoted/.test(jp?.text || '') && !chunks.some((c) => c.headingPath.includes('- not a heading')));
}

section('markdown: oversized sections split at paragraph boundaries');
{
  const paragraphs = Array.from({ length: 30 }, (_, i) =>
    `Paragraph ${i} of the long section explains one more of the many configuration values in enough words to count as a real paragraph for splitting purposes here.`);
  const text = `# Long Doc\n\nIntro paragraph long enough to stand as its own leading chunk of the long document under test today, with sufficient words to pass the floor.\n\n## The long section\n\n${paragraphs.join('\n\n')}\n`;
  const file = 'long.md';
  const extracted = extractByExtension(text, file);
  const doc = { id: 0, sourcePath: file, title: extracted.title, slug: null, text: extracted.text, sections: extracted.sections };
  const chunks = chunkDocs([doc], { targetTokens: 128, overlapPercent: 15 });
  const pieces = chunks.filter((c) => c.anchor === 'the-long-section');
  check('oversized section splits into multiple chunks', pieces.length >= 3, `${pieces.length} pieces`);
  check('every piece shares the section anchor and path',
    pieces.every((c) => c.headingPath.join('>') === 'The long section'));
  check('no piece exceeds the keep-together ceiling',
    pieces.every((c) => tokenize(c.text).length <= Math.floor(128 * 1.6)),
    JSON.stringify(pieces.map((c) => tokenize(c.text).length)));
  check('paragraphs are not split mid-sentence across pieces',
    pieces.every((c) => /here\.$/.test(c.text.trim()) || /section$/.test(c.text.trim())));
}
