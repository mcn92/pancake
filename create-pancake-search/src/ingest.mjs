// Ingestion: folder walk and URL crawl, HTML/Markdown/MDX extraction, chunking,
// dedupe, Docusaurus route mapping, and the public chunk shape.

import fs from 'node:fs/promises';
import fssync from 'node:fs';
import path from 'node:path';
import { CliError } from './common.mjs';

const MAX_CRAWL_BODY_BYTES = 2 * 1024 * 1024;
async function ingestFolder(root, source, log) {
  const docs = [];
  if (!fssync.existsSync(root)) {
    throw new CliError(`Source folder not found: ${root}\nNext: check --source or source.path in pancake.config.json.`, 1);
  }
  const files = await walk(root);
  for (const file of files) {
    const rel = path.relative(root, file).split(path.sep).join('/');
    if (!matchesSource(rel, source)) continue;
    try {
      const text = await fs.readFile(file, 'utf8');
      const extracted = extractByExtension(text, file);
      if (extracted.text.trim()) docs.push({ id: docs.length, sourcePath: rel, title: extracted.title || path.basename(file), slug: extracted.slug || null, text: extracted.text, sections: extracted.sections });
    } catch (error) {
      log(`warn: skipped unreadable file ${rel}: ${error.message}`);
    }
  }
  return docs;
}

async function walk(root) {
  const out = [];
  let entries;
  try {
    entries = await fs.readdir(root, { withFileTypes: true });
  } catch (error) {
    throw new CliError(`Unable to read source folder ${root}: ${error.message}`, 1);
  }
  for (const entry of entries) {
    const full = path.join(root, entry.name);
    if (entry.isDirectory()) out.push(...await walk(full));
    else if (entry.isFile()) out.push(full);
  }
  return out;
}

function matchesSource(rel, source) {
  const includes = source.include || ['**/*.{md,mdx,html,txt}'];
  const excludes = source.exclude || [];
  return includes.some((glob) => globMatch(rel, glob)) && !excludes.some((glob) => globMatch(rel, glob));
}

function globMatch(file, glob) {
  const normalized = glob.replace(/\\/g, '/');
  if (normalized.includes('{md,mdx,html,txt}')) return /\.(md|mdx|html|txt)$/i.test(file);
  if (normalized.startsWith('**/*.')) return file.toLowerCase().endsWith(normalized.slice(4).toLowerCase());
  if (normalized.startsWith('*.')) return !file.includes('/') && file.toLowerCase().endsWith(normalized.slice(1).toLowerCase());
  if (normalized.startsWith('**/') && normalized.endsWith('/**')) return file.includes(normalized.slice(3, -3));
  if (normalized.startsWith('**/')) return file.endsWith(normalized.slice(3)) || file.includes(normalized.slice(3).replace('/**', ''));
  return file === normalized || file.startsWith(`${normalized}/`);
}

// Aggregate pages that duplicate a whole site's content in one document —
// crawling them alongside the per-page versions puts every chunk in the
// corpus twice (near-duplicates survive exact-text dedupe because chunk
// boundaries differ). mdBook's print.html is the known offender.
const DEFAULT_URL_EXCLUDES = ['*/print.html'];

// URL patterns are matched against the URL's pathname, full-match, with '*'
// as the only wildcard — deliberately not the filesystem glob dialect the
// folder source uses ('**', braces), which does not map onto URLs.
function urlPatternMatcher(patterns) {
  const regexes = patterns.map((pattern) => new RegExp(
    `^${pattern.split('*').map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('.*')}$`,
  ));
  return (pathname) => regexes.some((regex) => regex.test(pathname));
}

function urlCrawlFilter(source) {
  const excluded = urlPatternMatcher([...DEFAULT_URL_EXCLUDES, ...(source.excludeUrl || [])]);
  const included = source.includeUrl?.length ? urlPatternMatcher(source.includeUrl) : null;
  return (href) => {
    const { pathname } = new URL(href);
    if (excluded(pathname)) return false;
    return included ? included(pathname) : true;
  };
}

// A pure redirect page: HTTP 200 whose HTML is a meta refresh
// (docs.astro.build's root is `<meta http-equiv="refresh"
// content="0;url=/en/getting-started/">` and nothing else).
function metaRefreshTarget(html, baseHref) {
  const meta = html.match(/<meta[^>]*http-equiv\s*=\s*["']?refresh["']?[^>]*>/i);
  if (!meta) return null;
  const url = meta[0].match(/content\s*=\s*["']?[^"'>]*?url\s*=\s*([^"'>\s;]+)/i);
  if (!url) return null;
  try { return new URL(url[1], baseHref); } catch { return null; }
}

// The seed the user typed is followed through redirects (bounded) before
// the crawl starts — both HTTP redirects and meta-refresh pages: sites
// routinely send their root to a canonical host or a localized landing page
// (docs.astro.build -> /en/getting-started/ via meta refresh), and a crawl
// seeded on the pre-redirect URL discovers one empty page, then fails with
// an error that blames the content. The final URL defines the crawl origin.
// Every other fetch keeps the strict redirect-skip: mid-crawl redirects are
// moved or aliased pages whose targets the crawl discovers through links.
async function resolveSeedUrl(seed, log) {
  let current = seed;
  for (let hop = 0; hop < 5; hop++) {
    let response;
    try {
      response = await fetch(current.href, {
        headers: { 'User-Agent': 'create-pancake-search/0.1.0' },
        redirect: 'manual',
        signal: AbortSignal.timeout(15000),
      });
    } catch {
      return current; // the crawl loop will surface the fetch error itself
    }
    const location = response.headers.get('location');
    if (response.status >= 300 && response.status < 400 && location) {
      await response.body?.cancel?.().catch(() => {});
      const next = new URL(location, current);
      log(`seed redirected: ${current.href} -> ${next.href}`);
      current = next;
      continue;
    }
    const type = response.headers.get('content-type') || '';
    if (response.ok && type.includes('text/html')) {
      let html = '';
      try {
        html = await readLimitedText(response, MAX_CRAWL_BODY_BYTES);
      } catch {
        return current;
      }
      const refresh = metaRefreshTarget(html, current.href);
      if (refresh && refresh.href !== current.href) {
        log(`seed redirected (meta refresh): ${current.href} -> ${refresh.href}`);
        current = refresh;
        continue;
      }
      return current;
    }
    await response.body?.cancel?.().catch(() => {});
    return current;
  }
  log(`warn: seed still redirecting after 5 hops; crawling ${current.href}`);
  return current;
}

async function ingestUrl(source, log) {
  const seed = await resolveSeedUrl(new URL(source.url), log);
  const seen = new Set();
  const queue = [seed.href];
  const docs = [];
  const allowed = urlCrawlFilter(source);
  let filtered = 0;
  while (queue.length && docs.length < (source.maxPages || 500)) {
    const href = queue.shift();
    if (seen.has(href)) continue;
    seen.add(href);
    let response;
    try {
      response = await fetch(href, {
        headers: { 'User-Agent': 'create-pancake-search/0.1.0' },
        redirect: 'manual',
        signal: AbortSignal.timeout(15000),
      });
    } catch (error) {
      log(`warn: skipped ${href}: ${error.message}`);
      continue;
    }
    if (response.status >= 300 && response.status < 400) {
      log(`warn: skipped redirect from ${href}`);
      continue;
    }
    const type = response.headers.get('content-type') || '';
    if (!response.ok || !type.includes('text/html')) continue;
    let html;
    try {
      html = await readLimitedText(response, MAX_CRAWL_BODY_BYTES);
    } catch (error) {
      log(`warn: skipped ${href}: ${error.message}`);
      continue;
    }
    const extracted = extractHtml(html);
    // A contentless meta-refresh page is a redirect wearing a 200: skip it
    // as a document (it would waste page budget as an empty doc) and follow
    // its target through the normal frontier filters. Pages with real
    // content and an incidental refresh tag are kept as documents.
    const refresh = metaRefreshTarget(html, href);
    if (refresh && extracted.text.split(/\s+/).filter(Boolean).length < 25) {
      log(`warn: ${href} is a meta-refresh redirect page -> ${refresh.href}`);
      if (refresh.origin === seed.origin && !seen.has(refresh.href)) {
        if (!allowed(refresh.href)) {
          seen.add(refresh.href);
          filtered++;
        } else if (queue.length + docs.length < (source.maxPages || 500)) {
          queue.push(refresh.href);
        }
      }
      continue;
    }
    docs.push({ id: docs.length, url: href, title: extracted.title || href, text: extracted.text, sections: extracted.sections });
    // Filters gate the queue, not the dequeue: a filtered link must never
    // occupy page budget (queue.length counts toward maxPages), or a large
    // excluded section starves discovery of the pages the crawl is for.
    // The seed the user typed bypasses the filters — it seeds the queue
    // directly above.
    for (const link of extractLinks(html, href, seed.origin)) {
      if (seen.has(link)) continue;
      if (!allowed(link)) {
        seen.add(link);
        filtered++;
        continue;
      }
      if (queue.length + docs.length < (source.maxPages || 500)) queue.push(link);
    }
  }
  if (filtered > 0) {
    const filters = [
      ...(source.includeUrl?.length ? [`include ${source.includeUrl.join(' ')}`] : []),
      `exclude ${[...DEFAULT_URL_EXCLUDES, ...(source.excludeUrl || [])].join(' ')}`,
    ];
    log(`URL filters skipped ${filtered} pages (${filters.join('; ')})`);
  }
  // A crawl that discovers almost nothing usually means the site renders its
  // navigation client-side; the crawler only follows server-rendered anchors.
  if (docs.length > 0 && docs.length < 5) {
    log(`warn: crawl discovered only ${docs.length} page${docs.length === 1 ? '' : 's'} from ${seed.href}. `
      + 'If the site builds its navigation with client-side JavaScript, the crawler cannot follow it.');
  }
  return docs;
}

async function readLimitedText(response, maxBytes) {
  const declaredLength = Number(response.headers.get('content-length') || '0');
  if (Number.isFinite(declaredLength) && declaredLength > maxBytes) {
    throw new Error(`response body exceeds ${maxBytes} bytes`);
  }

  const reader = response.body?.getReader?.();
  if (!reader) {
    const text = await response.text();
    if (Buffer.byteLength(text, 'utf8') > maxBytes) throw new Error(`response body exceeds ${maxBytes} bytes`);
    return text;
  }

  const chunks = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > maxBytes) {
      await reader.cancel().catch(() => {});
      throw new Error(`response body exceeds ${maxBytes} bytes`);
    }
    chunks.push(value);
  }
  return Buffer.concat(chunks.map((chunk) => Buffer.from(chunk))).toString('utf8');
}

function extractByExtension(text, file) {
  if (/\.html?$/i.test(file)) return extractHtml(text);
  if (/\.mdx?$/i.test(file)) return extractMarkdown(text);
  const title = (text.match(/^#\s+(.+)$/m) || [])[1];
  return { title, text: stripMarkdown(text) };
}

// GitHub-style heading slugs, deduplicated per document with -1/-2
// suffixes. Mirrors github-slugger — what Docusaurus and GitHub render —
// so Pancake's anchors match the framework's actual rendered ids: trim,
// lowercase, strip everything that is not a letter/number/space/hyphen/
// underscore (Unicode letters survive: "Über uns" -> "über-uns"), then
// replace EACH space with a dash without collapsing runs ("C++ & C#" ->
// "c--c", exactly as github-slugger emits). Inline code and link syntax
// are unwrapped first, approximating the rendered heading text the
// framework slugs. Verified against the real github-slugger by
// test/ingestion/anchors.test.mjs.
function makeSlugger() {
  const used = new Map();
  return (heading) => {
    const base = String(heading || '')
      .replace(/`([^`]*)`/g, '$1')
      .replace(/\[([^\]]+)]\([^)]*\)/g, '$1')
      .trim()
      .toLowerCase()
      .replace(/[^\p{L}\p{N} _-]/gu, '')
      .replace(/ /g, '-');
    const n = used.get(base) || 0;
    used.set(base, n + 1);
    return n === 0 ? base : `${base}-${n}`;
  };
}

// Strip inline markdown from a heading for display and path purposes.
// Underscores stay: snake_case identifiers (ACTION_QUERY_PARAMS) are far
// more common in developer-doc headings than _underscore emphasis_, and
// github-slugger keeps them.
const cleanHeading = (heading) => normalizeText(String(heading || '')
  .replace(/`([^`]*)`/g, '$1')
  .replace(/\[([^\]]+)]\([^)]*\)/g, '$1')
  .replace(/[*~]/g, ''));

// Split a markdown body into heading-delimited sections. Heading lines are
// recognized only outside code fences, so a `# comment` inside a fenced
// shell block never splits a section. Content before the first heading
// becomes a depth-0 preamble section.
function markdownSections(body) {
  const sections = [];
  let fence = null;
  let current = { depth: 0, heading: null, customId: null, lines: [] };
  for (const line of String(body).split('\n')) {
    const f = line.match(/^\s{0,3}(`{3,}|~{3,})/);
    if (f) {
      if (!fence) fence = f[1];
      else if (f[1][0] === fence[0] && f[1].length >= fence.length) fence = null;
      current.lines.push(line);
      continue;
    }
    const h = !fence && line.match(/^(#{1,6})\s+(.+?)\s*$/);
    if (h) {
      sections.push(current);
      current = { ...headingParts(h[2], h[1].length), lines: [] };
      continue;
    }
    // Setext headings: a line of === (depth 1) or --- (depth 2, two or
    // more so a stray list dash cannot promote its neighbor) directly
    // under a non-blank text line, outside fences.
    const setext = !fence && line.match(/^\s{0,3}(=+|-{2,})\s*$/);
    if (setext) {
      const prev = current.lines[current.lines.length - 1];
      if (prev && prev.trim() && !/^\s{0,3}(#|>|[-*+]\s|\d+[.)]\s)/.test(prev)) {
        current.lines.pop();
        sections.push(current);
        current = { ...headingParts(prev.trim(), setext[1][0] === '=' ? 1 : 2), lines: [] };
        continue;
      }
    }
    current.lines.push(line);
  }
  sections.push(current);
  return sections.map((s) => ({ ...s, body: s.lines.join('\n') }));
}

// Split a raw heading into display text, depth, and an explicit id when
// present ({#custom-id}, Docusaurus/Astro style — it wins over the slug).
function headingParts(rawHeading, depth) {
  let heading = rawHeading.replace(/\s+#+\s*$/, '');
  const custom = heading.match(/\s*\{#([A-Za-z0-9_-]+)\}\s*$/);
  if (custom) heading = heading.slice(0, custom.index);
  return { depth, heading: cleanHeading(heading), customId: custom ? custom[1] : null };
}

// Assign heading paths and anchors to raw sections. The path includes the
// section's own heading; a leading h1 that carries the document title is
// structural rather than sectional and stays out of the paths.
function sectionize(rawSections, title, textOf) {
  const slugger = makeSlugger();
  const stack = [];
  const out = [];
  let sawH1Title = false;
  for (const s of rawSections) {
    if (s.depth === 0) {
      const text = textOf(s);
      if (text) out.push({ headingPath: [], anchor: '', text });
      continue;
    }
    const anchor = s.customId || slugger(s.heading);
    const isTitleH1 = s.depth === 1 && !sawH1Title && title
      && s.heading.toLowerCase() === String(title).toLowerCase();
    if (s.depth === 1 && !sawH1Title) sawH1Title = true;
    while (stack.length && stack[stack.length - 1].depth >= s.depth) stack.pop();
    const headingPath = isTitleH1 ? [] : [...stack.map((e) => e.heading).filter(Boolean), s.heading];
    stack.push({ depth: s.depth, heading: isTitleH1 ? null : s.heading });
    const text = textOf(s);
    if (text || s.heading) out.push({ headingPath, anchor, heading: s.heading, text });
  }
  return out;
}

function extractMarkdown(text) {
  let body = String(text || '').replace(/\r/g, '');
  let slug = null;
  let title = null;
  const frontmatter = body.match(/^---\n([\s\S]*?)\n---(?:\n|$)/);
  if (frontmatter) {
    body = body.slice(frontmatter[0].length);
    slug = (frontmatter[1].match(/^slug:\s*["']?([^"'\n]+?)["']?\s*$/m) || [])[1] || null;
    title = (frontmatter[1].match(/^title:\s*["']?([^"'\n]+?)["']?\s*$/m) || [])[1] || null;
  }
  body = body
    .replace(/^import\s[^\n]*$/gm, ' ')
    .replace(/^export\s[^\n]*$/gm, ' ')
    .replace(/<\/?[A-Z][A-Za-z0-9.]*(?:\s[^>]*?)?\/?>/g, ' ');
  const rawSections = markdownSections(body);
  // Title fallback: the first level-1 heading, whichever syntax wrote it
  // (ATX or setext) — the section parser already normalized both.
  if (!title) title = rawSections.find((s) => s.depth === 1)?.heading || null;
  // Sections carry their own heading text as the first line so retrieval
  // (embedding and lexical alike) sees what the section is about.
  const sections = sectionize(rawSections, title,
    (s) => stripMarkdown(s.heading ? `${s.heading}\n${s.body}` : s.body).trim());
  return { title, slug, text: stripMarkdown(body), sections };
}

function extractHtml(html) {
  const title = normalizeText(decodeEntities((html.match(/<title[^>]*>([\s\S]*?)<\/title>/i) || [])[1] || ''));
  let body = (html.match(/<(main|article)[^>]*>([\s\S]*?)<\/\1>/i) || [])[2]
    || (html.match(/<body[^>]*>([\s\S]*?)<\/body>/i) || [])[1]
    || html;
  body = body.replace(/<(script|style|nav|header|footer|aside)[^>]*>[\s\S]*?<\/\1>/gi, ' ');
  const flattenHtml = (fragment) => normalizeText(decodeEntities(String(fragment)
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/(p|li|div|tr|pre)>/gi, '\n')
    .replace(/<[^>]+>/g, ' ')));
  // Heading tags delimit sections; an id attribute on the heading (or, as
  // fallback, on an anchor/element wrapping it) is the section anchor —
  // matching what the site's own anchor links use — with a derived slug
  // when none exists.
  const raw = [];
  let cursor = 0;
  let last = { depth: 0, heading: null, customId: null };
  for (const m of body.matchAll(/<h([1-6])([^>]*)>([\s\S]*?)<\/h\1>/gi)) {
    raw.push({ ...last, body: body.slice(cursor, m.index) });
    const attrId = attrValue(m[2], 'id') || attrValue(m[3], 'id') || null;
    last = { depth: Number(m[1]), heading: normalizeText(decodeEntities(m[3].replace(/<[^>]+>/g, ' '))), customId: attrId };
    cursor = m.index + m[0].length;
  }
  raw.push({ ...last, body: body.slice(cursor) });
  const pageTitle = title.split(/\s*[|·–—-]\s+/)[0] || title;
  const sections = sectionize(raw, pageTitle,
    (s) => (s.heading ? `${s.heading}\n${flattenHtml(s.body)}` : flattenHtml(s.body)).trim());
  return { title, text: flattenHtml(body), sections };
}

// Attribute values may be unquoted (minified HTML — nodejs.org's API docs
// serve href=fs.html); accept double-quoted, single-quoted, or bare.
const attrValue = (tag, name) => {
  const m = tag.match(new RegExp(`\\b${name}\\s*=\\s*("([^"]*)"|'([^']*)'|([^\\s"'<>=\`]+))`, 'i'));
  return m ? (m[2] ?? m[3] ?? m[4]) : null;
};

function extractLinks(html, baseHref, origin) {
  const links = [];
  for (const match of html.matchAll(/href\s*=\s*("([^"]*)"|'([^']*)'|([^\s"'<>=`]+))/gi)) {
    try {
      const url = new URL(match[2] ?? match[3] ?? match[4], baseHref);
      url.hash = '';
      if (url.origin === origin && /^https?:$/.test(url.protocol)) links.push(url.href);
    } catch {}
  }
  return links;
}

function stripMarkdown(text) {
  return normalizeText(text
    .replace(/```[\s\S]*?```/g, (block) => `\n${block}\n`)
    .replace(/!\[[^\]]*]\([^)]*\)/g, ' ')
    .replace(/\[([^\]]+)]\([^)]*\)/g, '$1')
    .replace(/[#>*_`~-]/g, ' '));
}

function decodeEntities(text) {
  return String(text || '')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

function normalizeText(text) {
  return String(text || '').replace(/\r/g, '').replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
}

// Section-aware chunking: a section stays one chunk when it fits (up to
// 1.6x the target, since splitting a coherent section costs more retrieval
// quality than a modestly long chunk), subdivides at paragraph boundaries
// when it does not, and falls back to token windows only inside a single
// oversized paragraph. Every chunk carries the section's heading path and
// anchor; subdivided pieces share their section's anchor. Sections under
// 25 tokens merge into the previous chunk of the same document (their
// content is preserved; their anchor is not a retrieval target). Documents
// without sections (plain .txt) keep the flat token windows.
function chunkDocs(docs, options) {
  const chunks = [];
  chunks.dropped = [];
  const target = options.targetTokens || 256;
  const overlap = Math.floor(target * ((options.overlapPercent || 15) / 100));
  for (const doc of docs) {
    const docTokens = tokenize(doc.text);
    if (docTokens.length < 25) {
      chunks.dropped.push({
        id: doc.id,
        title: doc.title,
        sourcePath: doc.sourcePath,
        url: doc.url,
        tokens: docTokens.length,
      });
      continue;
    }
    const base = () => ({
      id: chunks.length,
      docId: doc.id,
      title: doc.title,
      headingPath: [],
      url: doc.url || doc.sourcePath,
      anchor: '',
      sourcePath: doc.sourcePath || doc.url,
      slug: doc.slug || null,
      text: '',
    });
    if (!Array.isArray(doc.sections) || !doc.sections.length) {
      for (let start = 0; start < docTokens.length; start += Math.max(1, target - overlap)) {
        const slice = docTokens.slice(start, start + target);
        if (slice.length < 25 && chunks.length) {
          chunks[chunks.length - 1].text += ` ${slice.join(' ')}`;
          continue;
        }
        chunks.push({ ...base(), text: slice.join(' ') });
        if (start + target >= docTokens.length) break;
      }
      continue;
    }
    const docStart = chunks.length;
    const keepMax = Math.floor(target * 1.6);
    const push = (section, text) => chunks.push({
      ...base(),
      headingPath: section.headingPath,
      anchor: section.anchor || '',
      text: normalizeText(text),
    });
    for (const section of doc.sections) {
      const tokens = tokenize(section.text);
      if (!tokens.length) continue;
      if (tokens.length < 25) {
        if (chunks.length > docStart) chunks[chunks.length - 1].text += `\n${normalizeText(section.text)}`;
        else push(section, section.text);
        continue;
      }
      if (tokens.length <= keepMax) {
        push(section, section.text);
        continue;
      }
      let piece = [];
      let pieceTokens = 0;
      const flushPiece = () => {
        if (pieceTokens > 0) push(section, piece.join('\n\n'));
        piece = [];
        pieceTokens = 0;
      };
      for (const para of section.text.split(/\n{2,}/)) {
        const paraTokens = tokenize(para).length;
        if (!paraTokens) continue;
        if (paraTokens > keepMax) {
          flushPiece();
          const words = tokenize(para);
          for (let start = 0; start < words.length; start += Math.max(1, target - overlap)) {
            push(section, words.slice(start, start + target).join(' '));
            if (start + target >= words.length) break;
          }
          continue;
        }
        if (pieceTokens + paraTokens > target && pieceTokens > 0) flushPiece();
        piece.push(para);
        pieceTokens += paraTokens;
      }
      flushPiece();
    }
    // A document whose sections were all tiny still needs its 25-token
    // floor honored: merge a lone undersized chunk forward is impossible,
    // so it simply stays (the document itself passed the floor above).
  }
  return chunks;
}

function tokenize(text) {
  return normalizeText(text).split(/\s+/).filter(Boolean);
}

function dedupeChunks(chunks) {
  const seen = new Set();
  const out = [];
  for (const chunk of chunks) {
    const key = chunk.text;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ ...chunk, id: out.length });
  }
  return out;
}

function applySourceRoutes(chunks, config) {
  const baseUrl = config.source?.routeBaseUrl;
  if (!baseUrl) return;
  const routePrefix = String(config.source.routePrefix || '').replace(/^\/+|\/+$/g, '');
  const join = (...parts) => {
    const prefix = String(baseUrl || '/').endsWith('/') ? String(baseUrl || '/') : `${baseUrl}/`;
    const suffix = parts
      .map((part) => String(part || '').replace(/^\/+|\/+$/g, ''))
      .filter(Boolean)
      .join('/');
    return `${prefix}${suffix}`;
  };
  for (const chunk of chunks) {
    let route = String(chunk.sourcePath || chunk.url || '').replace(/\\/g, '/').replace(/^\.\//, '');
    route = route.replace(/\.(md|mdx|html|txt)$/i, '');
    // Docusaurus route conventions: NN- prefixes order the sidebar but are
    // stripped from routes; index/README docs become the directory route.
    route = route.split('/').map((segment) => segment.replace(/^\d+-/, '')).join('/');
    route = route.replace(/(^|\/)(index|readme)$/i, '$1').replace(/\/+$/, '');
    if (chunk.slug) {
      // Front-matter slug: absolute replaces the whole doc path (still under
      // routePrefix); relative replaces the last path segment.
      const slug = String(chunk.slug).replace(/\\/g, '/');
      route = slug.startsWith('/')
        ? slug
        : [route.split('/').slice(0, -1).join('/'), slug.replace(/^\.\//, '')].filter(Boolean).join('/');
    }
    chunk.url = join(routePrefix, route);
  }
}

function publicChunk(chunk) {
  return {
    id: chunk.id,
    docId: chunk.docId,
    title: chunk.title,
    headingPath: chunk.headingPath,
    url: chunk.url,
    anchor: chunk.anchor,
    sourcePath: chunk.sourcePath,
    text: chunk.text,
    preview: chunk.text.slice(0, 220),
  };
}

export {
  MAX_CRAWL_BODY_BYTES,
  ingestFolder,
  walk,
  matchesSource,
  globMatch,
  ingestUrl,
  resolveSeedUrl,
  urlCrawlFilter,
  readLimitedText,
  extractByExtension,
  extractMarkdown,
  extractHtml,
  extractLinks,
  stripMarkdown,
  decodeEntities,
  normalizeText,
  chunkDocs,
  tokenize,
  dedupeChunks,
  applySourceRoutes,
  publicChunk,
};
