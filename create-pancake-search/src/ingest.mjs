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
      if (extracted.text.trim()) docs.push({ id: docs.length, sourcePath: rel, title: extracted.title || path.basename(file), slug: extracted.slug || null, text: extracted.text });
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

async function ingestUrl(source, log) {
  const seed = new URL(source.url);
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
    docs.push({ id: docs.length, url: href, title: extracted.title || href, text: extracted.text });
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
  if (!title) title = (body.match(/^#\s+(.+)$/m) || [])[1];
  return { title, slug, text: stripMarkdown(body) };
}

function extractHtml(html) {
  const title = decodeEntities((html.match(/<title[^>]*>([\s\S]*?)<\/title>/i) || [])[1] || '');
  let body = (html.match(/<(main|article)[^>]*>([\s\S]*?)<\/\1>/i) || [])[2]
    || (html.match(/<body[^>]*>([\s\S]*?)<\/body>/i) || [])[1]
    || html;
  body = body
    .replace(/<(script|style|nav|header|footer|aside)[^>]*>[\s\S]*?<\/\1>/gi, ' ')
    .replace(/<h([1-3])[^>]*>/gi, '\n### ')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/p>/gi, '\n\n')
    .replace(/<[^>]+>/g, ' ');
  return { title, text: normalizeText(decodeEntities(body)) };
}

function extractLinks(html, baseHref, origin) {
  const links = [];
  for (const match of html.matchAll(/href=["']([^"']+)["']/gi)) {
    try {
      const url = new URL(match[1], baseHref);
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

function chunkDocs(docs, options) {
  const chunks = [];
  chunks.dropped = [];
  for (const doc of docs) {
    const tokens = tokenize(doc.text);
    if (tokens.length < 25) {
      chunks.dropped.push({
        id: doc.id,
        title: doc.title,
        sourcePath: doc.sourcePath,
        url: doc.url,
        tokens: tokens.length,
      });
      continue;
    }
    const target = options.targetTokens || 256;
    const overlap = Math.floor(target * ((options.overlapPercent || 15) / 100));
    for (let start = 0; start < tokens.length; start += Math.max(1, target - overlap)) {
      const slice = tokens.slice(start, start + target);
      if (slice.length < 25 && chunks.length) {
        chunks[chunks.length - 1].text += ` ${slice.join(' ')}`;
        continue;
      }
      chunks.push({
        id: chunks.length,
        docId: doc.id,
        title: doc.title,
        headingPath: [],
        url: doc.url || doc.sourcePath,
        anchor: '',
        sourcePath: doc.sourcePath || doc.url,
        slug: doc.slug || null,
        text: slice.join(' '),
      });
      if (start + target >= tokens.length) break;
    }
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
