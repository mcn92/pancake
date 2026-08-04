// Shared handler: serve an R2 object with real 206 Range support, immutable
// cache headers, edge caching, and CORS.
//
// Edge cache: identical ranges (the resident-tier open, the encoder, the
// offsets table) are cached at the colo keyed by url+range, so only the
// first visitor per colo pays the R2 read. The Cache API refuses 206
// responses, so ranges are stored as synthetic 200s under a range-suffixed
// key and re-wrapped on hit.
//
// CORS: the demo shards range traffic across Pages branch aliases to escape
// the browser's one-connection-one-isolate serialization (a single h2
// connection funnels every subrequest through one isolate's ~6-connection
// cap to R2). Cross-origin fetches from the shard origins need these
// headers; Range is not a CORS-safelisted header, so OPTIONS is answered
// and cached for a day per path.
//
// Objects over wrangler's 300 MiB upload ceiling are stored as fixed-size
// parts (<key>.p0, <key>.p1, ...) and stitched here; SPLIT_OBJECTS records
// the split geometry chosen at upload time.
const SPLIT_OBJECTS = {
    'pack/corpus.bin': { partSize: 209715200, total: 332294781, parts: 2 },
};

const CORS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Expose-Headers': 'Content-Range, Content-Length, Accept-Ranges, ETag',
};

// Cache-busting version segment: the client threads pack-manifest.json's
// packVersion (a content hash of the artifact) into every pack URL as
// /pack/vXXXX/<name>. The segment exists only to make immutable cache
// entries version-specific — it is stripped before the R2 key lookup, so
// the bucket layout never changes. The manifest itself is the version
// pointer and gets a short cache instead of immutable.
const VERSION_SEGMENT = /^v[0-9a-f]{6,}$/;
const MANIFEST_CACHE = 'public, max-age=300';
// Full-object reads above this are served but never edge-cached: a stray
// crawler doing a no-Range GET of the 332 MB corpus should not tee it into
// cache.put per request per colo. (The 45 MB encoder stays cacheable —
// that is the legitimate full-body read.)
const CACHE_MAX_FULL_BODY = 100 * 1024 * 1024;

// Single-range parser supporting bytes=a-b, bytes=a-, and the suffix form
// bytes=-n (which needs the object size to resolve, so it is converted by
// the caller once the size is known). Multipart ranges are ignored and
// served as full 200s per spec — the cache guard above bounds the damage.
function parseRange(request) {
    const header = request.headers.get('Range');
    if (!header) return null;
    let match = /^bytes=(\d+)-(\d*)$/.exec(header);
    if (match) return { offset: Number(match[1]), end: match[2] === '' ? undefined : Number(match[2]) };
    match = /^bytes=-(\d+)$/.exec(header);
    if (match) return { suffix: Number(match[1]) };
    return null;
}

function resolveSuffix(range, totalSize) {
    if (!range || range.suffix === undefined) return range;
    const length = Math.min(range.suffix, totalSize);
    return { offset: totalSize - length, end: totalSize - 1 };
}

function baseHeaders(contentType, etag) {
    return new Headers({
        'Content-Type': contentType || 'application/octet-stream',
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'public, max-age=31536000, immutable',
        ...(etag ? { ETag: etag } : {}),
        ...CORS,
    });
}

export function onOptions() {
    return new Response(null, {
        status: 204,
        headers: {
            ...CORS,
            'Access-Control-Allow-Methods': 'GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Range',
            'Access-Control-Max-Age': '86400',
        },
    });
}

// Bump CACHE_GENERATION to orphan every previously cached entry at once —
// needed when the caching policy itself changes (e.g. generation 1 stored
// the manifest as immutable, which would have pinned the version pointer
// for a year).
const CACHE_GENERATION = '2';

function cacheKeyFor(request, range) {
    const url = new URL(request.url);
    url.search = `?g=${CACHE_GENERATION}` + (range ? `&r=${range.offset}-${range.end ?? ''}` : '');
    return new Request(url.toString(), { method: 'GET' });
}

async function serveSplit(env, key, spec, range, waitUntil) {
    range = resolveSuffix(range, spec.total);
    const offset = range ? range.offset : 0;
    const end = range ? Math.min(range.end ?? spec.total - 1, spec.total - 1) : spec.total - 1;
    if (offset > end || offset >= spec.total) return new Response('range not satisfiable', { status: 416, headers: CORS });

    const slices = [];
    for (let p = Math.floor(offset / spec.partSize); p <= Math.floor(end / spec.partSize) && p < spec.parts; p++) {
        const partStart = p * spec.partSize;
        const from = Math.max(offset, partStart) - partStart;
        const to = Math.min(end, partStart + spec.partSize - 1) - partStart;
        slices.push({ key: `${key}.p${p}`, offset: from, length: to - from + 1 });
    }

    const headers = baseHeaders('application/octet-stream');
    headers.set('Content-Length', String(end - offset + 1));
    if (range) headers.set('Content-Range', `bytes ${offset}-${end}/${spec.total}`);

    if (slices.length === 1) {
        const object = await env.PACK.get(slices[0].key, { range: { offset: slices[0].offset, length: slices[0].length } });
        if (!object) return new Response('not found', { status: 404, headers: CORS });
        return new Response(object.body, { status: range ? 206 : 200, headers });
    }

    const { readable, writable } = new TransformStream();
    const pump = (async () => {
        try {
            for (const slice of slices) {
                const object = await env.PACK.get(slice.key, { range: { offset: slice.offset, length: slice.length } });
                if (!object) throw new Error(`missing part ${slice.key}`);
                await object.body.pipeTo(writable, { preventClose: true });
            }
            await writable.close();
        } catch (e) {
            await writable.abort(e);
        }
    })();
    waitUntil(pump);
    return new Response(readable, { status: range ? 206 : 200, headers });
}

export async function serveFromR2(context, prefix) {
    const { env, params, request } = context;
    const parts = Array.isArray(params.path) ? [...params.path] : [params.path];
    if (parts.length > 1 && VERSION_SEGMENT.test(parts[0])) parts.shift();
    const key = `${prefix}/${parts.join('/')}`;
    const isManifest = key === 'pack/pack-manifest.json';
    let range = parseRange(request);
    // Captured before suffix resolution mutates `range`: suffix reads are
    // never cached because their cache key needs an absolute offset.
    const hadSuffix = !!(range && range.suffix !== undefined);

    // Edge cache first: hits skip R2 (and the isolate connection cap) entirely.
    const cache = caches.default;
    const cacheKey = cacheKeyFor(request, range);
    const hit = hadSuffix ? null : await cache.match(cacheKey);
    if (hit) {
        const headers = new Headers(hit.headers);
        return new Response(hit.body, { status: range ? 206 : 200, headers });
    }

    let response;
    const split = SPLIT_OBJECTS[key];
    let fullBodySize = split ? split.total : 0;
    if (split) {
        response = await serveSplit(env, key, split, range, context.waitUntil.bind(context));
        range = resolveSuffix(range, split.total);
    } else {
        let object;
        if (range && range.suffix !== undefined) {
            // R2 supports suffix reads natively.
            object = await env.PACK.get(key, { range: { suffix: range.suffix } });
            if (object) range = resolveSuffix(range, object.size);
        } else if (range) {
            // The requested length may run past EOF; R2 clamps it (the split
            // path clamps explicitly instead). A wholly out-of-range offset
            // makes R2 return null, surfacing as 404 rather than the split
            // path's 416 — a known asymmetry, harmless to the reader.
            object = await env.PACK.get(key, {
                range: range.end === undefined ? { offset: range.offset } : { offset: range.offset, length: range.end - range.offset + 1 },
            });
        } else {
            object = await env.PACK.get(key);
        }
        if (!object) return new Response('not found', { status: 404, headers: CORS });
        fullBodySize = object.size;

        const headers = baseHeaders(object.httpMetadata?.contentType, object.httpEtag);
        if (range) {
            const end = range.end === undefined ? object.size - 1 : Math.min(range.end, object.size - 1);
            headers.set('Content-Range', `bytes ${range.offset}-${end}/${object.size}`);
            headers.set('Content-Length', String(end - range.offset + 1));
            response = new Response(object.body, { status: 206, headers });
        } else {
            headers.set('Content-Length', String(object.size));
            response = new Response(object.body, { status: 200, headers });
        }
    }

    if (isManifest) {
        // The manifest is the version pointer — it must be revalidatable so
        // clients pick up a rebuilt pack within minutes.
        response.headers.set('Cache-Control', MANIFEST_CACHE);
    }

    const cacheable = (response.status === 200 || response.status === 206)
        && !isManifest
        && !hadSuffix
        && (range || fullBodySize <= CACHE_MAX_FULL_BODY);
    if (cacheable) {
        // Store as a synthetic 200 (the Cache API rejects 206); the range
        // lives in the key. Content-Range survives in the stored headers and
        // is re-sent on hits.
        const [body, cacheBody] = response.body.tee();
        context.waitUntil(cache.put(cacheKey, new Response(cacheBody, { status: 200, headers: response.headers })));
        return new Response(body, { status: response.status, headers: response.headers });
    }
    return response;
}
