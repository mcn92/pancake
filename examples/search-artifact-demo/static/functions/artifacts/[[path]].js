// Cloudflare Pages serves static assets without Range support: a ranged
// request gets a 200 with the full body. This function restores real byte
// ranges for /artifacts/* by slicing the asset at the edge, so the browser's
// lazy range reads work as designed. The full asset is read from the edge
// asset store per request, which is acceptable at demo scale; large artifacts
// belong on a natively range-capable store (R2, S3, jsDelivr) instead.

const RANGE_RE = /^bytes=(\d+)-(\d+)?$/;

export async function onRequestGet(context) {
  const { request, env } = context;
  const assetResponse = await env.ASSETS.fetch(new Request(request.url));
  if (!assetResponse.ok) return assetResponse;

  const rangeHeader = request.headers.get('range');
  if (!rangeHeader) return assetResponse;

  const match = RANGE_RE.exec(rangeHeader.trim());
  if (!match) {
    return new Response('unsupported range', { status: 416 });
  }

  const body = new Uint8Array(await assetResponse.arrayBuffer());
  const start = Number(match[1]);
  const end = match[2] !== undefined ? Math.min(Number(match[2]), body.length - 1) : body.length - 1;
  if (start >= body.length || start > end) {
    return new Response('range not satisfiable', {
      status: 416,
      headers: { 'content-range': `bytes */${body.length}` },
    });
  }

  const slice = body.subarray(start, end + 1);
  return new Response(slice, {
    status: 206,
    headers: {
      'content-type': 'application/octet-stream',
      'content-range': `bytes ${start}-${end}/${body.length}`,
      'content-length': String(slice.byteLength),
      'accept-ranges': 'bytes',
      'cache-control': 'public, max-age=86400',
    },
  });
}
