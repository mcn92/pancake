// MCP server over .pancake knowledge packs: `create-pancake-search mcp
// --pack a.pancake --pack b.pancake` speaks Model Context Protocol on
// stdio, so any MCP client (Claude Code, Claude Desktop, an agent
// framework) can attach compiled packs as a retrieval tool — no vector
// database, embedding service, or retrieval backend. The pack carries the
// corpus, indexes, query encoder, and calibration; this server carries
// nothing but the reader.
//
// Design stances:
// - Provenance is first-class: every result names its pack, the pack's
//   immutable manifest identity (sha256), and the chunk's title, heading
//   path, anchor, and source — so an answer can cite the exact knowledge
//   state it was derived from.
// - Abstention is surfaced, not smoothed over: a pack that cannot answer
//   says so per pack (matchQuality 'none' with zero results), which is
//   exactly what a grounding layer must be able to tell a model.
// - Multi-pack search returns per-pack sections, never a cross-pack
//   fusion: distances are only comparable within one pack's encoder and
//   corpus, and keeping packs separate keeps provenance separable.
// - The protocol layer is hand-rolled: MCP stdio is newline-delimited
//   JSON-RPC 2.0 and this server needs four methods; a protocol SDK would
//   be this package's third runtime dependency for less code than this
//   comment block.
//
// stdout carries only protocol frames (the MCP stdio contract); all
// logging goes to stderr.

import path from 'node:path';
import readline from 'node:readline';

const PROTOCOL_VERSIONS = ['2025-06-18', '2025-03-26', '2024-11-05'];
const SERVER_INFO = { name: 'pancake-knowledge-packs', version: '0.7.0-dev' };
// Result text is chunk-sized by construction (compile targets ~256
// tokens); k stays small so one tool result cannot flood a context window.
const MAX_K = 20;

function toolDefinitions(packs) {
  const packNames = [...packs.keys()];
  const packDescription = packs.size === 1
    ? `Only one pack is mounted (${packNames[0]}); the parameter is optional.`
    : `Omit to search every mounted pack (${packNames.join(', ')}); results stay grouped per pack.`;
  return [
    {
      name: 'search',
      description: 'Search the mounted .pancake knowledge packs. Returns the most relevant '
        + 'chunks with full provenance (pack, immutable pack identity, title, heading path, '
        + 'source) plus a calibrated matchQuality per pack: "strong" means the pack answers '
        + 'this, "weak" means treat results with caution, "none" means this pack does not '
        + 'contain the answer — do not force a citation from a pack that says none.',
      inputSchema: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Natural-language question or keyword query.' },
          pack: { type: 'string', description: `Pack to search. ${packDescription}` },
          k: { type: 'number', description: `Results per pack, 1-${MAX_K} (default 5).` },
        },
        required: ['query'],
      },
    },
    {
      name: 'list_packs',
      description: 'List the mounted knowledge packs: name, immutable identity (sha256 of the '
        + 'pack manifest — cite it to pin the exact knowledge state an answer used), record '
        + 'count, encoder, and sample queries the pack was built to answer.',
      inputSchema: { type: 'object', properties: {} },
    },
    {
      name: 'get_record',
      description: 'Fetch one full record from a pack by the id a search result reported — '
        + 'the complete chunk text plus its provenance, integrity-verified from the pack.',
      inputSchema: {
        type: 'object',
        properties: {
          pack: { type: 'string', description: 'Pack name (optional when only one pack is mounted).' },
          id: { type: 'number', description: 'Record id from a search result.' },
        },
        required: ['id'],
      },
    },
  ];
}

function provenanced(packName, identity, result) {
  return {
    pack: packName,
    packIdentity: identity,
    id: result.id,
    title: result.title ?? null,
    headingPath: Array.isArray(result.headingPath) ? result.headingPath : [],
    anchor: result.anchor ?? null,
    source: result.url ?? result.sourcePath ?? null,
    distance: result.distance,
    text: result.text ?? result.preview ?? null,
  };
}

function resolvePack(packs, requested, toolName) {
  if (requested !== undefined) {
    if (typeof requested !== 'string' || !packs.has(requested)) {
      throw new Error(`${toolName}: unknown pack ${JSON.stringify(requested)} — mounted packs: ${[...packs.keys()].join(', ')}`);
    }
    return [requested];
  }
  return [...packs.keys()];
}

async function callSearch(packs, args) {
  if (typeof args?.query !== 'string' || !args.query.trim()) {
    throw new Error('search: query must be a non-empty string');
  }
  let k = 5;
  if (args.k !== undefined) {
    if (!Number.isSafeInteger(args.k) || args.k < 1 || args.k > MAX_K) {
      throw new Error(`search: k must be an integer between 1 and ${MAX_K}`);
    }
    k = args.k;
  }
  const names = resolvePack(packs, args.pack, 'search');
  const sections = [];
  for (const name of names) {
    const mounted = packs.get(name);
    const out = await mounted.search.query(args.query, { k });
    sections.push({
      pack: name,
      packIdentity: mounted.identity,
      matchQuality: out.matchQuality,
      ...(out.confidence !== undefined ? { confidence: out.confidence } : {}),
      results: out.results.map((r) => provenanced(name, mounted.identity, r)),
    });
  }
  const answered = sections.filter((s) => s.results.length > 0);
  return {
    query: args.query,
    packsSearched: names,
    ...(answered.length === 0 ? {
      note: 'No mounted pack contains an answer to this query. Say so rather than guessing.',
    } : {}),
    sections,
  };
}

async function callGetRecord(packs, args) {
  if (!Number.isSafeInteger(args?.id) || args.id < 0) {
    throw new Error('get_record: id must be a non-negative integer');
  }
  const names = resolvePack(packs, args.pack, 'get_record');
  if (names.length !== 1) {
    throw new Error('get_record: pack is required when more than one pack is mounted');
  }
  const mounted = packs.get(names[0]);
  const record = await mounted.search.record(args.id);
  const shaped = provenanced(names[0], mounted.identity, { ...record, id: args.id });
  delete shaped.distance;
  return shaped;
}

function callListPacks(packs) {
  return {
    packs: [...packs.entries()].map(([name, mounted]) => {
      const info = mounted.search.info();
      return {
        name,
        identity: info.identity,
        file: mounted.file,
        records: info.records,
        encoder: info.encoder?.model ?? info.encoder?.kind ?? null,
        hybridLexical: info.lexical !== null,
        sampleQueries: info.sampleQueries.slice(0, 5),
      };
    }),
  };
}

/**
 * Mount packs and serve MCP on stdio until stdin closes. `openPack` is
 * injected (the CLI passes pancake-wasm/complete's openPancakeFile) so
 * tests can stub it.
 */
export async function runMcpServer({ packPaths, openPancakeFile, stdin = process.stdin, stdout = process.stdout, log = (line) => process.stderr.write(`${line}\n`) }) {
  if (!Array.isArray(packPaths) || packPaths.length === 0) {
    throw new Error('mcp requires at least one --pack <file.pancake>');
  }
  const packs = new Map();
  for (const packPath of packPaths) {
    const resolved = path.resolve(packPath);
    const search = await openPancakeFile(resolved);
    const info = search.info();
    if (info.encoder && info.encoder.kind === 'external-transformers-v1') {
      await search.close();
      throw new Error(`${packPath}: kind-2 packs need a host encoder and cannot be mounted self-contained; `
        + 'compile packs with the inline encoder (the compile default) to serve them over MCP');
    }
    let name = info.name || path.basename(resolved).replace(/\.pancake$/, '');
    // Names address packs in every tool call; collisions get a stable
    // numeric suffix rather than silently shadowing an earlier mount.
    if (packs.has(name)) {
      let n = 2;
      while (packs.has(`${name}-${n}`)) n += 1;
      name = `${name}-${n}`;
    }
    packs.set(name, { search, identity: info.identity, file: resolved });
    log(`mounted ${name} (${info.records} records, identity ${info.identity.slice(0, 12)}…) from ${resolved}`);
  }

  const send = (message) => stdout.write(`${JSON.stringify(message)}\n`);
  const reply = (id, result) => send({ jsonrpc: '2.0', id, result });
  const replyError = (id, code, message) => send({ jsonrpc: '2.0', id, error: { code, message } });

  const rl = readline.createInterface({ input: stdin, crlfDelay: Infinity });
  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let message;
    try { message = JSON.parse(trimmed); } catch {
      replyError(null, -32700, 'parse error');
      continue;
    }
    const { id, method, params } = message;
    // Notifications (no id) get no response, per JSON-RPC.
    const isRequest = id !== undefined && id !== null;
    try {
      if (method === 'initialize') {
        const requested = params?.protocolVersion;
        reply(id, {
          protocolVersion: PROTOCOL_VERSIONS.includes(requested) ? requested : PROTOCOL_VERSIONS[0],
          capabilities: { tools: {} },
          serverInfo: SERVER_INFO,
        });
      } else if (method === 'ping') {
        if (isRequest) reply(id, {});
      } else if (method === 'tools/list') {
        reply(id, { tools: toolDefinitions(packs) });
      } else if (method === 'tools/call') {
        const toolName = params?.name;
        const args = params?.arguments ?? {};
        let result;
        if (toolName === 'search') result = await callSearch(packs, args);
        else if (toolName === 'list_packs') result = callListPacks(packs);
        else if (toolName === 'get_record') result = await callGetRecord(packs, args);
        else throw new Error(`unknown tool ${JSON.stringify(toolName)}`);
        reply(id, { content: [{ type: 'text', text: JSON.stringify(result, null, 1) }] });
      } else if (typeof method === 'string' && method.startsWith('notifications/')) {
        // initialized, cancelled, ... — nothing to do.
      } else if (isRequest) {
        replyError(id, -32601, `method not found: ${method}`);
      }
    } catch (err) {
      const text = err && err.message ? err.message : String(err);
      if (method === 'tools/call' && isRequest) {
        // Tool-level failures are results with isError, not protocol
        // errors — the model sees them and can correct the call.
        reply(id, { content: [{ type: 'text', text }], isError: true });
      } else if (isRequest) {
        replyError(id, -32603, text);
      } else {
        log(`error handling ${method}: ${text}`);
      }
    }
  }

  for (const mounted of packs.values()) await mounted.search.close();
}
