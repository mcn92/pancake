// The PSF1 wire-format constants (spec/COMPLETE_PROFILE.md), shared by the
// reader (index.mjs) and the builder (builder.mjs) so the format has exactly
// one definition. Environment-neutral.
export const MAGIC = 0x31465350; // "PSF1"
export const HEADER_BYTES = 64;
export const TABLE_ENTRY_BYTES = 48;
export const KINDS = { index: 1, corpus: 2, 'query-interp': 3, evaluation: 4, lexical: 5 };
export const KIND_NAMES = Object.fromEntries(Object.entries(KINDS).map(([name, kind]) => [kind, name]));
