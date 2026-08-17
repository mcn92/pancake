// Minimal BERT WordPiece tokenizer (uncased): lowercase, accent-strip via
// NFD, punctuation/CJK splitting, greedy longest-match with ## pieces,
// [CLS]/[SEP] framing. Parity-tested against the HF tokenizer on the wiki
// eval queries; pure JS so it runs wherever the reader runs.

export function createWordPiece(vocabText) {
  const vocab = new Map();
  vocabText.split('\n').forEach((token, id) => vocab.set(token, id));
  const CLS = vocab.get('[CLS]');
  const SEP = vocab.get('[SEP]');
  const UNK = vocab.get('[UNK]');
  const MAX_WORD = 100;

  const isPunct = (c) => {
    const cp = c.codePointAt(0);
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64)
      || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
    return /\p{P}/u.test(c);
  };
  const isCjk = (c) => {
    const cp = c.codePointAt(0);
    return (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf)
      || (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x20000 && cp <= 0x2a6df);
  };

  function basicTokenize(text) {
    const cleaned = text.normalize('NFD').replace(/\p{Mn}/gu, '').toLowerCase()
      .replace(/[\u0000\uFFFD\u0001-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, '');
    const out = [];
    let current = '';
    const flush = () => { if (current) { out.push(current); current = ''; } };
    for (const c of cleaned) {
      if (/\s/.test(c)) { flush(); continue; }
      if (isPunct(c) || isCjk(c)) { flush(); out.push(c); continue; }
      current += c;
    }
    flush();
    return out;
  }

  function wordpiece(word) {
    if (word.length > MAX_WORD) return [UNK];
    const pieces = [];
    let start = 0;
    while (start < word.length) {
      let end = word.length;
      let piece = null;
      while (start < end) {
        const candidate = (start > 0 ? '##' : '') + word.slice(start, end);
        if (vocab.has(candidate)) { piece = vocab.get(candidate); break; }
        end--;
      }
      if (piece === null) return [UNK];
      pieces.push(piece);
      start = end;
    }
    return pieces;
  }

  return {
    encode(text) {
      const ids = [CLS];
      for (const word of basicTokenize(text)) ids.push(...wordpiece(word));
      ids.push(SEP);
      return ids;
    },
  };
}
