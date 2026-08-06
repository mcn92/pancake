#!/usr/bin/env python3
"""Build a static knowledge-pack corpus: read documents from local JSONL or
stream Simple English Wikipedia, chunk them, embed with MiniLM-L6-v2 (mean
pooling, L2-normalized — the exact recipe transformers.js reproduces
client-side), and write:

  corpus.jsonl   one row per chunk: {id, title, url, text}
  vectors.f32    row-major float32 embeddings, dim 384, same order

Usage:
  python3 embed_corpus.py --input sample-corpus.jsonl --out data-sample
  python3 embed_corpus.py --limit 1000 --out data-full    # wiki slice
  python3 embed_corpus.py --out data-full                 # full wiki run

Input JSONL rows must have {title, text}; {id, url} are optional. The chunker
prefixes every chunk with its title so a chunk stays self-describing after
retrieval, and splits on sentence boundaries at a target chunk size — small
enough that MiniLM's 256-token window sees the whole chunk, large enough that a
hit carries usable context.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384
TARGET_CHARS = 800
MIN_CHARS = 120
MAX_TOKENS = 256

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_article(title, text):
    # Collapse blank runs; Simple English articles are short, so most become
    # one or two chunks.
    text = re.sub(r"\n{2,}", "\n", text).strip()
    if not text:
        return
    buf = []
    size = 0
    for sentence in SENT_SPLIT.split(text.replace("\n", " ")):
        sentence = sentence.strip()
        if not sentence:
            continue
        if size + len(sentence) > TARGET_CHARS and size >= MIN_CHARS:
            yield f"{title}: " + " ".join(buf)
            buf, size = [], 0
        buf.append(sentence)
        size += len(sentence) + 1
    # The final chunk is kept regardless of MIN_CHARS: a short tail is still
    # the only home its sentences have, and stub articles ARE short tails.
    if buf:
        yield f"{title}: " + " ".join(buf)


def iter_input_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("title"), str) or not isinstance(row.get("text"), str):
                raise ValueError(f"{path}:{line_no}: expected JSON object with string title and text")
            yield {
                "source_id": row.get("id", line_no - 1),
                "title": row["title"],
                "url": row.get("url", ""),
                "text": row["text"],
            }


def iter_wikipedia(limit):
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
    for i, article in enumerate(ds):
        if limit and i >= limit:
            break
        yield {
            "source_id": article.get("id", i),
            "title": article["title"],
            "url": article.get("url", ""),
            "text": article["text"],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="JSONL documents with {id?, title, text, url?}")
    parser.add_argument("--limit", type=int, default=0, help="document cap (0 = all)")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    documents = iter_input_jsonl(args.input) if args.input else iter_wikipedia(args.limit)
    dataset_name = str(args.input) if args.input else "wikimedia/wikipedia 20231101.simple"
    if args.input and args.limit:
        documents = (doc for i, doc in enumerate(documents) if i < args.limit)

    corpus_path = args.out / "corpus.jsonl"
    vectors_path = args.out / "vectors.f32"
    t0 = time.time()
    articles = 0
    chunks = 0
    batch_texts = []

    def flush(corpus_f, vectors_f):
        nonlocal batch_texts, chunks
        if not batch_texts:
            return
        with torch.no_grad():
            enc = tok([t for _, t in batch_texts], padding=True, truncation=True,
                      max_length=MAX_TOKENS, return_tensors="pt").to(device)
            out = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out * mask).sum(1) / mask.sum(1)
            emb = torch.nn.functional.normalize(emb, dim=1)
        vectors_f.write(emb.cpu().numpy().astype(np.float32).tobytes())
        for row, _ in batch_texts:
            corpus_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        chunks += len(batch_texts)
        batch_texts = []

    with open(corpus_path, "w", encoding="utf-8") as corpus_f, open(vectors_path, "wb") as vectors_f:
        for article in documents:
            articles += 1
            for text in chunk_article(article["title"], article["text"]):
                row = {"id": chunks + len(batch_texts), "title": article["title"],
                       "sourceId": article["source_id"], "url": article["url"], "text": text}
                batch_texts.append((row, text))
                if len(batch_texts) >= args.batch:
                    flush(corpus_f, vectors_f)
            if articles % 5000 == 0:
                rate = articles / (time.time() - t0)
                print(f"  {articles} articles / {chunks} chunks  ({rate:.0f} art/s)", flush=True)
        flush(corpus_f, vectors_f)

    dt = time.time() - t0
    print(f"done: {articles} articles -> {chunks} chunks in {dt:.0f}s")
    print(f"  {corpus_path}  {corpus_path.stat().st_size / 1e6:.1f} MB")
    print(f"  {vectors_path}  {vectors_path.stat().st_size / 1e6:.1f} MB (dim {DIM})")
    manifest = {"model": MODEL_ID, "dim": DIM, "pooling": "mean", "normalized": True,
                "maxTokens": MAX_TOKENS, "dataset": dataset_name,
                "articles": articles, "chunks": chunks}
    (args.out / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
