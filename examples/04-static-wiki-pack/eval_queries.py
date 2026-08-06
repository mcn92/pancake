#!/usr/bin/env python3
"""Build the pack's eval set: sample query texts, embed with the reference
fp32 encoder, and compute exact float brute-force top-10 over the full
corpus as ground truth.

Outputs (into --data):
  eval-queries.json   [{text, source}] — 180 sampled titles + 20 hand-written
  eval-gt.json        [[id, ...] x queries] — exact fp32 cosine top-10
"""
import argparse
import json
import random
from pathlib import Path

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384
K = 10

HAND_WRITTEN = [
    "how do plants turn sunlight into energy",
    "what causes earthquakes",
    "who wrote romeo and juliet",
    "why is the sky blue",
    "how does the human heart pump blood",
    "what is the largest planet in the solar system",
    "history of the roman empire",
    "how do vaccines protect against disease",
    "what language is spoken in brazil",
    "how are mountains formed",
    "what is the speed of light",
    "who painted the mona lisa",
    "how do bees make honey",
    "what caused the first world war",
    "how does photosynthesis work in the ocean",
    "what is the capital of australia",
    "how do computers store information",
    "what animals live in antarctica",
    "why do seasons change",
    "how is glass made",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data-full")
    parser.add_argument("--sampled", type=int, default=180)
    parser.add_argument("--no-hand", action="store_true", help="do not append the built-in Wikipedia-flavored hand queries")
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    rng = random.Random(20260803)
    titles = []
    seen = set()
    with open(args.data / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            title = json.loads(line)["title"]
            if title not in seen:
                seen.add(title)
                titles.append(title)
    sampled = rng.sample(titles, min(args.sampled, len(titles)))
    queries = [{"text": t, "source": "title"} for t in sampled]
    if not args.no_hand:
        queries += [{"text": t, "source": "hand"} for t in HAND_WRITTEN]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    with torch.no_grad():
        enc = tok([q["text"] for q in queries], padding=True, truncation=True,
                  max_length=256, return_tensors="pt").to(device)
        out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out * mask).sum(1) / mask.sum(1)
        emb = torch.nn.functional.normalize(emb, dim=1).cpu().numpy().astype(np.float32)

    vectors = np.memmap(args.data / "vectors.f32", dtype=np.float32, mode="r")
    vectors = vectors.reshape(-1, DIM)
    k = min(K, vectors.shape[0])
    gt = []
    block = 50
    for i in range(0, len(queries), block):
        scores = vectors @ emb[i:i + block].T          # (N, block); rows are unit vectors
        top = np.argpartition(-scores, k - 1, axis=0)[:k]  # (k, block) unsorted
        for c in range(scores.shape[1]):
            ids = top[:, c]
            order = np.argsort(-scores[ids, c])
            gt.append([int(x) for x in ids[order]])
        print(f"  gt {min(i + block, len(queries))}/{len(queries)}", flush=True)

    (args.data / "eval-queries.json").write_text(json.dumps(queries, indent=1) + "\n")
    (args.data / "eval-gt.json").write_text(json.dumps(gt) + "\n")
    emb.tofile(args.data / "eval-queries.f32")
    print(f"wrote {len(queries)} queries + exact top-{k} ground truth")


if __name__ == "__main__":
    main()
