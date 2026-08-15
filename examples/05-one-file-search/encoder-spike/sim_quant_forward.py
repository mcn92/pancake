#!/usr/bin/env python3
"""Simulate the fused-u8 inline encoder end to end: fake-quantize every
weight matrix (and embedding table) of the real MiniLM to block-affine u8 —
the same scheme the WASM kernel dequantizes on the fly — then run the true
six-layer forward and measure error compounding at the output.

  python3 sim_quant_forward.py [block=64]

Outputs (encoder-spike/real/):
  quant-queries.f32   the 200 wiki eval queries encoded by the quantized model
  fp32-queries.f32    same queries through the untouched model (sanity ref)
Prints per-query cosine stats between the two.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "04-static-wiki-pack" / "data-full"
OUT = HERE / "real"
OUT.mkdir(exist_ok=True)

BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 64
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def quantize_tensor_(w: torch.Tensor, block: int) -> int:
    """In-place block-affine u8 fake-quantization along the last dim.
    Returns the number of blocks quantized."""
    original_shape = w.shape
    flat = w.detach().reshape(-1, original_shape[-1])
    cols = flat.shape[1]
    nfull = cols // block
    count = 0
    for start in range(0, nfull * block, block):
        chunk = flat[:, start:start + block]
        lo = chunk.min(dim=1, keepdim=True).values
        hi = chunk.max(dim=1, keepdim=True).values
        scale = (hi - lo) / 255.0
        scale = torch.where(scale <= 0, torch.full_like(scale, 1e-12), scale)
        q = torch.clamp(torch.round((chunk - lo) / scale), 0, 255)
        flat[:, start:start + block] = q * scale + lo
        count += 1
    # Ragged tail (embedding tables whose dim isn't a block multiple): one block.
    if cols % block:
        chunk = flat[:, nfull * block:]
        lo = chunk.min(dim=1, keepdim=True).values
        hi = chunk.max(dim=1, keepdim=True).values
        scale = torch.where((hi - lo) <= 0, torch.full_like(hi, 1e-12), (hi - lo) / 255.0)
        flat[:, nfull * block:] = torch.clamp(torch.round((chunk - lo) / scale), 0, 255) * scale + lo
        count += 1
    return count


def encode(model, tokenizer, texts, batch=32):
    vectors = []
    for i in range(0, len(texts), batch):
        toks = tokenizer(texts[i:i + batch], padding=True, truncation=True,
                         max_length=256, return_tensors="pt")
        with torch.inference_mode():
            out = model(**toks).last_hidden_state
        mask = toks["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1)
        pooled = torch.nn.functional.normalize(pooled, dim=1)
        vectors.append(pooled)
    return torch.cat(vectors).numpy().astype("float32")


tokenizer = AutoTokenizer.from_pretrained(MODEL)
fp32 = AutoModel.from_pretrained(MODEL)
fp32.eval()
quant = AutoModel.from_pretrained(MODEL)
quant.eval()

blocks = 0
with torch.no_grad():
    for name, module in quant.named_modules():
        if isinstance(module, torch.nn.Linear):
            blocks += quantize_tensor_(module.weight, BLOCK)
        elif isinstance(module, torch.nn.Embedding):
            blocks += quantize_tensor_(module.weight, BLOCK)
print(f"[quant] block={BLOCK}: fake-quantized every Linear + Embedding ({blocks} blocks); "
      f"LayerNorm and biases stay fp32 (stored fp32 in the artifact)", flush=True)

queries = [q["text"] for q in json.loads((DATA / "eval-queries.json").read_text())]
ref = encode(fp32, tokenizer, queries)
sim = encode(quant, tokenizer, queries)
ref_file = OUT / "fp32-queries.f32"
sim_file = OUT / "quant-queries.f32"
ref.tofile(ref_file)
sim.tofile(sim_file)

cos = (ref * sim).sum(axis=1)
print(f"[compound] six-layer forward, quantized vs fp32, {len(queries)} real queries:")
print(f"  cosine mean {cos.mean():.6f}  p10 {np.percentile(cos, 10):.6f}  min {cos.min():.6f}")

# Sanity: fp32 re-encode vs the pack's shipped teacher embeddings.
packed = np.fromfile(DATA / "eval-queries.f32", dtype=np.float32).reshape(len(queries), 384)
pack_cos = (ref * packed).sum(axis=1)
print(f"[sanity] fp32 re-encode vs pack's shipped embeddings: mean {pack_cos.mean():.6f} min {pack_cos.min():.6f}")
