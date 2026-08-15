#!/usr/bin/env python3
"""Export real MiniLM-L6 layer-0 weight matrices plus the true activations
feeding each of them (captured with forward hooks on a real query), so the
fused-u8 kernel's quantization accuracy is measured on real distributions.

Outputs (encoder-spike/real/):
  <name>.w.f32     weight matrix, row-major f32 (rows x cols as in meta)
  <name>.x.f32     activation rows feeding that matrix (n x cols)
  meta.json        shapes
"""
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
OUT = HERE / "real"
OUT.mkdir(exist_ok=True)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)
model.eval()

layer = model.encoder.layer[0]
targets = {
    "q": layer.attention.self.query,
    "o": layer.attention.output.dense,
    "ffn_up": layer.intermediate.dense,
    "ffn_down": layer.output.dense,
}

captured: dict[str, torch.Tensor] = {}
hooks = []
for name, module in targets.items():
    def make_hook(key):
        def hook(_module, inputs, _output):
            captured[key] = inputs[0].detach()[0]  # (seq, cols)
        return hook
    hooks.append(module.register_forward_hook(make_hook(name)))

text = "how do volcanoes form and why do they erupt"
with torch.inference_mode():
    model(**tokenizer(text, return_tensors="pt"))
for hook in hooks:
    hook.remove()

meta = {}
for name, module in targets.items():
    weight = module.weight.detach()  # (rows, cols) applied as x @ W.T
    acts = captured[name]
    weight.numpy().astype("float32").tofile(OUT / f"{name}.w.f32")
    acts.numpy().astype("float32").tofile(OUT / f"{name}.x.f32")
    meta[name] = {"rows": weight.shape[0], "cols": weight.shape[1], "n": acts.shape[0]}
    print(f"[export] {name}: W {tuple(weight.shape)}, x {tuple(acts.shape)}")

(OUT / "meta.json").write_text(json.dumps(meta))
print(f"[export] wrote {OUT}")
