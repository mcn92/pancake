#!/usr/bin/env python3
"""Apply the cluster layout permutation (layout-perm.npy: new position ->
old id) to corpus.jsonl and vectors.f32, writing data-perm/. Chunk ids are
positional, so reordering the corpus IS the layout change — no format
support needed, and corpus/offsets/vectors all stay in lockstep.
"""
import json
import argparse
import numpy as np
from pathlib import Path

DIM = 384


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path(__file__).parent / "data-full")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data-perm")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    perm = np.load(args.src / "layout-perm.npy")
    vecs = np.memmap(args.src / "vectors.f32", dtype=np.float32, mode="r").reshape(-1, DIM)
    assert len(perm) == vecs.shape[0]

    print(f"permuting {len(perm)} rows")
    out = np.empty_like(vecs[:0], shape=(len(perm), DIM))
    block = 65536
    for i in range(0, len(perm), block):
        out[i:i + block] = vecs[perm[i:i + block]]
    out.tofile(args.out / "vectors.f32")

    lines = (args.src / "corpus.jsonl").read_bytes().splitlines()
    assert len(lines) == len(perm)
    with open(args.out / "corpus.jsonl", "wb") as f:
        for new_pos, old_id in enumerate(perm):
            row = json.loads(lines[old_id])
            row["id"] = new_pos
            f.write(json.dumps(row, ensure_ascii=False).encode() + b"\n")

    manifest = json.loads((args.src / "corpus-manifest.json").read_text())
    manifest["layout"] = "kmeans-chain"
    (args.out / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out}/{{vectors.f32, corpus.jsonl, corpus-manifest.json}}")


if __name__ == "__main__":
    main()
