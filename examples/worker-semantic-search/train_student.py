#!/usr/bin/env python3
"""Distill a tiny zero-runtime-dependency query encoder for the Worker demo.

The teacher is used only by this offline script. Runtime inference uses the
quantized PSTU artifact through student-embedder.mjs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer


MAGIC = b"PSTU"
VERSION = 1
HASH_SEED = 2166136261
DEFAULT_TEACHER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
TOKEN_RE = re.compile(r"[a-z0-9]+")

CURATED_HELDOUT = [
    "How does an edge isolate recover a saved vector index?",
    "Can I reclaim space after removing lots of records?",
    "How can searches be restricted to one tenant?",
    "What is the RAM cost of storing vectors as eight bit values?",
    "Can a snapshot be opened without repeating its construction settings?",
    "Does every index share the same WebAssembly memory?",
    "What happens when several requests arrive during a cold start?",
    "How do I change search accuracy for only one request?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--teacher", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--teacher-revision", default=DEFAULT_TEACHER_REVISION)
    parser.add_argument("--buckets", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.008)
    parser.add_argument("--max-features", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def fnv1a(text: str, seed: int = HASH_SEED) -> int:
    value = seed & 0xFFFFFFFF
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def extract_features(text: str, buckets: int, max_features: int) -> list[int]:
    tokens = TOKEN_RE.findall(str(text).lower())
    features: list[int] = []

    def push(feature: str) -> None:
        if len(features) < max_features:
            features.append(fnv1a(feature) % buckets)

    for token_index, raw_token in enumerate(tokens):
        if len(features) >= max_features:
            break
        token = raw_token[:48]
        push(f"w:{token}")
        padded = f"^{token}$"
        for width in range(3, 6):
            for start in range(0, len(padded) - width + 1):
                push(f"c{width}:{padded[start:start + width]}")
                if len(features) >= max_features:
                    break
            if len(features) >= max_features:
                break
        if token_index > 0:
            push(f"b:{tokens[token_index - 1][:48]}:{token}")
    return features


def clean_title(value: str) -> str:
    return re.sub(r"\s*[/|>]\s*", " ", str(value or "")).strip()


def first_sentence(value: str, max_words: int = 28) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    return " ".join(sentence.split()[:max_words])


def keywords(value: str, limit: int = 10) -> str:
    stop = {
        "about", "after", "also", "because", "before", "between", "from",
        "have", "into", "only", "pancake", "that", "their", "then", "there",
        "these", "they", "this", "through", "using", "when", "where", "which",
        "while", "with", "your",
    }
    result: list[str] = []
    for token in TOKEN_RE.findall(str(value).lower()):
        if len(token) < 3 or token in stop or token in result:
            continue
        result.append(token)
        if len(result) == limit:
            break
    return " ".join(result)


def dropout_words(text: str, randomizer: random.Random, rate: float = 0.22) -> str:
    words = str(text).split()
    kept = [word for word in words if randomizer.random() >= rate]
    if len(kept) < min(3, len(words)):
        kept = words[: min(len(words), 6)]
    return " ".join(kept)


@dataclass(frozen=True)
class QueryExample:
    text: str
    origin: int | None
    family: str


def make_examples(corpus: list[dict], seed: int) -> tuple[list[QueryExample], list[QueryExample]]:
    randomizer = random.Random(seed)
    train: list[QueryExample] = []
    heldout: list[QueryExample] = []
    seen_train: set[str] = set()
    seen_heldout: set[str] = set()

    def add(target: list[QueryExample], seen: set[str], text: str, origin: int, family: str) -> None:
        normalized = re.sub(r"\s+", " ", text).strip()
        key = normalized.lower()
        if len(normalized) >= 3 and key not in seen:
            target.append(QueryExample(normalized, origin, family))
            seen.add(key)

    for row, chunk in enumerate(corpus):
        title = clean_title(chunk.get("title") or chunk.get("docTitle") or f"section {row}")
        sentence = first_sentence(chunk.get("text") or chunk.get("preview") or "")
        terms = keywords(f"{title} {sentence}")

        training_forms = [
            (title, "title"),
            (f"what is {title}", "what"),
            (f"how does {title} work", "how"),
            (f"explain {title}", "explain"),
            (f"documentation for {title}", "docs"),
            (sentence, "sentence"),
            (terms, "keywords"),
            (dropout_words(f"{title} {sentence}", randomizer), "dropout"),
        ]
        for text, family in training_forms:
            add(train, seen_train, text, row, family)

        heldout_forms = [
            (f"where can i learn about {title}", "where"),
            (f"tell me about {title}", "tell"),
            (f"i need help with {title}", "help"),
        ]
        for text, family in heldout_forms:
            add(heldout, seen_heldout, text, row, family)

    for query in CURATED_HELDOUT:
        key = query.lower()
        if key not in seen_heldout:
            heldout.append(QueryExample(query, None, "curated"))
            seen_heldout.add(key)
    return train, heldout


class TeacherEncoder:
    def __init__(self, name: str, revision: str):
        self.name = name
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModel.from_pretrained(name, revision=revision)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int, max_length: int) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            outputs.append(functional.normalize(pooled, dim=1).cpu())
        return torch.cat(outputs, dim=0)


class StudentEncoder(nn.Module):
    def __init__(self, buckets: int, hidden: int, output: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(buckets, hidden, mode="mean")
        self.projection = nn.Linear(hidden, output)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.035)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, flat_features: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        pooled = torch.tanh(self.embedding(flat_features, offsets))
        return functional.normalize(self.projection(pooled), dim=1)


def feature_batch(feature_rows: list[list[int]], indexes: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor]:
    flat: list[int] = []
    offsets: list[int] = []
    for index in indexes:
        offsets.append(len(flat))
        row = feature_rows[index]
        flat.extend(row if row else [0])
    return torch.tensor(flat, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


def quantize_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maximum = np.max(np.abs(values), axis=1)
    scales = np.where(maximum > 0, maximum / 127.0, 1.0).astype("<f4")
    quantized = np.clip(np.rint(values / scales[:, None]), -127, 127).astype(np.int8)
    return scales, quantized


def quantized_forward(
    feature_rows: list[list[int]],
    embedding_scales: np.ndarray,
    embedding_quantized: np.ndarray,
    projection_scales: np.ndarray,
    projection_quantized: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    embedding = embedding_quantized.astype(np.float32) * embedding_scales[:, None]
    projection = projection_quantized.astype(np.float32) * projection_scales[:, None]
    results = np.zeros((len(feature_rows), projection.shape[0]), dtype=np.float32)
    for row_index, features in enumerate(feature_rows):
        used = features if features else [0]
        hidden = np.tanh(embedding[used].mean(axis=0))
        output = projection @ hidden + bias
        norm = np.linalg.norm(output)
        if norm > 0:
            output /= norm
        results[row_index] = output
    return results


def topk_overlap(student: np.ndarray, teacher: np.ndarray, docs: np.ndarray, k: int) -> float:
    student_top = np.argpartition(-(student @ docs.T), min(k, docs.shape[0]) - 1, axis=1)[:, :k]
    teacher_top = np.argpartition(-(teacher @ docs.T), min(k, docs.shape[0]) - 1, axis=1)[:, :k]
    total = 0.0
    for left, right in zip(student_top, teacher_top):
        total += len(set(left.tolist()).intersection(right.tolist())) / k
    return total / len(student)


def origin_recall(vectors: np.ndarray, docs: np.ndarray, examples: list[QueryExample], k: int) -> float | None:
    eligible = [(index, example.origin) for index, example in enumerate(examples) if example.origin is not None]
    if not eligible:
        return None
    top = np.argpartition(-(vectors @ docs.T), min(k, docs.shape[0]) - 1, axis=1)[:, :k]
    hits = sum(1 for index, origin in eligible if origin in top[index])
    return hits / len(eligible)


def export_model(path: Path, model: StudentEncoder, args: argparse.Namespace) -> dict:
    embedding = model.embedding.weight.detach().cpu().numpy().astype(np.float32)
    projection = model.projection.weight.detach().cpu().numpy().astype(np.float32)
    bias = model.projection.bias.detach().cpu().numpy().astype("<f4")
    embedding_scales, embedding_quantized = quantize_rows(embedding)
    projection_scales, projection_quantized = quantize_rows(projection)

    payload = bytearray()
    payload.extend(struct.pack(
        "<4s7I",
        MAGIC,
        VERSION,
        args.buckets,
        args.hidden,
        projection.shape[0],
        HASH_SEED,
        args.max_features,
        0,
    ))
    payload.extend(embedding_scales.tobytes())
    payload.extend(embedding_quantized.tobytes())
    while len(payload) % 4:
        payload.append(0)
    payload.extend(projection_scales.tobytes())
    payload.extend(projection_quantized.tobytes())
    payload.extend(bias.tobytes())
    path.write_bytes(payload)
    return {
        "embedding_scales": embedding_scales,
        "embedding_quantized": embedding_quantized,
        "projection_scales": projection_scales,
        "projection_quantized": projection_quantized,
        "bias": bias,
        "byte_length": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    args.out.mkdir(parents=True, exist_ok=True)

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("Corpus must be a non-empty JSON array")
    train_examples, evaluation_examples = make_examples(corpus, args.seed)
    validation_examples = [example for example in evaluation_examples if example.family == "where"]
    heldout_examples = [example for example in evaluation_examples if example.family != "where"]
    document_texts = [
        f"{row.get('docTitle', '')}. {clean_title(row.get('title', ''))}. {row.get('text', '')}"
        for row in corpus
    ]

    print(
        f"[data] documents={len(corpus)} train_queries={len(train_examples)} "
        f"validation_queries={len(validation_examples)} heldout_queries={len(heldout_examples)}"
    )
    teacher = TeacherEncoder(args.teacher, args.teacher_revision)
    started = time.perf_counter()
    # Materialize ordinary tensors outside inference mode. PyTorch deliberately
    # prevents inference tensors from being saved for the student's backward pass.
    document_vectors = torch.from_numpy(
        teacher.encode(document_texts, batch_size=16, max_length=256).numpy().copy()
    )
    train_teacher = torch.from_numpy(
        teacher.encode([example.text for example in train_examples], batch_size=64, max_length=96).numpy().copy()
    )
    validation_teacher = torch.from_numpy(
        teacher.encode([example.text for example in validation_examples], batch_size=64, max_length=96).numpy().copy()
    )
    heldout_teacher = torch.from_numpy(
        teacher.encode([example.text for example in heldout_examples], batch_size=64, max_length=96).numpy().copy()
    )
    print(f"[teacher] encoded in {time.perf_counter() - started:.1f}s dim={document_vectors.shape[1]}")

    train_features = [
        extract_features(example.text, args.buckets, args.max_features)
        for example in train_examples
    ]
    validation_features = [
        extract_features(example.text, args.buckets, args.max_features)
        for example in validation_examples
    ]
    heldout_features = [
        extract_features(example.text, args.buckets, args.max_features)
        for example in heldout_examples
    ]
    student = StudentEncoder(args.buckets, args.hidden, document_vectors.shape[1])
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(args.seed)
    temperature = 0.055
    best_state = copy.deepcopy(student.state_dict())
    best_epoch = 0
    best_validation = -math.inf
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(len(train_examples), generator=generator).tolist()
        running = 0.0
        student.train()
        for start in range(0, len(permutation), args.batch_size):
            indexes = permutation[start:start + args.batch_size]
            flat, offsets = feature_batch(train_features, indexes)
            targets = train_teacher[indexes]
            predicted = student(flat, offsets)

            cosine_loss = (1 - (predicted * targets).sum(dim=1)).mean()
            with torch.no_grad():
                teacher_distribution = functional.softmax((targets @ document_vectors.T) / temperature, dim=1)
            student_distribution = functional.log_softmax((predicted @ document_vectors.T) / temperature, dim=1)
            ranking_loss = functional.kl_div(student_distribution, teacher_distribution, reduction="batchmean")
            loss = 0.55 * cosine_loss + 0.45 * ranking_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
            optimizer.step()
            running += loss.item() * len(indexes)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            student.eval()
            with torch.inference_mode():
                flat, offsets = feature_batch(validation_features, range(len(validation_features)))
                prediction = student(flat, offsets)
                cosine = (prediction * validation_teacher).sum(dim=1).mean().item()
                agreement = topk_overlap(
                    prediction.numpy(),
                    validation_teacher.numpy(),
                    document_vectors.numpy(),
                    5,
                )
                validation_score = cosine + agreement
            if validation_score > best_validation + 1e-5:
                best_validation = validation_score
                best_epoch = epoch
                best_state = copy.deepcopy(student.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 5 if epoch > 1 else 1
            print(
                f"[train] epoch={epoch:03d} loss={running / len(train_examples):.4f} "
                f"validation_cos={cosine:.4f} teacher_top5_overlap={agreement:.4f}"
            )
            if epochs_without_improvement >= 20:
                print(f"[train] early stop; restoring epoch {best_epoch}")
                break

    student.load_state_dict(best_state)

    model_path = args.out / "student-model.bin"
    exported = export_model(model_path, student, args)
    heldout_quantized = quantized_forward(
        heldout_features,
        exported["embedding_scales"],
        exported["embedding_quantized"],
        exported["projection_scales"],
        exported["projection_quantized"],
        exported["bias"],
    )
    teacher_heldout_np = heldout_teacher.numpy()
    documents_np = document_vectors.numpy().astype("<f4")
    cosine_values = np.sum(heldout_quantized * teacher_heldout_np, axis=1)
    evaluation = {
        "heldoutQueries": len(heldout_examples),
        "validationQueries": len(validation_examples),
        "meanTeacherCosine": float(np.mean(cosine_values)),
        "p10TeacherCosine": float(np.quantile(cosine_values, 0.10)),
        "teacherTop1Agreement": topk_overlap(heldout_quantized, teacher_heldout_np, documents_np, 1),
        "teacherTop3Overlap": topk_overlap(heldout_quantized, teacher_heldout_np, documents_np, 3),
        "teacherTop5Overlap": topk_overlap(heldout_quantized, teacher_heldout_np, documents_np, 5),
        "studentOriginRecallAt5": origin_recall(heldout_quantized, documents_np, heldout_examples, 5),
        "teacherOriginRecallAt5": origin_recall(teacher_heldout_np, documents_np, heldout_examples, 5),
        "queries": [
            {
                "text": example.text,
                "family": example.family,
                "origin": example.origin,
                "teacherTop": int(np.argmax(teacher_heldout_np[index] @ documents_np.T)),
                "studentTop": int(np.argmax(heldout_quantized[index] @ documents_np.T)),
                "teacherCosine": float(cosine_values[index]),
            }
            for index, example in enumerate(heldout_examples)
        ],
    }

    (args.out / "docs-vectors.f32").write_bytes(documents_np.tobytes())
    (args.out / "student-evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "format": "pancake-distilled-student",
        "version": VERSION,
        "teacher": args.teacher,
        "teacherRevision": args.teacher_revision,
        "architecture": "hashed word/character n-grams -> mean pool -> tanh -> linear -> L2 normalize",
        "bucketCount": args.buckets,
        "hiddenDim": args.hidden,
        "outputDim": int(document_vectors.shape[1]),
        "hashSeed": HASH_SEED,
        "maxFeatures": args.max_features,
        "quantization": "symmetric int8, per row",
        "modelBytes": exported["byte_length"],
        "modelSha256": exported["sha256"],
        "trainQueries": len(train_examples),
        "heldoutQueries": len(heldout_examples),
        "validationQueries": len(validation_examples),
        "maxEpochs": args.epochs,
        "selectedEpoch": best_epoch,
        "seed": args.seed,
        "evaluation": {key: value for key, value in evaluation.items() if key != "queries"},
    }
    (args.out / "student-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[write] {model_path} ({exported['byte_length'] / 1024:.1f} KiB)")
    print(f"[write] {args.out / 'docs-vectors.f32'}")
    print(f"[eval] {json.dumps(manifest['evaluation'], sort_keys=True)}")


if __name__ == "__main__":
    main()
