# Distilled model notice

`assets/docs-student.bin` was trained by the accompanying
`train_student.py` pipeline to imitate query embeddings produced by
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
The teacher model is published under the Apache License 2.0. Teacher weights,
tokenizer files, PyTorch, and Transformers are not included in this repository's
runtime demo assets.

The student is domain-specific to the Pancake documentation corpus. Its
architecture, training procedure, quantization format, selected checkpoint,
teacher identifier, artifact hash, and held-out evaluation are recorded in the
source files and generated manifests in this directory.
