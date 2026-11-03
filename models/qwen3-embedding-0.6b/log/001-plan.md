+++
hypothesis = "The model can be prepared as a planned folder from its pinned Hugging Face metadata."
change = "Planned folder: weight pins, licence, attribution, README stub and this plan."
measurement = "Metadata only: repository tree, sizes and LFS sha256 at the pinned revision; nothing was downloaded or run."
verdict = "kept"
numbers = "No measurement: metadata only."
links = ["../model.toml"]
+++
# 001 plan (2026-10-30)

Created as a planned folder. Nothing was downloaded, converted or run.

Checked (metadata only): repository tree and sizes at the pinned revision, LFS sha256 from the Hugging Face API,
model card, config.json (if present), LICENSE (if present), and the kit's read-only `plan` output below.

Source choice: vendor. Night 1, port effort easy (guess from the architecture; confirm at port).

Not done: download, hash check against the files, conversion, MLX port, parity check, smoke test, benchmarks.

## Kit plan output (read-only, verbatim)

```
Plan for Qwen/Qwen3-Embedding-0.6B as qwen3-embedding-0.6b: embeddings, runs on the bundled embed worker, weights bf16.
Licence apache-2.0: open. Weights: 4 files, 1.20 GB to download after the user's yes.
Nothing was written. Show the user this plan; after their yes, run the same command without --plan.

A bundled runner serves this model: follow Path A in docs/ADD-A-MODEL.md (catalog add-local). Do not use the kit.
```

