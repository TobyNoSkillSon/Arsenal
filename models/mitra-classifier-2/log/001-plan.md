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

Source choice: vendor. Night 2, port effort medium (guess from the architecture; confirm at port).

Not done: download, hash check against the files, conversion, MLX port, parity check, smoke test, benchmarks.

## Kit plan output (read-only, verbatim)

```
repo       autogluon/mitra-classifier-2 at edada0d20759c58ada8c8605c25f22f6e98ea5f0
licence    Apache-2.0 (open)
task       tabular-classification; library none
params     75,670,026 (75.7M), counted from the safetensors headers
weights    model.safetensors: FP32 on disk; Arsenal serves FP16 (cast once at Get)
also pins  config.json
download   303 MB, once, into the Hugging Face cache; Arsenal's Get reuses it
runner     none bundled (arsenal-catalog: [architecture] autogluon/mitra-classifier-2 has no model_type in config.json, so no runner can be chosen.); you write one with the kit (Path B in docs/ADD-A-MODEL.md)

Tell the user in one message: the repo, licence, size, that you will write a small MLX runner for it, and that Arsenal will ask them once to approve its code. Wait for their yes. Then: kit/new-model.py scaffold autogluon/mitra-classifier-2 --id ID --kind KIND --op OP
```

