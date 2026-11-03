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
repo       google/timesfm-2.5-200m-pytorch at 1d952420fba87f3c6dee4f240de0f1a0fbc790e3
licence    Apache-2.0 (open)
task       time-series-forecasting; library timesfm
params     231,289,280 (231.3M), counted from the safetensors headers
weights    model.safetensors: FP32 on disk; Arsenal serves FP16 (cast once at Get)
also pins  config.json
download   925 MB, once, into the Hugging Face cache; Arsenal's Get reuses it
runner     none bundled (arsenal-catalog: [unsupported] model_type timesfm has no bundled runner in this Arsenal.); you write one with the kit (Path B in docs/ADD-A-MODEL.md)

Tell the user in one message: the repo, licence, size, that you will write a small MLX runner for it, and that Arsenal will ask them once to approve its code. Wait for their yes. Then: kit/new-model.py scaffold google/timesfm-2.5-200m-pytorch --id ID --kind KIND --op OP
```

