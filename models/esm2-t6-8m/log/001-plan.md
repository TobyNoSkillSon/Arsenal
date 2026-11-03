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
repo       facebook/esm2_t6_8M_UR50D at c731040fcd8d73dceaa04b0a8e6329b345b0f5df
licence    MIT (open)
task       fill-mask; library transformers
params     7,841,868 (7.8M), counted from the safetensors headers
weights    model.safetensors: FP32 on disk; Arsenal serves FP16 (cast once at Get)
also pins  config.json, special_tokens_map.json, tokenizer_config.json, vocab.txt
download   31 MB, once, into the Hugging Face cache; Arsenal's Get reuses it
runner     none bundled (arsenal-catalog: [unsupported] model_type esm has no bundled runner in this Arsenal.); you write one with the kit (Path B in docs/ADD-A-MODEL.md)

Tell the user in one message: the repo, licence, size, that you will write a small MLX runner for it, and that Arsenal will ask them once to approve its code. Wait for their yes. Then: kit/new-model.py scaffold facebook/esm2_t6_8M_UR50D --id ID --kind KIND --op OP
```

