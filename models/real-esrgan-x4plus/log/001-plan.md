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
repo       Comfy-Org/Real-ESRGAN at 5fd49b7b278836f48af63ecd314d0f98ab336105
licence    BSD-3-Clause (open)
task       not tagged; library diffusion-single-file
params     16,697,987 (16.7M), counted from the safetensors headers
weights    RealESRGAN_x4plus.safetensors: FP32 on disk; Arsenal serves FP16 (cast once at Get)
also pins  nothing  (no config.json: read sizes from the maker code)
download   67 MB, once, into the Hugging Face cache; Arsenal's Get reuses it
runner     none bundled (arsenal-catalog: [architecture] Comfy-Org/Real-ESRGAN has no readable config.json, so its architecture is unknown.); you write one with the kit (Path B in docs/ADD-A-MODEL.md)

Tell the user in one message: the repo, licence, size, that you will write a small MLX runner for it, and that Arsenal will ask them once to approve its code. Wait for their yes. Then: kit/new-model.py scaffold Comfy-Org/Real-ESRGAN --id ID --kind KIND --op OP
```

