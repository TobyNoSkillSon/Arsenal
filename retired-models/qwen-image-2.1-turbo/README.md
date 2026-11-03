# Qwen-Image-2.1-Turbo

Qwen's 8-step checkpoint of Qwen-Image-2.1: the same 7B single-stream transformer, Qwen3-VL 8B prompt encoder and RGBA VAE, distilled to run its saved sigma schedule (1.0 → 0.415, then 0) without guidance. It shares the `qwenimage` worker with `qwen-image-2.1`; the schedule comes from the checkpoint's `model_index.json` (`sample_sigmas`), which a step count cannot override, as in the maker's pipeline.

## Operations

- `generate`: `prompt`, optional `width`/`height` (multiples of 32, 256–2752; default 1024 × 1024; native up to 2K), `seed`, `n` (1–4). No steps, guidance or negative prompt: the maker's pipeline runs this checkpoint at CFG 1 on its saved schedule.
- `edit`: the same plus `images`, 1–10 reference images in order: edits, multi-subject composition, masks or circled/painted notes drawn into a reference. Each reference is resized (Lanczos, transparency kept) to about one megapixel; without `width`/`height` the output takes the last reference's aspect ratio at about one megapixel.
- Transparency: `generate` takes `transparent: true`, which wraps the prompt in the maker's format: it begins with "This is an RGBA image with transparency." and ends with "The image has alpha channel and the background is transparent." Writing those sentences into the prompt yourself works the same. The PNG keeps its alpha channel when at least 64 pixels are under half opacity; otherwise it is saved as RGB. `alpha` in each result says which.

## Sources and precision

Checkpoint: `Qwen/Qwen-Image-2.1-Turbo@fc13ba1986fcb905854dc8de2f52eee395f49b5b`; every file, SHA256 and byte count is pinned in `model.toml`. All weights ship BF16 and are served BF16 (`[precision]`; the VAE decoder's activations leave FP16's range, `log/002-format.md`). It ships BF16 only, with no quantized tier. Turbo's VAE equals the base model's FP32 VAE rounded to BF16, tensor for tensor.

Its prompt encoder is the base model's, tensor for tensor, but ships as one file instead of four shards, so Arsenal stores it separately for each model.

Implementation and verification are shared with `qwen-image-2.1` (`Workers/QwenImage`, its README and logs). Real-weight runs for this checkpoint are in `log/001-port.md`.

## Speed and memory

Quick checks on an M5 Max (21 Oct 2026, other apps running), one 1024 × 1024 image, model loaded (loading adds about 8 s once):

| Tier | Seconds per image (loaded model) | Peak memory (GB, MLX) |
|---|---|---|
| BF16 | 21 | 38.2 (all resident, measured on the base model's identical graph) |

The base model takes 69–73 s for the same image at 40 steps.

## Licence

Qwen Research License Agreement: non-commercial research and evaluation only. Commercial use, including images for client work, needs a separate licence from Alibaba (model-business@notice.qwencloud.com).
