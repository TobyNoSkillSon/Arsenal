# Qwen-Image-2.1

Alibaba Qwen's text-to-image and image-editing model: a 7B single-stream diffusion transformer conditioned by a Qwen3-VL 8B prompt encoder, with a 64-channel RGBA VAE. It ranked first among open-weight models on Artificial Analysis's text-to-image arena on 21 Oct 2026 (Elo 1042). Arsenal runs it natively in Swift/MLX (`Workers/QwenImage`); no Python runs.

## Operations

- `generate`: `prompt`, optional `width`/`height` (multiples of 32, 256–2752; default 1024 × 1024; native up to 2K: 2048 × 2048, 2752 × 1536), `steps` (default 40, the maker's), `seed`, `n` (1–4), and true classifier-free `guidance` (1–10, default 1 = off) with `negative_prompt`. Returns PNGs.
- `edit`: the same plus `images`, 1–10 reference images in order. One reference is an ordinary edit ("change the background to a sunset beach"); several compose subjects ("these three characters around a campfire"). Masks, circles and painted notes are not separate inputs: draw them into a reference or pass the mask as another image and name it in the prompt. Each reference is resized (Lanczos, transparency kept) to about one megapixel; without `width`/`height` the output takes the last reference's aspect ratio at about one megapixel.
- Transparency: `generate` takes `transparent: true`, which wraps the prompt in the maker's format: it begins with "This is an RGBA image with transparency." and ends with "The image has alpha channel and the background is transparent." Writing those sentences into the prompt yourself works the same. The PNG keeps its alpha channel when at least 64 pixels are under half opacity; otherwise it is saved as RGB (the VAE's alpha for an opaque image is 249–255, not exactly 255). `alpha` in each result says which.

Prompts are read as written. The maker recommends long, detailed descriptions and publishes a separate prompt-rewriting model (`Qwen/Qwen-Image-2.1-PE-T2I`, not in Arsenal).

## Sources and precision

Checkpoint: `Qwen/Qwen-Image-2.1@d26bb61231c349cf6b7896fa83353113880e1ba3`; every file, SHA256 and byte count is pinned in `model.toml`. Transformer and text encoder ship BF16, the VAE FP32. Get normalizes the snapshot to BF16 (`[precision]`; `log/002-format.md` records why BF16 and not FP16). It ships BF16 only, with no quantized tier. Lab memory peak (M5 Max, 21 Oct 2026, uncontrolled, not a Measured result): bf16 38.4 GB footprint at 1024 x 1024, all components resident (`lab/qwen-image/runs/base-bf16-neon.json`).

Implementation: diffusers `QwenImage21Pipeline` (commit `3421fed4a1655508436a5ca0c5179e88b0ec0afd`) and Transformers Qwen3-VL are the specification (`Workers/QwenImage/PORTED-FROM.md`). The prompt encoder, transformer, scheduler, VAE, prompt templates and condition-image handling follow it; the prefix (prompt and reference images) is computed once per request and reused by every step, as the maker's `use_kv_cache` does. The prompt encoder's final norm and LM head are never loaded.

## Speed and memory

Quick checks on an M5 Max (21 Oct 2026, other apps running), one 1024 × 1024 image, 40 steps, prompt encoder, transformer and VAE all resident as the worker keeps them (loading them adds about 8 s once):

| Tier | Seconds per image (loaded model) | Seconds per step | Peak memory (GB, MLX) |
|---|---|---|---|
| BF16 | 69 | 1.54 | 38.2 |

An edit adds the reference's tokens to the cached prefix. 2K images have four times the tokens and take several times longer; they were not timed.

## Verification

Static checks on small fixed tensors (`Workers/QwenImage/Tests`): the scheduler's sigmas, the 3-axis RoPE, the block-causal attention split, the prefix cache against a single-pass forward, the VAE's AvgDown/DupUp channel orders, the Qwen2-VL patch layout and the M-RoPE index walk. Real-weight runs and their figures are in `log/001-port.md`. No independent reference implementation was run.

## Licence

Qwen Research License Agreement: non-commercial research and evaluation only. Commercial use, including images for client work, needs a separate licence from Alibaba (model-business@notice.qwencloud.com).
