# Real-ESRGAN x4plus

Enlarges a photo four times, sharpening edges and cleaning blur, noise and JPEG artefacts as it goes. It is Tencent ARC's general-purpose Real-ESRGAN model (ICCV Workshops 2021), an RRDBNet trained on synthetic real-world degradations, in Comfy-Org's safetensors repackaging.

Input: `image` (PNG or JPEG), and optionally `scale` (2 or 4, default 4) or `target_height` (pixels; width keeps the aspect ratio). Output: `files`, one PNG (`id` `image`, name `upscaled.png`, `mime` `image/png`) with `path`, `bytes`, `sha256`, `width` and `height`.

Example request:

```json
{"image": {"path": "/absolute/path/to/photo.jpg"}, "scale": 4}
```

How it runs: the network always works at 4x. The image is processed in 256-pixel tiles, each reading 10 pixels of its neighbours (the maker's tile mode), so memory stays small: peak 1.2 GB for a 256-pixel photo and 1.6 GB for a 600-pixel one in the quick checks. Scale 2 and `target_height` resize the 4x result with a high-quality filter, as the maker's `outscale` does.

Limits: it invents plausible detail, so fine texture (grass, hair) can look smooth or painterly; faces get no special restoration and small text can be garbled; anime does better with the maker's anime model. Input at most 4 megapixels and 4096 pixels per side; output at most 64 megapixels. Transparency is flattened.

Parity: matches the maker's RRDBNet in PyTorch (MPS, float16) at cosine 0.99996 or better and PSNR above 51 dB on two photos, tiled and untiled (`log/003-parity.md`).

Source: [Comfy-Org/Real-ESRGAN](https://huggingface.co/Comfy-Org/Real-ESRGAN/tree/5fd49b7b278836f48af63ecd314d0f98ab336105), BSD-3-Clause; original code and weights by Xintao Wang et al., [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: one image file or a folder (`-i`); output goes to `results` by default. FP16 is the default, with a full-precision option. [Real-ESRGAN GitHub README, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not found in sources checked.
- Parameters people recommend: `-t` tile size (0 means no tiling, the default); `-s` outscale (default 4); `--face_enhance` GFPGAN face enhancement (off by default). No other tile or precision advice. [Real-ESRGAN GitHub README, undated, first-hand]
- Arsenal call: `POST /arsenal/v1/models/real-esrgan-x4plus/ops/upscale_image`, or the same operation through MCP. Vendor example: `python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --face_enhance` [Real-ESRGAN GitHub README, undated, first-hand]

**Good and bad**
- Good at: the maker places RealESRGAN_x4plus under "Inference general images". [Real-ESRGAN GitHub README, undated, first-hand]
- Bad at: anime, where the maker points to `RealESRGAN_x4plus_anime_6B`. Faces get no special handling unless `--face_enhance` is set. [Real-ESRGAN GitHub README, undated, first-hand]
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: in the ncnn-vulkan build (v0.2.0, x4plus weights), a JPG input saved as PNG showed tile artefacts; JPG output was fine. The reporter closed it as a duplicate of #19. [xinntao/real-esrgan-ncnn-vulkan issue #39, opened 2023-01-31, first-hand report]
- Fix or workaround: Reported by one user, not confirmed: the reporter scaled the 4x output down to the target size and said it worked. The idea came from another user; no maintainer replied. [issue #39, 2023-01-31; suggestion is opinion]

**Evidence**
- Strength: thin. The maker's README and one ncnn issue; no independent tests or Reddit threads found.
- Sources:
  - Real-ESRGAN, GitHub README, https://github.com/xinntao/Real-ESRGAN, undated (no release date on the page), first-hand.
  - Issue #39, https://github.com/xinntao/real-esrgan-ncnn-vulkan/issues/39, 2023-01-31, first-hand report.
