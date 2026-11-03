# Big LaMa

Big LaMa is the best model from LaMa (Large Mask Inpainting), made by Roman Suvorov and colleagues at Samsung AI Center (advimman) and published at WACV 2022. It is a 51-million-parameter convolutional generator with 18 residual blocks built on fast Fourier convolutions, which see the whole image at once. Given a photo and a mask, it fills the masked part with what plausibly lies behind it. Its strength is large holes over repeating or periodic backgrounds, at resolutions well above the 256×256 crops it was trained on.

## What it is for

The makers designed LaMa for large missing regions, complex geometric structure and high-resolution images. They used an image-wide receptive field in the network, a high-receptive-field perceptual loss and large training masks. Big LaMa was trained on 4.5 million images from the Places-Challenge dataset, which means scenes and places, not faces or documents.

- Removing objects, people, text overlays, watermarks and scratches from photos.
- Filling large holes over backgrounds such as walls, sky, grass, water, fences and tiled floors.
- Cleaning up the edges of a cut-out or an expanded canvas, where the missing area is masked.

## Inputs and outputs

### `inpaint` (default)

Fills every masked pixel of a photo and leaves every other pixel unchanged.

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file `{"path":"/absolute/path"}` | yes | The photo to fill. `image/png` or `image/jpeg`. At least 16 pixels on each side, at most 3840×2160 pixels in total (8,294,400). |
| `mask` | file `{"path":"/absolute/path"}` | yes | Same size as the photo: every pixel that is not black is filled. `image/png` or `image/jpeg`. |

Returns `files` with one entry, id `image`, label `Filled`: `inpainted.png`, an opaque 8-bit RGB PNG at the photo's own width and height, tagged with the photo's colour space. The entry gives `id`, `path`, `name`, `mime`, `bytes`, `sha256`, `width`, `height` and `label`.

The photo is read as 8-bit RGB with no colour-space conversion. Its EXIF orientation is applied, so the output comes back upright. Any alpha channel is dropped, not blended. Greyscale photos are repeated into three channels. 16-bit, float, CMYK and indexed files are first drawn into 8-bit RGB, and there any alpha is composited.

The mask is read as one 8-bit channel. Colour masks are converted to luma (ITU-R 601-2, as PIL's `"L"` mode does), and then any value above 0 marks a pixel to fill. Use white (255) on black (0). A colour that rounds to luma 0, such as a very dark blue, does not mark anything. Alpha in the mask is ignored, so a transparent PNG whose transparent pixels are black inside marks nothing there. A mask without its own EXIF orientation takes the photo's orientation when its size matches the photo as stored. After orientation, the mask must match the photo's width and height exactly, or the request is refused. Arsenal does not resize masks. A mask with no marked pixels is refused. Arsenal neither dilates nor feathers the mask, so masks should cover the whole object, including its shadow and a few pixels of edge.

The model works at the photo's own resolution in one untiled pass. Nothing is resized. The photo and the mask are padded on the bottom and right to a multiple of 8 by symmetric mirroring, and the padding is cropped from the result. The model's input is the photo with the hole zeroed, plus the mask as a fourth channel. Only masked pixels take the model's prediction.

## Run it

MCP: `arsenal_run(model:"big-lama", input:{"image":{"path":"/absolute/photo.jpg"},"mask":{"path":"/absolute/mask.png"}})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/big-lama/ops/inpaint" -H 'Content-Type: application/json' -d '{"image":{"path":"/absolute/photo.jpg"},"mask":{"path":"/absolute/mask.png"}}'`

There are no settings, prompt or seed: the same photo and mask give the same fill. One warm request on a small fixed test input took 0.32 s on an M5 Max, with 2.09 GB peak memory. That was a quick check with the Mac in use, not a controlled benchmark. Time and memory grow with the pixel count (see the large-photo figures below).

## Limits

- Photos above 8.3 megapixels (3840×2160) are refused, so downscale them first. Photos under 16 pixels on a side are also refused.
- The mask must be the photo's exact size. Every pixel that is not black is filled.
- Fills invent plausible texture. Lettering, faces and fine structure inside the hole are not recovered. The training data is Places scenes, so faces and documents are outside the model's training domain.
- At 3840×2160 the FP16 fill drifts toward a warmer tint compared with the makers' FP32 graph (31.6 dB inside the hole, mean 5.5 levels). At 1024×576 it matches closely (43.2 dB).
- One pass cannot be cancelled once the fill has started.
- The makers' optional refinement step (`refine=True` in their `bin/predict.py`) is not offered.

## Licence and credit

Apache-2.0: commercial use, modification and redistribution are allowed. Keep the licence and notices, and state your changes. The weights and the makers' reference code (advimman/lama) are Apache-2.0. The MLX Swift port that Arsenal adapts is MIT.

Made by Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park and Victor Lempitsky. Citation the makers give:

```bibtex
@article{suvorov2021resolution,
  title={Resolution-robust Large Mask Inpainting with Fourier Convolutions},
  author={Suvorov, Roman and Logacheva, Elizaveta and Mashikhin, Anton and Remizova, Anastasia and Ashukha, Arsenii and Silvestrov, Aleksei and Kong, Naejin and Goka, Harshith and Park, Kiwoong and Lempitsky, Victor},
  journal={arXiv preprint arXiv:2109.07161},
  year={2021}
}
```

## Links

- Model card: https://huggingface.co/mlx-community/LaMa-fp32/tree/92a388ff82beedfcf3de260d877e63e0c9c15dd8
- Code: https://github.com/advimman/lama
- Port: https://github.com/xocialize/mlx-lama-swift
- Paper: https://arxiv.org/abs/2109.07161
- Project page: https://advimman.github.io/lama-project/
- Licence: https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/LICENSE

## How Arsenal runs it

The checkpoint is `mlx-community/LaMa-fp32@92a388ff82beedfcf3de260d877e63e0c9c15dd8`, a single `model.safetensors` (204,316,668 bytes, sha256 `166187544dc18cce…`). All 837 of its tensors are bit-identical, in NHWC layout, to the generator in the widely mirrored `big-lama.pt` (sha256 344c77bb…), read through spandrel (log/001-port.md). The port is `xocialize/mlx-lama-swift@f085b8f4fddf6320a347e0e3ecb28fad20959b53` (MIT), vendored in `Workers/Restoration/Sources/LaMaNative`, with provenance in `Workers/Restoration/PORTED-FROM.md`. It runs in the native Swift `restoration` worker, one request at a time.

The upstream card says LaMa needs FP32 and that FP16 breaks it. The cause is the source's BatchNorm running variances, which reach 4.8e5 (`model.5.conv1.bn_l.running_var`), past FP16's 65,504. The unfolded graph on MPS collapses at both FP16 and BF16, filling the hole with black. Get therefore folds every BatchNorm into the convolution before it (scale = gamma/sqrt(var+eps), shift = beta − mean·scale). This is exact algebra: the folded graph at FP32 matches the unfolded one to 107 dB. It runs as recipe `cast:restoration-lama-v1` with `[precision] source = "raw"`, so Get hands the float32 file to this recipe, which folds the norms, stores FP16 (378 tensors, 51,057,027 parameters, 102 MB) and removes the FP32 download. FP16 is the only precision offered. The spectral FFTs run in float32, as in the makers' own half-precision path, and the spectral 1×1 convolutions run in FP16. The pipeline follows advimman/lama `bin/predict.py`: image /255, mask > 0, symmetric padding to a multiple of 8, generator on the zeroed image plus the mask, composite `mask·pred + (1−mask)·image` in float32 on the CPU, then `np.clip(x*255, 0, 255).astype(uint8)` truncation.

The gate (`gate/gate.toml`, `big-lama-native-v1`, static checks and visual review) checks the reference graph: spandrel 0.4.2 under torch 2.14.1 on MPS, captured at FP32. It requires exact checkpoint names and shapes, per-component fixed-tensor parity (`PhotoParityTests`) and one end-to-end image. Measured on 22 Oct 2026:

- Fixed 256×256 window with a hole, Swift FP16 against the provider graph: 38.0 dB over [0,1], max |d| 0.036.
- One 1024×576 photo end to end: 43.2 dB inside the hole, max 10 levels. Outside the hole the output is bit-identical to the input.
- 3840×2160 (the test photo upscaled, same mask): 31.6 dB inside the hole, max 18 levels, mean 5.5, with a visible shift toward a warmer tint. Outside the hole the output is identical. Running the spectral 1×1 convolutions in float32 changed nothing (31.65 dB), so the FFT block is not the cause. The drift was not localised further.
- Quick timing with the Mac in use: load plus one 1024×576 photo in 1.4 s, MLX peak 1.57 GB. At 3840×2160, 11.9 s in a debug build with MLX peak 6.9 GB (the provider on MPS took 2.3 s and 14.2 GB of driver memory).
- Arsenal API quick check on the M5 Max (`results/quick-m5-max-fp16-native-exact.json`): 0.32 s per warm request and 2.09 GB peak footprint on the small fixed test input.

Not measured: speed and memory under controlled conditions (the bench `big-lama-native-v1` is deferred), and photos above 3840×2160, which is why they are refused.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: A photo (PNG or JPEG, at least 16 px a side, up to 3840×2160) plus a same-size mask; every non-black mask pixel is filled. [README input table, 2026-10-31, first-hand]
- Prompts, reference audio or voice settings, if relevant: None; the model has no settings, prompt or seed. [README, 2026-10-31, first-hand]
- Parameters people recommend: Not found in sources checked.
- Arsenal call: `arsenal_run(model:"big-lama", operation:"inpaint", input:{"image":{"path":"/absolute/photo.jpg"},"mask":{"path":"/absolute/mask.png"}})` [README input table, 2026-10-31, first-hand]

**Good and bad**
- Good at: The maker reports an image-wide receptive field, large training masks, and generalisation to resolutions above the training resolution. [arXiv 2109.07161, v1 2021-09-15, first-hand (maker)]
- Bad at: Invented fills: lettering, faces and fine structure inside the hole are not recovered. [model.toml limits, first-hand (Arsenal's own note)]
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: FP16 training fails with a cuFFT error when the signal size is not a power of two (528 by 422 in the report). A training bug from a Windows 11 user. [advimman/lama issue 360, 2025-11-11, first-hand]
- Fix or workaround: Reported by one user, not confirmed: the run worked at precision 32 but hit CUDA out-of-memory there on their GPU. No reply (0 comments). [advimman/lama issue 360, first-hand (user report)]

**Evidence**
- Strength: none. The maker's paper and one training bug report with no reply. No user or independent test found.
- Sources:
  - arXiv 2109.07161, https://arxiv.org/abs/2109.07161, v1 2021-09-15, first-hand (maker).
  - advimman/lama issue 360, https://github.com/advimman/lama/issues/360, 2025-11-11, first-hand.
