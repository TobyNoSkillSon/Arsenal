# DDColor ModelScope

DDColor is an automatic photo colorizer from DAMO Academy, Alibaba Group (ICCV 2023). A ConvNeXt-L encoder feeds two decoders: a pixel decoder that restores spatial resolution and a query-based colour decoder that refines 100 learned colour queries against multi-scale image features. The network predicts only the colour (the a and b channels of Lab); the photo's own brightness is kept. This is the ModelScope checkpoint, 0.23B parameters, the makers' default model. It colours a black-and-white photo while leaving its detail and tones untouched.

## What it is for

The makers trained the ModelScope checkpoint on ImageNet with a data-cleaning scheme borrowed from BigColor, aimed at images beyond ImageNet, and list it as the default of their four checkpoints. The paper adds a colourfulness loss for richer colour and uses cross-attention between colour and image features to reduce colour bleeding.

- Black-and-white photos and film frames
- Old portraits, family photos and street scenes
- Colouring a single still pulled from black-and-white footage

## Inputs and outputs

### `colorize` (default)

Colours one photo and returns it at its own size.

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file `{"path":"/absolute/path"}` | yes | `image/png` or `image/jpeg`. The photo to colour. Only its brightness is used; colour photos work too. |
| `input_size` | integer | no | One of `256`, `512`, `768`, `1024`; default `512`. Square size, in pixels, at which colour is predicted; the photo keeps its own size and detail. |

Returns `files` with one entry: `colorized.png` (id `image`, label `Colorized`, mime `image/png`), an opaque 8-bit RGB PNG with exactly the width and height of the upright input.

How the photo is processed, following the makers' `ddcolor/pipeline.py`:

- Arsenal reads the file's own 8-bit values with no colour-space conversion, applies its EXIF orientation, drops any alpha channel (not blended), and replicates a greyscale image to RGB. 16-bit, float, CMYK and indexed images are first drawn into 8-bit RGB.
- The lightness channel L is taken from the full-size photo.
- A copy is resized to `input_size` × `input_size` with bilinear filtering. The resize is square and does not keep the aspect ratio, as in the makers' code. That copy is reduced to its L channel (a = b = 0), converted back to RGB and run through the network, which returns the a and b channels at `input_size` × `input_size`.
- The a and b channels are scaled to the full photo size with nearest-neighbour resizing, joined to the original full-size L, converted from Lab to RGB, and rounded to 8 bits.
- The PNG is tagged with the input's RGB colour space (sRGB for greyscale or non-RGB input).

The caller needs no preparation beyond a PNG or JPEG of at most 16 megapixels. A larger `input_size` gives the colour map more detail at more compute; 512 is the makers' default.

## Run it

MCP: `arsenal_run(model:"ddcolor-modelscope", input:{"image":{"path":"/absolute/path/photo.jpg"},"input_size":512})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/ddcolor-modelscope/ops/colorize" -H 'Content-Type: application/json' -d '{"image":{"path":"/absolute/path/photo.jpg"},"input_size":512}'`

There are no model settings; `input_size` is the only control. A warm request on a short fixed test input took 0.18 s and the first 0.47 s on an M5 Max, with load at 0.11 s and a peak of 1.7 GB across Arsenal's workers (measured, quick check on a busy Mac, 22 Oct 2026, results/quick-checks/night-2026-10-22.json).

## Limits

- Colour comes from a learned prior: skin, clothing and paint colours are plausible, not recovered from the photo.
- 16 megapixels per photo; larger images are refused.
- PNG and JPEG only.
- Colour is predicted on a square at most 1024 pixels wide and scaled up with nearest-neighbour resizing, so on large or very wide photos colour edges are coarser than the photo's detail and can show small blocks along sharp boundaries (derived from the pipeline, not measured).
- A colour photo loses its existing colour; only its brightness is kept.
- One photo per request; video is not handled by this operation.
- The makers note that the colourfulness loss can sometimes produce unpleasant colour blocks, such as red artifacts. Their artistic checkpoint, trained without it, is not offered here.

## Licence and credit

Apache-2.0 for the weights (upstream card and model.toml) and the makers' reference code: commercial use, modification and redistribution are allowed, provided the licence and notices are kept and changes are stated. The MLX Swift port it runs on is MIT.

Made by Xiaoyang Kang, Tao Yang, Wenqi Ouyang, Peiran Ren, Lingzhi Li and Xuansong Xie (DAMO Academy, Alibaba Group). FP16 MLX conversion by mlx-community; Swift port by xocialize. The makers ask for this citation:

```bibtex
@inproceedings{kang2023ddcolor,
  title={DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders},
  author={Kang, Xiaoyang and Yang, Tao and Ouyang, Wenqi and Ren, Peiran and Li, Lingzhi and Xie, Xuansong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={328--338},
  year={2023}
}
```

## Links

- Model card: https://huggingface.co/mlx-community/DDColor-modelscope-fp16/tree/479a36712efde407b455a2088e69993f02340f30
- Original weights: https://huggingface.co/piddnad/DDColor-models
- Code: https://github.com/piddnad/DDColor/tree/2adb63f2656ac41cbdf7b894cddd94121a3faf13
- Model zoo: https://github.com/piddnad/DDColor/blob/2adb63f2656ac41cbdf7b894cddd94121a3faf13/MODEL_ZOO.md
- Swift port: https://github.com/xocialize/mlx-ddcolor-swift
- Paper: https://arxiv.org/abs/2212.11613

## How Arsenal runs it

Checkpoint: `mlx-community/DDColor-modelscope-fp16@479a36712efde407b455a2088e69993f02340f30` (`model.safetensors`, 455,830,765 bytes), the FP16 conversion of `piddnad/DDColor-models` `ddcolor_modelscope.pth`. Every tensor was checked against that original: 566 of 573 bit-exact, the other seven (spectral-norm convolutions resolved with the stored u/v) within 7.6e-6; the extra `mean`/`std` are the model's ImageNet buffers, and no weight exceeds FP16's range (log/001-port.md). Port: `xocialize/mlx-ddcolor-swift@0bf684108316260a42c5c59db52e936c6a9336b6`, MIT, vendored as `Workers/Restoration/Sources/DDColorNative` with MLX's fused layer norm and precise softmax (provenance in `Workers/Restoration/PORTED-FROM.md`). It runs in the native Swift restoration worker, one request at a time.

Precision: FP16 only. The provider graph itself fails at FP16 on MPS (ab 7.7 dB against its FP32 capture; its LayerNorm overflows in the ConvNeXt stages) and holds at BF16 (43 dB). The MLX port at FP16 reaches 53.4 dB because MLX's fused layer norm accumulates in float32, so FP16 is kept; the Swift BF16 run read 35.2 dB. The colour-space math around the network (Lab conversions, resizes) runs in float32, as image processing, like the provider's cv2 code.

Gate (`gate/gate.toml`, status static checks and visual review): exact checkpoint names and shapes (575 tensors, `gate/header-check.json` passed), per-component fixed-tensor parity against the captured reference, and one end-to-end image against the provider pipeline. Reference: spandrel 0.4.2 provider graph on torch 2.14.1 MPS, FP32 capture. Fixed 512×512 input from a real photo, Swift FP16 against the provider: ab 53.4 dB PSNR, max |d| 4.2 of 47, every component 51.6 dB or more. One 1024×576 photo end to end: 51.9 dB, max 7 levels, against the provider pipeline's own output. The heavy `PhotoParityTests` repeat these.

Timing and memory: load plus one 1024×576 photo in 2.9 s through the adapter in the heavy test, MLX peak 1.39 GB (quick check, M5 Max, Mac in use, 22 Oct 2026). Arsenal quick check the same day: 0.18 s per warm request, 1.70 GB peak (`results/quick-m5-max-fp16-native-exact.json`). The controlled benchmark (`bench/bench.toml`) is deferred. Not measured: speed and memory under controlled conditions, colour quality on real archival photos beyond one desaturated test photo.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Black-and-white photos. A blogger ran it in Docker on an RTX 4090 and expects "just 1 second" (not timed). [akitaonrails.com, 2025-04-19, first-hand]
- Prompts, reference audio or voice settings, if relevant: No reference-image support; DDColor "does not natively support a reference image". [akitaonrails.com, 2025-04-19, first-hand]
- Parameters people recommend: Not found in sources checked.
- Arsenal call: `arsenal_run(model:"ddcolor-modelscope", operation:"colorize", input:{"image":{"path":"/abs/photo.jpg"},"input_size":512})`

**Good and bad**
- Good at: Archival and multi-object scenes, per the card; not tested here. [mlx-community card, undated, first-hand]
- Bad at: Video. The maintainer says it is "not specifically trained for video colorization". [piddnad/DDColor issue #35, opened 2024-04-28, maintainer reply 2024-05-03, first-hand]
- Against the obvious alternatives: Not found in sources checked. The one roundup read, RestorePhotosApp's, ranks its own app first. [RestorePhotosApp, undated, opinion]

**Complaints, bugs and fixes**
- Complaint or known bug: Colour flicker in video. Fix or workaround: No confirmed fix. The maintainer suggested an earlier checkpoint from a retrain, or no colourfulness loss, which "may help"; neither is reported to work. [issue #4, opened 2023-08-28, maintainer reply 2023-08-30, first-hand]
- Complaint or known bug: The Artistic checkpoint gave poor colour on one image; ModelScope worked. Fix or workaround: None; open, no maintainer reply. A second user reported the same 2026-03-30. [issue #68, opened 2025-11-28, first-hand (user test)]
- Complaint or known bug: No reference-image matching. Fix or workaround: Reported by one user, not confirmed: Reinhard global colour transfer after colouring; it changes the overall palette, not each object. [akitaonrails.com, 2025-04-19, first-hand]

**Evidence**
- Strength: thin; no independent quality test.
- Sources:
  - https://akitaonrails.com/en/2025/04/19/colorizing-black-and-white-images-with-ai/, 2025-04-19, first-hand
  - https://github.com/piddnad/DDColor/issues/4, #35, #68, dates as above, first-hand
  - https://huggingface.co/mlx-community/DDColor-modelscope-fp16, undated, first-hand
  - https://restorephotosapp.com/best-ai-photo-colorizer, undated, opinion
