+++
hypothesis = "Porting the checkpoint's image path (vision tower, processor, placeholders, M-RoPE) reproduces the makers' trained multimodal scoring at FP16."
change = "Vision.swift (Qwen3.5 vision encoder from Arsenal's Ovis port, multimodal rotary positions), Media.swift (PIL-faithful decode and bicubic, smart_resize, patchify), <image:N> placeholders, image-pad expansion and M-RoPE prefill; decide takes the makers' media list; processor_config.json pinned."
measurement = "Weight-free pixel, id, grid and position checks against the makers' processor; answers, vision and decoder states against the makers' trained path on MPS in BF16 and FP16, bounds frozen first; headless real check."
verdict = "kept"
numbers = { max_abs_dp_vs_maker_fp16 = 0.009112, max_abs_dp_vs_maker_bf16 = 0.002882, maker_bf16_vs_fp16_floor = 0.006282, argmax_flips = 0, branches = 6, worst_decoder_cosine = 0.999892, vision_cosine_probe = 0.99286, vision_floor_probe = 0.98556, blocks_vs_causal_diagnostic = 0.1628 }
links = ["README.md", "gate/gate.toml"]
+++

# 003 Image path (2 Nov 2026)

Reference. The makers document image questions; their code holds two image paths. `qev/mm_engine.py` (a serving engine)
is not touched by the checkpoint's `serve.patch` and never enters the block-mask context, so it scores images causally.
`train/common.batch_scores`, used for training and by `train/eval_mm.py` once `serve.patch` attaches the readout, scores
multimodal rows with the block mask, the image tokens inside the prefix block. The checkpoint declares the block mask
for its full-attention layers, and its readout notes say a checkpoint is served the way it was trained, so the port
follows `batch_scores`. On the fixed requests the two paths differ by up to 0.163 in probability.

Port. Pictures arrive as the makers' `media` list and are referenced by `<image:N>` in the state, once each, in order.
Each placeholder becomes the vision block in the chat prompt; the image-pad token is expanded to the grid's merged-patch
count; the vision tower's features replace those embeddings; the prefix runs with the three-axis rotary positions of
`get_rope_index`; each question suffix continues at the prefix position plus the rotary delta, as text. Decoding follows
PIL's `convert("RGB")`: stored RGB values, alpha dropped, no colour management.

Parity. Pixels identical to the makers' processor (PIL, torchvision and the maker agree to the level on the resized
probe); token ids, grids and positions identical. Answers within 0.0091 (FP16) and 0.0029 (BF16), no flips; decoder
states 0.99989 or better. The vision tower failed the frozen 0.999 bound on the probe screen (0.99286). Measured
afterwards, the makers' own BF16 and FP16 vision outputs differ more (0.98556 on that image): the features of flat colour
fields are small and precision-sensitive. The vision bound was revised to that floor, with the change and reason
recorded in `lab/evidence/elephant-1/bounds-mm-2026-11-02.json`.

Not ported: video frame lists (`{"type": "video", "frames": […]}`); URLs and base64 data in media (Arsenal passes local
files). Memory with pictures: see the real check.

## Review fixes (same day)

A fresh review found no structural bug. Fixed, each with a test: decode now follows PIL's `convert("RGB")` for grey
with alpha, 16-bit alpha and 16-bit grey (edge-case PNGs compared byte for byte); picture sizes are read from the header
before the token check and each picture is decoded only afterwards, one at a time; the resize is torchvision's int16
fixed-point antialiased bicubic (byte-identical to the maker's processor on every PNG fixture, including a textured
picture shrunk on both axes); normalisation is fused as torchvision does; the position-embedding interpolation sums in
float32 as transformers 5.19 does (probe vision 0.99286 → 0.99611 against the maker's FP16); placeholders read Python's
`\d` and refuse numbers past Int; vision special tokens in a question are refused; the heavy test answers through
`ElephantEngine.decide`.

A float32 run of the maker's vision tower (diagnostic, on the GPU) supports the precision hypothesis for the vision
bound: on the probe, this port's FP16 is at 0.99867 to float32, the maker's FP16 at 0.99898, the maker's BF16 at 0.97893.

JPEG: decoded by ImageIO rather than libjpeg-turbo, up to 62 levels apart at sharp colour edges (mean 1.3) on the
textured JPEG. Answers stay within 0.005 of the maker on the JPEG fixture and a real aerial photo; with PIL's decode of
the same files the vision tower matches at 0.99999 and 0.99996. Not fixed: matching libjpeg-turbo would need its
decoder in the runner.
