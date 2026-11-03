+++
hypothesis = "The MLX kit runner matches transformers VJEPA2Model.get_vision_features (with VJEPA2VideoProcessor) on the same weights."
change = "Port fixed once during this check: the centre crop takes the floor of (size - crop) / 2 (an upscaled 480 x 270 clip was cropped one pixel off)."
measurement = "lab/night2/vjepa2-vitl-fpc64-256/ref_parity.py vs runjson.py replies on m5-max, 2026-10-31: 4 frame sequences, clip and per-step embeddings."
verdict = "kept"
numbers = { embedding_cosine_min = 0.999897, per_step_cosine_min = 0.999311, embedding_max_abs_diff = 0.129 }
links = []
+++
# 003 Parity with transformers VJEPA2Model (2026-10-31)

Method: transformers 5.22.0 `VJEPA2VideoProcessor` and `VJEPA2Model.get_vision_features` (sdpa attention) on MPS at
float16, PyTorch 2.15.0, with the vendor's config.json and video_preprocessor_config.json (read as text from
facebook/vjepa2-vitl-fpc64-256 at b3c1679b) and the mlx-community FP16 weights that Arsenal serves (identical key
names; the Conv3d weight moved back from [out, t, h, w, in] to PyTorch's [out, in, t, h, w]; load_state_dict reported no
missing or unexpected keys). The vendor's FP32 weights were not downloaded, so this checks the port, not the
mlx-community conversion (its card reports encoder relative error 4.7e-3 at FP16 against PyTorch FP32).

Same input on both sides: frames decoded once by ffmpeg to PNG (lab/night2/vjepa2-vitl-fpc64-256/frames/), given to
the runner as `frames` and to the processor as a uint8 T x C x H x W tensor. Clips from Kinetics-mini val (the card's
archery example and two others). Embedding: last_hidden_state averaged over all tokens; per step: over each
2-frame step's 256 tokens.

| input | frame size | resize | tokens | embedding cosine | max abs diff | min per-step cosine |
|---|---|---|---|---|---|---|
| archery (card example), 16 frames | 480 x 360 | down to 389 x 292 | 2,048 | 0.999897 | 0.129 | 0.99974 |
| archery, portrait, 16 frames | 404 x 720 | down to 292 x 520 | 2,048 | 0.999904 | 0.106 | 0.99962 |
| bowling, 16 frames | 480 x 270 | up to 519 x 292 | 2,048 | 0.999908 | 0.099 | 0.99931 |
| archery (card example), first 64 frames | 480 x 360 | down | 8,192 | 0.999982 | 0.054 | 0.99974 |

Features reach 45 to 54 in magnitude. Preprocessing probe (probe_resize.py, probe_resize2.py): the runner's resize
and crop match the processor within 1 of 255 levels after the crop fix; before it, the upscaled bowling clip gave
cosine 0.99909 (crop offset 132 instead of the processor's 131).
Sense check (both sides agree): cosine archery vs portrait archery 0.7919, archery vs bowling 0.687.

Verdict: pass (embedding cosine >= 0.999 on every input).
