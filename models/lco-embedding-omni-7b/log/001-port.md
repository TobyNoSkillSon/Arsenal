+++
hypothesis = "A Swift MLX port of the Qwen2.5-Omni thinker in the Embed family, with the makers' prompt, processor and qwen-omni-utils media rules, reproduces LCO-Embedding-Omni-7B's vectors at FP16."
change = "New LCO adapter, language model, vision and audio towers in Workers/Embed; Whisper log-mel with Accelerate; registration for qwen2_5_omni_thinker with cast:embed-lco-omni-v1."
measurement = "Weight-free goldens against Transformers 5.22.1 and qwen-omni-utils; component and end-to-end parity against the makers' Transformers path on MPS, captured once in BF16 and once in FP16."
verdict = "kept"
numbers = { language_cosine_vs_maker_fp16 = 0.999994, vision_cosine_vs_maker_fp16 = 0.999895, audio_tower_cosine_vs_maker_fp16 = 0.995883, text_embedding_cosine_vs_maker_fp16 = 0.99999, image_embedding_cosine = 0.99995, audio_embedding_cosine = 0.99981, video_embedding_cosine = 0.99916, max_abs_hidden_fp16 = 7184 }
links = ["README.md", "gate/golden/golden.json"]
+++

# 001 Port (2 Nov 2026)

Swift MLX port in the Embed family (`Workers/Embed/Sources/EmbedWorker/LCO*.swift`), registered for
`model_type = qwen2_5_omni_thinker` with the cast recipe `cast:embed-lco-omni-v1` (BF16 checkpoint cast to FP16 at Get).

- Language model: Qwen2.5 7B with multimodal RoPE (sections 16/24/24), layout from mlx-swift-lm `Qwen25VL.swift`
  (`bd4b7434`, MIT); cos/sin cast to the activation dtype as Transformers does (the mlx-swift-lm copy keeps them FP32,
  which would promote FP16 attention to FP32). Only the last token is normed; the LM head is not loaded (1.1 GB saved).
- Vision: Qwen2.5-Omni ViT (separate q/k/v), window index ported from `transformers/vision_utils.py`; packed attention is
  run per segment, batching equal-length windows, so no sequence-by-sequence mask is built.
- Audio: Whisper log-mel in Swift with Accelerate (zero-padded 300 s semantics, global max - 8 floor); the encoder chunks
  at 200 frames with per-chunk attention, stride-2 pooling, FP16 clamp as the maker.
- Media rules: qwen-omni-utils `fetch_image`, `fetch_video` (fps 2, min 4 frames, even count, per-frame pixel budget) and
  the processor's hard-coded 1 s per temporal grid (25 positions).

Maker reference: Transformers 5.22.1, PyTorch 2.15.0 on MPS, the card's Transformers path. torchvision 0.29 dropped
`io.read_video` and torchcodec does not load against PyAV's FFmpeg 8.1, so the capture registers a PyAV reader in
qwen-omni-utils' torchvision slot (the removed reader was PyAV-based; same frames, same sampling).

Parity findings (2 Nov 2026; evidence in `lab/evidence/lco-embedding-omni-7b/`):

- Audio tower: cosine 0.9959 against the maker's FP16 run, median per-token cosine 0.9998. The differences sit on a few
  tokens (7, 24, 62, 68 of 80) where the maker's own BF16 and FP16 runs also disagree (per-token minimum 0.88), so it is
  the tower's own sensitivity, not a port error; the pooled audio embedding is at 0.9998.
- Video, first port: 0.983 end to end. That was a port error, not a matter of taste: AVFoundation's guess for the
  untagged clip was off by up to 22 levels on flat colour (blue disc R 81 against the drawn 60) and 69 at edges, while
  FFmpeg returned the drawn colours. Fixed in review (below).
- FP16 kept: the largest hidden-state magnitude is 7,184 (the Qwen2.5 massive activation), every output finite.

Review fixes (2 Nov 2026, fresh Opus review of 3ed9bcf0 and 272f65b5): the media front end now follows the makers'
readers (`LCOMediaIO.swift`), each with a weight-free test against the makers' processor output
(`Fixtures/lco_media_capture.py`, `lco-media-reference.json`).

- Video colour: Y'CbCr decoded by Arsenal with the track's tagged matrix and range, BT.601 limited when untagged
  (FFmpeg's default), chroma per 2×2 block, truncating like FFmpeg's fixed-point tables (rounding sat 1 to 2 levels
  high). Video pixels against the maker: cosine 0.99998, at most 2 levels; video vector 0.983 → 0.99916.
- Transparency: RGBA composited onto white, as qwen_omni_utils.to_rgb does (CoreImage had kept premultiplied black).
- Resize: PIL's antialiased bicubic (a = -0.5, widened support when shrinking, 8-bit between passes) in Swift; images
  keep their stored RGB values (PIL ignores ICC profiles). A 1000×750 image now matches the processor to one level.
- Memory: the token budget is computed from the track size before decoding; each sampled frame is resized as it is
  decoded and only the small frame is kept. Images and PDF pages are admitted from their size before decoding or
  rendering.
- PDF bound corrected: a US-letter page renders at 1,224×1,584 (twice its point size), 2,508 tokens, about 13 pages.
- User text containing a special token (`<|IMAGE|>`, `<|im_end|>` and the rest of the tokenizer's specials) is refused.
- Stereo audio: channels resampled separately and averaged, as librosa mono=True; AVAudioConverter's own downmix gain
  was not measured, the adapter no longer relies on it.
- Docs name the matched video path (the card's Transformers + qwen-omni-utils at 2 fps, not the sentence-transformers
  video processor).
