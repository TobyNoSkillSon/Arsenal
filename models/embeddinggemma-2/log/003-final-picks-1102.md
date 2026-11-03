+++
hypothesis = "A Swift MLX port of the full EmbeddingGemma 2 (text encoder, Gemma 4 vision and audio towers, processor rules) reproduces the maker's sentence-transformers vectors at FP16 for text, images, audio, video and interleaved inputs."
change = "New EmbeddingGemma2 adapter, vision and audio towers, media rules and log-mel in Workers/Embed; the text encoder gains a range trace; registration for embedding_gemma2 moves from the text-only BF16 path to the new adapter with cast:embed-gemma2-v2 (FP16); the folder returns from retired-models."
measurement = "Weight-free goldens against Transformers 5.22.1; component and end-to-end parity against sentence-transformers 6.2 / Transformers 5.22.1 on MPS, captured once in FP16 and once in BF16; the maker's FP16 run stressed against BF16."
verdict = "kept"
numbers = { text_embedding_cosine_vs_maker_fp16 = 0.999998, image_embedding_cosine = 1.0, audio_embedding_cosine = 0.999999, video_embedding_cosine = 0.999928, interleaved_embedding_cosine = 0.999999, min_cosine_vs_maker_bf16 = 0.999879, vision_tower_cosine = 0.999997, audio_tower_cosine = 0.999997, token_ids_identical = 14, max_abs_text_hidden_fp16 = 84, max_abs_vision_pooler_fp32 = 28489 }
links = ["../README.md", "../gate/golden/golden.json"]
+++

# 003 Full multimodal port (2 Nov 2026)

The model returns to the catalog with every modality its card documents. The retired port (log 001–002) served text
only, in BF16, and had never run trained weights.

Maker reference: the card's sentence-transformers path. Transformers 5.22.1 is the first release with
`embedding_gemma2`; its modeling and Gemma 4 processor files equal the main-branch commit 5f6629d8 except timestamped
video and audio-from-video, which are off by default. torchcodec does not load against this Mac's FFmpeg, so the capture
routes the video processor to Transformers' own PyAV reader. Evidence: `lab/evidence/embeddinggemma-2/`.

Port:
- Text encoder: the retired Swift graph (bidirectional, window 512 inclusive, per-layer embeddings, q/k/v norms, scale 1),
  unchanged apart from a lab trace; per-token projection, then the mean over every token as sentence-transformers pools.
- Vision: Gemma4VisionModel on real patches only (padding patches never change real outputs), axial RoPE x-x-y-y, 3×3
  pooling, the √768 scale kept in float32 up to the embedder's norm.
- Audio: the Gemma 4 feature extractor (semicausal 20 ms frames, |rfft 512|, HTK mel, log(+0.001)) in float64 on the CPU;
  the conformer with clipped linears, chunked attention (12 + 12 past), the relative shift as an index gather, causal
  depthwise convolution, FP16 clamps at 65,504 as the maker's.
- Processor: the chat template's order and placeholder rule, <bos> … <eos>, one <|image> … <image|> block per image and
  per video frame, <|audio> … <audio|> per clip; Pillow-identical decoding (EXIF applied), torchvision's uint8
  antialiased bicubic, 1 fps sampling with the 32-frame cap, FFmpeg-style Y'CbCr conversion.

Findings:
- The card says FP16 returns NaN or degraded vectors. Not reproduced: the maker's own FP16 run stayed within 0.9998 of BF16
  on an 11,409-token text, white and black images and 20 s of full-scale noise; FP16 is the default.
- The maker's processor cuts audio at 30 s (480,000 samples, 750 tokens), not the card's "minutes"; Arsenal refuses
  longer clips unless `truncate` is set.
- Default prefix: sentence-transformers adds none unless asked. Arsenal keeps its `document` default for text-only items
  only; items with media get a prefix only when a task is given (a default prefix on the interleaved test item moved it to
  0.981; without it 0.999999).
- Video differs from the maker by at most 2 levels per pixel (FFmpeg against AVFoundation rounding), 0.99993 end to end.

Fresh review (2 Nov 2026, a separate Opus session): four fixes, each with a test. Audio with 3–8 channels failed
(AVAudioFormat has no plain multichannel PCM target) and is now averaged at its own rate, then resampled; PDF pages now
render from their crop box with their /Rotate; `dimensions` is read with Int(exactly:) so 1e300 is refused instead of
trapping; video frames become patches as they decode instead of being held at full size. A new weight-free test shows the
blocked sliding-window attention used past about 1,500 tokens equals masked full attention. Heavy parity unchanged
(0.99993 to 1.0). Left as noted risks: CMYK JPEG colour, rotated-video behaviour of the maker's reader, decode time of long
videos, resampling of 44.1/48 kHz audio against librosa (not compared).

Not checked: inputs near 8,192 tokens through the towers, audio near 30 s, rotated or colour-tagged video, PDFs end to end (no maker path), Polish
retrieval quality, speed (deferred by the owner).
