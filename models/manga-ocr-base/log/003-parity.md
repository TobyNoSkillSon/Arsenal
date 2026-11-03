+++
hypothesis = "The MLX kit runner reproduces kha-white/manga-ocr-base as the maker's manga_ocr package runs it."
change = "Model.swift port: ViT encoder, BERT-Japanese decoder with cross-attention, HF 5.x beam search, manga_ocr post-processing, Pillow-exact preprocessing (Pixels.swift)."
measurement = "lab/night1/manga-ocr-base/parity.py: transformers 5.22 VisionEncoderDecoderModel on MPS float16 vs the runner's dumps (ARSENAL_PARITY_DIR), 6 fixed PNG inputs, before the FP16 cast (runner cast FP32 to FP16 at load)."
verdict = "kept"
links = ["results/quick-checks/kit-2026-10-30.json"]
numbers = { inputs = 6, beam4_tokens_identical = 6, greedy_tokens_identical = 5, greedy_fp16_ties = 1, first_logits_max_abs_diff = 0.0156, first_logits_min_cosine = 0.9999995, pixel_max_abs_diff = 0.00024 }
+++
# 003 Parity with the maker's code

Weights: model.safetensors from refs/pr/4 (82c77fb8, SFconvertbot's "Adding safetensors variant", parent = the planned
revision aa6573bd). Checked against the planned pytorch_model.bin (sha256 c63e0bb5...): all 263 tensors equal by value
and shape; the bin's 2 extra tensors (cls.predictions.decoder.weight/bias) are the tied copies of the word embeddings and
predictions.bias, and are equal to them.

Reference: the maker's manga_ocr/ocr.py path, run in ref-env: Image.convert("L").convert("RGB"), ViTImageProcessor
(PIL backend: bilinear 224x224, /255, mean/std 0.5), VisionEncoderDecoderModel.from_pretrained(float16).to("mps"),
generate(max_length=300), BertJapaneseTokenizer.decode(skip_special_tokens), post_process with jaconv.h2z.
generate() takes num_beams=4, no_repeat_ngram_size=3, length_penalty=2.0, early_stopping=True from config.json
(model.generation_config shows them), so the maker's decoding is 4-beam search, not greedy. The runner does the same
beam search; the greedy path (ARSENAL_PARITY_BEAMS=1, parity only) was checked too, as the brief asked.

Inputs: the maker's README examples 00, 03, 04, 07 and its package example image, converted once to PNG with PIL (so
both sides decode identical pixels), plus a 33x17 bicubic shrink of example 04.

| input | pixel max diff | first-step logits max abs diff / cosine | 4-beam tokens | greedy tokens |
|---|---|---|---|---|
| ex04 (bubble) | 0.00024 | 0.0078 / 0.9999998 | identical (35) | identical (36) |
| awkward 33x17 | 0.00024 | 0.0078 / 0.9999998 | identical (34) | differs at token 2 |
| ex07 | 0.00024 | 0.0156 / 0.9999998 | identical (29) | identical |
| ex00 | 0.00024 | 0.0078 / 0.9999998 | identical (12) | identical |
| ex03 | 0.00024 | 0.0078 / 0.9999998 | identical (25) | identical |
| example | 0.00024 | 0.0156 / 0.9999995 | identical (27) | identical |

The one greedy difference is an exact float16 tie in the reference itself: at step 2 tokens 912 and 858 both have logit
11.7656 in PyTorch; MPS argmax took 858, MLX took 912 (lower index). Not a port error (divergence.py).

The 4-beam texts equal the card's README answers for examples 00, 03, 04 and 07 character for character.

Not covered: JPEG inputs decode through ImageIO in the runner and libjpeg in PIL; on the JPEG originals the pixels
differ by up to 0.2 (normalized units), which still gave the same text on these examples. Transparent pixels are
composited over black (PIL ignores alpha). The torchvision image-processor backend (transformers 5 default when
torchvision is installed) was not compared; the PIL backend was.
