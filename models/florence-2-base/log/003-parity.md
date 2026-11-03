+++
hypothesis = "The MLX kit runner reproduces microsoft/Florence-2-base as the maker's own repo code runs it."
change = "Model.swift port: DaViT encoder, 2-D and temporal position embeddings, image projection, BART encoder-decoder, HF 5.x beam search with no-repeat 3-grams and forced BOS/EOS, BART byte-level decode with the processor's added tokens, Pillow-exact bicubic preprocessing (Pixels.swift)."
measurement = "lab/night2/florence-2-base/ref.py + parity.py: the maker's modeling/processing_florence2.py at the pinned revision on MPS float16 (transformers 5.22) vs the runner's dumps (ARSENAL_PARITY_DIR), 9 fixed requests, on the vendor FP16 weights."
verdict = "kept"
links = ["results/quick-checks/kit-2026-10-31.json"]
numbers = { requests = 9, tokens_identical = 9, texts_identical = 9, input_ids_identical = 9, first_logits_max_abs_diff = 0.0156, first_logits_min_cosine = 0.9999997, image_features_min_cosine = 0.99977, encoder_min_cosine = 0.9981 }
+++
# 003 Parity with the maker's code

Verdict: pass.

Reference: the maker's own remote code (repo-modeling_florence2.py, configuration and processing at revision 5ca5edf5),
copied as text and imported in ref-env with transformers 5.22, Florence2ForConditionalGeneration.from_pretrained(float16)
on MPS. That code was written for transformers 4.41, so four compatibility shims were needed, none touching the maths:
a getattr in the config's BART-CNN back-compat check; tying encoder/decoder embed_tokens and lm_head to `shared` by hand
(5.x skips the remote _tie_weights; the checkpoint holds only `shared`); adding the processor's 1,024 extra tokens with
add_tokens(special_tokens=True); and dropping None-valued image-processor arguments (4.41 read None as "use the default",
5.x skips the resize). timm is imported only for DropPath (identity at eval), so a stub stood in.

Decoding: the card's generate(max_new_tokens=1024, do_sample=False, num_beams=3), plus the language model's settings
from config.json text_config that transformers 4.41 copied into its generation_config (no_repeat_ngram_size 3,
early_stopping True, forced_bos_token_id 0, forced_eos_token_id 2, length_penalty 1.0). transformers 5.22 no longer
copies them (its generation_config shows only the token ids), so they were passed explicitly. The reference ran with
use_cache=False (the remote code's tuple cache does not fit 5.x caches); same decoding, recomputed each step.

Inputs: the card's car photo (car.jpg, converted once to PNG), the macOS Zebra, Parrot and Big Sur Coastline pictures
used in batch 1, a 33x17 shrink of the zebra, and a 640x240 PNG with two printed lines ("Fresh bread daily",
"Open 7am to 5pm") for OCR.

| request | input ids | image features cos | first-step logits max abs / cos | tokens | output |
|---|---|---|---|---|---|
| car | same | 0.99996 | 0.0156 / 0.99999995 | identical (14) | A green car parked in front of a yellow building. |
| awkward 33x17 | same | 0.99977 | 0.0156 / 0.9999998 | identical (13) | a spotted hyena in the sahara desert |
| zebra | same | 0.99997 | 0.0078 / 0.99999996 | identical (14) | A zebra standing in a field of tall grass. |
| parrot | same | 0.99998 | 0.0156 / 0.99999993 | identical (17) | A close up of a red parrot with a yellow beak. |
| coastline | same | 0.99993 | 0.0156 / 0.9999999 | identical (15) | A view of the ocean from the top of a cliff. |
| car, <DETAILED_CAPTION> | same | 0.99996 | 0.0078 / 0.99999995 | identical (30) | The image shows a green Volkswagen Beetle ... |
| car, <MORE_DETAILED_CAPTION> | same | 0.99996 | 0.0078 / 0.9999999 | identical (88) | The image shows a vintage Volkswagen Beetle ... |
| sign, <OCR> | same | 0.99999 | 0.0156 / 0.99999993 | identical (12) | Fresh bread dailyOpen 7am to 5pm |
| zebra, prompt "A zebra" (decoder prefix) | same | 0.99997 | 0.0156 / 0.9999997 | identical (14) | A zebra standing in a field of tall grass. |

Texts after post_process_generation are identical too. The car caption equals the card's own example caption.
BART encoder outputs agree at cosine 0.9981 to 0.9999 (fp16 on both sides).

Preprocessing: the runner's pixels equal PIL's own uint8 bicubic resize to 768x768 (max diff 0.00097, the float rounding;
checked on car and coastline), which is what the maker's transformers 4.41 CLIPImageProcessor does. The 5.22 processor in
the reference differs from PIL by one 8-bit level on 0.02% to 0.4% of pixels (max diff 0.035 after normalization); the
tokens agree regardless.

Not covered: JPEG/HEIC decoding (ImageIO vs PIL) and colour profiles; PNG inputs were compared. The torchvision fast
processor was not compared. Decoder-prefix prompts are an Arsenal addition (the maker's processor has no such option);
the reference forced the same decoder_input_ids [</s>, <s>, prompt tokens].
