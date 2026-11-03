+++
hypothesis = "The MLX kit runner reproduces microsoft/Florence-2-large as the maker's own repo code runs it."
change = "Same Model.swift as florence-2-base (the architecture reads every size from config.json): DaViT encoder, position embeddings, projection, BART encoder-decoder, HF 5.x beam search with no-repeat 3-grams and forced BOS/EOS, Pillow-exact bicubic preprocessing."
measurement = "lab/night2/florence-2-base/ref.py + parity.py run on the large snapshot: the maker's modeling/processing_florence2.py on MPS float16 (transformers 5.22) vs the runner's dumps (ARSENAL_PARITY_DIR), 9 fixed requests, on the vendor FP16 weights."
verdict = "kept"
links = ["results/quick-checks/kit-2026-10-31.json"]
numbers = { requests = 9, tokens_identical = 9, texts_identical = 9, input_ids_identical = 9, first_logits_max_abs_diff = 0.145, first_logits_min_cosine = 0.99991, image_features_min_cosine = 0.99953, encoder_min_cosine = 0.9977 }
+++
# 003 Parity with the maker's code

Verdict: pass.

Reference: identical to florence-2-base's (see that folder's 003-parity.md for the four transformers-5.x shims and the
decoding settings). The large repo's modeling and configuration files are byte-identical to base's at their pinned
revisions; processing_florence2.py differs only in blank lines. Same decoding: the card's generate(num_beams=3,
max_new_tokens=1024) plus text_config's no_repeat_ngram_size 3, early_stopping True, forced BOS/EOS. The repo's
top-level generation_config.json (num_beams 3, early_stopping false) belongs to the outer model; the maker's generate()
calls language_model.generate, which in transformers 4.41 used text_config's settings, so early_stopping is True.

Inputs: the same 9 requests as base (card car photo, Zebra, Parrot, Coastline, 33x17 zebra, two-line sign).

| request | image features cos | first-step logits max abs / cos | tokens | output |
|---|---|---|---|---|
| car | 0.99999 | 0.0156 / 0.9999995 | identical (14) | A green car parked in front of a yellow building. |
| awkward 33x17 | 0.99953 | 0.145 / 0.99991 | identical (13) | a cheetah in the sahara desert |
| zebra | 0.99999 | 0.0137 / 0.9999993 | identical (14) | A zebra standing in a field of tall grass. |
| parrot | 0.99998 | 0.0254 / 0.9999974 | identical (17) | A close up of a parrot's head with a green background. |
| coastline | 0.99996 | 0.0234 / 0.9999981 | identical (15) | A view of the ocean from the top of a cliff. |
| car, <DETAILED_CAPTION> | 0.99999 | 0.0840 / 0.99997 | identical (36) | The image shows a green Volkswagen Beetle ... along with a few trees. |
| car, <MORE_DETAILED_CAPTION> | 0.99999 | 0.0508 / 0.99998 | identical (105) | The image shows a vintage Volkswagen Beetle ... |
| sign, <OCR> | 0.99999 | 0.0469 / 0.9999963 | identical (14) | Fresh bread daily\nOpen 7am to 5pm\n |
| zebra, prompt "A zebra" (decoder prefix) | 0.99999 | 0.0195 / 0.9999994 | identical (14) | A zebra standing in a field of tall grass. |

Input ids identical on all 9. Texts after post_process_generation identical on all 9. The OCR line breaks are token
50118 ("\n"), as the card says this release writes them.

Preprocessing: as for base, the runner equals PIL's uint8 bicubic resize (the maker's 4.41 processor); the 5.22 processor
differs by one 8-bit level on a small share of pixels (max diff 0.035 after normalization).

Not covered: JPEG/HEIC decoding and colour profiles (PNG compared); the torchvision fast processor; decoder-prefix
prompts are an Arsenal addition checked by forcing the same decoder_input_ids in the reference.
