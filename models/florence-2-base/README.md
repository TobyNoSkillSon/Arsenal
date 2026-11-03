# Florence-2 Base

Writes an English caption for a photo. With a task token as the prompt it writes a detailed paragraph or reads the text in the image instead. Microsoft's Florence-2 Base: a DaViT image encoder and a BART-style encoder-decoder, 232M parameters, trained on 5.4 billion annotations. Decoding is the card's: 3-beam search, no repeated 3-grams.

Input: one image (`image`, PNG, JPEG or HEIC) and an optional `prompt`. Output: `caption`, English text.

The prompt can be:

- empty: a one-sentence caption
- `<DETAILED_CAPTION>`: two or three sentences
- `<MORE_DETAILED_CAPTION>`: a paragraph
- `<OCR>`: the text in the image
- any other text: the start of a caption for the model to continue

Example request:

```json
{"image": {"path": "/absolute/path/to/photo.jpg"}, "prompt": "<DETAILED_CAPTION>"}
```

On the card's car photo it returns "A green car parked in front of a yellow building." With `<DETAILED_CAPTION>` it returns "The image shows a green Volkswagen Beetle parked in front of a yellow building with two brown doors. The sky is visible in the background."

Limits: English only, not Polish. OCR drops line breaks and joins lines. Every image is squashed to 768x768, and a tiny image still gets a confident caption. The paragraph mode can invent details. Detection, grounding and segmentation are not offered here, because their boxes do not fit a caption.

Parity: on 9 fixed requests covering all four tasks and a caption start, the output tokens are identical to the maker's own Florence-2 code run with transformers on MPS at fp16. The first-step logits differ by at most 0.016 (cosine 0.9999999). Details are in log/003-parity.md.

Source: [microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base/tree/5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac), MIT. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: one image, with a task token as the text prompt, run through the card's processor. [Florence-2-base model card, undated (checked 2026-10-31), first-hand]
- Prompts, reference audio or voice settings, if relevant: the card also lists detection and grounding tokens (`<OD>`, `<CAPTION_TO_PHRASE_GROUNDING>`); the Arsenal entry does not offer them. [Florence-2-base model card, undated (checked 2026-10-31), first-hand]
- Parameters people recommend: `num_beams=3`, `do_sample=False`, `max_new_tokens=1024`, as the card's example uses. [Florence-2-base model card, undated (checked 2026-10-31), first-hand]
- Arsenal call: `POST /arsenal/v1/models/florence-2-base/ops/caption`, or the same operation through MCP. Vendor example: `processor(text=prompt, images=image, return_tensors="pt")`, then `model.generate(input_ids=..., pixel_values=..., max_new_tokens=1024, do_sample=False, num_beams=3)`. [Florence-2-base model card, undated (checked 2026-10-31), first-hand]

**Good and bad**
- Good at: the maker reports zero-shot COCO caption CIDEr 133.0, NoCaps 118.7 and TextCaps 70.1. [Florence-2-base model card, undated (checked 2026-10-31), first-hand]
- Bad at: Not found in sources checked. The card has no limitations section.
- Against the obvious alternatives: on 238 owner-labelled bird frames, Florence-2 caption named the bird type in 28% of frames, against 7% for BLIP-base. Detailed Florence-2 caption reached 72%. GPU time was 136 ms against 148 ms per image. [image-scoring-pipeline issue #421, 2026-09-25, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: the model fails on transformers newer than 4.49, and a reply on 2025-10-06 says it still does not work. Fix or workaround: none in the thread. [microsoft/Florence-2-base discussion #30, 2025-05-31, first-hand]

**Evidence**
- Strength: thin. Maker numbers only, plus one personal project and one bug thread.
- Sources:
  - Florence-2-base model card, https://huggingface.co/microsoft/Florence-2-base, undated (checked 2026-10-31), first-hand.
  - Discussion #30 "florence2 is broken above transformers 4.49", https://huggingface.co/microsoft/Florence-2-base/discussions/30, 2025-05-31, first-hand.
  - Issue #421 "Captions: Florence-2-base shadow trial against BLIP-base", https://github.com/synthet/image-scoring-pipeline/issues/421 (repo since renamed from image-scoring-backend), 2026-09-25, first-hand.
