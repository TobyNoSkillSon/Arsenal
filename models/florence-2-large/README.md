# Florence-2 Large

Writes an English caption for a photo. With a task token as the prompt it writes a detailed paragraph or reads the text in the image instead, one line per line. Microsoft's Florence-2 Large: a DaViT image encoder and a 12-layer BART-style encoder-decoder, 777M parameters, trained on 5.4 billion annotations. This is the maker's continued-pretraining release with a 4k context. Decoding is the card's: 3-beam search, no repeated 3-grams.

Input: one image (`image`, PNG, JPEG or HEIC) and an optional `prompt`. Output: `caption`, English text.

The prompt can be:

- empty: a one-sentence caption
- `<DETAILED_CAPTION>`: two or three sentences
- `<MORE_DETAILED_CAPTION>`: a paragraph
- `<OCR>`: the text in the image, with line breaks
- any other text: the start of a caption for the model to continue

Example request:

```json
{"image": {"path": "/absolute/path/to/sign.png"}, "prompt": "<OCR>"}
```

On the card's car photo it returns "A green car parked in front of a yellow building." On a sign with two printed lines, `<OCR>` returns "Fresh bread daily\nOpen 7am to 5pm\n".

Limits: English only, not Polish. The card warns this release saw little extra training. Every image is squashed to 768x768, and a tiny image still gets a confident caption. The paragraph mode can invent details. The card also lists detection, phrase grounding and segmentation; Arsenal does not offer them for this model, because their boxes do not fit a caption.

Parity: on 9 fixed requests covering all four tasks and a caption start, the output tokens are identical to the maker's own Florence-2 code run with transformers on MPS at fp16. The first-step logits agree at cosine 0.9999 or better. Details are in log/003-parity.md.

Source: [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large/tree/21a599d414c4d928c9032694c424fb94458e3594), MIT. Runs as FP16 through the Arsenal kit runner (MLX). It shares its runner code with Florence-2 Base.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: the card says the OCR task was updated to use line separators (`\n`). [HF model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: The task token is the prompt. In Arsenal only the caption tokens and `<OCR>` work; the card's grounding, detection and segmentation tasks are not offered here. [HF model card, undated, first-hand; Arsenal runner]
- Parameters people recommend: the task helper uses `max_new_tokens=1024`, `num_beams=3`, sampling off; `trust_remote_code=True`; float16 on CUDA, else float32. [HF model card, undated, first-hand]
- Arsenal call: `arsenal_run(model:"florence-2-large", operation:"caption", input:{"image":{"path":"/absolute/path/sign.png"},"prompt":"<OCR>"})`. Vendor example, for reference: the card's `run_example(task_prompt, text_input=None)` helper with task `<OCR>`. [HF model card, undated, first-hand]

**Good and bad**
- Good at: Not found in sources checked as independent results; the card reports only its own benchmarks. [HF model card, undated, first-hand]
- Bad at: heavily rotated text (90 degrees or more), tiny text after the 768x768 resize, and generative OCR "correcting" unfamiliar codes or numbers. [Curiosity OCR guide, undated, third-party] The card warns the release "might not be trained well". [HF model card, undated, first-hand]
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: an HF discussion poster reports weak German OCR and asks whether the English `tokenizer.json` can be replaced. Fix or workaround: Not found in sources checked; no replies. [HF discussion #91, opened 2024-12-18, opinion]
- Complaint or known bug: the card's code loads `AutoModelForCausalLM`, but its "Use this model" panel shows `AutoModelForMultimodalLM`. Fix or workaround: Not found in sources checked. [HF model card, undated, first-hand]

**Evidence**
- Strength: thin. Maker's card, one discussion with no replies, one third-party guide. No independent benchmark on the card.
- Sources: card https://huggingface.co/microsoft/Florence-2-large (undated); discussion https://huggingface.co/microsoft/Florence-2-large/discussions/91 (2024-12-18); guide https://docs.curiosity.ai/florence2-sharp/guides/ocr (undated).
