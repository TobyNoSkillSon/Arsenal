# UVDoc

Flattens a photo of a page. Give it a phone photo of a curled book spread, a folded sheet or a tilted document, and it returns the page resampled flat and straight, ready to read or to pass to OCR. A small convolutional network predicts a coarse sampling grid; the grid is upsampled to the photo's size and the photo is resampled through it, as PaddleOCR's document unwarping step does.

Input: one image (`image`, PNG, JPEG or HEIC). Output: `files`, the flattened page as a PNG at the photo's own size (`dewarped`), and the predicted grid as FP16 safetensors (`grid`, 45 by 31 points of x, y in -1 to 1).

Example request:

```json
{"image": {"path": "/absolute/path/to/page-photo.jpg"}}
```

On PaddleOCR's example photo of an open book on a desk it returns the two pages cropped to the spread, with the curved text lines straightened.

Limits: reads no text, so language does not matter. The output keeps the photo's size and aspect ratio, not the page's, and some background can stay at the edges. It sees the photo at 488 by 712, so fine local wrinkles stay. One page or one spread that fills most of the frame works best.

Parity: on the maker's example photo the predicted grid matches transformers' UVDocModel (MPS fp16) to within 0.0005 (cosine 1.000000), and the flattened image matches the maker's post-processing with cosine 0.99987. Details in log/003-parity.md.

Source: [PaddlePaddle/UVDoc_safetensors](https://huggingface.co/PaddlePaddle/UVDoc_safetensors/tree/bfa6aad03b549c7e2cf89cefd2fb02ba3bef2ae4), Apache-2.0, PaddlePaddle's own safetensors release of the weights in PaddlePaddle/UVDoc. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: one image. The card's example passes `image` through the model's image processor. [PaddlePaddle/UVDoc_safetensors model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not found in sources checked.
- Parameters people recommend: Not found in sources checked. The card gives no inference settings.
- Arsenal call: `POST /arsenal/v1/models/uvdoc/ops/dewarp`, or the same operation through MCP. Vendor example (transformers): `model = AutoModel.from_pretrained(model_path, device_map="auto")`, `image_processor = AutoImageProcessor.from_pretrained(model_path)`, `inputs = image_processor(images=image, return_tensors="pt").to(model.device)`, `outputs = model(**inputs)`, `result = image_processor.post_process_document_rectification(outputs.last_hidden_state, inputs["original_images"])` [PaddlePaddle/UVDoc_safetensors model card, undated, first-hand]

**Good and bad**
- Good at: the card reports one number, a CER of 0.179 on the DocUNet benchmark. It does not say whether lower is better, so the number alone shows no quality win. [PaddlePaddle/UVDoc_safetensors model card, undated, first-hand]
- Bad at: Not found in sources checked. The card has no limitations section.
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: the authors say the paper's MS-SSIM and AD figures for UVDoc were computed on only half the UVDoc benchmark, for their method and the related works. Fix or workaround: the erratum gives corrected full-benchmark figures; for UVDoc, MS-SSIM 0.785 and AD 0.119 (paper: 0.784 and 0.122). [tanguymagne/UVDoc GitHub README, erratum, undated, first-hand]

**Evidence**
- Strength: thin. One number from the maker's card and one erratum; no independent tests, Reddit or forum threads found.
- Sources:
  - UVDoc: Neural Grid-based Document Unwarping, arXiv 2302.02887, https://arxiv.org/abs/2302.02887, 2023-02-06 (v1), first-hand.
  - PaddlePaddle/UVDoc_safetensors, model card, https://huggingface.co/PaddlePaddle/UVDoc_safetensors, undated, first-hand.
  - tanguymagne/UVDoc, GitHub README, https://github.com/tanguymagne/UVDoc, undated, first-hand.
