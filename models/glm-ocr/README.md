# GLM-OCR

Reads the text in a document image or screenshot and returns it as plain text, keeping lines and paragraphs. Z.ai's GLM-OCR: a CogViT vision encoder and a GLM-0.5B decoder, trained for document OCR, and first overall on OmniDocBench V1.5 by the card's count (94.62). It runs the card's "Text Recognition:" prompt with greedy decoding, as the maker's generation config says.

Input: one image (`image`, PNG, JPEG or HEIC). Output: `text`.

Example request:

```json
{"image": {"path": "/absolute/path/to/page.png"}}
```

On a rendered delivery note it returned the heading, the address and date lines and the item rows, each on its own line, with blank lines between blocks.

Limits: text recognition only. The formula, table and JSON-extraction prompts are not offered, and there is no layout analysis. Table columns collapse to single spaces. Polish is not among the card's languages. A tiny image still gets an answer. A large page needs several GB of memory.

Parity: on 3 fixed inputs the output tokens are identical to transformers' GlmOcrForConditionalGeneration with its default processor, run on MPS at fp16. The first-step logits agree at cosine 0.999996 or better. Details are in log/003-parity.md.

Source: [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR/tree/2e85a62840ccac27daa451df36c736c4636b8628), MIT. The BF16 weights are cast to FP16 at Get and run through the Arsenal kit runner (MLX). The network code follows mlx-swift-lm's GlmOcr.swift (MIT) at the app's pinned revision.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: image paths or URLs. Markdown and JSON layout output; no plain-text output is named. [GLM-OCR README, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: "Table Recognition:" (#240 reporter) and "Text Recognition:" (#220 suggestion). The README names no prompts. [GitHub #240, 2026-08-28; #220, 2026-06-04, first-hand]
- Parameters people recommend: temperature 0 and top_k 1, one user's config. [GitHub #240, 2026-08-28, first-hand]
- Arsenal call: `POST /arsenal/v1/models/glm-ocr/ops/read`, or the same operation through MCP. Vendor SDK: `from glmocr import GlmOcr, parse`; `with GlmOcr() as parser: result = parser.parse("image.png")`. [GLM-OCR README, undated, first-hand]

**Good and bad**
- Good at: document OCR; the maker reports 94.62 on OmniDocBench V1.5, #1 overall. [GLM-OCR README, undated, first-hand]
- Bad at: table tails. Footer rows came out wrong for the #240 reporter (MLX). A second user got them right but an extra hallucinated row, on MLX and vLLM, at the SDK's default repetition_penalty 1.1. [GitHub #240, 2026-08-28 and 2026-09-22, first-hand]
- Bad at: Cantonese and Hong Kong traditional characters misread, per one user's list (vLLM). [GitHub #212, 2026-05-11, first-hand]
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: through Ollama, output repeats until a stop sequence is set. Still open. Fix or workaround: a stop sequence worked for potproject and CH3SH-LC (Ollama 0.34.1); CH3SH-LC found the native prompt alone still looped. No project reply. [GitHub #220, 2026-06-04, first-hand]
- Complaint or known bug: large tables cut off. Fix or workaround: raise `--max-model-len` and `--gpu-memory-utilization` (vLLM). Project responder JaredforReal said so; one user found output less accurate after. Closed 2026-04-07. [GitHub #182, 2026-04-02, first-hand]

**Evidence**
- Strength: thin. The accuracy figure is the maker's own; no independent benchmark was found.
