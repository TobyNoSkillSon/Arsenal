+++
hypothesis = "The MLX kit runner reproduces zai-org/GLM-OCR as transformers' own implementation runs it."
change = "Model.swift: mlx-swift-lm's GlmOcr.swift (pinned bd4b7434, MIT) made self-contained: CogViT vision encoder with axial 2-D RoPE and q/k RMSNorm, Conv3d patch embed as a matmul, 2x2 conv downsample, SwiGLU merger, GLM decoder with interleaved 3-D M-RoPE and its own KV cache; Glm46VImageProcessor preprocessing (smart_resize, Pillow-exact bicubic in Pixels.swift, patchify); the repo chat template for 'Text Recognition:'; greedy decoding."
measurement = "lab/night2/glm-ocr/ref.py + parity.py: transformers 5.22 GlmOcrForConditionalGeneration + Glm46VProcessor (default torchvision backend) on MPS float16 vs the runner's dumps (ARSENAL_PARITY_DIR), 3 fixed PNG inputs; the runner cast the BF16 originals to FP16 at load."
verdict = "kept"
links = ["results/quick-checks/kit-2026-10-31.json"]
numbers = { inputs = 3, tokens_identical = 3, texts_identical = 3, input_ids_identical = 3, first_logits_max_abs_diff = 0.074, first_logits_min_cosine = 0.999996, image_features_min_cosine = 0.99958, pixel_max_abs_diff = 0.0156 }
+++
# 003 Parity with the maker's code

Verdict: pass.

Reference: transformers 5.22 (the model's own library; config says transformers 5.17) GlmOcrForConditionalGeneration
from the pinned snapshot at float16 on MPS, AutoProcessor (Glm46VProcessor, default torchvision image backend; the PIL
backend could not load in ref-env because the processor's video part has no PIL backend). The pinned snapshot has no
chat template, so the repo's chat_template.jinja at the same revision was passed in; the rendered prompt is
"[gMASK]<sop><|user|>\n<|begin_of_image|><|image|><|end_of_image|>Text Recognition:<|assistant|>\n". generate(
max_new_tokens=8192, do_sample=False) as generation_config.json says (greedy, EOS 59246 or 59253), decoded with
skip_special_tokens.

Inputs: a 900x420 rendered delivery note (heading, address, date, a 3-column item table), the 640x240 two-line sign from
the Florence-2 checks, and a 33x17 shrink of the macOS Zebra picture.

| input | image grid | input ids | image features cos | first-step logits max abs / cos | tokens | output |
|---|---|---|---|---|---|---|
| delivery note | 30x64 | same (492) | 0.99958 | 0.047 / 0.999997 | identical (83) | Delivery note 2026-114 ... Total 398.00 |
| sign | 18x46 | same (219) | 0.99976 | 0.035 / 0.9999985 | identical (13) | Fresh bread daily Open 7am to 5pm |
| awkward 33x17 | 6x8 | same (24) | 0.99997 | 0.074 / 0.999996 | identical (4) | Zebra |

Decoded texts are identical too.

Preprocessing: the runner resizes with Pillow's uint8 bicubic (the processor's PIL backend); the reference's default
torchvision backend differs by one 8-bit level on about 0.02% of patch values (max diff 0.0156 after normalization).
smart_resize grids and token counts are identical on all three.

Not covered: JPEG/HEIC decoding and colour profiles (PNG compared); large pages (thousands of image tokens) and long
outputs; the formula, table and extraction prompts (not offered); the multi-token-prediction layer is not used (as in
transformers' generate).
