+++
hypothesis = "The MLX kit runner reproduces PaddlePaddle/UVDoc_safetensors as transformers' UVDocModel and UVDocImageProcessor run it."
change = "Model.swift port: ResNet head and stages, six dilated bridge blocks, head with reflect padding and PReLU, BGR input resized with bilinear align_corners, grid upsample and grid_sample (bilinear, zeros, align_corners) at the photo's size, truncating uint8 cast."
measurement = "lab/night2/uvdoc/ref.py (transformers 5.22 UVDocModel, MPS float16) vs the runner's grid.safetensors and dewarped.png (compare.py); post_check.py runs the maker's post-processing on the runner's own grid; paddle_match.py compares the weights with the planned Paddle file."
verdict = "kept"
links = ["results/quick-checks/kit-2026-10-31.json"]
numbers = { inputs = 1, grid_max_abs_diff = 0.00049, grid_cosine = 1.0, image_cosine = 0.99987, image_mean_abs_diff_255 = 0.90, same_grid_image_max_abs_diff_255 = 1 }
+++
# 003 Parity with the maker's code

Weights: the planned repo PaddlePaddle/UVDoc (16c3f0ea) has only Paddle inference.pdiparams, so the kit refused it. Used
PaddlePaddle's own sibling PaddlePaddle/UVDoc_safetensors at bfa6aad0 (model.safetensors sha256 65f980e3...), the
transformers release named on its card. paddle_match.py parsed the planned inference.pdiparams (sha256 81048889..., the
planned pin) tensor by tensor: every one of the 251 safetensors tensors (BatchNorm step counters aside) is present in the
Paddle file with identical FP32 values. The Paddle file has 8 more tensors (a 3-D point head: conv 128 to 32, BatchNorm,
PReLU, conv 32 to 3, 104,932 values) that the 2-D unwarping never uses and the transformers port drops. Same weights.

Input: PaddleOCR's card example doc_test.jpg (1100x708, an open book on a desk), converted once to PNG so both sides
decode identical pixels. Reference: AutoModel/AutoImageProcessor from the snapshot, float16 on MPS, under gpu.py;
post_process_document_rectification on the fp16 grid.

| check | result |
|---|---|
| grid [45, 31, 2] | max abs diff 0.00049, cosine 1.000000 |
| flattened image 708x1100x3 | cosine 0.99987, mean abs diff 0.90/255, max 45/255 (at text edges) |
| maker's post-processing on the runner's own grid | max abs diff 1/255, mean 0.015/255 |

Verdict: pass. The image differences come from the fp16 grid noise (0.0005 in -1..1 units is about a quarter pixel at
1100 wide); with the same grid the two post-processings agree to 1/255.

Bug found and fixed on the way: the first version resized with float32 matmul weight matrices on the GPU, which carried
errors of about 1e-3 (grid upsample off by up to 0.0014, image cosine 0.9988). Elementwise gathers and lerps fixed it
(upsample diff 2.4e-7, same as numpy).

Awkward input: a 33x17 shrink of the photo returns a 33x17 PNG (no crash). Images under 8x8 or over 50 megapixels get
invalid_request.

Not covered: JPEG inputs decode through ImageIO in the runner and libjpeg in PIL, so JPEG pixels can differ by a step.
