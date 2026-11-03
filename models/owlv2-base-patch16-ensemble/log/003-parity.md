+++
hypothesis = "The MLX kit runner reproduces google/owlv2-base-patch16-ensemble as transformers' Owlv2Processor and Owlv2ForObjectDetection run it."
change = "Model.swift port: torchvision-backend preprocessing (pad to square with black, Gaussian anti-alias blur folded into the bilinear resize, CLIP normalization), CLIP ViT-B/16 vision and causal text towers, class head with logit shift and scale, box head with the grid bias; CLIP BPE tokenizer copied from the clipseg runner (context 16)."
measurement = "lab/night2/owlv2-base-patch16-ensemble/ref.py (transformers 5.22, MPS float16, and float32 once as an adjudicating probe) vs the runner's ARSENAL_PARITY_DIR dumps (compare.py) on 2 fixed PNG inputs."
verdict = "kept"
links = ["results/quick-checks/kit-2026-10-31.json"]
numbers = { inputs = 2, logits_min_cosine = 0.99994, boxes_min_cosine = 0.99995, kept_boxes_same_set = true, kept_labels_identical = true, kept_box_max_abs_diff = 0.0013, max_score_diff = 0.0176 }
+++
# 003 Parity with the maker's code

Weights: the planned pin, model.safetensors sha256 e1e130b9... at cfd3195b, FP32 on disk, cast to FP16 by the runner at load.

Reference: Owlv2Processor (torchvision backend, torchvision 0.29.2 in ref-env) and Owlv2ForObjectDetection in float16 on
MPS under gpu.py. Decisive outputs: logits [3600, Q] and pred_boxes [3600, 4]; the reply is post_process_object_detection
on them (sigmoid of the best query, threshold, boxes times the padded side).

| input | queries | pixels max diff | logits cosine / max diff | boxes cosine | boxes above 0.1 | labels on them | box diff there | score diff |
|---|---|---|---|---|---|---|---|---|
| card's COCO cats 640x480 (upsampled, no blur) | a photo of a cat, a photo of a dog | 0.0001 | 0.99995 / 6.6 | 0.99988 | 2 and 2, same | identical | 0.0008 | 0.011 |
| book photo 1100x708 (blur kernel 3) | a book, a cable | 0.0003 | 0.99994 / 2.7 | 0.99995 | 9 and 9, same | identical | 0.0013 | 0.018 |

Verdict: pass (cosine >= 0.999; the detections the user sees are the same boxes with the same labels).

The large max logit differences sit in a handful of patches (0.7% of them, mostly empty background and the black padding),
where the best query label also flips at very low scores. A float32 run of the maker's model adjudicated it on the cats
photo: PyTorch fp16 vs fp32 logits max diff 4.3, mean 0.070, cosine 0.99997; MLX fp16 vs fp32 max diff 2.3, mean 0.073,
cosine 0.99998, boxes max diff 0.20 vs PyTorch fp16's 0.58. The port is as close to the fp32 model as PyTorch's own fp16.

The card's own run (fp32, older transformers) reports cat scores 0.665 and 0.614; the runner gives 0.658 and 0.623 with
boxes within 5 pixels.

Awkward input: a 33x17 shrink of the cats photo with phrase "cat. remote control." returns an empty detections list
(no crash). By code, not run: no labels and no phrase, or more than 512 queries, give invalid_request.

Not covered: queries longer than 14 word pieces, which the maker's processor does not truncate (the reference would
fail); the runner cuts them. JPEG decoding differs slightly between ImageIO and PIL.
