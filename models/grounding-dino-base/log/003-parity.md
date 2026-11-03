+++
hypothesis = "The MLX kit runner reproduces IDEA-Research/grounding-dino-base as transformers' GroundingDinoProcessor and GroundingDinoForObjectDetection run it."
change = "Model.swift, Backbones.swift, Pixels.swift: Swin-B backbone (window 12, always partitioned), BERT with the per-phrase mask and position ids, input projections with GroupNorm, sine position embeddings, six fusion / text-enhancer / deformable encoder layers, two-stage top-900 query selection, six decoder layers with iterative box refinement; multi-scale deformable attention's grid_sample written as gathers; uint8 antialiased bilinear resize (PIL algorithm, copied from the clipseg runner)."
measurement = "lab/night2/grounding-dino-base/ref.py and probe_ref.py (transformers 5.22, MPS float16; float32 as an adjudicating probe) vs the runner's ARSENAL_PARITY_DIR dumps (compare.py, probe_compare.py, tri.py) on 2 fixed PNG inputs."
verdict = "kept"
links = ["results/quick-checks/kit-2026-10-31.json"]
numbers = { inputs = 2, encoder_vision_min_cosine = 0.99987, encoder_text_min_cosine = 0.999999, kept_detections_same = true, score_max_diff_vs_fp32 = 0.0024, box_max_diff_vs_fp32 = 0.0011 }
+++
# 003 Parity with the maker's code

Weights: the planned pin, model.safetensors sha256 5548f844... at 12bdfa31, FP32 on disk, cast to FP16 by the runner.

Reference: GroundingDinoProcessor (torchvision backend) and GroundingDinoForObjectDetection on MPS in float16 under
gpu.py, plus the same in float32 once per input to adjudicate. Inputs: the card's COCO cats photo (640x480 PNG, prompt
"a cat. a remote control.") and an 1100x708 book photo (prompt "book. cartoon drawing. black cable.").

| check | cats | book |
|---|---|---|
| input_ids | identical | identical |
| pixel values (normalized) | max diff 0.0175 (one 8-bit step at a few pixels: torch's uint8 resize vs the PIL algorithm) | same |
| encoder text states | cosine 0.9999998 | cosine 0.9999991 |
| encoder image states | cosine 0.99996 | cosine 0.99987 |
| top-900 proposals | 892 of 900 the same | - |
| detections above threshold vs torch fp16 | same 4 boxes and labels, scores within 0.002, boxes within 0.001 | same 4 boxes and labels; one box drifts in torch fp16 (below) |
| detections vs torch fp32 | score max diff 0.0009, box 0.0002, same set | score max diff 0.0024, box 0.0011, same set |

Verdict: pass. The per-query arrays (logits [900, L], boxes [900, 4]) cannot be compared index by index: the 900 queries
are the top-k of float16 scores with many exact ties, and a one-ulp difference reorders them, which changes which learned
query embedding each proposal gets. PyTorch's own fp16 run reorders them the same way against its fp32 run (106 of 900 in
the same place; MLX 201). On the book photo PyTorch fp16's second "book" box drifts from its fp32 run (0.314 from x 21
vs 0.304 from x 32 pixels) while MLX's stays with fp32 (0.306 from x 33.5); the first book box (0.489 vs fp32 0.492),
the cartoon (0.589 vs 0.592) and the cable (0.820 vs 0.819) agree. The runner sorts float16 scores with ties in index
order, as torch.topk does.

Bug found with a per-layer probe: encoder layer 0's text enhancer differed (cosine 0.9993) while its inputs matched.
A float64 numpy recomputation showed the maker's encode_sinusoidal_position_embedding casts its output back to the dtype
of the int64 position ids, so the text position embedding truncates to 0 everywhere except cos(0) = 1. The runner
reproduces that cast; afterwards every encoder sub-layer matched (cosine 0.99996 to 0.9999998).

Labels: the maker labels a box with the decoded tokens above text_threshold (0.25), which can be a partial phrase
("cartoon"); the runner labels it with the whole query whose tokens score highest ("cartoon drawing"). Scores and boxes
are the maker's.

Awkward input: a 33x17 shrink of the cats photo with labels ["cat"] returns two cat boxes (0.57 and 0.32): the processor
upsamples it to 1333x687, so the blurry cats are still found. By code, not run: no labels and no phrase, more than 128
things or more than 256 tokens give invalid_request.

Not covered: batched or padded inputs (Arsenal sends one image), prompts near 256 tokens, JPEG decoding differences.
