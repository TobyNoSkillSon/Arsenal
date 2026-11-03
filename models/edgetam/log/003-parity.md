+++
hypothesis = "The MLX kit runner reproduces the maker's EdgeTAM mask logits and IoU predictions at FP16."
change = "Runner: images are decoded to their stored 8-bit RGB values (no colour conversion, as PIL does) and resized to 1024x1024 with the maker's antialiased bilinear (Resize.swift; the resize is copied from the CLIPSeg runner) instead of CoreGraphics scaling; multimask_output false now picks the mask as the maker does (stability check, delta 0.05, threshold 0.98) instead of always mask 0; Kit.swift and Wire.swift refreshed from kit/runner (the kit's GPU self-test needs them). The network code (Net.swift) is unchanged. A lab-only logits dump hook used for this check was removed afterwards."
measurement = "transformers EdgeTamModel + Sam2 processor (yonigozlan/EdgeTAM-hf, revision c266ce53) on MPS at float16 against the kit runner's 256x256 low-resolution mask logits, five image and prompt pairs, 2026-10-31, M5 Max. Two variants: the same 1024x1024 pixel tensor fed to both networks, and each side's own image loading."
verdict = "kept"
numbers = { cases = 5, min_cosine_same_pixels = 0.999994, min_cosine_own_loading = 0.999964, max_abs_logit_same_pixels = 0.203, min_mask_iou_own_loading = 0.9879 }
links = ["results/quick-checks/kit-2026-10-31.json"]
+++
# 003 Parity with the maker's code

Decisive output: the low-resolution mask logits (256x256, before thresholding) of the masks the op returns, and their
predicted IoU. Reference: transformers 5.22 `EdgeTamModel` with `yonigozlan/EdgeTAM-hf` (a transformers conversion of
the maker's edgetam.pt; the runner loads `mlx-community/EdgeTAM-fp16`, an MLX conversion of the same file), MPS, FP16.
Inputs: the SAM 2 notebook truck photo (1800x1200 JPEG, sRGB) with a window click, a whole-truck box, and a wheel box
with a background click on the tyre (single mask); a 640x360 Display P3 photo of a rainbow arch with one click (single
mask); a 33x17 PNG with one click. Scripts: lab/night2/edgetam/{ref.py,drive_try.py,cases.json}; numbers in parity.json.

| case | masks | cosine, same pixels | max abs logit | min mask IoU | cosine, own loading | max abs logit | IoU maker / MLX | min mask IoU |
|---|---|---|---|---|---|---|---|---|
| truck window click | 3 | 0.999999 | 0.156 | 0.9976 | 0.999998 | 0.469 | 0.046 0.487 0.761 / 0.046 0.487 0.758 | 0.9958 |
| truck box | 3 | 0.999994 | 0.172 | 0.9954 | 0.999993 | 0.238 | 0.946 0.978 0.946 / 0.946 0.978 0.946 | 0.9965 |
| wheel box minus tyre | 1 | 0.999999 | 0.203 | 0.9990 | 0.999998 | 0.406 | 0.901 / 0.901 | 0.9913 |
| arch click (Display P3) | 1 | 0.999994 | 0.148 | 0.9999 | 0.999964 | 0.492 | 0.947 / 0.950 | 0.9997 |
| 33x17 PNG click | 3 | 0.999994 | 0.148 | 0.9863 | 0.999994 | 0.164 | 0.351 0.312 0.092 / 0.352 0.314 0.091 | 0.9879 |

Verdict: pass. Cosine 0.99996 or better on every case, both with the same pixels and through each side's own image
loading; the best mask agrees everywhere.

Found and fixed on the way. The 26 Oct runner resized with CoreGraphics and converted colours to sRGB: through its own
loading it was at cosine 0.979 on the 33x17 image and 0.9989 on the Display P3 photo, where the maker's single-mask check
also picked a different output. With the maker's resize and raw RGB values those became 0.999994 and 0.99996. The rest
of the own-loading gap is the JPEG decoder (Apple's and libjpeg's differ by a few levels: 0.24 to 0.66 in normalized
pixel units on these photos); the PNG matches exactly.
