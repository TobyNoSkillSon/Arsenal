+++
hypothesis = "The MLX kit runner reproduces the maker's SAM 2.1 mask logits and IoU predictions at FP16."
change = "None; comparison only. A lab-only logits dump hook used for this check was removed from the runner afterwards."
measurement = "transformers Sam2Model + Sam2Processor (facebook/sam2.1-hiera-large, same pinned revision) on MPS at float16 against the kit runner's 256x256 low-resolution mask logits, six image and prompt pairs, 2026-10-31, M5 Max. Two variants: the same 1024x1024 pixel tensor fed to both networks, and each side's own image loading."
verdict = "kept"
numbers = { cases = 6, min_cosine_same_pixels = 0.999997, min_cosine_own_loading = 0.999907, max_abs_logit_same_pixels = 0.945, best_mask_identical = true }
links = ["results/quick-checks/kit-2026-10-31.json"]
+++
# 003 Parity with the maker's code

Decisive output: the low-resolution mask logits (256x256, before thresholding) of the masks the op returns, and their
predicted IoU. Reference: transformers 5.22 `Sam2Model` and `Sam2Processor` on the same pinned checkpoint, MPS, FP16
(the reference casts the FP32 checkpoint to FP16 on load, as the runner does).
Inputs: the SAM 2 notebook truck photo (1800x1200 JPEG, sRGB) with a window click, a whole-truck box, a wheel box with a
background click on the tyre (single mask) and two wheel boxes (one mask each); a 640x360 Display P3 photo of a rainbow
arch with one click (single mask); a 33x17 PNG with one click. Scripts: lab/night2/sam2/{ref.py,drive_try.py,cases.json};
numbers in parity-large.json.

| case | masks | cosine, same pixels | max abs logit | min mask IoU | cosine, own loading | max abs logit | IoU maker / MLX | min mask IoU |
|---|---|---|---|---|---|---|---|---|
| truck window click | 3 | 1.000000 | 0.938 | 0.9986 | 0.999987 | 4.250 | 0.838 0.047 0.622 / 0.835 0.047 0.626 | 0.9979 |
| truck box | 3 | 0.999999 | 0.473 | 0.9951 | 0.999981 | 2.352 | 0.982 0.986 0.993 / 0.982 0.986 0.993 | 0.9830 |
| wheel box minus tyre (single) | 1 | 0.999999 | 0.672 | 0.9952 | 0.999993 | 4.906 | 0.888 / 0.889 | 0.9933 |
| arch click, Display P3 (single) | 1 | 0.999998 | 0.484 | 0.9994 | 0.999907 | 1.422 | 0.972 / 0.974 | 0.9950 |
| two wheel boxes (one mask each) | 2 | 0.999999 | 0.812 | 0.9992 | 0.999992 | 5.266 | 0.978 0.945 / 0.978 0.946 | 0.9973 |
| 33x17 PNG click | 3 | 0.999997 | 0.945 | 0.9988 | 0.999997 | 0.945 | 0.086 0.148 0.412 / 0.086 0.148 0.412 | 0.9988 |

Verdict: pass. Cosine 0.999997 or better with the same pixels and 0.999907 or better through each side's own image
loading, on every case; the predicted IoU agrees to about 0.003 and the best mask is the same everywhere.
The runner decodes images to their stored RGB values (no colour conversion, as PIL does) and resizes with the maker's
antialiased bilinear; an earlier build that converted the Display P3 photo (run on the tiny model) to sRGB fell to cosine 0.777 there, because the
maker's single-mask check then chose a different output. The remaining own-loading gap is the JPEG decoder (Apple's and
libjpeg's differ by a few levels); the PNG matches exactly. The masks were also checked by eye: the window click gives the
window pane, the two boxes give one wheel each.
