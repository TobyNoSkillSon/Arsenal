+++
hypothesis = "The kit runner reproduces the maker's transformers ViTPose keypoints at FP16."
change = "Kit runner files (Kit.swift, Wire.swift, main.swift, Package.swift) refreshed to the current kit so the build passes its GPU self-test; Model.swift unchanged from the 26 Oct install."
measurement = "transformers VitPoseForPoseEstimation + VitPoseImageProcessor + post_process_pose_estimation, whole image as the person box, MPS float16, against the kit runner's full replies, 2026-10-30, M5 Max."
verdict = "kept"
numbers = { crop_max_px = 0.93, crop_mean_px = 0.35, full_photo_max_px = 2.21, full_photo_mean_px = 1.23, max_score_diff = 0.021, tiny_max_px = 7.1 }
links = ["results/quick-checks/kit-2026-10-30.json"]
+++
# 003 Parity with the maker's code

Weights: the vendor repo usyd-community/vitpose-base-simple at the pinned revision, as the 26 Oct port already used,
not the community MLX build named in 001-plan.md. The op schemas now use the x-arsenal- keywords.

Decisive output: the 17 keypoints (x, y in image pixels) and their scores. Inputs: a public COCO val2017 photo of one
skier (000000000785, 640x425), a 290x380 crop around her, and an awkward 33x17 downscale. Both sides treat the whole
image as the person box. Scripts: lab/night1/vitpose-base-simple/ref.py and lab/night1/tools/runjson.py.

| input | max distance | mean distance | max score diff |
|---|---|---|---|
| skier crop 290x380 | 0.93 px (0.2% of diagonal) | 0.35 px | 0.012 |
| full photo 640x425 | 2.21 px (0.3% of diagonal) | 1.23 px | 0.021 |
| tiny 33x17 (awkward) | 7.1 px | 1.2 px | 0.19 |

Verdict: pass on real inputs. Every joint lands within a fraction of one heatmap cell (the 64x48 heatmap cell is about
17 px on the full photo). The 33x17 input is upsampled about tenfold into the 256x192 crop, its heatmaps are nearly
flat, and the two resampling paths pick different peaks for the weakest joints; it still answers without error.
The FP16-cast stored weights give keypoints identical to the FP32 originals loaded as FP16.
