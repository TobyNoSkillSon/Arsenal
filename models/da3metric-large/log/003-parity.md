+++
hypothesis = "The kit runner with Arsenal's vendored MLXDA3 reproduces the maker's DA3 Metric Large depth at FP16."
change = "None; comparison only."
measurement = "depth_anything_3 DepthAnything3Net (da3metric-large) on MPS float16 autocast against the kit runner's raw depth (no focal length given), 2026-10-30, M5 Max."
verdict = "kept"
numbers = { arch_cosine = 0.9996768, skier_cosine = 0.9998493, arch_non_sky_cosine = 0.9993856, skier_non_sky_cosine = 0.9997185, min_sky_cosine = 0.999897 }
links = ["results/quick-checks/kit-2026-10-30.json"]
+++
# 003 Parity with the maker's code

Inputs: `photo.jpg` (a macOS Portrait background, a rainbow arch on a lawn, 640x360) and a public COCO val2017 photo
of a skier (000000000785, 640x425). Scripts: lab/night1/da3/{ref.py,view2.png} and lab/night1/tools/runjson.py; the
reference is depth_anything_3 at ByteDance-Seed/Depth-Anything-3@3d835ec, built from its registry config with the
pinned weights (strict load: no unexpected keys), with its InputProcessor at 504 upper_bound_resize, on MPS with
float16 autocast as the maker's api.forward does off CUDA. Compared at the processed resolution, float32 on both sides.
The FP16-cast stored weights give maps identical to the FP32 originals loaded as FP16.

| photo | cosine | non-sky cosine | non-sky mean abs | sky fill, maker vs runner | reference range | sky cosine |
|---|---|---|---|---|---|---|
| arch 504x280 | 0.99968 | 0.99939 | 0.35 | 40.5 vs 38.9 | 3.0 to 49.7 | 0.99995 |
| skier 504x336 | 0.99985 | 0.99972 | 0.018 | 17.02 vs 17.02 | 1.3 to 21.1 | 0.99990 |

Verdict: pass (cosine >= 0.999). The compared depth is the canonical output before the focal-length scale, which is
plain arithmetic (focal_px x depth / 300, utils/alignment.py). Sky pixels take the 99th percentile of non-sky depth:
the maker samples 100,000 of the arch photo's 104,894 non-sky pixels at random, the runner uses all of them, so the
fill differs by 4% there; with fewer non-sky pixels than the sample size (the skier) both agree. A few pixels at the
sky boundary flip between sky and non-sky in half precision and carry the largest differences.

Metric check: the skier photo with focal_length 530 returns depth_unit meters; a 33x17 image with mode metric and no
focal length returns invalid_request.

After review fixes (same day): mode "relative" with a focal length returns depth_unit relative, and a focal_length of
90,000,000 on the 33x17 image returns invalid_request instead of writing infinities. The metre scale now uses the mean
of the width and height resize ratios, as the maker averages fx and fy.
