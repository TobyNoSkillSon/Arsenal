+++
hypothesis = "The MLX kit runner reproduces the maker's EoMT panoptic segmentation at FP16."
change = "None; comparison only. A lab-only dump hook (logits and panoptic map) used for this check was removed from the runner afterwards."
measurement = "transformers EomtForUniversalSegmentation + EomtImageProcessorFast and post_process_panoptic_segmentation (same pinned revision) on MPS at float16 against the kit runner, four images, 2026-10-31, M5 Max. Two variants: the same padded 640x640 pixel tensor fed to both networks, and each side's own image loading."
verdict = "kept"
numbers = { images = 4, labels_and_segments_identical = 4, argmax_identical_queries = "200 of 200 on every image", min_mask_cosine_same_pixels = 0.999992, min_class_cosine_same_pixels = 0.999994, min_pixel_agreement = 0.9982, min_mask_cosine_own_loading = 0.998764 }
links = ["results/quick-checks/kit-2026-10-31.json"]
+++
# 003 Parity with the maker's code

Decisive outputs: the class decision of every query and the panoptic result (which segments, with which labels, and the
per-pixel map at the photo's size); also the raw mask logits (200 x 160 x 160) and class logits (200 x 134).
Reference: transformers 5.22 `EomtForUniversalSegmentation`, its image processor and `post_process_panoptic_segmentation`
with default thresholds, MPS, FP16. Scripts and inputs: lab/night2/eomt-coco-panoptic-large-640/{ref.py,parity.json}.

| image | kept queries / segments | same labels and segments | mask cosine, same pixels | class cosine | map agreement | mask cosine, own loading | class cosine | map agreement |
|---|---|---|---|---|---|---|---|---|
| COCO cats on a couch (640x480) | 5 / 5 | yes | 0.999992 | 1.000000 | 0.9999 | 0.999971 | 0.999998 | 0.9998 |
| SAM 2 truck photo (1800x1200) | 4 / 4 | yes | 0.999996 | 0.999994 | 0.9999 | 0.999728 | 0.999518 | 0.9998 |
| rainbow arch, Display P3 (640x360) | 4 / 4 | yes | 0.999998 | 0.999999 | 0.9999 | 0.998764 | 0.998991 | 0.9988 |
| 33x17 PNG | 3 / 2 | yes | 0.999997 | 0.999997 | 0.9982 | 0.999994 | 0.999997 | 0.9982 |

Verdict: pass. On every image the argmax class of all 200 queries, the queries kept above 0.8, the segment labels and
their order are identical; segment scores agree to 0.0006; the panoptic maps agree on 99.8% of pixels or more (the rest is
segment borders). With the same pixels the mask logits are at cosine 0.99999 or better. Through each side's own image
loading the arch photo drops to cosine 0.9988 for the raw mask logits (labels and segments still identical): its JPEG has
4:2:0 chroma, and Apple's decoder and libjpeg differ by up to 40 levels on about 1% of its pixels, along the saturated
colour edges of the rainbow. The maps were also checked by eye on the cats photo: each cat, the couch and both remotes are
traced on their objects.
