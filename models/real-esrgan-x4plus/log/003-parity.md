+++
hypothesis = "The MLX kit runner matches the maker's RRDBNet and RealESRGANer inference (PyTorch MPS, float16) on the same checkpoint."
change = "None: measurement of the Model.swift port before the FP16 cast."
measurement = "lab/night1/real-esrgan-x4plus/ref.py vs kit try outputs, compare.py, m5-max, 2026-10-30: two photos, 8-bit RGB outputs."
verdict = "kept"
numbers = { cosine_min = 0.999960, psnr_db_min = 51.87, max_abs_diff_levels = 19 }
links = []
+++
# 003 Parity with the maker's RRDBNet (2026-10-30)

Method: BasicSR's `RRDBNet` (num_feat 64, num_block 23, num_grow_ch 32, scale 4), written out from
`basicsr/archs/rrdbnet_arch.py`, loaded strictly from `RealESRGAN_x4plus.safetensors` (original FP32 file, before the cast),
run as RealESRGANer does: RGB / 255, model.half() on MPS, clamp 0..1, x255, round, uint8; for the larger input its
`tile_process` with tile 256 and tile_pad 10 (the same tiling the runner uses). PyTorch 2.15.0, M5 Max.
The kit runner (MLX, FP16) produced the PNG it returns for the same file. Compared on all 8-bit RGB values.
Scripts: lab/night1/real-esrgan-x4plus/ref.py and compare.py. Inputs (sRGB PNG): coast256.png (Big Sur Coastline desktop
thumbnail, 256x256, one tile) and rock600.png (Catalina Rock thumbnail, 600x600, 3x3 tiles).

| input | output | max abs diff (of 255) | mean abs diff | pixels > 2 levels off | PSNR | cosine |
|---|---|---|---|---|---|---|
| coast256 (no tiling) | 1024x1024 | 9 | 0.229 | 0.44% | 53.66 dB | 0.999990 |
| rock600 (3x3 tiles) | 2400x2400 | 19 | 0.274 | 1.27% | 51.87 dB | 0.999960 |

Both sides run float16 through 23 residual-in-residual blocks, so rounding differs in a few pixels; most differences are
one level. The rock600 tile crossing (x, y = 1024 at output) was inspected at 100%: no seam.
The FP16-cast files gave a byte-identical PNG for coast256 (the runner loads weights as float16 either way).

Not checked: the tiled output against an untiled run on large images (tile_pad 10 is the maker's choice; RRDBNet's
receptive field is larger, so tiles can differ slightly from a whole-image pass, as in the maker's own tile mode).

Verdict: pass (cosine >= 0.999 on both inputs).
