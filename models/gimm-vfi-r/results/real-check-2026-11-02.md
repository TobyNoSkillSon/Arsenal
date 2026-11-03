# Real check, 2 Nov 2026

One run of the shipped restoration worker (`Workers/.build/helpers/arsenal-worker-restoration`, debug build of this
branch) driven over Arsenal's wire protocol on an isolated profile (ARSENAL_SUPPORT_DIR / ARSENAL_CACHE_DIR in a
temporary folder), under the GPU lock. M5 Max, Mac in use. No speed figures are claimed.

- `convert` (Get's recipe `cast:restoration-gimm-vfi-r-v1`) in a fresh worker on an APFS clone of the pinned
  `gimmvfi_r_arb.pt`: one FP16 `model.safetensors`, 39,619,636 bytes, 387 tensors.
- Load at FP16 with `architecture = "gimm_vfi_r"`, then `interpolate_video` on the golden's two 281x240 frames with
  `t = [0.25, 0.5, 0.75, 0, 1]`: five PNGs at 281x240. The three in-between frames match the makers' frames at
  62.7, 62.6 and 61.8 dB PSNR (mean difference 0.03 of an 8-bit level; at most 3, 3 and 9 levels on a handful of
  pixels); `t = 0` and `t = 1` return the inputs exactly. Looked at the t = 0.5 frame: the dog sits between the two
  poses, the ball between its positions, no seams.
- `interpolate_video` on a 4-frame, 4 fps, 844x720 H.264 clip of the makers' demo frames with `factor = 2`: an
  844x720 MP4 of 8 frames at 8 fps. Looked at its first interpolated frame at full size: the leaping dog is mid-motion
  and sharp on the head and ball; the fast front paw is slightly smeared.
- `t = [1.5]` is refused as invalid_request; Unload exits 0. Worker MLX peak during the run: 6.2 GB (the 844x720 clip).
- Not checked: the app-level Get/Load through `arsenal-testserver --real` (its catalog reads committed folders from a
  fixed model list; this model was not yet committed), frames above a megapixel, cancellation mid-pair.

Rerun after the review fixes (same build tree, same profile): Get, Load and the frames match the first run (the frames
at 62.8, 62.7 and 61.8 dB against the makers'); Get at BF16 is refused (`unsupported_recipe`).
A 6-frame 844x720 clip at 30000/1001 fps with `factor = 2`: 12 frames out. The frames at source times are the source
frames (36 to 42 dB against them, H.264 re-encoding only); each in-between frame comes from its pair (about 18 dB from
both neighbours, as expected for this large motion), with ghosting on the fast front leg.

