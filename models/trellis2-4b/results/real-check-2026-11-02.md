# Real check, 2 Nov 2026

Requests through Arsenal's real service on an isolated profile (`arsenal-testserver --real`, private QA catalog from
the committed folder, Hugging Face client on loopback so Get could only import local copies). M5 Max, Mac in use, on
AC. No speed or memory figures are claimed.

- Get: `microsoft/TRELLIS.2-4B`, the TRELLIS-image-large structure decoder and the DINOv3 encoder (pinned mirror)
  imported from the local Hugging Face cache, FP16 (DINOv3 keeps its residual stream in FP32, as declared). BiRefNet,
  the background-removal step, came from a temporary copy of the installed app's original files.
- `image_to_3d` on a 1024 x 1280 photo of a man, three runs, all HTTP 200 with a GLB mesh and a 512 x 512 preview:
  - default settings, seed 0: 75,779 vertices, 108,268 faces; extents 0.316 x 1.006 x 0.568, matching Arsenal's
    22 Oct mesh (0.315 x 1.007 x 0.567); the preview shows the same frontal, textured figure. This is the golden.
  - default settings, seed 1, and resolution 512 with 20,000 faces, seed 1: the figure comes out turned about 45
    degrees, with fragments of the photo's crack line floating beside the face. Same code, so this is the model's
    sampling.
- Not checked: a maker reference (Microsoft's code needs CUDA kernels), resolution 1536, int8, mesh quality beyond
  the preview.
- DINOv3 now comes from Meta's own `facebook/dinov3-vitl16-pretrain-lvd1689m` at `ea8dc286` (re-pinned later the same
  day). Meta's repository is gated and the Hugging Face account on this Mac has not accepted its terms, so the pins
  were checked only against the local mirror copy: `model.safetensors` matches Meta's published LFS pointer, but the
  SHA-256s of the two small files, `config.json` and `preprocessor_config.json`, are the mirror's and are unverified
  against Meta's tree. If they differ, every TRELLIS.2 Get fails its integrity check; the first real Get with an
  accepted token confirms them.
