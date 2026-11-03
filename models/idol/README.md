# IDOL

Turns one picture of an object into a 3D Gaussian splat: a standard 3D Gaussian Splatting PLY (positions, degree-0
colour, opacity, scales, rotations; y-up) with 32,768 to 1,048,576 Gaussians, plus a rendered preview and the camera
IDOL estimates for the picture (yaw, pitch, distance, field of view). The operation declares the host BiRefNet
remove_background pre-step and selects its foreground PNG; the adapter itself refuses an image without transparency.

`image_to_3d` takes `image`, `mode` (synthetic: renders and generated images, crop 1.2; real: photos, crop 1.7),
`crop_margin`, `num_gaussians` (default 262,144), `steps` (50), `cfg_strength` (3), `rescale_t` (3) and `seed` (42),
the defaults of IDOL's own sample.py. Outputs: `splat` (application/x-ply) and `preview` (PNG, 768 px, the splat seen
from the estimated direction and framed to fill the image), `gaussians` and `camera`. The Make 3D window shows the
preview and the Gaussian count; the Mac's own viewers do not open splat PLYs, any 3DGS viewer does.

## Pipeline and precision

The prepared 1024 px cutout feeds DINOv3 ViT-H+/16 (1 + 4 + 4,096 tokens of 1,280) and the FLUX.2 VAE encoder (a sample
of its posterior, 2×2-patchified to 4,096 tokens of 128, five zero rows in front). The denoiser, a rectified-flow
transformer over 8,192 latent tokens and one camera token, runs 50 guided Euler steps. The octree decoder places
num_gaussians / 32 anchors by descending eight levels with systematic resampling; the Gaussian decoder turns each into 32
Gaussians. Everything runs in MLX in the Geometry worker (`Workers/Geometry/Sources/MLXIdol`); the octree bookkeeping,
PLY writer and preview rasterizer are host code.

- IDOL's own modules: FP16 weights, cast once at Get from the FP32 originals. idol-sdk runs its transformer blocks in
  FP16 and keeps the small embedders, adaLN, RePo and in/out projections in FP32; here those weights are FP16 too, with
  their activations in float32 as in idol-sdk (parity within BF16 tolerance, results/parity-2026-11-02.md). The
  denoiser's fixed latent positions stay FP32 (98 KB): their embedding resolves 2^16 cycles per unit and is computed
  once on the host.
- DINOv3 ViT-H+: BF16, as idol-sdk runs it. Not FP16: with the trained weights its residual stream reaches about
  137,000 from layer 22, past FP16's 65,504 (measured on a gallery picture, 2 Nov).
- FLUX.2 VAE encoder: FP16. In MLX it is closer to FP32 in FP16 (0.4 % relative RMS) than in BF16 (3.1 %); idol-sdk's
  own BF16 run is 0.7 % from FP32.
- int8 and int4 are not offered.

## Sources and licences

- `speridlabs/idol@2731c229`: config.json, denoiser, decoder, decoder_gs (FP32 safetensors, Apache-2.0, ungated).
  IDOL's bundled BiRefNet is not downloaded: Arsenal's own BiRefNet removes the background.
- `facebook/dinov3-vith16plus-pretrain-lvd1689m@c807c9ee`: gated, Meta's DINOv3 License. Each user accepts Meta's terms
  on its Hugging Face page (Meta approves each request; one request covers the DINOv3 collection) and Arsenal downloads
  with the user's token, as for FLUX 3 Klein and TRELLIS.2's ViT-L. Its model.safetensors SHA-256 `3e1d4d18…1a4d`
  rebuilds the repository's Git LFS pointer id `fba5c041…` exactly. The licence text is LICENSE-DINOv3.md here and
  `Licenses/LICENSE-Meta-DINOv3.md` in the app (the same file as the ViT-L's).
- `black-forest-labs/FLUX.2-klein-4B@e7b7dc27` `vae/`: Apache-2.0, ungated. IDOL's config names FLUX.2-dev's VAE
  (non-commercial); with IDOL's own BF16 cast the two are bit-identical in all 251 tensors and the same in configuration,
  so klein is used (`results/vae-swap-2026-11-02.md`).
- Code: `speridlabs/idol-sdk@4702a7f0` (Apache-2.0), see Workers/Geometry/PORTED-FROM.md.

## Checks

Parity against idol-sdk (`results/parity-2026-11-02.md`): every module within BF16 tolerance on captured tensors, the
50-step sampler from idol-sdk's own noise and image tokens reproduces its final latent (cosine 0.99994), and the
trained end-to-end runs are in `results/end-to-end-2026-11-02.md`. Weight-free: IdolTests (host arithmetic, request
limits, Get's precision policy, the whole adapter at miniature size with random weights), gate/no-weights.json, and the
gated-flow rows in MetaGatedModelsTests.

## Usage notes

A seed matters: like idol-sdk itself, some seeds stretch the object along the camera's line of sight (on the armchair,
seed 42 in MLX gave a chair about twice as deep as wide; seed 7 and idol-sdk's own seed-42 run did not). The preview
looks right either way, because it is seen from the picture's direction; turn the splat in a viewer, and try another
seed if it is too deep. A seed does not reproduce idol-sdk's result for the same seed: the random numbers come from MLX.
