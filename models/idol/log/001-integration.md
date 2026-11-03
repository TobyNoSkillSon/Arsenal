+++
hypothesis = "idol-sdk's modules, DINOv3 ViT-H+ and the FLUX.2 VAE encoder run in MLX within BF16 tolerance of the reference, with the Apache klein VAE in place of FLUX.2-dev's."
change = "Port IDOL to the Geometry family (MLXIdol) and expose image_to_3d with a splat PLY and a rendered preview."
measurement = "Per-module parity on captured CPU tensors, trained-weight DINOv3 against FP32, the 50-step sampler from idol-sdk's own noise, four trained end-to-end runs and idol-sdk seed comparison on MPS."
verdict = "kept"
links = ["../README.md", "../results/parity-2026-11-02.md", "../results/end-to-end-2026-11-02.md", "../results/vae-swap-2026-11-02.md"]
numbers = { denoiser_velocity_rel_rms = 0.0015, sampler_latent_cosine = 0.99994, vae_tensors_identical = 251, end_to_end_runs = 4 }
+++

# 001 Integration (2 Nov 2026)

- Ported idol-sdk to MLX in the Geometry family (MLXIdol): denoiser, octree and Gaussian decoders, sampler, DINOv3
  ViT-H+, preparation, PLY and a preview rasterizer; the VAE encoder is Flux3Klein's (a `moments` accessor added).
- Get recipe `cast:geometry-idol-v1`: IDOL FP32 → FP16 (latent positions FP32), DINOv3 FP32 → BF16, VAE → FP16.
- DINOv3 ViT-H+ pin: Meta hides the LFS hash from accounts without access; it was read once this Mac's access request was
  approved (2 Nov, the request covers the whole DINOv3 collection) and checked against the public pointer id.
- First trained run found a chair twice as deep as wide; traced stage by stage against idol-sdk's own MPS run to the
  sampler's noise draw (IDOL's own seed variation), not the port.
- Still to do at integration: rebuild and re-sign the catalog with this folder; a reviewed install.
