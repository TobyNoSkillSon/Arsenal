+++
hypothesis = "FLUX 3 Image Klein runs natively in Swift/MLX on the bundled flux worker, ported from mflux and served as an add-on folder because its licence is not open."
change = "New Flux3Klein library in the flux family (models/flux2-klein-4b/runners/swift/Sources/Flux3Klein): Qwen3-VL-4B tap encoder, per-mode and single-stream DiT with flow-map time conditioning, host RoPE and schedule, FLUX.2 VAE encoder (decoder reused), Pillow-exact LANCZOS reference preparation, 16-step pipeline. The flux worker loads it when a folder's architecture is flux3-image-klein and serves generate and edit. This schema-3 folder pins the BF16 diffusers files at 337337621b8526e120f2275d250242f4a6bedd75."
measurement = "Component parity against mflux 9669af51 on a tiny random-initialised checkpoint in the vendor layout, M5 Max, 1 Nov 2026: every check within its BF16 bound (log/002-parity.md). Real weights were not available (gated repository, licence not accepted on this Mac): no real-weight parity, no generation, no memory or speed figure."
verdict = "inconclusive"
numbers = "No real-weight measurement yet; the parity table is in 002-parity.md."
links = ["../README.md", "002-parity.md", "../gate/gate.toml"]
+++

Reference: mflux (MIT) at commit 9669af51, `src/mflux/models/flux3` plus its shared Qwen3-VL text tower and FLUX.2 VAE, run once in a sealed lab environment. Python only captured references; nothing serves the model in Python.

Source choice: the diffusers layout of the release (transformer/, text_encoder/, vae/), as mflux reads it. The BFL-format duplicates (model.safetensors, ae.safetensors) and the FP8/NVFP4 variants are not downloaded. The text encoder files are byte-identical to Qwen/Qwen3-VL-4B-Instruct at ebb281ec (same git blob ids for all 14 files), which pinned the SHA-256 of the small, non-LFS files without access to the gated copies. The transformer and VAE configs are not pinned: the runner reads every dimension from the tensor shapes. The release VAE is float32; the loader casts it to BF16 tensor by tensor. The transformer and text encoder are BF16 as released.

Precision: BF16 only. mflux documents feed-forward outputs near 3e5 in BF16 runs, past FP16's range, so no FP16 trial was run on random weights (it could not show that overflow). Where the port differs from mflux's arithmetic, it follows the BF16 choices of the vendor's PyTorch stack: per-head RMSNorm and RoPE in BF16 (mflux casts q and k to float32), fused RMSNorm and BF16 attention in the text encoder (mflux uses float32 for both), BF16 GroupNorm in the VAE (mflux normalizes row strips in float32), and a BF16 VAE encoder stream (mflux's float32 image input promotes it). The Euler state stays float32 as in mflux (a [1, tokens, 128] tensor).

Not ported: the prompt upsampler (black-forest-labs/flux-3-image-klein-pup, a separate 9B model the maker says to always use), the NSFW classifier of the reference code, Klein Base (undistilled, refused at load), LoRA and quantized tiers.
