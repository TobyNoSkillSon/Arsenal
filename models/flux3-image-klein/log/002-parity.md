+++
hypothesis = "Each Swift component reproduces mflux on the same weights and fixed inputs within BF16 rounding."
change = "Tiny random-initialised FLUX 3 checkpoint in the vendor layout (text encoder 32+2 layers of width 64, transformer 2+2 blocks of width 64, VAE channels 32), saved once and loaded by mflux's own loader and by the Swift loader; mflux outputs captured once (lab/flux3/scripts/capture_tiny.py) into Workers/Flux/Tests/Flux3KleinTests/Fixtures/tiny."
measurement = "scripts/test.sh --heavy --filter Flux3Parity on an M5 Max, 1 Nov 2026, Mac in use: 8 tests, 37 checks, all within bounds. Exact: seeded noise, timestep sinusoid, time vectors, modulations, text RoPE, embeddings, Euler step, patchify. Transformer blocks and velocity within one BF16 ulp (0.0078 at magnitude 2)."
verdict = "kept"
numbers = { velocity_max_abs_diff = 0.00781, text_encoder_max_abs_diff = 0.188, vae_decode_max_abs_diff = 0.0977, pixels_max_levels = 9, checks = 37 }
links = ["001-port.md", "../gate/gate.toml"]
+++

Bounds are BF16 floors, not tuned to pass: where the port's arithmetic differs from mflux's, the bound is set by how far mflux itself moves under the same kind of change (lab/flux3/scripts/numerics_probe.py). On this 32-layer random text encoder, swapping only mflux's float32 attention for BF16 moves its output by 0.156, and only its float32 RMSNorm for the fused one by 0.219; the port differs by 0.188. mflux's own two VAE decoders (row strips with float32 GroupNorm, its default, and full image) differ by 0.111 on the decode input and by 0.078 (about 10 levels) on the pipeline image; the port differs by 0.098 and 0.078 (9 levels).

| Component | Shape | max abs diff | ref max abs | dtype | bound | result |
|---|---|---|---|---|---|---|
| noise seed 7 (64x64) | 1x16x128 | 0 | 3.61 | float32 | 0 | pass |
| noise seed 42 (48x32) | 1x6x128 | 0 | 3.45 | float32 | 0 | pass |
| timestep sinusoid | 1x256 | 0 | 1 | bfloat16 | 0 | pass |
| timestep sinusoid t=0 | 1x256 | 0 | 1 | bfloat16 | 0 | pass |
| time vector image | 1x64 | 0 | 0.383 | bfloat16 | 0.004 | pass |
| time vector context | 1x64 | 0 | 0.398 | bfloat16 | 0.004 | pass |
| early modulation image | 1x1x192 | 0 | 0.126 | bfloat16 | 0.002 | pass |
| early modulation image_cond | 1x1x192 | 0 | 0.11 | bfloat16 | 0.002 | pass |
| early modulation txt | 1x1x192 | 0 | 0.169 | bfloat16 | 0.002 | pass |
| single modulation image | 1x1x192 | 0 | 0.16 | bfloat16 | 0.002 | pass |
| single modulation image_cond | 1x1x192 | 0 | 0.128 | bfloat16 | 0.002 | pass |
| single modulation txt | 1x1x192 | 0 | 0.16 | bfloat16 | 0.002 | pass |
| text RoPE cos | 1x80x16 | 0 | 1 | bfloat16 | 0 | pass |
| text RoPE sin | 1x80x16 | 0 | 1 | bfloat16 | 0 | pass |
| text encoder layer 0 | 1x80x64 | 0.0312 | 4.03 | bfloat16 | 0.04 | pass |
| text encoder taps 4..32 | 1x80x512 | 0.188 | 5.53 | bfloat16 | 0.25 | pass |
| embed image | 1x16x64 | 0 | 1.7 | bfloat16 | 0.02 | pass |
| embed references | 1x16x64 | 0 | 1.93 | bfloat16 | 0.02 | pass |
| embed text | 1x80x64 | 0 | 2.17 | bfloat16 | 0.02 | pass |
| mode block 0 image | 1x16x64 | 0.000977 | 1.7 | bfloat16 | 0.02 | pass |
| mode block 0 image_cond | 1x16x64 | 0.00391 | 1.93 | bfloat16 | 0.02 | pass |
| mode block 0 txt | 1x80x64 | 0.00781 | 2.14 | bfloat16 | 0.02 | pass |
| mode block 1 image | 1x16x64 | 0.00391 | 1.69 | bfloat16 | 0.02 | pass |
| mode block 1 image_cond | 1x16x64 | 0.00391 | 1.93 | bfloat16 | 0.02 | pass |
| mode block 1 txt | 1x80x64 | 0.00781 | 2.14 | bfloat16 | 0.02 | pass |
| single block 0 | 1x112x64 | 0.00781 | 2.14 | bfloat16 | 0.02 | pass |
| single block 1 | 1x112x64 | 0.00781 | 2.14 | bfloat16 | 0.02 | pass |
| velocity, edit (2 references) | 1x16x128 | 0.00781 | 1.95 | bfloat16 | 0.03 | pass |
| velocity, text to image | 1x16x128 | 0.00781 | 1.95 | bfloat16 | 0.03 | pass |
| denoise step (Euler jump) | 1x16x128 | 0 | 3.6 | float32 | 0 | pass |
| patchify + pack | 2x6x128 | 0 | 1.54e+03 | float32 | 0 | pass |
| VAE encode → tokens | 1x16x128 | 0.0259 | 1.74 | bfloat16 | 0.05 | pass |
| VAE tokens → decode | 1x64x64x3 | 0.0977 | 2.2 | bfloat16 | 0.12 | pass |
| pipeline reference tokens | 1x16x128 | 0.0215 | 1.64 | bfloat16 | 0.05 | pass |
| pipeline latents after 4 steps | 1x16x128 | 0.00806 | 3.72 | float32 | 0.03 | pass |
| pipeline decode | 1x64x64x3 | 0.0781 | 2.05 | bfloat16 | 0.12 | pass |
| pipeline pixels (8-bit levels) | 64x64x3 | 9 | 255 | int32 | 12 | pass |

Default-suite checks against the same capture (no GPU): the 16-step and five other schedules within one float32 ulp, text/image/reference RoPE ids exact, RoPE tables within 1.2e-7 including the release's 128-wide head at text position 4095 and a 1024² grid, prompt positions exact, Pillow 13 LANCZOS bit-exact on four resizes, numpy reflect/symmetric padding, and mflux's canvas and crop sizes.

End to end on the release's file layout (lab/flux3/phase-b.sh in dry-run mode, 1 Nov 2026): the tiny random weights split into the release's shard names, with a full-size embedding so the real Qwen3-VL tokenizer is used. mflux and the Swift lab tool ran the five Phase B cases at 16 steps from the same seeds: three 512 x 512 text-to-image prompts, an edit from a 512 x 512 reference, and an edit from a 500 x 340 reference (LANCZOS fit, reflect padding to 512 x 352, crop back). PSNR 43.95 to 44.27 dB, at most 19 levels on one pixel, mean 1.1 levels. Tokenizer ids matched mflux on all 7 prompts (Unicode, a one-character prompt, and a 3000-word prompt truncated at 4096). The debug flux worker served load, generate (2 images), edit and a refused guidance request over the wire from the same snapshot.

What this does not show: random tiny weights do not exercise the release's activation ranges, its 4096-token prompts or 1024² images. FLUX.2's port passed tiny checks and still differed at full size (27.5 dB). Real-weight parity and image review are the gate's remaining items (lab/flux3/phase-b.sh).
