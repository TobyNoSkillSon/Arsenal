+++
hypothesis = "Qwen-Image-2.1 can serve in FP16 like the other native families."
change = "Ran each component at FP16 and BF16 on the same fixed inputs (Workers/QwenImage qwen-image-lab components; M5 Max, 21 Oct 2026, machine in use) and recorded activation ranges."
measurement = "Prompt encoder: FP16 finite, last-layer max |h| 711 (BF16 720). Transformer at 1024 x 1024: FP16 finite, pre-clip residual max 9,272 (prefix) and 651 (target); velocity vs BF16 relative L2 0.0075-0.012. VAE decoder: BF16 activations reach 264,192 after the fourth up block, past FP16's 65,504, and the FP16 decode is NaN."
verdict = "rejected"
numbers = { vae_decoder_max_abs_bf16 = 264192, fp16_max = 65504, transformer_prefix_residual_max = 9272, encoder_last_layer_max = 720 }
links = ["../README.md"]
+++

FP16 fails in the VAE decoder: its third and fourth up blocks hold values far beyond the FP16 range (5,664 then 264,192 in BF16 on a fixed N(0, 0.64) latent), so an FP16 decode overflows to NaN. The prompt encoder and transformer stay finite in FP16, but the model serves one half format, so it stays BF16 (`[precision] dtype = "bf16"`). M1/M2 Macs run BF16 without native BF16 arithmetic, more slowly; only M5 Max was tested.

Raw captures (fixed inputs, per-component outputs, comparison JSON) are local under `lab/qwen-image/components/`.
