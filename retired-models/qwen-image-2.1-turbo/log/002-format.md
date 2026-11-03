+++
hypothesis = "Qwen-Image-2.1-Turbo can serve in FP16 like the other native families."
change = "Decoded the same fixed latent with this checkpoint's VAE at BF16 and FP16 (Workers/QwenImage qwen-image-lab components; M5 Max, 21 Oct 2026)."
measurement = "BF16 decoder activations reach 5,664 after the third up block and 264,192 after the fourth, past FP16's 65,504; the FP16 decode is NaN. Turbo's VAE is the base model's FP32 VAE rounded to BF16, tensor for tensor (238 of 238), so the base model's FP16 evidence (prompt encoder and transformer finite) carries over."
verdict = "rejected"
numbers = { vae_decoder_max_abs_bf16 = 264192, fp16_max = 65504 }
links = ["../README.md", "../../qwen-image-2.1/log/002-format.md"]
+++

FP16 overflows in the VAE decoder, so the checkpoint serves BF16 (`[precision] dtype = "bf16"`). M1/M2 Macs run BF16 without native BF16 arithmetic, more slowly; only M5 Max was tested.
