+++
hypothesis = "Songs with four or more lyric lines fail with NaN latents because the FP16 lyric encoder overflows."
change = "The lyric encoder computes in BF16 (upstream's precision; FP16 overflows) and returns FP16; everything else stays FP16."
measurement = "Real FP16 weights, conditioning and the 8-step diffusion only (no planner codes, no VAE decode), 30 s, three seeds per input: before, 4 and 8 lines gave NaN encoder output and NaN latents; after, every input is finite."
verdict = "kept"
numbers = { last_layer_residual_instrumental = 47678, last_layer_residual_2_lines = 37552, last_layer_residual_4_lines = 110900, last_layer_residual_8_lines = 163782, last_layer_residual_24_lines = 469171, fp16_max = 65504, encoder_output_max = 58.6, dit_residual_max_seen = 22784 }
links = ["../../../Workers/Music/Sources/MusicWorker/Vendor/ACE/Encoders.swift", "../../../Workers/Music/Tests/MusicTests/LyricEncoderPrecisionTests.swift"]
+++

On 31 Oct 2026 a website task asked for songs with lyrics over HTTP; 11 of 13 requests failed with `generation failed: the diffusion produced NaN or infinite latents`. The two that passed were an instrumental and a two-line lyric; every request with four or more lines failed, at any seed, duration 30 or 40 s, on three fresh worker loads. The 1 Nov test night passed because its only request was `{"prompt": "speech"}`, which encodes `[Instrumental]`.

Measured with the stored FP16 snapshot on an M5 Max (1 Nov 2026): the lyric encoder's last layer (residual stream, before the final RMS norm) reaches 47,678 on `[Instrumental]` and 37,552 on two lines, under FP16's 65,504, and 110,900 on four lines, 163,782 on eight and 469,171 on 24 when computed in FP32. In FP16 those became `inf`, the final norm turned them into NaN and the DiT's cross-attention spread NaN through every layer. Upstream runs BF16, whose range covers these values, so the encoder now computes in BF16 (Arsenal never runs FP32; a first fix in FP32 was replaced the same day). The normed output stays at 58.6, so it returns to FP16 for the DiT. The DiT's own residual stream peaked at 22,784 in these runs, inside FP16 with about 2.9x headroom; it was checked at t = 1 only, without planner codes.

Not checked here: the planner's codes path (the default `thinking`), the VAE decode, songs longer than 30 s, and the int8/int4 tiers (their quantized matmuls promote to FP32 inputs the same way, by MLX's type rules, but were not run).
