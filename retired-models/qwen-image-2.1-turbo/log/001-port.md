+++
hypothesis = "Qwen-Image-2.1-Turbo runs on the qwenimage worker unchanged, with its saved 8-step sigma schedule."
change = "Folder for the Turbo checkpoint on the shared Workers/QwenImage runner; the schedule comes from model_index.json sample_sigmas."
measurement = "Prompt encoder identical to Qwen-Image-2.1's, tensor for tensor (750 of 750); VAE equals the base FP32 VAE rounded to BF16 (238 of 238). Real weights on an M5 Max (21 Oct 2026, Mac in use, release build): 1024 x 1024, 8 steps, model loaded: BF16 21 s per image (encoder freed before denoising, peak 21.9 GB); int8 17 s with every component resident, peak 24.0 GB. A 768 x 768 edit at int8 took 15 s."
verdict = "kept"
numbers = { bf16_seconds = 21.2, bf16_peak_gb_encoder_freed = 21.9, int8_seconds = 17.2, int8_peak_gb = 24.0, edit_768_int8_seconds = 15.1, steps = 8 }
links = ["../README.md", "002-format.md", "../../qwen-image-2.1/log/001-port.md"]
+++

Images inspected at full size: the neon-sign prompt at BF16 and int8 rendered "QWEN" and "IMAGE 2.1" correctly on a rainy storefront, the two tiers nearly identical; an int8 edit of a dawn hero background added a lighthouse with its beam on the right hill and kept the sky, horizon and empty space.

With every component resident, BF16 peaks like the base model (38.2 GB measured there, same graph); this BF16 run freed the prompt encoder before denoising, so its 21.9 GB peak is lower than the worker's. The int8 component checks of the base model apply to Turbo's prompt encoder (identical weights); Turbo's transformer was not separately compared. Timings are quick checks on a busy Mac; GPU contention from other apps varied between runs.
