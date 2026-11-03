+++
hypothesis = "Qwen-Image-2.1 runs natively in Swift/MLX from the vendor's diffusers implementation, with generate and multi-reference edit."
change = "New qwenimage family (Workers/QwenImage): Qwen3-VL prompt encoder, block-causal single-stream DiT with a prefix KV cache, RGBA VAE, flow-matching scheduler; schema-3 folder with generate and edit."
measurement = "Header inventory matched every loaded tensor (text encoder 748/750, transformer 297/297, VAE 226/238; the LM head, final norm and 12 temporal convolutions are never read). Real weights on an M5 Max (21 Oct 2026, Mac in use): 1024 x 1024, 40 steps, model loaded: BF16 69 s per image (62.4 s denoising, 1.54 s/step), peak 38.2 GB with every component resident; int8 73 s (1.80 s/step), peak 24.0 GB. Edits at int8: one reference 87 s, two references 106 s."
verdict = "kept"
numbers = { bf16_seconds = 69.0, bf16_step_seconds = 1.54, bf16_peak_gb = 38.2, int8_seconds = 73.2, int8_step_seconds = 1.80, int8_peak_gb = 24.0, edit_one_reference_int8_seconds = 86.9, edit_two_references_int8_seconds = 106.0 }
links = ["../README.md", "../gate/header-check.json", "002-format.md"]
+++

Static checks pass on small fixed tensors (`Workers/QwenImage/Tests`): scheduler sigmas against the diffusers formula, 3-axis RoPE, the segment-wise block-causal attention against one masked call, the prefix cache against a single-pass forward of the whole joint sequence, the VAE's AvgDown3D/DupUp3D channel orders, the Qwen2-VL patch layout and the M-RoPE walk; every tier runs both operations on random weights.

Images were inspected at full size: a neon sign rendering "QWEN IMAGE 2.1" exactly with wet-street reflections (BF16 and int8 nearly identical); an edit of that image to a blue "ARSENAL" sign that kept the street, camera and lighting; a two-reference composition placing a generated apple on that street; a transparent RGBA apple (alpha kept; a few faint specks near its edge).

int8 component check on fixed inputs against BF16: prompt encoder relative L2 0.042, cosine 0.99913; transformer velocity at t = 1000/500/100 relative L2 0.013/0.012/0.017, cosine ≥ 0.99986: passes the 0.05/0.999 policy. int4: encoder 0.30/0.958, transformer 0.10-0.13/0.992-0.995: fails, offered with the quality warning.

Timings are quick checks on a busy Mac (other apps held 57-76 GB), not controlled measurements. Peak figures are MLX peaks for one 1024 x 1024 image with the prompt encoder, transformer and VAE all resident, as the worker serves them. No independent reference implementation was run; the diffusers pipeline was the specification.
