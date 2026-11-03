+++
hypothesis = "The qualified rotate-half RoPE can be delivered in the model folder without changing Qwen3-TTS outputs."
change = "Move the shader body from compiled Swift to kernels/qwen-fused-rope.metal; use shared ArsenalKernels preflight, ExactGate and stock fallback."
measurement = "Real bf16 VoiceDesign service cycles in Fast and Exact; M5 Max, gpulock, local HF import, no controlled benchmark."
verdict = "kept"
numbers = { exact_check_outputs = 24, exact_check_elements = 27648, differing_elements = 0, functional_modes = 2, fast_cycle_s = 15.041, exact_cycle_s = 2.847, network_weight_bytes = 0, worker_mb_after_unload = 0 }
links = ["kernels/manifest.toml", "kernels/qwen-fused-rope.metal", "docs/evidence/mb-v2-voices/qwen3-tts-voicedesign-bf16-fast.json", "docs/evidence/mb-v2-voices/qwen3-tts-voicedesign-bf16-exact.json"]
+++

The active M5 exact replacement is `qwen.rotate-half-rope`; Fast inherits this exact kernel. Fixed checks cover bf16, fp16 and fp32, batch 1/2, lengths 1/3, unequal query/key heads and broadcast rotary tables. Compilation is preflighted through Metal before MLX sees the instantiation; compile or equality failure serves the stock expression and reports stock with its reason. The source digest, model/dtype, check revision, OS and GPU key persisted verdicts.

Both full speech WAVs have identical SHA256 `631939a3ad3563951ccffe2dc0adb9972d0f9a113e464df52f36d5a0f8afba23`. Each cycle exercised description → saved voice → speak → list voices → unload; residency returned to zero. These are functional figures with different cold/warm cache states, not a speed comparison, ranking or Measured badge. Other qualified Vireo components remain on in their modes; this slice externalizes RoPE, not every existing sampler/depth shader.
