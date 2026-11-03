+++
hypothesis = "MiniMax Music 3's 8.6B global language model can run in FP16 like the rest of the pipeline."
change = "Captured the maker's residual maxima on PyTorch MPS in BF16 (its trained dtype): the short parity prompt and the maker's own long end-to-end example prompt, plus 150 sampled frames."
measurement = "Per-layer max |residual| from transformers' Qwen3ForCausalLM output_hidden_states; 68-token prompt (reference capture) and 1,380-token prompt with 150 frames (lm_range_probe)."
verdict = "rejected"
numbers = { short_prompt_max = 55040, long_prompt_max = 65024, fp16_max = 65504, frame_pass_max = 984 }
links = ["README.md", "log/001-port.md"]
+++

FP16 was rejected for the global language model. From layer 17 on, a few channels of its residual stream carry massive activations: 55,040 on a 68-token prompt and 65,024 on the maker's own 1,380-token example (positions 60, 144, 358 and others, in the classifier-free row), 480 below FP16's limit, and growing with prompt length; prompts may reach 5,000 tokens. The maker's FP16 run of the short prompt stayed finite only because it is short. The model therefore computes in BF16, its trained dtype, as the precision rule allows when FP16 fails: Get still stores FP16 (exact for BF16 values in FP16's normal range), Load casts the language model back to BF16, and its hidden states are cast to FP16 where the local model takes them (frame passes stay under 1,000). The local model, fusion encoder, flow transformer and vocoder run FP16.

The maker's FP16 vocoder produced NaN in every block: its snake activation adds 1e-9 to the scale, which FP16 cannot hold, so a zero scale divides by zero. Arsenal computes the snake in float32 and returns FP16. Its parity floor uses the shared FP32-source policy (BF16 round-to-nearest versus BF16 truncation of the same weights) because the maker's FP16 vocoder output is not finite.
