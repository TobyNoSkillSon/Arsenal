+++
hypothesis = "The vendor hybrid runner preserves System One semantics on Apple Silicon."
change = "Added pinned offline torch-MPS adapter and image inputs."
measurement = "Historical, superseded by log/003 and log/004 (torch MPS compared with itself; MPS causal attention is faulty in bf16). M5 Max, 8 Oct 2026, bf16: max absolute probability difference 0.0; 24/24 argmax agreement across Noul, Choice, Score, batch and image inputs."
verdict = "kept"
numbers = { max_abs_probability_difference = 0.0, argmax_agreement = 24, questions = 24 }
links = ["gate/gate.toml", "gate/parity-bf16.json"]
+++
# d1-3B reference runner

Historical: this torch MPS runner is removed. Its gate compared MPS with MPS; log/003 shows the MPS kernel was wrong, and log/004 describes the MLX runner that replaced it.

Native MLX supports the LFM2/LFM2-VL backbone, but Arsenal’s native decision engine supports Laya/Von only. Porting the decision tree and multimodal readout is a new numerical implementation, not a dispatcher addition. Python mlx-vlm also supports the backbone; neither supplies the vendor’s hybrid System One tree. Select the vendor’s torch MPS hybrid implementation for reference semantics, with reference parity passed. No claim that torch is faster than an unmeasured MLX port.

The deleted original inputs were replaced with a fresh, manually labeled 32-ad/24-section private snapshot before inference. The short product-path probe interleaves both models and restores Laya’s load/Keep Hot state; the inputs and labels were kept privately. See docs/notes/d1-3b.md for the method and optimization handoff. No controlled benchmark or energy run.
