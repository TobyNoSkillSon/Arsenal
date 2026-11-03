+++
hypothesis = "The existing family MLX Swift pin supports the vendor Qwen3.8 hybrid text architecture without a Python fallback."
change = "Selected qwen3_5 from the pinned MLXLLM registry and inspected its convolution, gated-delta recurrence, full attention, RoPE and sanitizer."
measurement = "Source inspection and vendor config match; real smoke is the next check."
verdict = "inconclusive"
numbers = { layers = 64, linear_layers = 48, full_attention_layers = 16, checkpoint_context = 262144, default_context = 8192, model_billion = 27 }
links = ["runners/native/README.md", "../../Workers/Chat/PORTED-FROM.md"]
+++

Pinned mlx-swift-lm `bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57` registers `qwen3_5` as Qwen35Model.
Its nested text configuration decodes the vendor dimensions and RoPE parameters; every fourth layer is full
attention, the other 48 use GatedDeltaNet with convolution state plus a recurrent MambaCache. Full-attention
layers use KVCacheSimple. The model sanitizer maps `model.language_model` keys, moves PyTorch convolution
axes, shifts the vendor RMSNorm weights when unsanitized convolution/MTP weights are present, and discards
vision/MTP weights for text-only inference. The vendor's 64 layer_types agree with the pinned interval rule.

Full-attention output gating is sigmoid in both this native implementation and the HF Transformers
Qwen3_5Attention reference; the config's swish field is not a reason to substitute a different operation.
The native runtime reads generation_config.json's EOS IDs. Its tokenizer uses the vendor Jinja file with
`enable_thinking=false`; this is explicit non-thinking text output, not hidden reasoning returned as prose.

Keep native Swift: no dependency pin changes and no Python environment are needed. The full checkpoint still
has to be hash-verified, loaded and answered through the real Arsenal supervisor; metadata alone is not proof.
