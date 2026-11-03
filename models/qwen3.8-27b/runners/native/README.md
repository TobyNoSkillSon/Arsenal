# Native text runner

`Workers/Chat` → `arsenal-worker-chat`, ArsenalWire v1, existing family pins unchanged.
MLXLLM's Qwen35Model loads the vendor HF safetensors directly; no intermediate float conversion.
The text-only sanitizer removes vision/MTP modules. Linear attention keeps convolution/recurrent state,
full attention keeps request-local KV caches. Context checks bound total prompt+output before allocation.

Streamed text, system/user/assistant roles, temperature/top-p/seed/max_tokens/stop and token counts are supported.
Tools, tool-call history, strict schemas, JSON mode, images and videos are not advertised or silently ignored.
Conversion is uncalibrated MLX affine group 64 from bf16. No Fast/Exact or MTP speculation is offered.
