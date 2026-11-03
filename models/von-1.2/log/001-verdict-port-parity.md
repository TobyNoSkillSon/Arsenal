+++
hypothesis = "Verdict's Von port, driven over ArsenalWire, keeps its pinned-SDK parity at fp32 and its 16-bit rule."
change = "Ported Von (network, .pt reader, VonStore fp16 conversion, tokenizer) into arsenal-worker-decisions with a convert op."
measurement = "Verdict Von SDK gate (von-sdk 1.2.2) at fp32/fp16/int8/int4 through the wire runner (M5 Max, gpulock)."
verdict = "kept"
numbers = { fp32_max_dp = 0.0, fp32_violations = 0, fp32_mlx_peak_mb = 2821.0, fp32_load_s = 0.82, fp16_max_dp = 0.0069, fp16_violations = 0, fp16_mlx_peak_mb = 1599.0, fp16_load_s = 0.26, int8_max_dp = 0.0707, int8_violations = 0, int8_mlx_peak_mb = 1250.0, int8_load_s = 0.23, int4_max_dp = 0.4182, int4_violations = 0, int4_mlx_peak_mb = 1078.0, int4_load_s = 0.24 }
links = ["gate/gate.toml", "gate/parity-von-sdk.json", "gate/smoke-suite-v1.json", "../../Workers/Decisions/PORTED-FROM.md"]
+++

## What was done
Same port as the Laya models (shared engine, `Workers/Decisions/PORTED-FROM.md`): Verdict's Von network (ModernBERT encoder +
option-marker scorer), its PyTorch-zip reader, fast tokenizer and calibration, behind the ArsenalWire runner. The vendor ships
fp32 only (`option_marker.pt`); the worker's `convert` op makes Verdict's stored fp16 (`cast:von-fp16-1`, VonStore: encoder
fp16 round-to-nearest-even, scorer head and final norm f32, ~1 s on the CPU) and int8/int4 views are quantized at load from it.

## Measurement
`scripts/mlock gpulock decisions -- python3 Workers/Decisions/scripts/parity.py von von-1.2` (M5 Max, 2 Oct 2026, the whole run a few
seconds). Verdict's pinned-SDK gate (`lab/tests/test_von_helper.py`, von-sdk 1.2.2, fixture `von-1.2-sdk.json.gz`):

| dtype | gate | violations | max |Δp| vs SDK | MLX peak |
|---|---|---|---|---|
| fp32 (lab only, not offered) | Δp ≤ 0.0001 in 3 modes | 0 | 0.0 | 2821.0 MB |
| fp16 | same argmax/side, |Δp| ≤ 0.1 | 0 | 0.0069 | 1599.0 MB |
| int8 | valid answers (lossy by design) | 0 | 0.0707 | 1250.0 MB |
| int4 | valid answers (lossy by design) | 0 | 0.4182 | 1078.0 MB |

The int4 drift is Verdict's known behaviour (its THEORY notes report up to ~1.0 |Δp| at 4 bits on near-ties); Arsenal offers int4
because Verdict does, and the gate here is validity, not agreement.

## Rejected
The repository also holds a `model.safetensors`; it is not the file von-sdk or Verdict's gate reads, so it is not pinned or used.
