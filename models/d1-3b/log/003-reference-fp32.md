+++
hypothesis = "The vendor System One code on torch MPS bf16 is a faithful reference for d1-3B."
change = "Compared it with the same pinned vendor code in float32 on the CPU; rebased the gate reference to fp32 CPU."
measurement = "M5 Max, 8 Oct 2026, torch 2.12.0: MPS bf16 vs fp32 CPU max |dp| 0.726 and 23/24 argmax on the gate set; MPS causal scaled_dot_product_attention in bf16/fp16 has 0.41-0.46 relative error at 100-400 tokens."
verdict = "kept"
numbers = { mps_gate_max_dp = 0.726, mps_gate_argmax = 23, cpu_bf16_gate_max_dp = 0.0172, cpu_bf16_gate_argmax = 24, questions = 24, mps_causal_sdpa_rel_err_400 = 0.408, mps_full_sdpa_rel_err_400 = 0.0034 }
links = ["gate/reference-check.json", "gate/parity.py", "gate/gate.toml"]
+++
# The torch MPS reference was wrong

The first runner (log/001) ran LiquidAI's code on torch MPS in bf16, and its gate compared that runner with the same code on MPS, so a kernel error on both sides passed with 0.0. Run in float32 on the CPU, the same pinned vendor code disagrees with it: max |dp| 0.726 and 23/24 argmax on the 7 gate cases (11 requests, 24 questions), 91/96 on the probe ads and 67/72 on the requirement sections. The worst case is the red 64×64 picture, which MPS called blue (0.70) while fp32 CPU, CPU bf16 and MLX all say red (0.98).

Cause, isolated with random tensors against float64: torch 2.12.0's MPS `scaled_dot_product_attention` with `is_causal=True` in bf16 or fp16 has 0.41-0.46 relative error from about 100 tokens up (16 tokens: 0.003); non-causal bf16 is 0.003, fp32 is exact, matmul and depthwise conv1d on MPS match the CPU. The vendor code uses causal SDPA for the trunk and for every plain chain, so every state of 100+ tokens was affected.

Reference now: the pinned vendor `modeling_d1` from the checkpoint folder, float32, CPU (`gate/parity.py`). The bf16 floor is the same vendor code in bf16 on the CPU (correct kernels): 0.0172 max |dp|, 24/24 on the gate; 0.030 / 95 of 96 on the ads, 0.024 / 72 of 72 on the requirements. Any bf16 runtime should sit near that floor; the MPS runner sat 25-40× above it.

Effect on earlier results: the Polish probe in log/002 was run on the MPS runner. Against fp32 its d1 counts were off by up to three items per question (Polish Noul 27/32 there, 25/32 in fp32). `gate/reference-check.json` holds the per-question counts for fp32, the MLX runner, the MPS runner and Laya.
