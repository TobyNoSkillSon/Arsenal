+++
hypothesis = "Retain gated selective scan/causal convolution with stock fallback and Exact opt-out."
change = "Ported accepted paths and preserved the source rejection decisions."
measurement = "Historical Vireo source evidence on M5 Max, September 2026; no new Arsenal measurement."
verdict = "kept"
numbers = { historical_input_s = 22.8, historical_stock_rtf = 4.3, historical_metal_rtf = 0.21, historical_metal_vs_stock_snr_db = 80, historical_first_gate_s = 3.3, historical_cached_gate_s = 0.05 }
links = ["Workers/Sound/PORTED-FROM.md", "docs/evidence/mb-v2-sound/vireo-history-excerpts.txt"]
+++

Kept in Vireo: fused Metal selective scan and causal convolution; on a 22.8-second input RTF 4.3→0.21, roughly 80 dB SNR against stock. Kernel unit bounds are 1e-3 relative and the whole-model bound is ≥40 dB SNR. A forced non-finite output was rerun on stock, bit-identical, with a sticky failed verdict. Three load/use/unload cycles returned MLX active memory to zero.

Arsenal ports the gate and runtime fallback, and disables these inexact kernels in Exact mode. The FP32 vendor source is cast once to FP16, widened losslessly for model arithmetic at load. This history does not prove a Arsenal inference, a quantized Arsenal cycle or performance on another chip.

Source: Vireo `lab/perf/NOTES.md`, read from the local research working tree with HEAD `9366da5e14dcafe846fc41565c94f4f145eaf7cc` (the lab note is not Git-tracked); excerpts are retained in Arsenal evidence. No historical result is placed in Arsenal `results/` or used for its badge/ranking.
