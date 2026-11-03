+++
hypothesis = "BiRefNet's int8 and int4 tiers are worth keeping: faster or meaningfully smaller than FP16 at the same masks."
change = "model.toml drops [derive] int8 and int4; BiRefNet ships FP16 only. The int8/int4 table results are removed; the 16 Oct intake and its quality binding stay as evidence for the FP16 figures."
measurement = "Quick check, M5 Max, 22 Oct 2026, Mac in use (lab/night-2026-10-22/peak.py): per precision one ordinary Load and two requests through the installed app's API; peak = summed macOS phys_footprint_peak of the Arsenal worker processes (MiB, shown here in GB of 10^9 bytes); the timed request is the second, one sample. Not the controlled ABBA protocol. Rows: lab/night-2026-10-22/peak-results.jsonl; summary: lab/night-2026-10-22/tiers.md."
verdict = "rejected"
numbers = { fp16_seconds = 0.478, int8_seconds = 0.473, int4_seconds = 0.468, fp16_peak_gb = 5.69, int8_peak_gb = 5.26, int4_peak_gb = 5.17 }
links = ["../README.md", "002-intake.md"]
+++

# Int8 and int4 withdrawn

22 October 2026. The owner asked to decide per model whether int8 and int4 are worth keeping and to drop them where they are slower or worse. One 1024 × 1280 photo through `remove_background`: FP16 took 0.48 s with a 5.7 GB peak, int8 0.47 s and 5.3 GB, int4 0.47 s and 5.2 GB. The packed weights are a small share of the peak: the 1024 × 1024 activations dominate, so quantizing saves at most 9 % of memory and no time. An earlier run the same night (04:52) gave 5.5, 5.5 and 5.4 GB with the same order of times. Outputs stayed close (mean absolute pixel difference at most 0.006 int8 and 0.08 int4, of 255), so the tiers go for being not worth keeping, not for quality. Both tiers are removed rather than offered; a copy already on someone's disk shows as withdrawn and can be deleted.
