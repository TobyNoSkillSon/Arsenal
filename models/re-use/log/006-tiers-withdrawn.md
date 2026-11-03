+++
hypothesis = "RE-USE's int8 and int4 tiers are worth keeping: faster or meaningfully smaller than BF16."
change = "model.toml drops [derive] int8 and int4; RE-USE ships BF16 only. The int8/int4 table results are removed; the 15 Oct intake and its quality binding stay as evidence for the BF16 figures."
measurement = "Quick check, M5 Max, 22 Oct 2026, Mac in use (lab/night-2026-10-22/peak.py): per precision one ordinary Load and two requests through the installed app's API; peak = summed macOS phys_footprint_peak of the Arsenal worker processes (MiB, shown here in GB of 10^9 bytes); the timed request is the second, one sample. Not the controlled ABBA protocol. Rows: lab/night-2026-10-22/peak-results.jsonl; summary: lab/night-2026-10-22/tiers.md."
verdict = "rejected"
numbers = { bf16_seconds = 0.873, int8_seconds = 1.14, int4_seconds = 1.129, bf16_peak_gb = 1.45, int8_peak_gb = 1.22, int4_peak_gb = 1.27 }
links = ["../README.md", "004-half-intake.md"]
+++

# Int8 and int4 withdrawn

22 October 2026. The owner asked to decide per model whether int8 and int4 are worth keeping and to drop them where they are slower or worse. One 3.3 s speech-and-music clip through `enhance`: BF16 took 0.87 s with a 1.45 GB peak; int8 1.14 s and 1.22 GB; int4 1.13 s and 1.27 GB. Both quantized tiers are slower for about 0.2 GB less memory on a 9.6M-parameter model, so both are removed rather than offered. Audio stayed close (relative L2 0.013 int8, 0.036 int4).
