+++
hypothesis = "NuExtract3's int8 and int4 tiers keep the FP16 output while running faster and smaller."
change = "model.toml drops [derive] int8 and int4; NuExtract3 ships FP16 only."
measurement = "Quick check, M5 Max, 22 Oct 2026, Mac in use (lab/night-2026-10-22/peak.py): per precision one ordinary Load and two requests through the installed app's API; peak = summed macOS phys_footprint_peak of the Arsenal worker processes (MiB, shown here in GB of 10^9 bytes); the timed request is the second, one sample. Not the controlled ABBA protocol. Rows: lab/night-2026-10-22/peak-results.jsonl; summary: lab/night-2026-10-22/tiers.md."
verdict = "rejected"
numbers = { fp16_seconds = 3.331, int8_seconds = 2.193, int4_seconds = 1.672, fp16_peak_gb = 12.88, int8_peak_gb = 8.53, int4_peak_gb = 5.80, int8_text_similarity = 0.2595, int4_text_similarity = 0.1967 }
links = ["../README.md", "001-integration.md"]
+++

# Int8 and int4 withdrawn

22 October 2026. The owner asked to decide per model whether int8 and int4 are worth keeping and to drop them where they are slower or worse. One document through `read_document`: FP16 took 3.3 s with a 12.9 GB peak; int8 2.2 s and 8.5 GB; int4 1.7 s and 5.8 GB. Both quantized tiers changed the text: similarity 0.26 (int8) and 0.20 (int4) to the FP16 text, so both fail the check and are removed rather than offered. The `extract` request answered HTTP 500 at every precision in this run, FP16 included; that is a separate fault, not part of this decision.
