+++
hypothesis = "Chandra OCR 2's int8 and int4 tiers are worth keeping: faster or smaller than FP16 at the same OCR text."
change = "model.toml drops [derive] int8 and int4; Chandra OCR 2 ships FP16 only."
measurement = "Quick check, M5 Max, 22 Oct 2026, Mac in use (lab/night-2026-10-22/peak.py): per precision one ordinary Load and two requests through the installed app's API; peak = summed macOS phys_footprint_peak of the Arsenal worker processes (MiB, shown here in GB of 10^9 bytes); the timed request is the second, one sample. Not the controlled ABBA protocol. Rows: lab/night-2026-10-22/peak-results.jsonl; summary: lab/night-2026-10-22/tiers.md."
verdict = "rejected"
numbers = { fp16_seconds_per_page = 7.773, int8_seconds_per_page = 8.119, int4_seconds_per_page = 6.31, fp16_peak_gb = 20.4, int8_peak_gb = 17.18, int4_peak_gb = 13.96, int8_text_similarity = 1.0, int4_text_similarity = 0.2494 }
links = ["../README.md", "001-integration.md"]
+++

# Int8 and int4 withdrawn

22 October 2026. The owner asked to decide per model whether int8 and int4 are worth keeping and to drop them where they are slower or worse. One page through `read_document`: FP16 took 7.8 s with a 20.4 GB peak. Int8 gave the same text but took 8.1 s, no faster, for a peak only 16 % smaller (17.2 GB). Int4 was faster (6.3 s, 14.0 GB) but its OCR text changed: similarity 0.25 to the FP16 text. Int8 goes for being not worth keeping, int4 for failing the check; both are removed rather than offered.
