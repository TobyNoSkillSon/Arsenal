+++
hypothesis = "Kokoro 82M's int8 tier is worth keeping: faster or smaller than FP16."
change = "model.toml drops [derive] int8; Kokoro ships FP16 and int4. The int8 table result is removed; the 16 Oct intake and its quality binding stay as evidence for the FP16 and int4 figures."
measurement = "Quick check, M5 Max, 22 Oct 2026, Mac in use (lab/night-2026-10-22/peak.py): per precision one ordinary Load and two requests through the installed app's API; peak = summed macOS phys_footprint_peak of the Arsenal worker processes (MiB, shown here in GB of 10^9 bytes); the timed request is the second, one sample. Not the controlled ABBA protocol. Rows: lab/night-2026-10-22/peak-results.jsonl; summary: lab/night-2026-10-22/tiers.md."
verdict = "rejected"
numbers = { fp16_speak_seconds = 0.048, int8_speak_seconds = 0.04, int4_speak_seconds = 0.035, fp16_peak_gb = 1.40, int8_peak_gb = 1.43, int4_peak_gb = 1.45 }
links = ["../README.md", "010-voice-heldout-r5.md"]
+++

# Int8 withdrawn

22 October 2026. The owner asked to decide per model whether int8 and int4 are worth keeping and to drop them where they are slower or worse. One short `speak` request plus `voices`: FP16 took 0.062 s in all with a 1.40 GB peak; int8 0.059 s and 1.43 GB; int4 0.045 s and 1.45 GB. At 82M parameters the weights are a sliver of the peak, so int8 saves no memory and almost no time; it is removed rather than offered. Int4 stays: it was the fastest. Audio quality was not compared in this run (sampled output).
