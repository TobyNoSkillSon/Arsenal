+++
hypothesis = "Identical component inputs and frozen FP16 bounds can isolate quantization errors from generated-audio differences."
change = "Capture teacher-forced component replays, activation checks and balanced warm samples."
measurement = "Busy M5 Max quick check; historical component policy, superseded by log 005."
numbers = "Per-component bounds, warm samples and physical peaks are retained in the linked intake result."
verdict = "inconclusive"
links = ["../results/intake/intake-2026-10-15T11-22-27Z-534d1300-91ab-4f17-bdc4-6a9a49e08b33.json"]
+++

# Intake component gate, 15 October 2026

Round 2 replaces the final waveform/spectrum precision verdict with independent teacher-forced component checks. Token IDs/mask, original conditioning, exact latents/times at steps 0/3/7 and the production decoder input are saved once. The SAME-L decoder replay uses the same input dtype as the serving decoder. Candidate prompt and denoiser outputs never contaminate subsequent component checks. A/A reference repeats were identical; bounds were frozen from that floor plus one FP16-format ULP before the low-bit candidates were evaluated.

Run: `intake-2026-10-15T11-22-27Z-534d1300-91ab-4f17-bdc4-6a9a49e08b33`. Full metrics/bounds, activation checks and all warm samples are in `../results/intake/<run>.json`; `../gate/evidence/intake-latest.json` links the exact run. Binary captures stay local with a hashed fixture manifest. An earlier interrupted attempt is retained separately and is not table evidence.

The FP16 baseline is reproducible and finite. It does not qualify the initial cast of the vendor's FP32-only SAME-L source: an original decoder component reference remains absent. Saved original PCM and the older round-1 spectral/waveform screen cannot fill that component gap. No FP32 model reference was executed.

Both new low-bit recipes pack eligible DiT, T5Gemma (including embedding) and SAME-L matrices. Every output archive hash matches round 1 and its source-header coverage. Int8 relative L2: prompt 0.010392; denoiser 0.015139/0.012016/0.014325; decoder 0.007849. Int4: prompt 0.171593; denoiser 0.211073/0.162202/0.228570; decoder 0.097293. The corresponding bounds are 0.000790; 0.000718/0.000720/0.000722; 0.000713. All outputs/recorded activations are finite, with no clamp events, but numerical and activation bounds fail. This leaves the recipes unqualified rather than proving perceptual unsuitability. One saved five-second audio sanity artifact per tier has no clipped samples; log-mel differences are diagnostic only (int8 0.607 dB, int4 4.748 dB), without a listening-quality verdict.

Warm five-second/eight-step generation medians (min–max): FP16 0.243 s (0.230–0.258), int8 0.278 s (0.271–0.282), int4 0.240 s (0.239–0.240). Candidate-paired FP16 medians are 0.255 s and 0.232 s respectively; the pooled FP16 median must not imply an int4 speed win. Each candidate has four balanced ABBA BAAB samples. Within-pair baseline drift is +1.34%/−1.02%. Per-worker lifetime physical peaks, including Load self-tests, are 5.726/4.704/3.533 GB; cold/load latency is separate.

No new cast or whole-part quant recipe is offered, and the existing catalog recipe remains unchanged. The bounded run completed without swap growth; all lab weight copies were removed. No install or manifest change was made.
