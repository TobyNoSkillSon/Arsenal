<!-- SPDX-License-Identifier: MIT -->
# RE-USE

Weights come from NVIDIA's own repository, `nvidia/NVIDIA-NemotronLabs-RE-USE@31f82f2efad1095034492f236e8dac9239ba713b`, under OpenMDW-1.1 (open, commercial use allowed). They replace `faraday/re-use-mlx`, an MLX conversion of NVIDIA's earlier noncommercial release; the two checkpoints hold different weights. Get casts NVIDIA's PyTorch file into the port's canonical file (MLX names, channels-last convolutions). The FP32 source is cast once to BF16 (the earlier weights failed their FP16 source-format check; FP16 has not been rechecked on these); the engine widens losslessly for fp32 computation. BF16 is the only tier: int8 and int4 are withdrawn, because a quick check on the M5 Max (22 Oct 2026, Mac in use) found them slower than BF16 (1.14 s and 1.13 s against 0.87 s) for a peak only about 0.2 GB smaller (1.22 and 1.27 GB against 1.45 GB) (`log/006-tiers-withdrawn.md`).

Native fp16 Fast Get→Load→enhance→Unload passed on this Mac; see `gate/smoke-evidence.json`. This is a functional smoke, not a full quality suite or Measured badge. Port provenance and gate details: `Workers/Sound/PORTED-FROM.md`. Historical Vireo numbers are not Arsenal measurements.

## Usage notes (2 Nov 2026)

RE-USE cleans up recorded speech: background noise, reverb, clipping, band limits and codec damage. Send `enhance`
an audio file (`audio: {path}`); `mode` is `denoise` (default) or `restore`, which keep the input's rate, or `upscale`,
which returns 48 kHz. Inputs at 8, 16, 22.05, 24, 32, 44.1 or 48 kHz run at their own rate; other rates are resampled
to the nearest one (inputs above 48 kHz are read at 48 kHz). Audio is cleaned in 5-second windows (`chunk_seconds`, a load setting). Overlapping talkers: the
model keeps the main speaker and suppresses the others.

Parity: on a fixed 1 s, 16 kHz test signal, Arsenal's BF16 output matches NVIDIA's own code run on the GPU in FP16 at
39.4 dB SNR (correlation 0.99996); `results/real-check-2026-11-02.md`, golden in `gate/golden/`.

