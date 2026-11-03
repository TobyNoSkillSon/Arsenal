<!-- SPDX-License-Identifier: MIT -->
# DeepFilterNet3

DeepFilterNet3 is Hendrik Schröter's low-complexity speech enhancement network for full-band (48 kHz) audio, built to run in real time. It has about 2.2M parameters and two paths: one estimates gains for 32 ERB bands across the whole spectrum, the other predicts complex deep-filter coefficients that clean the lowest 96 STFT bins (up to 4.8 kHz) over five frames, recovering speech harmonics that a gain alone cannot. Frames are 20 ms with a 10 ms hop and two frames of lookahead.

## Usage notes (2 Nov 2026)

**What it does.** Removes background noise from speech at 48 kHz. Small and fast, it suits calls, voice notes, meetings and long recordings. On the maker's own noisy sample (`assets/noisy_snr0.wav`, 10.6 s) both the maker's code and Arsenal raise the SNR against the clean original from −2.98 dB to 0.78 dB (measured, 2 Nov 2026).

**Inputs.** One operation, `enhance`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `audio` | file | yes | `{"path":"/absolute/path"}`. Any file AVFoundation reads, any rate, up to eight channels, up to 10 minutes. |
| `mode` | string | no | `denoise` only (the default). |
| `atten_lim_db` | number | no | 0–100 dB, default 0 (no limit). The maker's attenuation limit: remove at most this much noise by mixing the original back in, e.g. 12 to keep some ambience. |
| `post_filter` | boolean | no | Default `false`. The maker's post filter (`--pf`), which slightly over-attenuates very noisy sections. |
| `stream` | boolean | no | Default `false`. With `true` the PCM is streamed after the whole clip is cleaned. |

The model works at 48 kHz, the only rate the maker supports; Arsenal resamples other rates in and back out and cleans each channel on its own. The result is a WAV with the input's rate (48 kHz for inputs above it, which are read at 48 kHz), channel count and length; the maker's STFT delay is compensated as `--compensate-delay` / `pad=True` does.

MCP: `arsenal_run(model:"deepfilternet3", input:{"audio":{"path":"/absolute/path/call.wav"},"atten_lim_db":12})`

HTTP: `curl -s "$ARSENAL_URL/arsenal/v1/models/deepfilternet3/ops/enhance" -H 'Content-Type: application/json' -d '{"audio":{"path":"/absolute/path/call.wav"}}'`

**Limits.**
- Denoising only: no reverb removal, clipping repair or bandwidth extension.
- Trained for speech; music under the noise is suppressed with it.
- Up to 10 minutes per clip. Each channel is cleaned in one pass over the whole file, so memory grows with length; the peak for a 10-minute file is not measured (estimated at a few GB).
- Exact digital silence longer than about 78 s used to turn the output into NaN on the GPU; the running normalisation is now clamped at the smallest normal float, which changes nothing for audible input.
- Arsenal cleans a whole file and then returns it. The maker's live microphone use (its LADSPA plugin for PipeWire) is not offered.
- The maker's post filter turns to NaN in FP16 (its 1e-12 epsilon underflows); Arsenal computes it in float32.

**Parity result.** Checked against the maker's own code (Rikorose/DeepFilterNet `d375b2d8`, deepfilternet and libdf 0.5.6, PyTorch 2.5.1 on MPS with the network in FP16), captured once on a fixed 1 s, 48 kHz clip; every network component got the maker's own input tensors.
- Without weights, on the CPU in the default suite: libdf STFT, ERB and complex features (also over 120 s, 12,002 frames), synthesis and the deep filter all within 1e-4 relative L2 of the maker.
- Network, FP16, relative L2: encoder outputs 0.0004–0.0017, ERB mask 0.0027, deep-filter coefficients 0.0026, enhanced spectrum 0.00064. End to end: 0.00052 plain, 0.00053 with the post filter, 0.00038 with a 12 dB attenuation limit. Bounds were frozen before the run (0.02–0.05).
- One real Get, Load, enhance and Unload in an isolated profile (2 Nov 2026): no network traffic, the maker's 10.6 s noisy sample came back at the same rate and length, 0.0007 relative L2 from the maker's own output on the same file.

## How Arsenal runs it

Weights: `mlx-community/DeepFilterNet-mlx@220d5dfb7266352272d74c2a7d2025c59e07b391`, `v3/model.safetensors` (FP32, 8.7 MB; MIT card), plus `v3/config.json`. The maker publishes DeepFilterNet3 only on GitHub (`models/DeepFilterNet3.zip`), which Arsenal's Get cannot pin, so this conversion is pinned instead. It was checked tensor by tensor against the maker's `model_120.ckpt.best` from that archive (zip SHA-256 `49c52edc…84d2`): 118 of 118 names and shapes match and every tensor is bit-identical (max abs difference 0); the maker's 15 batch-norm step counters are the only tensors it drops. Get casts it once to FP16 (`cast:deepfilternet3`), checking each tensor finite and below FP16's maximum; at the small end 51 of 2.17M values flush to zero, mostly running variances of dead channels where batch norm's epsilon dominates.

The network runs in FP16 on the GPU. The STFT, ERB and complex feature normalisation, mask and deep-filter application, post filter and attenuation limit are float32 numerical intermediates. The GRUs and libdf's running normalisation are sequential in time; each runs as one Metal kernel launch over all frames, so every model computation runs on the GPU. Only reading the file, resampling and splitting channels happen on the CPU.

The network layers come from the mlx-audio-swift port (`Blaizzy/mlx-audio-swift@0dda61b`, MIT). Arsenal replaced its CPU parts (an Accelerate GRU loop, sequential feature loops, and a closed-form running mean that overflows after about 90 s), wrote libdf's analysis and synthesis exactly, kept every activation in the half dtype, and fixed an aliasing bug in the deep filter. Details: `log/002-port.md`; evidence: `gate/golden/golden.json`.

## Licence and credit

The maker's code and models are dual-licensed MIT or Apache-2.0; the pinned conversion's card says MIT. Cite: Schröter, Escalante-B., Rosenkranz, Maier, *DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement*, Interspeech 2023 (https://arxiv.org/abs/2305.08227).

## Links

- Code and models: https://github.com/Rikorose/DeepFilterNet
- Pinned weights: https://huggingface.co/mlx-community/DeepFilterNet-mlx/tree/220d5dfb7266352272d74c2a7d2025c59e07b391
