<!-- SPDX-License-Identifier: MIT -->
# MossFormer2 SE 48K

MossFormer2_SE_48K is the full-band speech enhancement model of Alibaba Speech Lab's ClearerVoice-Studio. It removes background noise from speech at 48 kHz, so the top of the spectrum that 16 kHz denoisers throw away is kept. The network is a MossFormer2 mask network: 24 blocks that pair gated single-head attention (quadratic within 256-frame groups, linear across the clip) with a gated FSMN memory, 55.3M parameters. It reads Kaldi log-mel features with their deltas and predicts a magnitude mask over a 48 kHz STFT.

## Usage notes (2 Nov 2026)

**What it does.** Cleans background noise out of speech. The maker reports, on VoiceBank+DEMAND at 48 kHz, PESQ 3.15, STOI 0.95 and SI-SDR 19.36 dB, against 1.97, 0.92 and 8.39 dB for the noisy input (ClearerVoice-Studio `clearvoice/README.md`, Model Performance).

**Inputs.** One operation, `enhance`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `audio` | file | yes | `{"path":"/absolute/path"}`. Any file AVFoundation can read, any rate, up to eight channels, up to 10 minutes. |
| `mode` | string | no | `denoise` only (the default). Restore and upscale are not offered. |
| `one_time_decode_length` | integer | no | 1–60 s, default 20 (the maker's default). Clips up to this length are cleaned in one pass. |
| `decode_window` | integer | no | 1–20 s, default 4 (the maker's default). Longer clips are cleaned in windows of this length at a 3/4 stride; 1/8 of a window is dropped at each inner edge. |
| `stream` | boolean | no | Default `false`. With `true` the PCM is streamed after the whole clip is cleaned, not while it is made. |

Each channel is cleaned on its own at 48 kHz: other rates are resampled in and back out, as the maker's reader and writer do. The result is a WAV with the input's rate (48 kHz for inputs above it, which are read at 48 kHz), channel count and length.

MCP: `arsenal_run(model:"mossformer2-se-48k", input:{"audio":{"path":"/absolute/path/call.wav"}})`

HTTP: `curl -s "$ARSENAL_URL/arsenal/v1/models/mossformer2-se-48k/ops/enhance" -H 'Content-Type: application/json' -d '{"audio":{"path":"/absolute/path/call.wav"}}'`

**Limits.**
- Denoising only: it does not remove reverb, repair clipping or raise the sample rate.
- It is trained for speech; music or other wanted sounds under the noise are suppressed with it.
- Up to 10 minutes per clip; split longer recordings.
- BF16 only. The FP16 cast overflows inside the network (below), and M1 and M2 Macs have no BF16 arithmetic, so they run it more slowly.
- Two deliberate differences from the maker: the fbank dither is 0 instead of the maker's random one-step dither, so a clip always gives the same result; and when a long clip's length fits the window stride exactly, Arsenal pads one more stride so the last 0.5 s is cleaned (the maker leaves it silent).

**Parity result.** Checked against the maker's own code (ClearerVoice-Studio `6b3774dc`, PyTorch 2.15 on MPS in BF16), captured once on a fixed 1 s, 48 kHz harmonic-plus-noise clip. Every component got the maker's own input tensors.
- Weight-free, on the CPU in the default test suite: features (Kaldi fbank with deltas) 1.9e-7 relative L2, STFT → mask → iSTFT 2.6e-7, the 21 s windowed decode 1.9e-7.
- Network, BF16: encoder 1.4e-4, first attention block 0.016, block 23 0.016, computation block 0.016, mask 0.0078 (cosine 0.99997); the whole clip end to end 0.013 (cosine 0.99992). Bounds were frozen before the run (mask 0.03, cosine 0.999; waveform 0.05).
- For scale: the maker's network run in BF16 on the CPU and on the GPU differs by 0.0056 at the mask; casting its weights to BF16 by rounding versus truncation moves the mask by 0.022.
- One real Get, Load, enhance and Unload ran in an isolated profile on 2 Nov 2026: the maker's 18.4 s stereo 44.1 kHz sample, forced through the windowed path, came back at the same rate, channels and length, with no network traffic. The quietest tenth of its frames fell from −53.8 dB to −87.6 dB while speech stayed at about −22 dB (measured).

## How Arsenal runs it

Checkpoint: `alibabasglab/MossFormer2_SE_48K@eff8c97925c8bec812af707814b3e5d777fd4503`, `last_best_checkpoint.pt` (FP32 PyTorch state dict, 221.6 MB). Get converts it once with a non-executing pickle reader (`cast:mossformer2-se-48k`) into one BF16 safetensors file of about 110 MB: 904 parameters, every shape checked; the 24 rotary and one positional frequency buffers are checked against their formula and recomputed at run time. Runner: the Sound family of the bundled native worker, sequential only, macOS 26 or later.

FP16 was tried first and rejected: in the maker's own half-precision run the gated FSMN product in block 1 reaches about 1.9e7 and the output turns to NaN. BF16 keeps every activation finite. Norm statistics and the affine step after them, framing, FFTs and overlap-add run in float32 as numerical intermediates; every learned weight and matrix product is BF16.

The network is the mlx-audio-swift port (`Blaizzy/mlx-audio-swift@0dda61b`, MIT) with fixes found by the parity run: the maker's intra-block norm is GroupNorm over channels and time, not a per-frame LayerNorm; its LayerNorm, ScaleNorm and channel norm epsilons are 1e-6, 1e-5 and 1e-5; and GlobalLayerNorm must round once, after its affine step. The front end follows the maker's torchaudio Kaldi fbank rather than the port's HTK mel bank. Details: `log/001-port.md`; evidence: `gate/golden/golden.json`.

## Licence and credit

The weights are Apache-2.0 per the pinned model card; ClearerVoice-Studio's code is Apache-2.0. Made by Alibaba's Speech Lab (Institute for Intelligent Computing). Paper: https://arxiv.org/abs/2506.19398.

## Links

- Model card: https://huggingface.co/alibabasglab/MossFormer2_SE_48K/tree/eff8c97925c8bec812af707814b3e5d777fd4503
- Code: https://github.com/modelscope/ClearerVoice-Studio
- MossFormer2 paper: https://arxiv.org/abs/2312.11825
