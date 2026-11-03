<!-- SPDX-License-Identifier: MIT -->
# UniPASE

UniPASE is a generative universal speech enhancer from Xiaobin Rong and colleagues (IEEE TASLP 2026). It cleans speech that has several problems at once (background noise, reverb, clipping, codec damage, a narrow band, short dropouts) and can return it at up to 48 kHz. It does not filter the noisy waveform: a WavLM Large fine-tuned for enhancement (DeWavLM-Omni) reads the speech, a Vocos adapter turns its noisy acoustic layer and its cleaned phonetic layer into a clean acoustic layer, a Vocos vocoder speaks that layer at 16 kHz, and, for any output rate above 16 kHz (22.05 to 48 kHz), a TF-GridNet PostNet restores the band above 8 kHz. 546.5M parameters (measured over the four checkpoints: 315.5M, 114.5M, 113.7M and 2.8M).

## Usage notes (2 Nov 2026)

**What it does.** Universal speech enhancement in one pass: denoising, dereverberation, declipping, codec and band-limit repair, packet-loss concealment, and bandwidth extension to any rate above 16 kHz (22.05–48 kHz). The authors report first place in the objective evaluation of the URGENT 2026 challenge (arXiv 2604.14606; their claim, not measured here). The model card names English, Chinese, Spanish, French and German; Polish is not named.

**Inputs.** One operation, `enhance`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `audio` | file | yes | `{"path":"/absolute/path"}`. Any file AVFoundation can read, any rate, up to eight channels, 0.1 s to 10 minutes. |
| `mode` | string | no | `denoise` (default) and `restore` run the same universal pass and keep the input's rate (inputs above 48 kHz are read at 48 kHz); `upscale` returns `rate`, 48 kHz by default, never below the input's rate. |
| `rate` | integer | no | Output rate: 8000, 16000, 22050, 24000, 32000, 44100 or 48000 (the maker's `--sr_out`). Above 16 kHz the PostNet restores the upper band; at 16 kHz or below no PostNet runs. |
| `enable_plc` | boolean | no | Default `true` (the maker's default). 20 ms packets that are digital silence are treated as lost and filled in. |
| `one_time_decode_length` | integer | no | 1–60 s, default 20. Longer clips go through the maker's long-form path: 8 s segments at a 4 s hop, joined at their centres. |
| `stream` | boolean | no | Default `false`. With `true` the PCM is streamed after the whole clip is enhanced. |

Each channel is enhanced on its own, as one mono file of the maker's scripts, and its peak is then matched to the input channel's peak, as both of the maker's inference scripts do. The result is a WAV at the output rate with the input's channel count and length at that rate.

MCP: `arsenal_run(model:"unipase", input:{"audio":{"path":"/absolute/path/call.wav"},"mode":"upscale"})`

HTTP: `curl -s "$ARSENAL_URL/arsenal/v1/models/unipase/ops/enhance" -H 'Content-Type: application/json' -d '{"audio":{"path":"/absolute/path/call.wav"},"rate":48000}'`

**Limits.**
- Speech only: music and other wanted sounds are removed with the noise.
- It re-synthesises the voice. The makers designed it for low hallucination, but on badly damaged input it can change details of the speech rather than only remove noise.
- Packet-loss concealment sees only dropouts of digital silence (at least 99% of a 20 ms packet below 1e-7); quiet but non-zero gaps are left as they are.
- Channels are enhanced independently; stereo is not modelled jointly.
- 10 minutes per clip at most. The one-pass path normalises over the whole clip, so up to `one_time_decode_length` the clip is treated as one utterance; past it, each 8 s segment is.
- Not covered: the makers' training scripts and the evaluation-only `Vocoder_WavLM-L24.pt` (not part of UniPASE); arbitrary output rates outside the seven listed.

**Differences from the maker's code**, kept on purpose:
- The maker's long-form script crashes on clips it would split into one segment (6–10 s) or none (under 6 s); Arsenal enhances those in one piece.
- The maker's long-form script slices and writes at the input rate even when another output rate is asked for; Arsenal slices at the output rate, so long clips are correct at every rate.
- The maker's `--enable_plc` flag is always on from the command line (an argparse `type=bool`); here it is a real switch.
- Stereo: the maker's scripts take mono (the long-form one only the first channel); Arsenal enhances every channel.
- Odd rates whose ratio to 16 or 48 kHz does not reduce (44,099 Hz, say) would need a sinc kernel of hundreds of millions of taps in torchaudio's resampler; Arsenal converts those with AVAudioConverter instead. Every standard rate uses the maker's resampler.

**Parity result.** Checked against the maker's own code (Xiaobin-Rong/unipase `857b60ad`, PyTorch 2.15 on MPS in FP16), captured once on a fixed 3 s, 48 kHz clip: the maker's noisy sample `fileid_295.wav` with 100 ms of digital silence, so the PLC path runs. Every component got the maker's own input tensors; bounds were frozen before the run.
- Weight-free, on the CPU in the default test suite: torchaudio resampling at 7 rate pairs at most 1.7e-7 relative L2, the PostNet's STFT, band split, crossfade and iSTFT 3.0e-7, the long-form segmentation and stitch 6.0e-8, the Vocos ISTFT 9.1e-6; the packet-loss mask and WavLM's position buckets exact.
- Networks, FP16: WavLM CNN 8.3e-4, layer 24 0.0068, L24 0.0068 (cosine 0.99998; bound 0.05), adapter 0.0017 (cosine 0.9999985), vocoder waveform 0.0069 (cosine 0.99998; bound 0.1), PostNet output 4.0e-4. End to end on the clip: STFT magnitude 0.022 (bound 0.116), waveform 0.063 (cosine 0.998).
- For scale: the maker's own network in BF16 differs from its FP16 run by 0.053 at L24, 0.32 at the vocoder waveform and 0.116 in STFT magnitude at the output. The vocoder turns tiny differences into phase drift, which is why the end-to-end check is on magnitude.
- One real Get, Load, enhance and Unload ran in an isolated profile on 2 Nov 2026, with no network traffic: the maker's 12.36 s, 48 kHz sample through the long-form path came back at the same rate and length. On the maker's three samples Arsenal's output is within 0.7–4.6% (STFT magnitude) of the maker's own published enhanced files, with the same distance to the clean recordings (log-spectral distance 1.23, 1.95 and 1.68 dB against the maker's 1.23, 1.93 and 1.60; noisy input 3.01, 2.76 and 2.00). Details: `results/real-check-2026-11-02.md`.

## How Arsenal runs it

Weights: `Xiaobin-Rong/unipase@f0b4d4c4411fe08fc2dddbf2d9f33260c27ac4a0`, `DeWavLM-Omni.pt`, `Adapter.pt`, `Vocoder_DWO-L1.pt` and `PostNet.pt` (FP32 PyTorch state dicts, 2.19 GB). Get converts them once with a non-executing pickle reader (`cast:unipase`) into one FP16 safetensors file of 1.09 GB: 1,033 parameters, every shape and FP16 range checked; WavLM's weight-normalised position convolution is folded in float32; the unused final encoder LayerNorm and the vocoder's Hann window are checked and dropped. Runner: the Sound family of the bundled native worker, sequential only, macOS 26 or later.

FP16 throughout: in the maker's own FP16 run every activation stays finite, the largest at a component boundary being 1,053, so BF16 was not needed. Norm statistics, GELU, softmax, the LSTM cell state, FFTs and resampling are float32 intermediates, rounded once; every learned weight and matrix product is FP16.

The port is Arsenal's own, written after the maker's PyTorch modules (WavLM, Vocos and ESPnet's TF-GridNet as the maker adapted them); details in `log/001-port.md`, evidence in `gate/golden/golden.json`.

## Licence and credit

The weights are Apache-2.0 per the pinned model card. The maker's code is MIT (its root LICENSE), with parts under their own licences in its `licenses/` folder: Cisco's PASE (Apache-2.0), ESPnet's TF-GridNet (Apache-2.0), Microsoft's WavLM (MIT) and WavTokenizer's Vocos (MIT). Made by Xiaobin Rong, Zheng Wang, Yushi Wang, Jun Gao and Jing Lu. Paper: https://arxiv.org/abs/2604.14606.

## Links

- Model card: https://huggingface.co/Xiaobin-Rong/unipase/tree/f0b4d4c4411fe08fc2dddbf2d9f33260c27ac4a0
- Code: https://github.com/Xiaobin-Rong/unipase
- Demo: https://xiaobin-rong.github.io/unipase_demo/
