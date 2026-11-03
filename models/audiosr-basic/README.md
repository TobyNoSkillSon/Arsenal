# AudioSR (basic)

AudioSR is an audio super-resolution model by Haohe Liu, Ke Chen, Qiao Tian, Wenwu Wang and Mark D. Plumbley. It is a latent-diffusion model: a VAE, a UNet and a HiFi-GAN vocoder, 672M parameters across the three networks it runs that look at the mel spectrogram of band-limited audio and generate the missing high band. The basic checkpoint is the general one for any kind of sound. What it does best is give back a full-band 48 kHz file from audio whose top end was cut off, while keeping the original low band.

## What it is for

The makers trained it to take any input whose bandwidth is between 2 kHz and 16 kHz and output audio with 24 kHz bandwidth at a 48 kHz sampling rate, for sound effects, music and speech alike. Their training data was simulated by low-pass filtering.

- Old recordings and transfers with no treble
- Phone calls, voice notes and low-rate files (8, 16, 22.05 kHz) brought up to 48 kHz
- Restoring the top end of output from other generators (the paper names AudioLDM, FastSpeech 2 and MusicGen)
- Speech, music and effects in the same model

## Inputs and outputs

### `enhance` (default)

Rebuilds the missing high frequencies of one clip and returns it at 48 kHz.

| Field | Type | Required | Notes |
|---|---|---|---|
| `audio` | file | yes | `{"path":"/absolute/path"}`, `audio/*`. Up to 60 s. |
| `steps` | integer | no | 10–100, default 50. DDIM sampling steps; more is slower. Capped at 100 so a 60 s clip stays well inside the operation's time limit. |
| `guidance_scale` | number | no | 1–10, default 3.5. Classifier-free guidance scale. |
| `seed` | integer | no | 0–4294967295, default 42. Changes the generated high band (and, as in the reference, the input low-pass filter type). |
| `stream` | boolean | no | Default `false`. Ignored: the result is one file. |

Returns `files` with one entry, `audio`: a mono 16-bit PCM WAV at 48 kHz (`audio/wav`), the same length as the input, peak-normalised to 0.5 (about −6 dBFS). The entry carries `id`, `path`, `name`, `mime`, `bytes` and `sha256`.

The input can be any file macOS's AVFoundation can decode, at any sample rate and channel count. Arsenal resamples it to 48 kHz and averages all channels to mono before the model sees it; a file longer than 60 s, or shorter than about 2 ms (100 samples at 48 kHz), is refused rather than cut. The clip is processed whole, not in chunks: it is padded to a multiple of 5.12 s, so memory grows with length (about 13 GB of GPU memory at 60 s). For a longer recording, split it into clips of up to 60 s and join the results. The model finds the input's own frequency cutoff and low-passes there, so no cutoff has to be given. Upstream warns that cutoffs unlike a low-pass filter, such as the holes MP3 compression leaves near the cutoff, are filled poorly; low-pass filtering such a file below its cutoff before sending it helps.

## Run it

MCP: `arsenal_run(model:"audiosr-basic", input:{"audio":{"path":"/absolute/path/old-record.wav"}})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/audiosr-basic/ops/enhance" -H 'Content-Type: application/json' -d '{"audio":{"path":"/absolute/path/old-record.wav"}}'`

The model declares no settings; `steps`, `guidance_scale` and `seed` go in each request. Keep `seed` to repeat a result; change it to get a different high band. On an M5 Max a 5.12 s clip took 3.5 s warm and a 60 s clip 36.7 s at the default 50 steps (quick checks, 22 Oct 2026, Mac in use, measured).

## Limits

- Output is mono only; stereo input is mixed down.
- Up to 60 s per clip, processed whole.
- Inputs whose cutoff differs from a plain low-pass (MP3 and other codec damage), or with strong noise or reverb, can fail, per upstream.
- It does not remove noise, reverb or clipping; it only adds the high band.
- A diffusion model: each seed gives a different high band, and the share of energy it adds varies widely (on one clip the reference put 3.8% of the energy above 8 kHz with seed 42 and 55% with seed 43).
- Measured on a 1916 record: the regenerated band above about 10 kHz sounds noise-like. Listen before keeping a result.
- A seed does not reproduce the PyTorch reference's output sample for sample.

## Licence and credit

The checkpoint is Apache-2.0 (SPDX `Apache-2.0`) per the pinned model card; the AudioSR code is MIT. Apache-2.0 allows commercial use, modification and redistribution, with the licence and notices kept and changes stated; it includes a patent grant. Made by Haohe Liu, Ke Chen, Qiao Tian, Wenwu Wang and Mark D. Plumbley, who ask for this citation:

```bibtex
@inproceedings{liu2024audiosr,
  title={{AudioSR}: Versatile audio super-resolution at scale},
  author={Liu, Haohe and Chen, Ke and Tian, Qiao and Wang, Wenwu and Plumbley, Mark D},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing},
  pages={1076--1080},
  year={2024},
  organization={IEEE}
}
```

## Links

- Model card: https://huggingface.co/haoheliu/audiosr_basic/tree/74a47f49061a1e788e968cc43ad45c0b6243f37d
- Code: https://github.com/haoheliu/versatile_audio_super_resolution
- Paper: https://arxiv.org/abs/2309.07314
- Samples: https://audioldm.github.io/audiosr

## How Arsenal runs it

Checkpoint: `haoheliu/audiosr_basic@74a47f49061a1e788e968cc43ad45c0b6243f37d` (`pytorch_model.bin`, FP32, 6.18 GB). The file also carries a CLAP model, EMA copies and a duplicate condition VAE that inference never uses. Get converts the three networks the pipeline runs (VAE 223M, UNet 258M, HiFi-GAN vocoder 190M parameters *(unverified)*) once into FP16 safetensors (`cast:audio-audiosr-v1`, a non-executing pickle reader), about 1.3 GB. Arsenal offers FP16 only. Runner: the native Swift MLX audio worker, sequential only, macOS 26 or later; the operation times out at 590 s.

Behaviour follows audiosr 0.0.7 `super_resolution` with its command-line defaults (50 DDIM steps, guidance 3.5, seed 42): the clip is centred, peak-normalised to 0.5 and padded to a multiple of 5.12 s; its energy cutoff is detected and it is low-passed there (filter family picked by the seed, as the reference does); the VAE-encoded low-passed mel conditions DDIM (v-prediction, η = 1, classifier-free guidance); the decoded mel keeps the input's low mel bins; HiFi-GAN renders 48 kHz; the input's STFT below the cutoff replaces the generated low band; the result is peak-normalised to 0.5 and trimmed to the input length.

Steps are capped at 100 (the reference allows any count): a 60 s clip took 36.7 s at the default 50 steps on an M5 Max (quick check, 22 Oct), so 100 steps is about 75 s there, which leaves roughly eight times headroom for slower Macs inside the 590 s operation timeout. `stream` is accepted and ignored, as for the other enhancement models.

Differences from the reference: stereo input is mixed down rather than keeping only the first channel; a clip whose cutoff sits at Nyquist is passed through unfiltered where the reference raises; step counts whose DDIM grid reaches t = 1000 read the last schedule entry where the reference raises; the Gaussian noise comes from MLX's generator, with one key per draw split from the seed, so a seed always gives the same Arsenal result but does not reproduce PyTorch's output sample for sample.

Gate (`gate/gate.toml`, evidence `gate/evidence/native/parity.json`, 22 Oct 2026, M5 Max): reference is audiosr 0.0.7 on PyTorch MPS under FP16 autocast, DSP against SciPy 1.17 / librosa 0.9.2 test vectors (filter type per seed, four low-pass families, `resample_poly`, mel basis), cosine schedule checked against the checkpoint's `alphas_cumprod` within 1e-6. Fixture: 5.12 s of a 1916 record (16 kHz source) as 48 kHz mono, detected cutoff 7125 Hz, elliptic low-pass. Component relative L2 on the reference's own inputs: UNet 0.0013, VAE decode 0.0010, vocoder 0.0019, VAE encode mean 0.0126 and std 0.0181, low-pass mel 9.5e-5. End to end, replaying the reference's noise: waveform 0.022, log-mel 0.0057. Share of energy above 8 kHz: reference seed 42 3.8%, seed 43 55%; port replaying seed 42's noise 3.7%, port with its own seed 42 noise 15%. That spread is the noise-like band measured on 21 Oct with the reference, not a port error.

Memory and time (quick checks, Mac in use, not the controlled benchmark, which is deferred in `bench/bench.toml`): 5.12 s clip 3.5 s warm with 3.3 GB MLX peak; 60 s clip 36.7 s warm with 12.7 GB MLX peak; Load plus two short requests through Arsenal's API peaked at 3.83 GB summed process footprint (`results/quick-m5-max-fp16-native-exact.json`). Research log: `log/001-port.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Band-limited audio, 2 to 16 kHz bandwidth in; 24 kHz bandwidth at 48 kHz sampling rate out. [arXiv 2309.07314, submitted 2023-09-13, first-hand (maker)]
- Prompts, reference audio or voice settings, if relevant: Not found in sources checked.
- Parameters people recommend: Not found in sources checked for defaults. One user tried 50 random seeds on a demo speech clip: seed 11 gave LSD about 1.29 against about 1.66 for the default seed, and no seed matched the demo's stability. [GitHub issue 84, 2025-11-23, first-hand (user test)]
- Arsenal call: `arsenal_run(model:"audiosr-basic", operation:"enhance", input:{"audio":{"path":"/absolute/path/old-record.wav"},"steps":50,"guidance_scale":3.5,"seed":42})` [ops/enhance.input.json, 2026-10-31, first-hand]

**Good and bad**
- Good at: The paper says it works as a plug-and-play module for AudioLDM, Fastspeech2 and MusicGen output, based on a subjective evaluation. [arXiv 2309.07314, first-hand (maker claim)]
- Bad at: Not found in sources checked beyond the complaints below.
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: Broken or garbled words in speech output on the speech checkpoint (CUDA), against the demo clip. [GitHub issue 84, 2025-11-23, first-hand]
- Complaint or known bug: Abrupt noises in silent parts and between voiced regions, on the speech and basic modes. [GitHub issue 64, 2024-09-05, first-hand]
- Fix or workaround: Reported by one user, not confirmed: seed tuning lowered LSD on one clip only. Neither issue has a reply (0 comments). [GitHub issue 84, first-hand (user report)]

**Evidence**
- Strength: thin. The maker's paper and two GitHub reports, neither with a maintainer reply. No independent comparison found.
- Sources:
  - arXiv 2309.07314, https://arxiv.org/abs/2309.07314, submitted 2023-09-13, first-hand (maker).
  - GitHub issue 84, https://github.com/haoheliu/versatile_audio_super_resolution/issues/84, 2025-11-23, first-hand.
  - GitHub issue 64, https://github.com/haoheliu/versatile_audio_super_resolution/issues/64, 2024-09-05, first-hand.
