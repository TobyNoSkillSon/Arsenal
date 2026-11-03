# BS-RoFormer vocals (viperx 1296)

A Band-Split RoPE Transformer (BS-RoFormer) checkpoint trained by viperx for one target: the voice. BS-RoFormer, from Wei-Tsung Lu, Ju-Chiang Wang, Qiuqiang Kong and Yun-Ning Hung, splits the complex spectrogram into frequency bands, runs a stack of Transformers across time within each band and across bands, and estimates a mask for the target source. This checkpoint has 160M parameters and separates a stereo mix into vocals and everything else.

## What it is for

The architecture, trained on MUSDB18HQ plus 500 extra songs, ranked first in the music separation track of the Sound Demixing Challenge 2023. This checkpoint is the vocals specialist that python-audio-separator lists as "BS-Roformer-Viperx-1296"; the 12.9628 in its file name is the validation SDR viperx recorded *(unverified: no published card states the dataset or metric)*.

- Pull vocals, speech or narration out of a song or a soundtrack.
- Make an instrumental or a music-and-effects bed with the voice removed.
- Clean a voice track before transcription or voice work.

## Inputs and outputs

### `separate_audio` (default)

Split a mix into vocals and instrumental and return the stems asked for.

| Field | Type | Required | Notes |
|---|---|---|---|
| `audio` | file `{"path":"/absolute/path"}` | yes | Accepts `audio/*`; path 1 to 4096 characters. |
| `stems` | array of strings | no | Each one of `vocals`, `instrumental`; 1 to 2 items, unique. Default `["vocals","instrumental"]`. |

Returns `files`: one 16-bit PCM WAV per requested stem, stereo, 44.1 kHz, with `id` and `name` set to the stem (`vocals`, `vocals.wav`), plus `path`, `mime` (`audio/wav`), `bytes` and `sha256`. `instrumental` is the mix minus the vocals, not a second network output.

Arsenal decodes the file with AVFoundation (any container `AVAudioFile` opens), resamples to 44.1 kHz and converts to stereo; more than two channels are downmixed by averaging, and how mono is spread to two channels is AVFoundation's default *(unverified)*. Input is limited to ten minutes. As python-audio-separator 0.45.0 does with its default MDXC settings, the mix is scaled down to a 0.9 peak if it is louder, processed in chunks of 800 STFT frames (about 8 s; 256 frames, 2.55 s, for clips under ten seconds; a clip shorter than one chunk runs whole), overlap-added with a Hamming weight with the last chunk aligned to the end, and each stem is scaled to at most a 0.9 peak before writing. Output loudness can therefore be lower than the input's.

## Run it

MCP: `arsenal_run(model:"bs-roformer-viperx-1296", input:{"audio":{"path":"/absolute/path/song.m4a"},"stems":["vocals"]})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/bs-roformer-viperx-1296/ops/separate_audio" -H 'Content-Type: application/json' -d '{"audio":{"path":"/absolute/path/song.m4a"},"stems":["vocals","instrumental"]}'`

There are no settings. Measured on an M5 Max in use (quick checks): a 4 s clip in 0.63 s warm, a 60 s clip in 8.6 s warm. The request times out after 590 s.

## Limits

- Two stems only: vocals and everything else. No drums, bass or other split.
- Ten minutes of input per request.
- python-audio-separator's adjustable segment size, overlap, batch size and output formats (FLAC, MP3) are not exposed; Arsenal uses its defaults and writes WAV.
- The SDR in the file name is the trainer's own figure; no independent benchmark of this checkpoint was found.

## Licence and credit

MIT, as declared by the hosting repository `Politrees/UVR_resources`, matching Ultimate Vocal Remover's model repository; viperx published no separate licence for this checkpoint. MIT allows commercial use with the copyright and licence notice kept. python-audio-separator asks integrators of UVR-trained models to credit UVR and its developers (Anjok07 and aufr33).

Checkpoint trained by viperx; architecture by Wei-Tsung Lu, Ju-Chiang Wang, Qiuqiang Kong and Yun-Ning Hung. For the architecture, cite Lu, Wang, Kong and Hung, "Music Source Separation with Band-Split RoPE Transformer", arXiv:2309.02612, 2023.

## Links

- Model files: https://huggingface.co/Politrees/UVR_resources/tree/929e057b81aa49bc2e6490bef8671f47b2c120f6
- Mirror's code repository: https://github.com/Politrees/UVR_resources
- Reference inference: https://github.com/nomadkaraoke/python-audio-separator
- Paper: https://arxiv.org/abs/2309.02612

## How Arsenal runs it

Checkpoint: `model_bs_roformer_ep_368_sdr_12.9628.ckpt` with its training YAML, the viperx release distributed by Ultimate Vocal Remover and python-audio-separator, mirrored byte for byte (same SHA-256) at `Politrees/UVR_resources@929e057b81aa49bc2e6490bef8671f47b2c120f6`. The checkpoint is a PyTorch state dict in FP32; Get converts it once (`cast:audio-bs-roformer-v1`, a non-executing pickle reader) into an FP16 safetensors snapshot. Every weight fits FP16 (largest magnitude 1.67). Arsenal offers FP16 only: at 160M parameters a quantized tier would save little and run no faster.

Runner: the native Swift MLX audio worker, sequential requests only, macOS 26 or later. The network runs in FP16 on the GPU; STFT and iSTFT run in float32.

Gate (`gate/gate.toml`, evidence `gate/evidence/native/parity.json`), qualified against python-audio-separator 0.45.0 with torch 2.14.1 on MPS and FP16 weights, 22 Oct 2026, M5 Max. 699 source tensors checked (675 FP16 parameters and 24 rotary buffers equal to theta 10 000). On a 4.0 s 44.1 kHz stereo mix of speech over generated music, relative L2 against the reference: STFT 5e-7, iSTFT from the reference mask 3e-7, band split 4.9e-4, first layer 6.2e-4, mask 1.5e-3, chunk output 5.6e-4; end to end, vocals waveform 5.0e-4 and log-mel 1.4e-3, instrumental waveform 7.8e-4 and log-mel 9.3e-4. Pass thresholds: STFT and iSTFT below 1e-4, band split below 5e-3, one axial block below 1e-2, mask and chunk audio below 5e-2, end-to-end waveform and log-mel below 1e-2.

Speed and memory, quick checks on a Mac in use: the 4 s parity clip ran in 0.63 s warm at 1.32 GB MLX peak, a 60 s clip in 8.6 s at 1.67 GB. Through the installed Arsenal API (22 Oct 2026): load 0.30 s, warm `separate_audio` 0.21 s, peak 1.48 GB summed over Arsenal's worker processes (`results/`). Separation quality on real material was measured earlier with the reference implementation in the lab, not with this port. Log: `log/001-port.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: A third-party packaging of the ViperX checkpoint returns vocals and "other" (mix minus vocals) [puar-playground/bs-roformer, undated, first-hand].
- Prompts, reference audio or voice settings: Not applicable.
- Parameters people recommend: Not found in sources checked. The ZFTurbo training repo gives no chunk or overlap advice for this model [Music-Source-Separation-Training README, undated, first-hand].
- Arsenal call: `arsenal_run(model:"bs-roformer-viperx-1296", input:{"audio":{"path":"/absolute/path/song.m4a"},"stems":["vocals"]})`

**Good and bad**
- Good at: Not found in sources checked. The only quality figure is the SDR in the file name, 12.9628, which no outside test confirms [Politrees/UVR_resources file listing, undated, first-hand].
- Bad at: Not found in sources checked.
- Against the obvious alternatives: Not found in sources checked. MVSEP's news post gives 10.87 vocal SDR on its Multisong set for ViperX weights (checkpoint not named), not comparable with the file-name figure [MVSEP news 41, 2024-03-29, first-hand].

**Complaints, bugs and fixes**
- Complaint or known bug: A GGUF repackaging of checkpoint 317 (SDR 12.9755, not this 1296 model) reports F16 losing about 30 dB of vocal-stem SNR against F32. Fix or workaround: none confirmed; the card publishes only F32 [thepatch GGUF card, measured 2026-10-02/03, first-hand].
- Complaint or known bug: The GGUF card says no licence is stated upstream and lists it as unknown, while Arsenal's model.toml declares MIT. Fix or workaround: none found [thepatch GGUF card, undated, first-hand].

**Evidence**
- Strength: thin. No independent test of the 1296 checkpoint was found. The only figures are the file-name SDR and MVSEP's numbers on other sets.
- Sources:
  - Politrees/UVR_resources, https://huggingface.co/Politrees/UVR_resources, undated, first-hand (file present at the pinned revision).
  - ZFTurbo/Music-Source-Separation-Training, https://github.com/ZFTurbo/Music-Source-Separation-Training, undated, first-hand.
  - puar-playground/bs-roformer, https://huggingface.co/puar-playground/bs-roformer, undated, first-hand.
  - MVSEP news 41, https://mvsep.com/news/41, 2024-03-29, first-hand.
  - thepatch/bs-roformer-viperx-317-GGUF, https://huggingface.co/thepatch/bs-roformer-viperx-317-GGUF, undated (measured 2026-10-02/03), first-hand.
