# HTDemucs fine-tuned

HTDemucs fine-tuned (`htdemucs_ft`) is Hybrid Transformer Demucs from Meta (Simon Rouard, Francisco Massa, Alexandre Défossez): a U-Net that works on the waveform and the spectrogram at once, with a cross-domain Transformer at its centre that attends within each domain and across the two. The fine-tuned release is a bag of four such networks, each fine-tuned for one source, together 168M parameters. It splits a stereo music mix into four stems: drums, bass, other and vocals.

## What it is for

The makers trained it on MUSDB HQ plus 800 extra songs and fine-tuned one network per source, trading speed for quality: upstream says separation takes four times longer than plain `htdemucs` but "might be a bit better". The paper reports 9.20 dB SDR on the MUSDB test set for its best per-source fine-tuned variant, which used sparse attention that was not released.

- Split a song into drums, bass, other and vocals for remixing, sampling or practice.
- Remove one instrument family from a mix, for example a drumless or bassless backing track.
- Isolate vocals for transcription or lyric alignment when a four-way split is also wanted.

## Inputs and outputs

### `separate_audio` (default)

Separate a music file into the four stems and return the ones asked for.

| Field | Type | Required | Notes |
|---|---|---|---|
| `audio` | file `{"path":"/absolute/path"}` | yes | Accepts `audio/*`; path 1 to 4096 characters. |
| `stems` | array of strings | no | Each one of `drums`, `bass`, `other`, `vocals`; 1 to 4 items, unique. Default `["drums","bass","other","vocals"]`. |
| `mode` | string | no | `extract` or `remove`. Default `extract`. |

Returns `files`: one 16-bit PCM WAV per returned stem, stereo, 44.1 kHz, with `id` and `name` set to the stem (`vocals`, `vocals.wav`), plus `path`, `mime` (`audio/wav`), `bytes` and `sha256`. `extract` returns the listed stems. `remove` returns every stem not listed, each as its own file; it does not mix them into one track (sum the files yourself for, say, a drumless backing track), and removing all four is refused.

Arsenal decodes the file with AVFoundation (any container `AVAudioFile` opens), resamples to 44.1 kHz and converts to stereo before the model sees it. A file with more than two channels is downmixed by averaging; how a mono file is spread to two channels is AVFoundation's default *(unverified)*. Input is limited to ten minutes. The audio is cut into segments of the model's training length (7.8 s, from the pinned configuration), overlapping by a quarter and blended with a triangular weight; all four networks run on each segment and their outputs are combined with the per-stem weights shipped in the checkpoint.

## Run it

MCP: `arsenal_run(model:"htdemucs-ft", input:{"audio":{"path":"/absolute/path/song.flac"},"stems":["vocals"],"mode":"remove"})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/htdemucs-ft/ops/separate_audio" -H 'Content-Type: application/json' -d '{"audio":{"path":"/absolute/path/song.flac"},"stems":["drums","bass","other","vocals"]}'`

There are no settings. A warm request on a 3.3 s clip took 0.40 s on an M5 Max that was in use (quick check, measured); longer files scale with the number of segments. The request times out after 590 s.

## Limits

- Four fixed stems. Upstream's six-source model (adds guitar and piano) and two-stem mode are not offered here.
- Upstream's `shifts` (random time-shift averaging) and adjustable `overlap` are not exposed; Arsenal runs one pass with a fixed 25% overlap.
- It was trained on music; upstream documents no results on speech-only recordings or film soundtracks.
- Ten minutes of input per request.
- Upstream Demucs is archived and no longer maintained by Meta; fixes go to the author's fork.
- Trained-output quality of this port has not been compared against the PyTorch reference.

## Licence and credit

MIT, as declared by the MLX conversion's card and by the original Demucs repository. Commercial use is allowed; keep the copyright and licence notice.

Made by Simon Rouard, Francisco Massa and Alexandre Défossez at Meta. MLX conversion by iky1e, mirrored by mlx-community; Swift MLX port by Kyle Howells. Cite:

```bibtex
@inproceedings{rouard2022hybrid,
  title={Hybrid Transformers for Music Source Separation},
  author={Rouard, Simon and Massa, Francisco and D{\'e}fossez, Alexandre},
  booktitle={ICASSP 23},
  year={2023}
}
@inproceedings{defossez2021hybrid,
  title={Hybrid Spectrogram and Waveform Source Separation},
  author={D{\'e}fossez, Alexandre},
  booktitle={Proceedings of the ISMIR 2021 Workshop on Music Source Separation},
  year={2021}
}
```

## Links

- Model card: https://huggingface.co/mlx-community/demucs-mlx-fp16/tree/908d2d05cf3035bf4017fc0f52be64163689a61f
- Code: https://github.com/facebookresearch/demucs (archived; maintained fork https://github.com/adefossez/demucs)
- Swift MLX port: https://github.com/kylehowells/demucs-mlx-swift
- Paper: https://arxiv.org/abs/2211.08553
- Earlier Hybrid Demucs paper: https://arxiv.org/abs/2111.03600

## How Arsenal runs it

Checkpoint: `mlx-community/demucs-mlx-fp16@908d2d05cf3035bf4017fc0f52be64163689a61f` (`htdemucs_ft.safetensors` and `htdemucs_ft_config.json`). Native port: `kylehowells/demucs-mlx-swift@c8482739c621b90a64bf1a9f013712f6fea44eda`. Exact files, SHA-256 digests and byte counts are in `model.toml`. All four fine-tuned submodels run with the checkpoint bag weights; this is not the first submodel relabelled as the bag. The recurrent LSTM projections are packed with the same rule as the other weights.

Runner: the native Swift MLX audio worker, sequential requests only, macOS 26 or later. FP16 is the only offered precision: below 1B parameters, 8-bit would save under 1 GB and run no faster. The upstream MLX card reports FP16 output identical to FP32 within 3.1e-5 per sample.

FP16 group-norm fix (`log/002-fp16-group-norm.md`): every clip is padded to the 7.8 s training segment, and the time branch's first DConv normalises one group of 6 × 85,995 values; the FP16 sum overflowed and NaN reached every stem. Statistics, affine and activation now run in float32 and return the input dtype. With the trained weights a 3.2 s clip separated in 1.15 s and the four stems summed back to the mix at 21.2 dB SNR, with speech landing in vocals (quick check).

Quick check, 22 Oct 2026, M5 Max 40-core GPU, Mac in use: load 0.93 s; warm `separate_audio` on a 3.3 s speech-and-music clip 0.40 s including the API round trip (first 0.43 s); peak memory 1.56 GB, the sum of `phys_footprint_peak` over Arsenal's worker processes (`results/`).

Gate (`gate/gate.toml`): status `no-weights`. The no-weight fixture builds a shrunk version of the real graph and runs the ordinary worker operation, which checks wiring and shapes, not trained output, speed or numerical fidelity. No parity run against PyTorch Demucs has been made, so trained-weight quality is unqualified; the benchmark is deferred (`bench/bench.toml`). Native source closure is recorded in `Workers/Audio/PORTED-FROM.md`. Logs: `log/001-integration.md`, `log/002-fp16-group-norm.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Full music mixes. The maker says it takes "4 times more time" than htdemucs and "might be a bit better". [facebookresearch/demucs README, read 2026-10-31, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not applicable; none found in sources checked.
- Parameters people recommend: Not found in sources checked.
- Arsenal call: `arsenal_run(model:"htdemucs-ft", operation:"separate_audio", input:{"audio":{"path":"/absolute/path/song.flac"},"stems":["vocals"],"mode":"extract"})`

**Good and bad**
- Good at: On a vendor table, not re-run: vocals about 8.9 dB against 8.1 for htdemucs, drums about 9.5 against 8.4. [AI Stem Splitter benchmark, 2026-04-28, opinion (vendor)]
- Bad at: Speed, which the maker puts at 4 times the plain model's time. [facebookresearch/demucs README, read 2026-10-31, first-hand]
- Against the obvious alternatives: The vendor's per-stem table puts BS-RoFormer (no extra data) about 1.3 dB ahead on average (9.8 against 8.5). Its plain htdemucs figures disagree (about 9.0 in its TL;DR, 7.7 per stem). [AI Stem Splitter, 2026-04-28, opinion (vendor)]

**Complaints, bugs and fixes**
- Complaint or known bug: Poor output after upsampling 48 kHz input to 96 kHz. Fix or workaround: None confirmed; the reporter says shift and overlap settings are not the cause, even at extremes. No replies. [facebookresearch/demucs issue #609, opened 2024-07-11, first-hand (user)]
- Complaint or known bug: The maker says the repo is no longer maintained. Fix or workaround: None found; the README's linked fork is also "not actively maintained". [facebookresearch/demucs README, read 2026-10-31, first-hand]

**Evidence**
- Strength: thin; maker README, one user report, a secondary vendor comparison; no listening test or PyTorch parity check.
- Sources:
  - https://github.com/facebookresearch/demucs, read 2026-10-31; GitHub archived flag true (date not read), first-hand
  - https://github.com/facebookresearch/demucs/issues/609, opened 2024-07-11, no replies, first-hand
  - https://aistemsplitter.org/blog/htdemucs-vs-bs-roformer-vs-spleeter-2026-benchmark, 2026-04-28, opinion (vendor)
