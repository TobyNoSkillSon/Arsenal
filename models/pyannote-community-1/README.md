# Pyannote Community-1

Pyannote Community-1 is pyannote's open speaker-diarization pipeline: it finds who spoke when in a recording, including stretches where several people talk at once. It chains a small PyanNet segmentation network, a WeSpeaker ResNet34 speaker-embedding network and VBx clustering, about 8.1 million parameters in all. Arsenal runs a Swift and MLX port of the makers' pyannote.audio 4.0.0 pipeline. Its strength is accurate speaker turns and speaker counts in meetings and conversations, with an exclusive one-speaker-at-a-time view that lines up easily with a transcript.

## What it is for

pyannote built Community-1 to improve speaker assignment and counting over its previous open pipeline (speaker-diarization-3.1), and to simplify reconciling diarization with transcription timestamps through exclusive diarization.

- Speaker turns for meetings, interviews, podcasts and calls
- Labelling a transcript from `parakeet-v3-ultra` or `qwen3-asr-1.7b` by speaker, using `exclusive_segments`
- Finding overlapping speech
- Counting speakers, or splitting a recording into a known number of speakers
- RTTM files for diarization tools and scoring, and per-speaker embeddings for matching speakers across recordings

## Inputs and outputs

### `diarize` (default)

Finds the speakers in one audio file and returns when each one speaks.

| Field | Type | Required | Notes |
|---|---|---|---|
| `audio` | file object `{"path": string}` | yes | Path 1–4096 characters. Accepts `audio/*`. Arsenal types the file by extension: `.wav`, `.flac`, `.mp3`, `.m4a`, `.aiff`, `.caf`; others (`.aif`, `.ogg`, `.opus`, video files) are refused. macOS AVFoundation decodes it. |
| `num_speakers` | integer | no | 1–32. Exact number of speakers, when known. |
| `min_speakers` | integer | no | 1–32. Lower bound on the number of speakers; ignored when `num_speakers` is set. |
| `max_speakers` | integer | no | 1–32. Upper bound on the number of speakers; ignored when `num_speakers` is set. Must not be below `min_speakers`. |

With none of the three set, the number of speakers is found automatically, with no upper cap.

Returns JSON:

- `segments` (array, at most 100000): the regular diarization, `{"start": seconds, "end": seconds, "speaker": string}`. Overlapping speech appears as overlapping segments with different speakers.
- `exclusive_segments` (array, at most 100000): the same shape, with at most one speaker at any time. Use it to assign transcript words or sentences to speakers.
- `speakers` (array of strings, at most 1024): the labels used, `SPEAKER_00`, `SPEAKER_01`, … in pyannote's order.
- `rttm` (string): the regular diarization in RTTM, one line per segment, `SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>`, where `<file>` is the input's file name without extension, spaces replaced by underscores.
- `speaker_embeddings` (array, at most 1024): one 256-dimensional centroid per entry of `speakers`, in the same order; `null` when no speech segment was long enough to embed.

Times are seconds from the start of the file, rounded to milliseconds. A file with no detected speech returns empty arrays and an empty `rttm`.

Audio: the pipeline works on 16 kHz mono, and the makers' loader averages the channels of stereo or multichannel files and resamples other rates to 16 kHz. Arsenal does the same, so the caller sends the file as it is: it decodes to 16 kHz with macOS's converter (not torchaudio, so samples can differ slightly from pyannote's) and averages the channels. A file longer than 600 seconds (ten minutes) is refused, not truncated. The whole file is processed at once.

## Run it

MCP: `arsenal_run(model:"pyannote-community-1", input:{"audio":{"path":"/absolute/path/meeting.m4a"},"min_speakers":2,"max_speakers":5})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/pyannote-community-1/ops/diarize" -H 'Content-Type: application/json' -d '{"audio":{"path":"/absolute/path/meeting.m4a"},"num_speakers":2}'`

A path may name any file on this Mac; Arsenal clones it into its inbox before the model reads it. `arsenal_upload` is only for a file that is not on this Mac. When a speaker count forces k-means, it runs with a fixed seed (42). In a quick check on an M5 Max (22 Oct 2026, Mac in use, short fixed test clip), a warm request took 0.16 s and the worker peaked at 0.41 GB.

## Limits

- Ten minutes per file and at most 32 requested speakers.
- Not streaming: the whole recording is processed at once.
- Speaker labels are per file. `SPEAKER_00` in one file is not the same person as `SPEAKER_00` in another; compare `speaker_embeddings` to match people across files.
- The makers' own benchmark diarization error rates range from 8.9% (REPERE) and 11.2% (VoxConverse) to 44.6% (AVA-AVD) and 46.8% (Ego4D): expect more errors on far-field, noisy or many-speaker recordings.
- Trained-weight output and FP16 precision quality in Arsenal are not yet checked against pyannote.audio.

## Licence and credit

CC-BY-4.0 (Creative Commons Attribution 4.0). Commercial use, modification and redistribution are allowed if you credit pyannote, link the licence and say what you changed. The makers' Hugging Face repository is gated behind a contact form; Arsenal downloads the ungated `pyannote-community` mirror of the same files. Made by pyannote (Hervé Bredin, Alexis Plaquet; pyannoteAI); the speaker embedding is WeSpeaker's (Wang et al., 2023, trained on VoxCeleb, CC BY 4.0) and the clustering follows BUT Speech@FIT's VBx. Training and tuning used GENCI's Jean Zay supercomputer. The makers ask for these citations:

```bibtex
@inproceedings{Plaquet23,
  author={Alexis Plaquet and Hervé Bredin},
  title={{Powerset multi-class cross entropy loss for neural speaker diarization}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}
@inproceedings{Wang2023,
  title={Wespeaker: A research and production oriented speaker embedding learning toolkit},
  author={Wang, Hongji and Liang, Chengdong and Wang, Shuai and Chen, Zhengyang and Zhang, Binbin and Xiang, Xu and Deng, Yanlei and Qian, Yanmin},
  booktitle={ICASSP 2023, IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
@article{Landini2022,
  author={Landini, Federico and Profant, J{\'a}n and Diez, Mireia and Burget, Luk{\'a}{\v{s}}},
  title={{Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization: theory, implementation and analysis on standard tasks}},
  year={2022},
  journal={Computer Speech \& Language},
}
```

## Links

- Model card (pinned mirror): https://huggingface.co/pyannote-community/speaker-diarization-community-1/tree/8a527374977391da736e0daaef26855d949d9685
- Release post: https://www.pyannote.ai/blog/community-1
- Code: https://github.com/pyannote/pyannote-audio

## How Arsenal runs it

This is the makers' pipeline as run by pyannote.audio 4.0.0, ported to Swift with MLX for the networks:

1. PyanNet segmentation (SincNet, four bidirectional LSTM layers, powerset classifier for up to three local speakers, two at once) over 10 s windows every 1 s.
2. Speaker counting: the mean number of active local speakers per 16.875 ms frame across overlapping windows.
3. WeSpeaker ResNet34 embeddings for each window and local speaker, from Kaldi 80-bin filterbanks, masked to that speaker's frames with overlap excluded unless too short.
4. Clustering exactly as `VBxClustering`: centroid-linkage agglomerative initialisation at threshold 0.6, PLDA transform, VBx with Fa 0.07 and Fb 0.8, constrained assignment; when a speaker count is requested and VBx disagrees, scikit-learn KMeans (k-means++, three initialisations, seed 42).
5. Reconstruction of regular and exclusive diarizations, labelled `SPEAKER_00`, `SPEAKER_01`, … in pyannote's order, times rounded to milliseconds.

The SciPy linkage, scikit-learn KMeans and NumPy random generator are ported decision for decision and checked against those libraries on fixed inputs (identical labels on 60 linkage and 357 of 358 KMeans cases; the remaining case is a float32 tie decided by BLAS summation order). Resampling uses macOS rather than torchaudio, so input samples can differ slightly.

Checkpoint: `pyannote-community/speaker-diarization-community-1@8a527374977391da736e0daaef26855d949d9685`, the ungated mirror of the gated `pyannote/speaker-diarization-community-1`; both list the same files and byte sizes. Port references: `pyannote/pyannote-audio@a1ed3bb0440d33d18622e1cb6b138431cfaf4f7a` (tag 4.0.0), asteroid-filterbanks `ParamSincFB`, SciPy 1.16 `_hierarchy`, scikit-learn 1.9 `KMeans`. No Core ML, Python or pickle execution is involved.

The release is PyTorch Float32 checkpoints plus NumPy PLDA arrays. Get converts them once into one FP16 `model.safetensors` using fixed tensor descriptors read from the archive headers (`gate/header-check.json`, `gate/header-map.json`). The PLDA transform is clustering data and stays FP16. SincNet's window and time vectors are recomputed, not stored. FP16 storage of the PLDA arrays and SincNet cut-off frequencies is the main precision risk; it is unqualified until the trained-weight check. Arsenal offers this model at FP16 only: below 1B parameters, 8-bit would save under 1 GB and run no faster.

Verification: the no-weight fixture runs the real graph with a shrunk window and ResNet through the ordinary worker for automatic, fixed and bounded speaker counts. Unit tests rebuild PyTorch/NumPy containers from random weights, convert and load every tier, and compare clustering with SciPy, scikit-learn and pyannote's VBx references. None of this qualifies trained output. The 22 Oct quick check (load 1.18 s, warm request 0.161 s, peak 395 MiB summed over Arsenal's worker processes) is in `results/`; trained output against pyannote.audio waits for the trained-weight trial. Research log: `log/001-integration.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Mono 16 kHz; other files are resampled on load. [model card, pyannote-community/speaker-diarization-community-1, undated, first-hand]
- Parameters people recommend: `num_speakers` fixes the count; `min_speakers` and `max_speakers` bound it. The card's `exclusive_speaker_diarization` is an output; Arsenal returns it as `exclusive_segments`. [model card, undated, first-hand; pyannote blog, 2026-10-30, first-hand]
- Arsenal call: `arsenal_run(model:"pyannote-community-1", input:{"audio":{"path":"/absolute/path/meeting.m4a"}})` [ops/diarize.input.json, checked 2026-10-31, first-hand]

**Good and bad**
- Good at: On the card's DER table (2025-09), community-1 beats 3.1 on 10 of 12 sets, e.g. AVA-AVD 44.6 against 49.7, overlap included. [model card, first-hand]
- Bad at: REPERE (phase 2) is worse than 3.1: 8.9 against 7.9. The blog's "across all key metrics" claim does not hold here. [model card; pyannote blog, 2026-10-30, first-hand]
- Against the obvious alternatives: The blog gives no benchmark table, only an internal 15x training speed-up. No independent comparison found. [pyannote blog, 2026-10-30, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: Install error on pyannote 4: `NameError: AudioDecoder`, FFmpeg and TorchCodec failing (HF Inference Support #5851, opened 2025-10-30). Fix or workaround: a reply suggests conda-forge ffmpeg 6 and uninstalling `av`; no one reports it working. [HF Inference Support #5851, opinion]
- Complaint or known bug: Overlapped segments assigned to one speaker (#1987, closed 2026-10-05 as cannot_reproduce, wontfix). Peak CUDA VRAM 9.54 GB against 1.59 GB for 3.3.2 (#1963, open, wontfix). Fix or workaround: Not found in sources checked. [pyannote-audio issues, 2025-12 to 2026-10, opinion]

**Evidence**
- Strength: thin. Accuracy claims are the maker's; no independent trained-weight test found.
- Sources:
  - Model card, https://huggingface.co/pyannote-community/speaker-diarization-community-1, last updated 2025-09, first-hand (maker).
  - Blog, https://www.pyannote.ai/blog/community-1, published 2026-10-30, first-hand (maker).
  - Issues #1987 and #1963, https://github.com/pyannote/pyannote-audio/issues, opinion.
  - HF Inference Support #5851, https://huggingface.co/spaces/huggingface/InferenceSupport/discussions/5851, opened 2025-10-30, opinion.
