# TIGER DnR

Splits a film or video soundtrack into dialogue, music and sound effects. It is the cinematic version of TIGER from Tsinghua University (ICLR 2025): three TIGER networks of 1.4M parameters each, one per stem, working on 57 frequency bands of 44.1 kHz audio, trained for the Divide and Remaster (DnR) task.

Input: `audio` (any file AVFoundation opens), and optionally `stems` (`dialog`, `effects`, `music`; `dialogue`, `speech`, `effect`, `sfx` are accepted too) and `mode` (`extract`, the default, returns the listed stems; `remove` returns the ones not listed, for example a soundtrack without music as separate dialog and effects files). Only the networks for the returned stems run. `prompt`, `seed` and `steps` are accepted by the audio kind but ignored. Output: `files`, one 44.1 kHz mono 16-bit WAV per stem (`id` and name `dialog` / `dialog.wav`, label "Dialogue"), with `path`, `mime` `audio/wav`, `bytes`, `sha256` and `order`.

Example request:

```json
{"audio": {"path": "/absolute/path/to/clip.mp4"}, "stems": ["dialog"]}
```

How it runs: the audio is resampled to 44.1 kHz mono and cut as the makers' `wav_chunk_inference` does: 12-second windows every 4 seconds, zero-padded at both ends, each window separated (STFT window 2048, hop 512; 8 separator repeats), and the overlapping results summed and divided by three. Samples outside -1...1 are clipped when the WAV is written.

Limits: three fixed stems; sung vocals can split between dialog and music. Stereo is mixed to mono and the stems come back mono. Input is 46 ms to 5 minutes. English-language film audio in training; other languages, Polish included, were not tested. The Hugging Face card for this repository shows the makers' Apollo heading by mistake; the weights load into TIGERDNR exactly.

Parity: matches the maker's PyTorch TIGERDNR (MPS) at cosine 0.9999998 or better on every stem of a 15-second clip, against float32 and float16 autocast, and the runner's 60-second dialogue for the makers' example clip matches their published output at cosine 0.999999 (`log/003-parity.md`).

Source: [JusperLee/TIGER-DnR](https://huggingface.co/JusperLee/TIGER-DnR/tree/b7a59560bbca10febbcd46fb01600f868e587f57), Apache-2.0; code [JusperLee/TIGER](https://github.com/JusperLee/TIGER), paper [arXiv 2410.01469](https://arxiv.org/abs/2410.01469). By Mohan Xu, Kai Li, Guo Chen and Xiaolin Hu. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Not found in sources checked. The card names only the script's `--audio_path` argument.
- Prompts, reference audio or voice settings, if relevant: Not found in sources checked.
- Parameters people recommend: the paper's DnR appendix uses 6-second windows at 50% overlap at inference, not the README body's 12-second, 4-second-hop windows. [arXiv 2410.01469 v3, 2026-02-27, first-hand]
- Arsenal call: `POST /arsenal/v1/models/tiger-dnr/ops/separate_audio`, or the same operation through MCP. Vendor example: `python inference_dnr.py --audio_path test/test_mixture_466.wav`, from the JusperLee/TIGER code. [JusperLee/TIGER-DnR model card, undated (checked 2026-10-31), first-hand]

**Good and bad**
- Good at: in the paper's cinematic table (SI-SDR, dB), TIGER scores 7.4 music, 15.5 speech, 6.5 effects. BSRNN scores 5.5, 13.8 and 1.8, and uses 52.8 M parameters against TIGER's 1.40 M. [arXiv 2410.01469 v3, 2026-02-27, first-hand]
- Bad at: on the TIGER speech model, not DnR (issue #11, opened 2025-04-14), a user found other languages "didn't work well" and speakers with very different voices were not separated. The maintainer suggested fine-tuning. [JusperLee/TIGER issue #11, 2025-04-14, first-hand]
- Against the obvious alternatives: Table 9 also lists Conv-TasNet* (0.3, 8.5, 2.0 dB) and MRX* (4.2, 12.3, 5.7 dB), starred as the DnR paper's numbers. TIGER is higher on all three. [arXiv 2410.01469 v3, 2026-02-27, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: running the speech script after `requirements.txt` fails with "No module named 'pkg_resources'" on Python 3.10 (issue #40, opened 2026-06-10). Fix or workaround: Reported by one user, not confirmed: a manual `pip install setuptools==69.5.1` after `requirements.txt` solves it, per the reporter. No maintainer reply; still open. [JusperLee/TIGER issue #40, 2026-06-10, first-hand]

**Evidence**
- Strength: thin. Maker's paper and card, two user issues, no independent DnR test.
- Sources: huggingface.co/JusperLee/TIGER-DnR; arxiv.org/html/2410.01469v3; github.com/JusperLee/TIGER/issues/11 and /40.
