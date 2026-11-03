# Audiobox Aesthetics

Rates a speech, music or sound clip the way human listeners would, on four axes, with no reference recording:
Content Enjoyment (CE), Content Usefulness (CU), Production Complexity (PC) and Production Quality (PQ), each about 1 to 10.
Made by Meta (Tjandra et al., 2025): a WavLM Base+ encoder with four small heads, 103.9M parameters.

## Input and output

`score_quality` takes one audio file (WAV, MP3, M4A, FLAC, AIFF; any rate, mono or stereo) and returns:

- `CE`, `CU`, `PC`, `PQ`: the four predicted ratings
- `seconds`: the clip's length

The clip is mixed to mono, resampled to 16 kHz, cut into 10-second windows (the last one zero-padded and masked)
and the window scores are averaged by length, as the maker's `infer.py` does.

## Example

```json
{"audio": {"path": "/System/Library/Sounds/Glass.aiff"}}
```

returns about `{"CE": 3.85, "CU": 7.95, "PC": 1.65, "PQ": 8.14, "seconds": 1.65}`: a clean single chime, so high
production quality, very low complexity and middling enjoyment. Twelve seconds of macOS `say` speech scored
CE 5.4, CU 6.6, PC 1.5, PQ 6.5.

## Limits

- Scores predict human ratings; compare clips of the same kind rather than reading one number alone.
- A long file gets the length-weighted average of its windows, not its worst stretch.
- Clips under about a second are mostly padding and score less reliably (0.3 s of the chime: CE 3.4 against 3.9 for the whole).
- Non-16 kHz files go through macOS's resampler, not the maker's torchaudio one; on the chime that moved scores by up to 0.1.
- It does not read language; Polish speech is rated like any speech, though its training speech was mostly English.

## Parity

Checked against the maker's own PyTorch code on MPS at FP16, on three 16 kHz clips (1.65 s chime, 12.2 s speech,
0.3 s chime): max abs difference 0.0022 on the 1-10 scale, cosine over the 12 scores 0.99999998. Pass. See `log/003-parity.md`.

Source: [facebook/audiobox-aesthetics](https://huggingface.co/facebook/audiobox-aesthetics/tree/9b1dd8e5df9af7216e836a98974fe3b82c56ded6), CC-BY-4.0
(WavLM parts MIT). Code: github.com/facebookresearch/audiobox-aesthetics. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Python takes a file path or a tensor with its sample rate; the CLI reads JSONL with optional `start_time` and `end_time`. Needs Python 3.9+ and PyTorch 2.2+. [model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not applicable.
- Parameters people recommend: Not found in sources checked.
- Arsenal call: `POST /arsenal/v1/models/audiobox-aesthetics/ops/score_quality`, or the same operation through MCP. Vendor example: `from audiobox_aesthetics.infer import initialize_predictor`, then `predictor.forward([{"path": "/path/to/a.wav"}])`. [model card, undated, first-hand]

**Good and bad**
- Good at: Scores speech, music and sound on four axes with no reference clip. The paper says "comparable or superior" to human MOS and existing methods. [arXiv 2502.05139, 2025-02-07, first-hand]
- Bad at: On music, a Cyanite review found a low-quality MP3 rated only slightly below its studio version on PQ, and female-led tracks scored higher on CE, CU and PQ. Speech and sound not tested there. [Cyanite review, 2026-02-12, first-hand; vendor sells music-AI tools]
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: PyTorch deprecation warning for mixing bool and float masks in WavLM attention. The reporter says results are correct but the path may break in a future PyTorch. Fix or workaround: none shown. [facebookresearch/audiobox-aesthetics issue #17, opened 2026-10-30, open, no replies, first-hand]

**Evidence**
- Strength: thin. Maker's card and paper, one music-only vendor review, and one issue read in full. No speech user reports found.
- Sources:
  - facebook/audiobox-aesthetics model card, https://huggingface.co/facebook/audiobox-aesthetics, undated, first-hand.
  - Meta Audiobox Aesthetics paper, https://arxiv.org/abs/2502.05139, 2025-02-07, first-hand.
  - Cyanite review, https://cyanite.ai/blog/meta-audiobox-aesthetics-review/, 2026-02-12, first-hand.
  - Issue #17, https://github.com/facebookresearch/audiobox-aesthetics/issues/17, 2026-10-30, first-hand.
