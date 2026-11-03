# CLAP HTSAT Fused

Labels a sound or music clip with whichever of your own text descriptions fits it best, with no training.
LAION's contrastive language-audio model (Wu et al., 2022): an HTSAT Swin audio encoder with feature fusion and an
English RoBERTa text encoder, mapped into one 512-number space. 153.6M parameters, trained on LAION-Audio-630K.

## Input and output

`label` takes:

- `audio`: one file (WAV, MP3, M4A, FLAC, AIFF; any rate, mono or stereo; mixed to mono and resampled to 48 kHz)
- `labels`: 1 to 256 candidate descriptions, e.g. "a dog barking", "rain on a window", "jazz piano"
- `return_embedding` (optional): also return the clip's 512-number embedding, for search

and returns every label with `score` (probability among the given labels), `similarity` (cosine, -1 to 1, comparable
across requests), sorted best first, plus `top`, `seconds` and, when asked, `audio_embedding`.

Clips up to 10 seconds are repeated to fill 10 seconds; longer clips are summarised from a shrunken copy of the whole
plus three 10-second crops, as the maker's feature extractor does (fixed crops here, random in the maker's code).

## Example

```json
{"audio": {"path": "/System/Library/Sounds/Glass.aiff"}, "labels": ["a bell", "a dog barking", "speech", "music"]}
```

The macOS Glass chime comes back as "a bell" 0.63, "a dog barking" 0.36, speech 0.01, music 0.002. Twelve seconds of
macOS `say` speech came back as speech 0.997.

## Limits

- Scores are relative to the labels you give; add a catch-all label when none may fit.
- Very short clips are weak: 0.3 s of the same chime came out as music 0.52, bell 0.28.
- English labels only; Polish labels score poorly. It does not transcribe speech.
- Files not at 48 kHz go through macOS's resampler.

## Parity

Checked against transformers' ClapModel and ClapProcessor on MPS at FP16, on three 48 kHz clips (1.65 s chime, 0.3 s
chime, 12.2 s speech): audio embedding cosine 0.999999 on each, cosine similarities within 0.0007, the same top label
and the same ranking every time. Pass. See `log/003-parity.md`.

Source: [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused/tree/365dea6ef167def6676140ed93bbc43f84dabb28), Apache-2.0.
Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: one audio clip plus candidate text labels, scored against each other. [CLAP model card, undated (checked 2026-10-31), first-hand]
- Prompts, reference audio or voice settings, if relevant: the LAION README's ESC-50 script wraps each label as "This is a sound of <label>"; no rule on wording is given. [LAION-AI/CLAP README, undated (checked 2026-10-31), first-hand]
- Parameters people recommend: the LAION README advises fusion checkpoints (`630k-fusion-best.pt`) for variable-length general audio. No sampling settings. [LAION-AI/CLAP README, undated (checked 2026-10-31), first-hand]
- Arsenal call: `POST /arsenal/v1/models/clap-htsat-fused/ops/label`, or the same operation through MCP. Vendor example: `audio_classifier(audio, candidate_labels=["Sound of a dog", "Sound of vaccum cleaner"])`. [CLAP model card, undated (checked 2026-10-31), first-hand]

**Good and bad**
- Good at: the card's abstract claims state-of-the-art zero-shot classification, with no numbers. A third-party table gives 91.0 ESC-50 accuracy, checkpoint not named. [laion/clap-htsat-fused model card, undated (checked 2026-10-31), first-hand; arXiv 2607.25085 Table 2, submitted 2026-07-27, first-hand]
- Bad at: the same table averages 50.1 over eight benchmarks; FSD50K is 21.5 mAP. [arXiv 2607.25085 Table 2, submitted 2026-07-27, first-hand]
- Against the obvious alternatives: MS-CLAP averages 61.0% in that table. [arXiv 2607.25085 Table 2, submitted 2026-07-27, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: the same clip gives different embeddings when batched (issue #90, still open). The maintainer (2023-04-10) calls variation over 10 s expected (random segments) and under 10 s a bug they will fix. Fix or workaround: none confirmed; the maintainer names the `laion_clap` pip package as corrected, with no version or report that it worked. [LAION-AI/CLAP issue #90, 2023-04-02, first-hand]

**Evidence**
- Strength: thin. The card, one third-party table (some numbers not strict zero-shot, per its caption), one user issue.
- Sources: huggingface.co/laion/clap-htsat-fused; github.com/LAION-AI/CLAP (README and issues/90); arxiv.org/html/2607.25085.
