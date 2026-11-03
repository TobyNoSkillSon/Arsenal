# Mel-Band RoFormer Vocals (Kim)

Kimberley Jensen's vocal model pulls the singing out of a song and leaves the instrumental. It is a Mel-Band RoFormer (228 million parameters): the stereo spectrum is cut into 60 overlapping bands spaced like human pitch hearing, and six blocks of rotary-attention transformers look along time and across bands before predicting a complex mask for the vocals. The instrumental is the mix minus the vocals. Arsenal runs a Swift and MLX port of the maker's own code in FP16.

## What it is for

- Vocals out of a full song, for karaoke, remixes, covers or lyrics transcription
- An instrumental bed without the voice
- Speech or narration out of music and sound effects
- Long tracks: it works through 8 s chunks, so memory stays flat

The maker says the model does slightly better than the paper's equivalent because it was trained on more data.

## Inputs and outputs

### `separate_audio`

| Field | Type | Notes |
|---|---|---|
| `audio` | file `{"path": string}` | Required. Any audio macOS decodes; resampled to 44.1 kHz stereo (mono is copied to both channels). Up to ten minutes. |
| `stems` | array of `vocals`, `instrumental` | Which files to return; both by default. |
| `overlap` | integer 2–8 | The maker's `num_overlap`: how many 8 s chunks cover each moment (default 2, the release's setting). Higher smooths the seams between chunks and takes proportionally longer; the maker suggests no more than 8. 1 is refused: without overlap the maker's fades leave two silent-vocal samples at every 8 s seam. |

Returns `files`: one 16-bit 44.1 kHz stereo WAV per stem, with ids `vocals` and `instrumental`.

## Usage notes (2 Nov 2026)

What it does: two-stem vocal separation, exactly as the maker's `inference.py`: 8 s chunks (the training length), linear cross-fades of 0.8 s, the clip reflect-padded at both ends when longer than 8 s, and the instrumental as the mix minus the vocals. Inputs: one audio file, optional stems and overlap. Limits: two stems only, backing vocals count as vocals, ten minutes per clip, and a clip shorter than 8 s still costs one full chunk. The maker writes float WAV; Arsenal writes 16-bit PCM, so a stem that would pass full scale is scaled down to just under it instead of clipping (stems within full scale are untouched).

Source: `KimberleyJSN/melbandroformer` at `ac9b0614`, one FP32 PyTorch state dict (913 MB). The repository has no configuration; the maker's own (`configs/config_vocals_mel_band_roformer.yaml` in KimberleyJensen/Mel-Band-Roformer-Vocal-Model) is built in, and Get checks every one of the checkpoint's 672 learned tensors against it by name and shape before writing an FP16 snapshot (`cast:audio-mel-band-roformer-v1`, 456 MB). The network runs in FP16; the STFT, the mask averaging over overlapping bands and the inverse STFT run in float32.

Port: the attention, feed-forward and band layers are shared with Arsenal's BS-RoFormer port (the same upstream code). New here: the band layout, computed from librosa's Slaney mel filterbank with the maker's two forced corners and identical to the maker's own indices; the output norm that closes every transformer; and the averaging of masks where bands overlap.

Parity: checked against the maker's own code (Mel-Band-Roformer-Vocal-Model at `25f44ffb`), run once on the Mac's GPU with the network in FP16, on a fixed 12 s stereo clip (a macOS voice over synthesized chords and a kick). Relative error on the same inputs: band features 0.27%, first axial block 0.08%, whole trunk 0.50%, mask 0.03%, one 8 s chunk 0.09%, and the maker's full chunked `demix_track` on the whole clip 0.09%; the inverse STFT from the maker's own mask matches to 4e-7. Bounds were set before the comparison (chunk and demix 5%). On that clip both the maker's vocals and Arsenal's score 28.8 dB SDR against the true voice, against 1.6 dB for the unseparated mix (derived, one synthetic clip). Not checked: real commercial recordings, a benchmark such as MUSDB18, speed and memory figures.

## Licence and credit

MIT, from the model card. Credit Kimberley Jensen for the model; the architecture is Wang, Lu and Won, "Mel-Band RoFormer for Music Source Separation" (ByteDance, 2023), implemented by lucidrains (BS-RoFormer, MIT) and trained with ZFTurbo's Music-Source-Separation-Training.
