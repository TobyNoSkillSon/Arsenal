# DiariZen Large v2

DiariZen finds who spoke when in a recording, including speech where up to four people talk at once. It was built by BUT Speech@FIT (Brno University of Technology) for meetings and far-field room audio. A WavLM Large speech encoder, structurally pruned from 316.6 to 63.3 million parameters, feeds four Conformer layers and a powerset head that marks which of up to four local speakers are active every 20 ms in 16-second windows. Speaker embeddings from WeSpeaker ResNet34 and VBx clustering then link the windows' speakers across the whole recording, so the number of speakers is found automatically. Arsenal runs a Swift and MLX port of the maker's pipeline in FP16 (76 million parameters with the embedding model).

## What it is for

- Meetings, panels and far-field room recordings
- Overlapping speech, up to four voices at the same moment
- Recordings with many speakers: the number is found automatically
- Exclusive one-speaker-at-a-time turns for aligning a transcript
- RTTM export

The card reports diarization error rates at collar 0 of 13.9% on AMI, 10.1% on AISHELL-4, 10.8% on AliMeeting, 16.7% on NOTSOFAR-1, 15.8% on MSDWild, 14.5% on DIHARD3, 11.0% on RAMC and 9.1% on VoxConverse, against 22.4%, 12.2%, 24.4%, –, 25.3%, 21.7%, 22.2% and 11.3% for pyannote 3.1.

## Inputs and outputs

### `diarize`

| Field | Type | Notes |
|---|---|---|
| `audio` | file `{"path": string}` | Required. Any audio macOS decodes, resampled to 16 kHz; only the first channel of a multichannel file is used, unmixed, as the maker's pipeline does. Up to ten minutes. |

There are no other options: the maker's pipeline decides the number of speakers itself and takes no speaker count.

Returns the audio kind's diarization result, the same shape as `pyannote-community-1`'s (without its speaker embeddings), so callers can swap models:

- `segments`: `{"start", "end", "speaker"}` in seconds, in time order; overlapping speech gives overlapping segments. Speakers are `SPEAKER_00`, `SPEAKER_01`, … in the maker's cluster order.
- `exclusive_segments`: the same clusters with at most one speaker at a time (the most active one), made the way pyannote 4.0 makes them. This is an Arsenal addition; the maker's pipeline returns only the regular diarization.
- `speakers`: the labels that occur.
- `rttm`: the segments as RTTM lines.

## Run it

MCP: `arsenal_run(model:"diarizen-large-v2", input:{"audio":{"path":"/absolute/path/meeting.m4a"}})`

HTTP: `curl -s "$ARSENAL_URL/arsenal/v1/models/diarizen-large-v2/ops/diarize" -H 'Content-Type: application/json' -d '{"audio":{"path":"/absolute/path/meeting.m4a"}}'`

## Limits

- At most four speakers active together inside one 16-second window; a fifth simultaneous voice is not marked.
- Ten minutes per recording (a ten-minute file took about a minute on an M5 Max in use; 590 s per request).
- Segments end with the recording. The maker's pipeline lets the last zero-padded window's speech run past the end of the file (its AMI example ends at 30.392 s of 30 s); Arsenal ends such segments at the recording's length. Nothing inside the recording changes.
- The speaker count cannot be set; similar voices can be merged. On Arsenal's synthetic three-voice clip both the maker's pipeline and this port give two of the macOS voices the same label.
- Speakers are numbered per recording, not identified, and no speaker embeddings are returned.
- Not streaming: the whole recording is processed at once.
- Trained on far-field English and Mandarin meetings and conversations (AMI, AISHELL-4, AliMeeting, NOTSOFAR-1, MSDWild, DIHARD3, RAMC, VoxConverse). Polish is untested here.

## Usage notes (2 Nov 2026)

What it does: speaker diarization of a whole recording with overlapping speech, the maker's pipeline end to end (segmentation, median filter, speaker count, WeSpeaker embeddings, VBx clustering, reconstruction). Inputs: one audio file, first channel, 16 kHz after decoding. Limits: ten minutes per file, four simultaneous speakers per 16 s window, no speaker-count option.

Source: `BUT-FIT/diarizen-wavlm-large-s80-md-v2` at `027b3e22` (FP32 PyTorch state dict, 278 MB, with its config and PLDA files) and `pyannote/wespeaker-voxceleb-resnet34-LM` at `837717dd` (27 MB) as an asset. Get reads both checkpoints in place, folds the positional convolution's weight norm and writes one FP16 snapshot (`cast:audio-diarizen-v1`, about 152 MB). The networks run in FP16, and the PLDA tables are stored in FP16 like Community-1's (the loader keeps every stored tensor at 16-bit; clustering still reproduces the maker exactly on its own embeddings); waveform normalisation, the 25-layer weighted sum, fbank features and all clustering run in Float32 or Double, as DSP and host steps.

Port: the segmentation network is new (pruned WavLM with per-layer head lists from the maker's configuration, Conformer, powerset head). WeSpeaker, PLDA, AHC and VBx reuse Community-1's ports, which match DiariZen's pyannote fork; DiariZen's own clustering steps (10% clean-frame filter, unnormalised centroids, Hungarian assignment, the maker's renumbering) are ported as they are, including a quirk where a speaker left without a cluster becomes cluster 0.

Parity: checked against BUT's own DiariZen code with its pyannote fork, run once on the Mac's GPU in half precision (torch 2.15; the pinned torch 2.1.1 computes the model wrongly on Apple GPUs) on a 27.5 s three-voice clip and the maker's 30 s AMI example. On one 16 s window: feature extractor within 0.3% relative error, WavLM layers within 1.1%, the weighted layer mix within 1.9%, Conformer within 1.0%, powerset scores within 0.14%, and the frame-by-frame speaker decision identical on all 799 frames; speaker embeddings at cosine 0.99992 or better. Over whole files the speaker decisions agree on 100% of frames for the three-voice clip and 99.2% for AMI, and the speaker count on 100% of frames for both. Host steps replayed on the maker's own values reproduce its filtered segmentation, speaker count, clustering and segments exactly. End to end, before the end-of-recording clipping above: the three-voice clip gives exactly the maker's segments. On the AMI clip the clustering is sensitive to rounding: the maker's own pipeline differs from itself by 16.7% DER when only its embedding network runs in FP32 instead of FP16. The asserted bound is that the port lies within 16.7% DER of at least one of the maker's two runs; it lies 4.3% from the FP32-embedding run and 21.0% from the FP16 run. Bounds for the components were set before the comparison; the whole-file agreement bounds were added after review, below the measured values. Not checked: accuracy on a benchmark, recordings of real ten-minute meetings, other languages.

## Licence and credit

The model is CC BY-NC 4.0: non-commercial use only, as the card asks, with credit to BUT Speech@FIT. Arsenal makes no money, so it is accepted here; recheck before any revenue. The VBx PLDA files in the same repository are CC BY 4.0, and the WeSpeaker ResNet34 embedding (`pyannote/wespeaker-voxceleb-resnet34-LM`, a pyannote wrapper around WeSpeaker's VoxCeleb model) is CC BY 4.0. Arsenal converts the weights to FP16. Cite Han et al., "Leveraging self-supervised learning for speaker diarization" (ICASSP 2025), "Fine-tune Before Structured Pruning: Towards Compact and Accurate Self-Supervised Models for Speaker Diarization" (2025) and "Efficient and Generalizable Speaker Diarization via Structured Pruning of Self-Supervised Models" (2025).
