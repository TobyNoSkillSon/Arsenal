# Smart Turn v3

Smart Turn v3 is Pipecat's open semantic turn detector: it listens to the end of a stretch of speech and says whether the speaker has finished their turn, judging from the raw audio (grammar, tone and pace) rather than from a transcript. It is the encoder of Whisper Tiny with a linear classifier on top, about 8 million parameters. The pinned snapshot is a safetensors conversion of the makers' unquantized GPU checkpoint (`smart-turn-v3.2-gpu.onnx`). Its strength is a single fast decision on whether a voice agent should reply now or keep listening.

## What it is for

The makers built it for voice agents that must decide when to respond, and optimised it for fast single-pass inference alongside a lightweight voice-activity detector such as Silero, so it only runs when speech pauses.

- Deciding when a voice assistant should answer, instead of waiting for a fixed silence timeout
- Not cutting off someone who pauses mid-sentence ("I can't seem to, um ...")
- Endpointing dictation or live transcription; Arsenal's realtime `semantic_vad` turn detection calls this same operation

## Inputs and outputs

### `end_of_turn` (default)

Scores whether the speech at the end of an audio file is a finished turn.

| Field | Type | Required | Notes |
|---|---|---|---|
| `audio` | file object `{"path": string}` | yes | Path 1–4096 characters. Accepts `audio/*`. Arsenal types the file by extension: `.wav`, `.flac`, `.mp3`, `.m4a`, `.aiff`, `.caf`; others (`.aif`, `.ogg`, `.opus`, video files) are refused. macOS AVFoundation decodes it. |
| `threshold` | number | no | 0–1, default 0.5. `prediction` is true when `probability` is above it. |

Returns JSON: `probability` (number, 0–1), the model's probability that the turn is complete, and `prediction` (boolean), `probability > threshold`.

Audio: the model takes 16 kHz mono PCM and reads exactly 8 seconds. Arsenal converts any file it accepts to 16 kHz mono Float32 (channels averaged), then keeps the final 8 seconds, or pads a shorter clip with silence at the beginning so the speech sits at the end, as the makers specify. It then normalises the clip to zero mean and unit variance and computes an 80-bin log-mel spectrogram (400-sample FFT, 160-sample hop, 800 frames). Files up to 600 seconds are accepted; only the last 8 seconds affect the result. The makers recommend sending the whole current turn, from where the user started speaking, and re-running on the whole turn if more speech arrives; earlier turns are not needed.

## Run it

MCP: `arsenal_run(model:"smart-turn-v3", input:{"audio":{"path":"/absolute/path/turn.wav"},"threshold":0.5})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/smart-turn-v3/ops/end_of_turn" -H 'Content-Type: application/json' -d '{"audio":{"path":"/absolute/path/turn.wav"}}'`

A path may name any file on this Mac; Arsenal clones it into its inbox before the model reads it. `arsenal_upload` is only for a file that is not on this Mac. In a quick check on an M5 Max (22 Oct 2026, Mac in use), a warm request took 0.03 s and the worker peaked at 0.22 GB.

## Limits

- Uses only the final eight seconds of audio.
- Not a voice-activity detector: it assumes the clip contains speech and is meant to run after a VAD reports silence. It is not designed for very short audio segments.
- The makers list 23 languages: Arabic, Bengali, Chinese, Danish, Dutch, German, English, Finnish, French, Hindi, Indonesian, Italian, Japanese, Korean, Marathi, Norwegian, Polish, Portuguese, Russian, Spanish, Turkish, Ukrainian and Vietnamese. On their v3 test set accuracy ranges from 97.1% (Turkish) and 94.3% (English) down to 84.1% (Bengali) and 81.3% (Vietnamese). Those figures were published for v3.0; the v3.2 checkpoint served here may differ.
- No text conditioning: it cannot be told that the user is reading out a phone number, card number or address.

## Licence and credit

BSD-2-Clause. Commercial use, modification and redistribution are allowed; redistributions must keep the copyright notice, the licence conditions and the disclaimer. Made by Pipecat (Daily); contributors listed by the makers are Marcus, Eli, Mark and Kwindla, with audio datasets from Liva AI, Midcentury and MundoAI. The MLX safetensors conversion is by mlx-community. The makers request no specific citation.

## Links

- Model card (pinned conversion): https://huggingface.co/mlx-community/smart-turn-v3/tree/74af1538ebace37b0fcc667a6a8a3d902d73eee0
- Original model card: https://huggingface.co/pipecat-ai/smart-turn-v3
- Code: https://github.com/pipecat-ai/smart-turn
- Release post: https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/

## How Arsenal runs it

Checkpoint: `mlx-community/smart-turn-v3@74af1538ebace37b0fcc667a6a8a3d902d73eee0`. The selected snapshot's config names `smart-turn-v3.2-gpu.onnx` as its source; it is not being silently described as a different revision. Native port: `Blaizzy/mlx-audio-swift@244458d7b61e54ff135073ddc9ed76782bb20758`. Exact files, SHA-256 digests and byte counts are in model.toml; only metadata and safetensors headers were fetched during integration.

The release is FP32; Get casts it to FP16. FP16 is the only offered precision: below 1B parameters, 8-bit would save under 1 GB and run no faster. Checked live at FP16 on 19 Oct: end-of-turn probability 0.98 for a complete sentence, 0.02 for a sentence cut mid-word, and 0.97 for a phrase cut after a whole word.

Verification: the no-weight fixture constructs a shrunk version of the actual graph and uses the ordinary worker operation. This verifies wiring and shapes, not trained output, speed, memory, or numerical fidelity. Native source closure and shared-host prerequisites are recorded in `Workers/Audio/PORTED-FROM.md` in the source repository. The 22 Oct quick check (load 1.39 s, warm request 0.028 s, peak 206 MiB summed over Arsenal's worker processes) is in `results/`; accuracy against the makers' ONNX model is not yet measured. Research log: `log/001-integration.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Before Pipecat PR #3857 (merged 2 Mar 2026), 8 kHz audio cut accuracy from 98.0% to 60.0% in one commenter's 100-sample test. [pipecat-ai/pipecat #3844, 2026-02-26, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not found in sources checked.
- Parameters people recommend: Speko's lowest false-cutoff threshold at 85%+ END-recall; no fixed value. [Speko benchmark, 2026-08-07, first-hand]
- Arsenal call: `arsenal_run(model:"smart-turn-v3", operation:"end_of_turn", input:{"audio":{"path":"/absolute/path/turn.wav"},"threshold":0.5})`

**Good and bad**
- Good at: Real English turns: 94.0% accuracy and 1.0% false cutoffs on 200 clips (v3.2). [Speko benchmark, 2026-08-07, first-hand] Maker's English test: 94.31% on 2,846 samples. [Daily announcement, 2025-09-11, first-hand]
- Bad at: Synthetic TTS turns: 62.4% accuracy and 57.7% false cutoffs. [Speko benchmark, 2026-08-07, first-hand] Second-stage Incomplete precision: 52% to 59%. [pipecat-ai/smart-turn #41, 2026-07-08, first-hand]
- Against the obvious alternatives: On real clips v3.2 beat LiveKit Intl (87.0%); on TTS it trailed Turnsense (98.8%). [Speko benchmark, 2026-08-07, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: 8 kHz audio, as above. Fix or workaround: PR #3857 resamples to 16 kHz (merged 2 Mar 2026); a commenter's retest went from 59% to 94%. [pipecat-ai/pipecat #3844, PR #3857, first-hand]
- Complaint or known bug: 512 complete turns predicted Incomplete in one dataset. Fix or workaround: none shown; open. [pipecat-ai/smart-turn #41, 2026-07-08, first-hand]
- Complaint or known bug: Short Hindi replies ("haan", "hmm") end turns early. Fix or workaround: none; the maintainer says "hmm" was in training, maybe pronounced differently; open. [pipecat-ai/smart-turn #18, 2025-07-16, first-hand]

**Evidence**
- Strength: thin. Few third-party tests.
- Sources:
  - Daily announcement, https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/, 2025-09-11, first-hand
  - Speko benchmark, https://benchmarks.speko.ai/blog/voice-agent-turn-taking-benchmark, 2026-08-07, first-hand
  - pipecat #3844 and PR #3857, https://github.com/pipecat-ai/pipecat/issues/3844, first-hand
  - smart-turn #41, https://github.com/pipecat-ai/smart-turn/issues/41, first-hand
  - smart-turn #18, https://github.com/pipecat-ai/smart-turn/issues/18, first-hand
