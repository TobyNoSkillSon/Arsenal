# Voice Gender Classifier (ECAPA-TDNN)

Labels the speaker in a recording as male or female, with the probability of each label: a 15.5M-parameter ECAPA-TDNN speaker network with a two-class head, fine-tuned on VoxCeleb2.

Input: `audio`, one audio file (wav, mp3, m4a or flac); it is resampled to 16 kHz mono. Output: `label`, `probabilities` (male and female) and `duration_s`.

Example request:

```json
{"audio": {"path": "/absolute/path/to/clip.wav"}}
```

Limits: two labels only, as the VoxCeleb training data labels them; the card warns the data may not represent the global population. One speaker per clip works best. Clips under 0.03 s are refused, and silence scores male at 0.81, so read the label with its probabilities. The card's 98.7% accuracy was not measured here, no language was tested (Polish included), and parity with the maker's code has not been run.

Source: [JaesungHuh/voice-gender-classifier](https://huggingface.co/JaesungHuh/voice-gender-classifier/tree/db1222153bd60337e900be22add7af180452adc0), MIT. Runs as FP16 through the Arsenal kit runner (MLX).
