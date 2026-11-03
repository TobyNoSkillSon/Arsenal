# Fish Audio Drama 3

Expressive speech in the voice of a short reference clip, with inline delivery cues and up to 16 speakers. Built with Fish Audio.

## Source

`mlx-community/fish-audio-drama-3-4bit` at `cc639e27aba7edb73429d3e3806f471dcd79d316`: routed experts 4-bit, other projections and the audio tower 8-bit (MLX affine, group 64), converted from Fish Audio's block-FP8 `fishaudio/drama-3`. Norm gammas and the codec are float32 on disk; the runner casts the codec to FP16 in memory. The 16-bit release, `mlx-community/fish-audio-drama-3-bf16` (72 GB), is listed as the source but has no runner here.

Licence: Fish Audio Community License 1.0. Free for research and non-commercial use; commercial use only under US$500k annual revenue; no hosted speech or cloning service without a Fish Audio licence; never imitate an identifiable person's voice without consent; anyone given cloning must be bound by the same terms. Show "Built with Fish Audio" wherever it is distributed.

## Run

Get (imports the 4-bit snapshot from the Hugging Face cache when present) → Load → speak through the native voices worker. A request clones `reference = { path, text }` (your own clip, up to 30 seconds, with its exact transcript), uses a saved voice from `clone_voice`, or, with neither, draws an unprompted voice. There are no preset voices, no speed control and no instructions: cues go inline, e.g. `[whispers]`, speakers as `<|speaker:0|>` to `<|speaker:15|>`. Output is 44.1 kHz mono, streamed. Sampling defaults follow the reference recipe: temperature 0.7, top-k 30, top-p 0.7, 1,024 frames per chunk.

## Checked

Fixed-tensor parity against mlx-audio `fish_drama3` dad2087f on the same weights (M5 Max, 22 Oct 2026, Mac in use): every backbone layer replayed on the reference's own inputs agrees to rel-L2 ≤ 0.014; teacher-forced decode keeps per-step semantic-logit rel-L2 under 0.045 over 32 steps with no drift, and P(EOS) matches. Reference-clip codes agree 100% at float32 and 93% at the FP16 runtime codec. Quick check, not a benchmark: about 25.5 GB peak footprint, about 1.2× real time for a 4 s sentence. Details: `log/001-native-port.md`.
