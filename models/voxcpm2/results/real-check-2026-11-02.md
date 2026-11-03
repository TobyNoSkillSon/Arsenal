# Real check, 2 Nov 2026

One run through Arsenal's real service on an isolated profile (`arsenal-testserver --real`, private QA catalog from the
committed folder at 80f759b5, repeated at 4dab3af3 after the review fixes with all six WAVs byte-identical, Hugging Face client on loopback so Get could only import the local copy), under the GPU
lock, headless. M5 Max on mains power, Mac in use. No speed or memory figures are claimed. Intelligibility was judged by
Qwen3-ASR-1.7B (the installed app's FP16 snapshot, APFS-cloned into a temp profile and run in the Audio worker binary
with the language set), not by listening.

- Get: the two BF16/FP32 safetensors cloned from the local Hugging Face cache and cast to FP16 (`cast:voxcpm2`, 4.96 GB snapshot); Load at fp16.
- `speak`, Polish, no reference (seed 7), 4.16 s: "Dzień dobry. Jutro rano jedziemy pociągiem do Poznania na spotkanie z klientem."
- `speak`, English, no reference, 6.08 s: "Good morning. Tomorrow we take the early train to the city for a meeting with a client."
- `speak`, Polish, cloned from a 5 s clip (macOS voice Zosia) without its transcript, 4.16 s: "Dzień dobry, jutro rano jedziemy pociągiem do Poznania na spotkanie z klientem."
- `speak`, Polish, the same clip with its transcript (reference + continuation), 4.48 s: "Dzień dobry. Jutro rano jedziemy pociągiem do Poznania na spotkanie z klientem."
- `speak`, English, voice described as "A calm older man, deep voice", 5.28 s: "Good morning. Tomorrow we take the early train to the city for a meeting with a client."
- `speak`, English, cloned from a 4 s clip (macOS voice Samantha) with its transcript, `cfg_value` 2.5, `inference_timesteps` 12, 4.96 s: "Good morning. Tomorrow we take the early train to the city for a meeting with a client."
- Every transcript is word for word; the only difference is a comma or full stop after "Dzień dobry". The maker's own FP16 outputs (zero-shot Polish and English, Polish clone, Polish ultimate clone) transcribe the same way.
- `clone_voice` with the Polish clip and transcript saved a voice; `voices` lists it.
- `clone_mode: continuation` without a transcript is refused with 422 `invalid_request`. The first two runs found it returned 500 (first as `engine_error`, then as `invalid_input`, which the service does not map); both fixed with a test.
- No swap added (2,093 MB before and after); no leftover processes.
- Second run at c6f213ba (chained requests):
  - Saved voice: `clone_voice` (Polish clip with transcript), then `speak` with the returned voice id on seed 7, gave a WAV byte-identical to the same clip and transcript passed inline.
  - Long text: an 842-character Polish paragraph (three chunks of at most 400 characters) gave 52.6 s in a fresh voice and 47.8 s in the saved voice. ASR, saved voice: word for word except "siódmej" heard as "siedmiej" twice. Fresh voice: five words off ("siedmiej" twice, "odjeżdza", "Miastie", "obu drzecie" for "o budżecie"). The maker's own code reading the same paragraph in one call, with no chunking, slips the same way: "Miastie" in the same place, "siódmej" heard as "7" or "8", "telefonom", and a stray "Tak." at the end. These are the model's own Polish weak spots, not chunk joins: no join shows dropped or repeated words.
- Streaming: through the voices worker binary on the Get snapshot, `stream: true` and the file output (`out_dir`) on the same seed gave byte-identical PCM, for one chunk (4.16 s) and three chunks (52.6 s). The service's ops endpoint always uses the streamed path.
- Not checked: whether the cloned and designed voices sound like the clip or the description (ASR checks words, not timbre); real recordings as references; texts near the 5,000-character cap; the app's own Get and Load (headless rule).

Raw result, outputs and transcripts: lab/evidence/voxcpm2 (`real-check-result.json`, `real-check-out/`, `asr-real-check.json`).
