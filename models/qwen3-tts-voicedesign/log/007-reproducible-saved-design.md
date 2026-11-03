+++
hypothesis = "A saved selected preview can be reproduced after a worker restart by preserving its design seed and instructions."
change = "Record the actual random-or-requested seed, model revision, dtype, mode, preview text/language and retained audio; restore the saved seed and description on speech; allow name in the model design schema."
measurement = "Bounded gpulock functional proof, local copied weights, bf16 Fast; existing signed catalog reused without re-signing."
verdict = "kept"
numbers = { selected_preview = 2, exact_wav_matches = 2, worker_mb_after_unload = 0, network_download_bytes = 0 }
links = ["docs/evidence/mb-v2-fix-f/qwen-saved-design.json", "docs/evidence/mb-v2-fix-f/qwen-saved-design.log"]
+++

ElevenLabs generated three previews without an explicit seed. Preview #2 was saved using `text-to-voice`; same-text saved speech and same-text speech after unload/reload both matched the retained native WAV byte-for-byte (SHA256 `86799cc8384620820605e78a337b957511cebed07a78c4096016a16574c32db5`). The chosen description, generated seed, revision and audio survive transient preview cleanup. No cloning model was needed.

This proves replay for the same text, language, revision, dtype and mode, not subjective quality, voice consistency for arbitrary new text, or cross-dtype equivalence. Older saved description-only voices remain readable but cannot acquire the lost original seed; legacy preview tokens without a recorded recipe must be redesigned. No benchmark or Measured badge.
