+++
hypothesis = "OpenBMB VoxCPM2 runs in the voices worker as a native MLX port (mlx-audio-swift e8450402, corrected against the maker's code), in FP16, matching the maker's components and generation on fixed tensors."
change = "Added the VoxCPM2 module (MiniCPM-4 LMs, LocEnc, LocDiT with CFG-Zero* Euler, AudioVAE V2), the maker's four prompt layouts, length cap and bad-case retry, a cast:voxcpm2 FP16 Get cast with a range check, the voices-worker load/speak/clone_voice/voices path, weight-free tests and a heavy parity suite, the folder and golden."
measurement = "The maker's code (OpenBMB/VoxCPM cce58a5) with the maker's weights (openbmb/VoxCPM2 32279eff), torch 2.15.0 on MPS, everything including the AudioVAE in float16, captured once on a 5 s Polish clip; the port per component and a teacher-forced replay of the maker's 28-step generation; one real run through the isolated service."
verdict = "kept"
numbers = { vae_encode_rel_l2 = 0.0047, vae_decode_rel_l2 = 0.0013, locenc_rel_l2 = 0.00095, lm_prefill_rel_l2 = 0.0014, dit_rel_l2 = 0.0015, cfm_rel_l2 = 0.0013, teacher_forced_worst_rel_l2 = 0.0131, stop_decisions_agree = 28, final_audio_rel_l2 = 0.0012 }
links = ["README.md", "gate/golden/golden.json", "results/real-check-2026-11-02.md"]
+++

# Final picks, 2 Nov 2026

Runtime choice (AGENTS.md order): no bundled runner served VoxCPM2; mlx-audio-swift has a Swift MLX port (option 2) at e8450402, the last commit before its move to mlx-swift 0.33. The voices worker vendors an older mlx-audio-swift (01dec7c9), so the port is its own module (`Workers/Voices/Sources/VoxCPM2`) behind VireoCore's `SpeechModel`, special-cased in the Runner like Drama 3. Served weights: mlx-community/VoxCPM2-bf16; its 577 language-model tensors are bit-identical to the maker's checkpoint and its 234 AudioVAE tensors equal the maker's weight-norm-fused FP32 values (max 1.8e-7). Get casts all of it to FP16; FP16 passed every check, so BF16 is not kept.

Bugs found in the Swift port, each fixed with a test:
- The AudioVAE's stride-5 encoder conv padded 6 samples instead of the maker's 2·3 − 1 = 5, shifting every reference encoding by one sample (`strideFiveCausalConvPadsLikeTheMaker`).
- Sample-rate buckets used ≥ where torch.bucketize uses > (`sampleRateBucketsFollowTorchBucketize`).
- No continuation cloning: a reference transcript was ignored. Added continuation and reference + continuation, the maker's default for a clip with its transcript (`promptLayoutsMatchTheMaker`).
- Prompt text and target were tokenized separately; the maker tokenizes them as one string, and the ids differ at the seam ("książki.Dzień"). Found by the parity run (last LM hidden 11.9% off before, 1.0% after; `tokensMatchTheMaker`).
- Activations were silently promoted to FP32 by float32 masks, noise and time embeddings; all now stay in the weight dtype.
- No length cap or bad-case retry: the port ran to 8,192 patches when the stop head missed. Now min(6 × target tokens + 10, max_tokens), retried with seed + 1 up to three times, as the maker's pipeline does.

Reference note: the maker's code forces float32 on MPS, saying half precision there glitches the diffusion loop. The capture overrode that (VOXCPM_MPS_DTYPE=float16) as the fingerprint design asks. The components agreed with the MLX port within 0.1–1.3%, and the maker's own FP16 generation stopped normally (28 steps).

Not ported: the maker's ZipEnhancer denoiser (a separate ModelScope model) and its wetext text normalizer; streaming per patch (audio is emitted per text chunk of about 400 characters).

Not checked: speed and memory, long texts, real (non-synthetic) reference recordings, languages other than Polish and English.

Review fixes (fresh Opus review, 2 Nov), each with a test in VoxCPM2Tests or VoxCPM2ParityTests:
- Text opening with "(" and no ")" crashed the worker on a force unwrap; the description prefix is now parsed safely (`descriptionPrefixNeverCrashes`).
- The chunker knew only Latin and CJK sentence marks and never cut a long sentence; it now splits at Devanagari, Burmese, Arabic, Urdu and Khmer marks, then at spaces, then mid-run, and every chunk's prompt plus length cap must fit the 8,192-token context before anything is generated (`chunkerHandlesEveryScript`).
- Chunk k's retry seeds collided with chunk k+1's; chunk seeds are now well separated (`chunkSeedsAreIndependent`).
- Preset voices, whitespace-only or over-long text, clips under 0.5 s (speak and clone_voice) and over-long prompts were engine errors (HTTP 500); now invalid_request (422) (`unservableRequestsAreRefusedAsInvalidRequest`).
- Text is capped at 5,000 characters, an estimate of what finishes within 590 s from the real check's timing.
- Get's cast copied every file in the snapshot and left partial files on failure; it now copies only model.toml's pinned files and removes what it wrote on failure or cancellation (`castCopiesOnlyPinnedFilesAndCleansUp`).
- A long-chunk FP16 range check of the AudioVAE decoder (60 s): stage peaks at most 17, audio within 0.48% of FP32 (`audioDecoderStaysInFP16RangeOnALongChunk`).

Further checks (coordinator, 2 Nov): saved-voice speech, a three-chunk Polish paragraph and stream versus file output (results/real-check-2026-11-02.md). The voices family's request checks now return invalid_request (HTTP 422) for every model (RequestErrorCodeTests).
