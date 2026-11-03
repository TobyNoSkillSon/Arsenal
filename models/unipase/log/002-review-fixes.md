+++
hypothesis = "The fresh review's findings on the UniPASE port are real and each can be fixed without changing the parity-checked networks."
change = "Bounded the sinc kernel (AVAudioConverter for rates it cannot reduce, in and out), refused unknown request options, moved the per-channel and long-form plumbing behind an injectable segment function, checked cancellation before each network stage, computed the Hann window in torch's float32 arithmetic, corrected the licence lines (maker code MIT, weights Apache-2.0), the long-form comment and the PostNet's rates, and declared the implicit and upscale rates in the op schema."
measurement = "New default tests for each item; the Sound default suite; the parity test rerun; a heavy FP16 range check on a minute of near-silence with a full-scale burst."
verdict = "kept"
numbers = { sound_default_tests = 46, L24_rel_l2 = 0.0068, adapter_rel_l2 = 0.0017, vocoder_rel_l2 = 0.0069, postnet_rel_l2 = 4.0e-4, end_to_end_stft_magnitude_rel_l2 = 0.022, fp16_range_max_abs = 657, traced_tensors_finite = 32 }
links = ["README.md", "../../Workers/Sound/Tests/SoundTests/UniPASETests.swift"]
+++

Each finding was checked against the code and the maker's repository before the fix: 44,099 Hz → 16 kHz would have built a kernel of 16,000 × 44,133 floats (2.8 GB); the repository root LICENSE is MIT, with Apache-2.0 (PASE, ESPnet) and MIT (WavLM, WavTokenizer) files under `licenses/`; the long-form comment described the two branches the wrong way round; 22.05 kHz output also runs the PostNet.

Results (2 Nov 2026, M5 Max, Mac in use): the Sound default suite passes (46 tests). Parity is unchanged on the rerun (L24 0.0068, adapter 0.0017, vocoder 0.0069, PostNet 4.0e-4; end-to-end STFT magnitude 0.022). The new FP16 range check, one 60 s pass to 48 kHz of 1e-5 noise with a 1 s full-scale clipped burst, keeps all 32 traced tensors finite; the largest is 657 (the PostNet's deconvolution), against FP16's 65,504.

The float32 Hann window matches the window saved in the maker's vocoder checkpoint (torch.hann_window) in all but 41 of 1,280 values, at most 6e-8 off; the double formula it replaces differed in 820 values, up to 1.8e-7. It does not widen the Vocos ISTFT margin, though: the weight-free golden moves from 9.12e-6 to 9.19e-6 relative L2 (bound 1e-5). That residual comes from the FFT arithmetic, not the window.
