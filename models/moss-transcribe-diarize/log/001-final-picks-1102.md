+++
hypothesis = "OpenMOSS-Team/MOSS-Transcribe-Diarize runs in the audio worker as an MLX port of the maker's remote code, writing the maker's transcript token for token."
change = "Added the Whisper-Medium encoder, VQ adaptor, processor (chunks, token counts, time markers, prompt), greedy generation with chunked prefill, the maker's transcript parser, transcribe and diarize ops, a BF16→FP16 Get cast, the golden and tests."
measurement = "The maker's remote code (snapshot 10304d92) with its GitHub helpers (79c675f0), transformers 5.22.1, torch 2.15.0 on MPS in FP16, captured once on a 29 s and a 58 s two-speaker recording; the port per component and end to end; one real run through the isolated service."
verdict = "kept"
numbers = { features_relative_l2 = 0.00014, whisper_relative_l2 = 0.0076, adaptor_relative_l2 = 0.0030, transcript_tokens = 184, transcript_token_identical = 1 }
links = ["README.md", "gate/golden/golden.json"]
+++

# Final picks, 2 Nov 2026

Runtime choice (AGENTS.md order): no bundled runner served this architecture. The audio worker already had a Qwen3 text decoder with KV cache (Qwen3-ASR's, from mlx-audio-swift) and Whisper's log-mel frontend (Smart Turn's); this port reuses both and adds a Whisper-Medium encoder (Smart Turn's encoder classes are private and sized for Whisper-tiny), the 4× merge and VQ adaptor, the maker's processor and parser.

Source: the pinned revision is the current `10304d92` (the planned folder's 30 Oct pin was `704aa4a9`; the coordinator's download took the newer one). The planned folder in planned-models/moss-transcribe-diarize is superseded by this one.

Reference: the maker's README runs the model in float32 on CPU and bfloat16 on CUDA; the reference here is the same code with the weights in float16 on MPS, as the fingerprint design asks. It transcribed the fixture perfectly.

Bugs found while porting: none in the numerics. Build fixes: an optional closure parameter is escaping in Swift, so streaming takes a flag and a non-escaping closure.

Not checked: real recordings, languages other than English (a Polish clip is in the real check, as a probe), recordings longer than a minute, speed and memory.
