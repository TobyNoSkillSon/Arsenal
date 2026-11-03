+++
hypothesis = "nvidia/diar_streaming_sortformer_4spk-v2 runs in the audio worker as an MLX port of NeMo's SortformerEncLabelModel, matching NeMo's own streaming output."
change = "Added the Sortformer v2 network, NeMo frontend, streaming driver and post-processing, a .nemo Get conversion, diarize and diarize_session, the golden and tests."
measurement = "NeMo 3.0.0 on MPS (FP16 network, float32 frontend) captured once on a 27.5 s three-voice clip; MLX port compared per component and end to end; one real run through the native worker on an isolated profile."
verdict = "kept"
numbers = { features_max_abs = 0.000396, pre_encode_relative_l2 = 0.00152, encoder_relative_l2 = 0.0061, head_max_abs = 0.0020, cache_step_spkcache_relative_l2 = 0.0, card_probability_max_abs = 0.0022, low_probability_max_abs = 0.0093, stress_probability_max_abs = 0.0716, stress_probability_mean_abs = 0.0035, weights_downloaded_by_worker = 0 }
links = ["README.md", "gate/golden/golden.json"]
+++

# Final picks, 2 Nov 2026

Runtime choice (AGENTS.md order): no bundled runner served this architecture. The audio family already had the speech-swift Sortformer pieces for Nemotron 3 (cache updater, config, mel frontend), but Nemotron 3 is a different network (RoPE transformer, 8 speakers, 10 ms output). mlx-audio-swift (0dda61b2, MIT) has a Swift Sortformer port written for the mlx-community v2.1 conversion; its modules were adapted here and corrected against NeMo 3.0 source: NeMo's Transformer encoder has no positional embedding (mlx-audio adds sinusoidal ones), its streaming loop is NeMo's `forward_streaming` (chunks with left/right context) rather than mlx-audio's own chunking, and the frontend is NeMo's FilterbankFeatures (preemphasis 0.97, n_fft 512, no normalisation). The cache update reuses `SortformerStateUpdater` from the Nemotron folder unchanged; it was compared step by step with NeMo's `streaming_update`/`_compress_spkcache` and with the captured state (speaker cache identical after a compressing step).

Weights: the maker's `.nemo` (tar of model_config.yaml and an FP32 torch checkpoint). Get reads the checkpoint in place (TorchCheckpoint gained a byte-range init), keeps NeMo's tensor names, transposes convolutions and casts to FP16 (235.5 MB). 117,711,368 learned parameters, matching the card's 117M.

Reference: NeMo 3.0.0 + torch on MPS, network `model.half()`, frontend float32. Two capture fixes were needed and are recorded in the golden: NeMo creates the empty streaming state in float32, which a half network cannot concatenate (cast to half, no logic change), and `model.half()` had rounded the frontend's filterbank before it was restored to float32 (re-captured).

Bug found and fixed during parity: the port first fed float32 probabilities to the cache scoring; NeMo scores the network-dtype (half) probabilities. With the 32-frame stress cache this moved some frames by up to 0.087; using half-rounded probabilities brings it to 0.072 (bound 0.08, mean 0.0035), and the heavy parity test keeps that bound.

Not checked: DER on a real benchmark, real recordings with four speakers, languages other than English, the app-level Get/Load path (arsenal-testserver's real mode serves a fixed id list that does not include this model; the real check drove the shipped worker binary directly), speed and memory figures (none claimed).
