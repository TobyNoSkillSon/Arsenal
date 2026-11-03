+++
hypothesis = "DeepFilterNet3 can join the Sound family as an enhancement model from the mlx-audio-swift port, checked against the maker's own PyTorch code."
change = "None landed. Parked on 2 Nov 2026 after the MossFormer2 SE 48K port used this worker's time box; groundwork recorded below."
measurement = "No model run. Archive fetched and its Git blob hash checked; maker package install checked with a dry run; candidate Hugging Face sources listed from the API."
verdict = "inconclusive"
numbers = { archive_bytes = 7986207 }
links = []
+++

State on 2 Nov 2026 (final-picks batch B). Nothing here is a model yet. (Moved from planned-models/ when the model landed; see 002-port.md.)

What is ready:

- The maker's checkpoint: Rikorose/DeepFilterNet `models/DeepFilterNet3.zip` at `d375b2d8309e0935d165700c91da9de862a99c31` (main, 17 Oct 2024), 7,986,207 bytes, SHA-256 `49c52edc8947ae1f9bf50d81530beaf3a2c3245aeaf34b6f31ff535cd22284d2`, Git blob `ea53bac00d064378451d0201d573f44f05f1d746` (matches GitHub). It holds `DeepFilterNet3/checkpoints/model_120.ckpt.best` (8,714,073 bytes) and `config.ini` (sr 48000, fft 960, hop 480, nb_erb 32, nb_df 96, df_order 5, df_lookahead 2, conv_lookahead 2, lsnr -15..35). It was fetched into a scratch folder that this worker deleted at the end; refetch it the same way.
- The maker reference installs: `deepfilternet==0.5.6` with `deepfilterlib==0.5.6` resolves on Python 3.11 arm64 (needs torch and torchaudio beside it). Run it with `-I -B` from a venv outside the download folder, on MPS in half precision.
- The Swift port: Blaizzy/mlx-audio-swift `0dda61b29c308175e302dceef77c3bd84551e92b`, `Sources/MLXAudioSTS/Models/DeepFilterNet` (about 3,300 lines: config, DSP, layers, forward, a streamer). It loads `mlx-community/DeepFilterNet-mlx` `v3/` safetensors.

What blocks a quick landing:

1. Weights source. Arsenal's Get pins Hugging Face files only; the maker publishes DeepFilterNet3 on GitHub, not on Hugging Face. Candidates (API listing, 2 Nov 2026): `mlx-community/DeepFilterNet-mlx@220d5dfb7266352272d74c2a7d2025c59e07b391` `v3/model.safetensors` (8,682,709 bytes, card licence MIT, converted by the port's author with `convert_deepfilternet.py`), the same file in `iky1e/DeepFilterNet3-MLX`, and `fal/DeepFilterNet3@efd275812b477d9e3cbf7e830d7d5123576918b9` `model.safetensors` (8,684,112 bytes, card Apache-2.0). None is in the local Hugging Face cache; each is under 10 MB. Needs the coordinator's choice and download. Recommended: the mlx-community `v3/` file, with the parity run against the maker's own checkpoint proving the conversion.
2. The port's GRUs run their recurrence on the CPU with Accelerate (`vDSP_mmul` in DeepFilterNetLayers.swift and DeepFilterNetForward.swift). Arsenal serves models on MLX only, so those loops must become MLX (MLXNN GRU or an MLX step loop) before it ships.
3. FP16 check: weights are FP32 (about 2.2M parameters); cast to FP16 and check its range on the maker's capture, BF16 only if FP16 fails.

Next steps, in order: get the weights into the Hugging Face cache; capture the maker reference once (libdf STFT and ERB/complex features with their running normalisation, encoder, ERB decoder mask, DF decoder coefficients, deep-filter output, waveform) on a fixed 1 s 48 kHz clip under `lab/evidence/deepfilternet3/`; vendor the port into `Workers/Sound/Sources/DeepFilterNet3/` as MossFormer2 SE was, replacing the CPU GRU with MLX; add `models/deepfilternet3/` with `atten_lim_db` (the maker's attenuation limit, `df.enhance(..., atten_lim_db=...)`), the `post_filter` option and 48 kHz resampling declared in the ops schema; goldens for features and deep filtering without weights; then the heavy parity and one real cycle.
