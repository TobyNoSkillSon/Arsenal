# Vireo TTS port

Copied from the sibling `../Vireo` project, commit `9366da5e14dcafe846fc41565c94f4f145eaf7cc`:

- `Sources/VireoEngine/{MLXSpeech,FastPath,Quantization,KernelGate,RuntimeCaches}.swift`.
- `Sources/VireoEngine/Vendor/{MLXAudioTTS,MLXAudioCodecs,MLXAudioCore,SmallMGEMM}` (DACVAE omitted).
- Speech-only subset of `Sources/VireoCore/{Generators,Tiers,DiagnoseReport}.swift`.
- `Resources/models.json` and `voices.json` metadata. Full upstream model metadata is retained in `metadata/models.json`; the shipped `Sources/VoicesWorker/Resources/models.json` contains only the five fields decoded by `SpeechSpec` (tested against the retained source). Developer pin/import/smoke scripts use the full metadata.

Changes: remove non-TTS engines; route vendored logs to stderr; persist gates in Arsenal support, not Vireo; frame stdio instead of HTTP/playback. Numerical engine and quantization paths retained. Kernel child inherits the lock-owning worker's process group.

Vendor sources: mlx-audio-swift MIT `01dec7c9bdce3088a6b6b7ab9f2e403458195efb`; SmallMGEMM from Vella Apache-2.0 `40510bdcd0d00f283d3bdb6f656707cadeff7535`. Licence texts in `Licenses/`, Vireo licence in `LICENSE`.

## Fish Audio Drama 3

- `Sources/FishS1Codec/`: the Fish S1 DAC from mlx-audio-swift MIT `01dec7c9bdce3088a6b6b7ab9f2e403458195efb` (`Sources/MLXAudioCodecs/FishS1DAC`), rewritten to match the mlx-audio Python codec MIT `dad2087f790dfe96b07d5aac1405db4b578b411f` (`mlx_audio/codec/models/fish_s1_dac`) key for key, with its streaming decode ported. Changes: `lab/drama3-port/codec-notes.md`.
- `Sources/FishDrama3/`: our port of mlx-audio `mlx_audio/tts/models/fish_drama3` at the same commit. The backbone adapts mlx-swift-lm MIT `bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57` `Libraries/MLXLLM/Models/{Qwen35,Qwen3Next}.swift` to take frame embeddings, with float32 zero-centred norms as in the reference.
- Weights and the model itself are under the Fish Audio Community License 1.0 (`models/fish-drama-3/model.toml`); no voice clip ships.

## OpenBMB VoxCPM2

- `Sources/VoxCPM2/`: mlx-audio-swift MIT `e8450402ca60780a4532829eac92bbb2e57cfb6c` (`Sources/MLXAudioTTS/Models/VoxCPM2/{MiniCPM,VoxCPMEncoder,VoxCPMDiT,VoxAudioVAE,VoxCPM2Config,VoxCPM2Model}.swift`), checked line by line against the maker's OpenBMB/VoxCPM Apache-2.0 `cce58a5` (`src/voxcpm/model/voxcpm2.py`, `modules/{minicpm4,locenc,locdit,audiovae}`); the generation loop, prompt layouts, length cap and bad-case retry follow the maker's code. Licence texts: `Licenses/mlx-audio-swift-MIT.txt`, `Licenses/VoxCPM-Apache-2.0.txt`.
- Changes from the Swift port: the stride-5 AudioVAE encoder conv pads 2·padding − output_padding as the maker does (the port shifted the encoding by one sample); sample-rate buckets follow `torch.bucketize`; continuation cloning and reference + continuation (the port had zero-shot and reference only); prompt text and target tokenized as one string; the maker's length cap and seed + 1 retry; activations kept in FP16 (the port promoted to FP32 through float32 time embeddings, masks and noise); rotary embedding in float32 cast back, MLX's fused causal mask; no quantization or Hub loading (the worker loads the FP16 Get cast); the protocol adapted to VireoCore `SpeechModel`.
- Weights: mlx-community/VoxCPM2-bf16 (Apache-2.0, `models/voxcpm2/model.toml`), no voice clip ships.
