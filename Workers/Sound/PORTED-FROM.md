<!-- SPDX-License-Identifier: MIT -->
# Sound provenance

Copied from Vireo commit `9366da5e14dcafe846fc41565c94f4f145eaf7cc`, never modified in place:

- `Sources/VireoEngine/StableAudio3Engine.swift` → `Sources/SoundEngine/StableAudio3Engine.swift`.
- `Sources/VireoEngine/Enhance/**` → `Sources/SoundEngine/Enhance/**`.
- `Sources/VireoEngine/Vendor/StableAudio3/*.swift` → `Sources/StableAudio3/*.swift`; original third-party headers retained. Derived from soniqo/speech-swift `2af9c6c` (Apache-2.0), with Vireo's SAME-S/MLX adaptations.
- FastPath, KernelGate and RuntimeCaches → SoundEngine; gate child registry reduced to RE-USE, support files use Arsenal directories, Exact explicitly disables the inexact fp16 residual stream and RE-USE Metal kernels.
- Quantization → SoundEngine (sound/enhancement rules and generic helpers only); AudioMetrics and Sound/Enhance request protocols extracted from VireoCore.
- Framing and PCM adapters adapted from Arsenal VoicesWorker; no HTTP, UI, playback, model-server or process polling code.

Vireo-owned adaptations follow the V-family MIT licence; upstream SpeechLens and speech-swift code stays Apache-2.0. Notices are in `licenses/`. **Model terms are independent of Arsenal's MIT licence.** Stable Audio 3 weights: Stability AI Community License plus Gemma Terms; RE-USE: OpenMDW-1.1 (NVIDIA's repository `nvidia/NVIDIA-NemotronLabs-RE-USE` at `31f82f2e`, text in `licenses/RE-USE-NVIDIA-OpenMDW-1.1.txt`); it superseded the earlier NVIDIA NSCLv1 non-commercial terms.

## Kept paths

SA3 Fast uses the gated fp16 residual stream; Exact uses gated fp32 copies only where reference arithmetic promoted weights. Medium also keeps bit-exact strided decoder windows. Every path retains the source's self-test and sticky non-finite-output fallback. RE-USE Fast uses gated Metal selective scan/causal convolution; Exact uses stock MLX. No new speed claim is made here.

Runtime accepts standard load fields, per-model settings, and dtype. Generic op schemas live in the model/kind folders. Sound streams contain PCM16 **after** whole-clip generation (not low-latency incremental diffusion). WAVs preserve stereo; enhancement preserves input channels/rate unless explicitly upscaled.

Model instructions declare the native packaged engine; accepted kernels currently remain embedded in that engine, not independently replaceable folder Metal. The folder-kernel runtime adapter is a separate team delivery and must not be inferred from these ports.

## MossFormer2 SE 48K

- `Sources/MossFormer2SE/MossFormer2Layers.swift` and `MossFormer2Network.swift`: from Blaizzy/mlx-audio-swift `0dda61b29c308175e302dceef77c3bd84551e92b` (`Sources/MLXAudioSTS/Models/MossFormer2SE`, MIT, `licenses/mlx-audio-swift-MIT.txt`). Changed: the intra-block norm is GroupNorm(1) as in the maker, not a per-frame LayerNorm; maker epsilons; float32 norm statistics with one rounding; dtype-safe constants; float32 positions; no Hugging Face download or STS protocol.
- `Sources/MossFormer2SE/MossFormer2DSP.swift` and `MossFormer2Weights.swift`: written for Arsenal after the maker's ClearerVoice-Studio `6b3774dc` (Apache-2.0): Kaldi fbank, STFT/iSTFT, the windowed decode and the checkpoint name map.
- `Sources/SoundEngine/Checkpoint/TorchCheckpoint.swift`: copy of the Audio family's non-executing PyTorch checkpoint reader at `32db88da`.

## UniPASE

- `Sources/UniPASE/` and `Sources/SoundEngine/Enhance/UniPASE/UniPASEEngine.swift`: written for Arsenal after the maker's Xiaobin-Rong/unipase `857b60ad45d37a2cf6d7a89883ec9fc4fc164b45` (MIT; its `licenses/` folder carries the adapted parts' licences: Cisco's PASE and ESPnet's TF-GridNet, Apache-2.0; Microsoft's WavLM and WavTokenizer's Vocos, MIT). The weights are Apache-2.0 per the model card. No upstream code is copied: the networks, torchaudio's sinc resampler, the Vocos ISTFT, the PostNet's STFT and crossfade, and the inference scripts' long-form segmentation and peak matching are re-implemented on MLX and checked against the maker's code (`models/unipase/gate/golden/`). The cast reads the maker checkpoints with `TorchCheckpoint.swift`.

## DeepFilterNet3

- `Sources/DeepFilterNet3/DeepFilterNetConfig.swift`, `DeepFilterNetDSP.swift`, `DeepFilterNetLayers.swift`, `DeepFilterNetForward.swift`: from Blaizzy/mlx-audio-swift `0dda61b29c308175e302dceef77c3bd84551e92b` (`Sources/MLXAudioSTS/Models/DeepFilterNet`, MIT, `licenses/mlx-audio-swift-MIT.txt`). Removed: version 1, the streamer, CPU feature loops and the Accelerate GRU. Changed: half-dtype folded batch norm and transposed-conv weights.
- `DeepFilterNetModel.swift` (libdf STFT, features, deep filter, post filter, attenuation limit) and `DeepFilterNetKernels.swift` (Metal GRU and running mean): written for Arsenal after the maker's Rikorose/DeepFilterNet `d375b2d8` (MIT or Apache-2.0).
