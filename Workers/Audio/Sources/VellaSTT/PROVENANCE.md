# VellaSTT provenance

Vella's dictation runtime for Parakeet TDT and Qwen3-ASR, copied from Vella `6d311195eb83cbcd033071b03bdf0cd5d45fa807` (`Projects/Vella`, MIT, `Workers/Audio/LICENSE-upstream-Vella`). Vella's `Worker/Sources/MLXAudioSTT` started as mlx-audio-swift `01dec7c9bdce3088a6b6b7ab9f2e403458195efb` (MIT, `Workers/Audio/LICENSE-upstream-mlx-audio`); Parakeet's frontend follows mlx-audio 0.5.1's Python filterbank (MIT, `Workers/Audio/LICENSE-upstream-mlx-audio-python`). Vella's pins of mlx-swift, mlx-swift-lm and swift-transformers equal Arsenal's, so the code compiles unchanged against the one native graph.

| Folder here | Vella source | Contents |
|---|---|---|
| `MLXAudioCore/DSP.swift` | `Worker/Sources/MLXAudioCore/DSP.swift` | windows, STFT, mel filters |
| `SmallMGEMM/SmallMGEMM.swift` | `Worker/Sources/SmallMGEMM/SmallMGEMM.swift` | small-M tile, GEMV and native quantized tile kernels |
| `Parakeet/` | `MLXAudioSTT/Parakeet/` minus `ParakeetRuntime.swift` | loader, TDT decoding, fused encoder, compiled TDT decoder, kernels |
| `Qwen3ASR/` | `MLXAudioSTT/Qwen3ASR/` minus `Qwen3ASRRuntime.swift` | audio tower, Qwen3 text decoder, pipelined decode |
| `Shared/` | `MLXAudioSTT/Shared/{Generation,STTOutput}.swift`, `Shared/Nemo/` | generation types, NeMo layers, alignment |
| `Gate/` | `MLXAudioSTT/Gate/{FastPath,KeptLevers}.swift` | the optimized-path protocol and Vella's kept levers |
| `Precision/` | `MLXAudioSTT/Precision/` | Vella's load-time derived precision (unused by the Arsenal adapters) |

Not copied: Whisper, Nemotron, the runtime registry, `FastPathGate` (Vella's child-process self-test with persisted verdicts), the Vella workers and app. `ArsenalShims.swift` stands in for the two names the copied code uses from them: VellaWire's `Recipe` and a silent `FastPathGate.debug`, plus `KeptLevers(enabled:)`.

## Local edits

- Imports of `MLXAudioCore`, `SmallMGEMM` and `VellaWire` removed: those files are in this module. `MLXAudioCore.computeMelSpectrogram` is called unqualified.
- `ParakeetModel`: `keptLevers` is public; `fromConfig(_:)` builds the graph from config.json without weights (`fromDirectory` now calls it); `parameterName(checkpointKey:)` exposes the checkpoint key remap; `lastResult` keeps the aligned tokens and sentences of the last `generate` for timestamps.
- `Qwen3ASRModel.loadTokenizer(_:)`: the tokenizer construction of `fromModelDirectory`, unchanged, as its own public function; its synthesis no longer prints the generated path, because stdout carries the worker protocol.
- `ParakeetVocabulary.read(from:)` no longer defaults to Vella's support directory.
- `Qwen3ASRModel.decoderFiniteProbe(_:)`: the optimized decoder's per-step finiteness check sums the logits in f32. Vella sums them in the logits' dtype, which is safe at BF16 but overflows at FP16 (Qwen3-ASR logits sum to about ±1.5M), so every FP16 step read as non-finite.

Arsenal loads weights through IntegrationKit's precision tier, not `fromDirectory`/`fromModelDirectory`. The adapters (`AudioWorker/VellaASRAdapters.swift`) choose Vella's levers explicitly: the exact components on Metal family Apple 9, tail block sizing for Parakeet v3 Ultra, the inexact GEMM components off.
