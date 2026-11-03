# MLXTIPS

Repository: https://github.com/mnmly/mlx-swift-tips
Revision: 222f5fa5133b6ee9272f2878fd84b7ec3cff8a17

Imported files: SentencePieceTokenizer.swift, TIPSModel.swift, TextEncoder.swift, Tokenizer.swift, VisionTransformer.swift

Local adaptations: common Arsenal MLX pin; native adapter; no network/download path; streamed precision installation; bounded cancellation. Checkpoint mapping and source identity are in each model folder.

Additional adaptations: FP16 default; quantized SAM lookup tables use Embedding calls rather than raw packed storage; no unused DINO register parameter for Nano; decoder fused QKV stays one checkpoint-compatible Linear; LocateAnything uses the pinned tokenizer API and cancellation between bounded decoding steps. Tests use the exact same native adapter graphs.

`SentencePieceTokenizer.swift` is copied from the existing Arsenal MLXAudioCore port: Blaizzy/mlx-audio-swift 01dec7c9bdce3088a6b6b7ab9f2e403458195efb via Vireo 9366da5e14dcafe846fc41565c94f4f145eaf7cc; source terms in LICENSE-tokenizer. `tokenizer.model` is a non-weight asset from google/tipsv2-b14 ed1e4dc6b74bf3935ae099e9d5eb30fa96528454 (same tokenizer for L/14). The worker reads it from the model folder's pinned weights; the copy used by no-download lab checks is Workers/Perception/Tests/PerceptionTests/Fixtures/tokenizers/tipsv2, retained under the model source Apache-2.0 terms.
