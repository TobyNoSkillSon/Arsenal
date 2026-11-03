# MLXLocateAnything

Repository: https://github.com/mnmly/mlx-swift-LocateAnything
Revision: 5245a70e736fcc46ae1d0a8baf0732854b5ce577

Imported files: Configuration.swift, Language.swift, LocateAnything.swift, MetalBicubic.swift, PBD.swift, Pipeline.swift, Processing.swift, TokenizerSetup.swift, Vision.swift

Local adaptations: common Arsenal MLX pin; native adapter; no network/download path; streamed precision installation; bounded cancellation. Checkpoint mapping and source identity are in each model folder.

Additional adaptations: FP16 default; quantized SAM lookup tables use Embedding calls rather than raw packed storage; no unused DINO register parameter for Nano; decoder fused QKV stays one checkpoint-compatible Linear; LocateAnything uses the pinned tokenizer API and cancellation between bounded decoding steps. Pipeline.swift exposes its termination and box-grammar parse (`boxes(generated:)`) and tokenization of decoded output for the adapter’s parser fixtures. Tests use the exact same native adapter graphs.

Tokenizer files are non-weight files of mlx-community/LocateAnything-3B-bf16 b409aade8f6aae828a0c8e0f4190bf28eb8f0244, listed in the model folder's pinned weights; serving reads them from that verified snapshot. Byte-identical copies for no-download lab checks live in Workers/Perception/Tests/PerceptionTests/Fixtures/tokenizers/locateanything.
