# MLXRFDETR

Repository: https://github.com/mnmly/mlx-swift-rf-detr
Revision: 83941f0461313cdd1991d6873e205f613ff1be31

Imported files: Backbone/Backbone.swift, Backbone/Projector.swift, Config.swift, Decoder/Decoder.swift, Decoder/DecoderLayer.swift, Decoder/DecoderMLP.swift, Decoder/DecoderSelfAttention.swift, Decoder/GridSampleKernel.swift, Decoder/KeypointInitializer.swift, Decoder/MSDeformableAttention.swift, Decoder/Transformer.swift, Decoder/Utilities.swift, Documentation.docc/MLXRFDETR.md, Model.swift, PostProcess.swift, Processor.swift, Segmentation/SegmentationHead.swift, Variant.swift, WeightMap.swift

Local adaptations: common Arsenal MLX pin; native adapter; no network/download path; streamed precision installation; bounded cancellation. Checkpoint mapping and source identity are in each model folder.

Additional adaptations: FP16 default; quantized SAM lookup tables use Embedding calls rather than raw packed storage; no unused DINO register parameter for Nano; decoder fused QKV stays one checkpoint-compatible Linear; LocateAnything uses the pinned tokenizer API and cancellation between bounded decoding steps. Tests use the exact same native adapter graphs. RF-DETR two-stage selection picks the top queries on the host in torch.topk order (ties by position), so equal FP16 scores pair with the same learned queries as the maker; sizes come from config.json.
