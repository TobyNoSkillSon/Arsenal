# MLXTrellis

Source: xocialize/mlx-trellis2-swift@f0b8717511fb9573b901118e65d1fd7feb1174ce.

Generation and mesh closure only. Adaptations use shared streamed NativePrecisionTier, FP16 neural arithmetic and native affine Linear modules. No engine, network client, encoder, app, or runtime weight downloader. See the geometry family provenance and model README.

Flow transformer blocks, sampling steps, remeshing slices, simplification iterations, UV rasterization chunks and atlas inpainting rings call the cancellation checkpoint. The preview samples the atlas at row `v·size`, the same orientation the bake writes and glTF export reads.
