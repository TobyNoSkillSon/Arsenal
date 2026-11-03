# Depth Anything 3 Mono Large

Relative depth from one image.

Checkpoint: `depth-anything/DA3MONO-LARGE@f465978e618db8cc79c83b8bbf24964857db1875`. The published checkpoint is FP32; Get casts tensor by tensor to FP16. Spatial kernels, normalization and non-matrix tensors stay FP16. Quantization quality needs the approved trained-weight trial.

## Operation

`estimate_depth` returns a safetensors file and a 16-bit grayscale depth PNG per view. The PNG normalizes positive finite depth to its per-view range; safetensors retains the actual values as FP16 (any mask as uint8, 1 = valid). Invalid pixels are identified by the mask where the model predicts one. Tensor names describe depth, confidence, optional point map, normals, mask and rays as available. Depth units are relative, with no physical scale. `views` names file IDs, original image dimensions and actual depth dimensions. Depth, points, normals and companion maps with NaN, infinity outside MoGe's mask, or no positive finite depth fail with `invalid_model_output` before any file is written.

DA3 uses the maker's DINOv2/DPT configuration and ImageNet preprocessing with its default `process_res` of 504 and `upper_bound_resize`, on the patch grid (504 is the value this port uses). Mono-Large uses the single DPT head with its sky branch; as in the reference, sky pixels (probability ≥ 0.3) take the 99th percentile of non-sky depth when both regions exceed 10 pixels, and the sky map is returned.

## Qualification limits

Random graphs prove wiring and shapes only. Trained output, FP16 casting and quantization quality are unqualified. The model folder now selects its registered architecture explicitly through the merged contracts-r2 dispatch seam; the original checkpoint config remains intact.

## Provenance and no-weight checks

Swift port: `mnmly/mlx-swift-da3@496dfb1d1f0969c5dc2518aad59ea21abb1ea88a`. The family retains licences and detailed adaptations beside its imported sources. Source pins, exact byte sizes and checkpoint SHA256 are in `model.toml`.

All 406 active parameter names and shapes match the pinned header after the explicit model-local map. 0 tensors are excluded. All 406 active source tensors are FP32 and require the declared FP16 cast; a shape match is not cast-quality evidence. Exact Range responses read 48872 header bytes and no payload bytes.

FP16/int8/int4 random graphs pass every declared operation through the native worker and the production artifact validator. Int8/int4 pack every eligible matrix in the shrunk graph; norms and spatial kernels remain half. Ten family tests pass, including fixed transposed-convolution orientation, bicubic half-precision behavior, and stored int8/int4 conversion/load with nontrivial biases. Four malformed requests, cancellation and unloading pass. These checks do not establish trained output or precision quality.

Current checks use the tracked FP32 source metadata, the shared FP16/int8/int4 serving-tier enumeration, original checkpoint configs and immediate requests without pacing. Explicit per-tensor header cast maps record the F32-to-F16 Get plan; they do not establish trained cast quality. Folder architecture selects the adapter, including when the checkpoint carries an unrelated model_type.

Arsenal offers this model at FP16 only: below 1B parameters, 8-bit would save under 1 GB and run no faster.
