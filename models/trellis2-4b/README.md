# TRELLIS.2 4B

Generate a textured GLB and a 512-pixel mesh preview from one image. The material includes base color, metallic, roughness and opacity. The operation declares the host BiRefNet remove_background pre-step and selects its foreground PNG. BiRefNet must be added with installed weights; missing dependencies return dependency_not_installed without an automatic download. The host cleans up its intermediate cutout after the target finishes or fails.

The adapter independently requires a background-removed RGBA image. Direct worker calls reject RGB, opaque, gray-alpha and empty-mask inputs with a clear remove_background error. The host passes its cutout through this same image-path input. No RMBG-2.0 code or weights are used.

`image_to_3d` supports voxel resolutions 512, 1024 and 1536; the latter two run the vendor's 512-to-1024-flow cascade, with the vendor token-count backoff. The default is 1024, 12 Euler steps per stage, a 1024 texture atlas and a 120,000-face target. The schema exposes a seed, steps, resolution, texture size and face target. Outputs are `mesh` (GLB with embedded base-color/opacity and metallic/roughness PNGs) and `preview` (PNG rasterized from that mesh). Output files share the host's 64 MiB aggregate limit.

The graph runs DINOv3 conditioning, sparse-structure flow/decoder, shape flow/decoder and texture flow/decoder in MLX. Pixel decoding, sparse index bookkeeping, xatlas UV charts and preview rasterization use native host code. Neural parameters and activations use FP16, with float accumulation for normalization and geometric arithmetic. Precision exception: the DINOv3 residual stream and its LayerScale stay float32, beyond the FP16-only rule's accumulator allowance. The trained encoder reaches about 153,000 from the first block, past FP16's 65,504; in FP16 the condition became NaN and no voxel was occupied. With the float32 residual the condition matches the vendor capture at cosine 0.9999996 (log/002-dinov3-float32-residual.md). There is no Python serving process, auxiliary engine or runtime downloader.

## Checkpoints and precision

Primary weights: `microsoft/TRELLIS.2-4B@af44b45f2e35a493886929c6d786e563ec68364d`, ungated, MIT. All generation weights are safetensors; the other shipped files are their JSON configs and pipeline.json. Unused shape/texture encoders are excluded from Get. Five flow checkpoints are BF16, two latent decoders are FP16. Get streams a cast into a complete FP16 base snapshot, then derives group-64 affine int8/int4 packs from that base. All nine components participate. No dense model is reconstructed from stored packs; convolutions, norms and unaligned small matrices remain FP16.

The structure decoder is `microsoft/TRELLIS-image-large@25e0d31ffbebe4b5a97464dd851910efc3002d96`, `ckpts/ss_dec_conv3d_16l8_fp16.safetensors`, with mixed FP16 convolution and FP32 normalization tensors.

DINOv3 comes from Meta's gated original `facebook/dinov3-vitl16-pretrain-lvd1689m@ea8dc2863c51be0a264bab82070e3e8836b02d51`. Each user accepts Meta's DINOv3 License on its Hugging Face page (Meta approves each request) and Arsenal downloads it with the user's token, as for FLUX 3 Klein; until then the Get waits on the gate. Its model.safetensors SHA256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179` is the official file's: it reproduces the repository's published LFS pointer exactly. The licence text ships as LICENSE-DINOv3.md in this folder and in the app's Licenses. Its FP32 tensors are cast individually to FP16 at Get.

Component pins, byte sizes and SHA256s are in model.toml. Both are `[[asset]] into` components: Get stages them only while converting the FP16 base, writes them to components/trellis-structure-decoder and components/trellis-dinov3, and keeps no separate FP32 copy; int8/int4 pack them with the rest, and Load reads only those converted copies.

## No-weight verification and limits

The independent architecture inventory matches all 4,265 required checkpoint tensor names and shapes. Five BF16 flows contain 3,200 tensors; 453 auxiliary tensors are FP32 and require the planned FP16 cast. The remaining 612 are FP16. Exact HTTP Range requests read checkpoint headers only. This is structural evidence, not FP16 conversion-quality evidence. gate/header-check.json records component counts and eligibility: 1,350 matrices containing 6,900,707,136 of 7,786,541,462 parameter values are eligible for int8/int4 packing.

The fixture shrinks widths/blocks and physical grids while exercising every stage, subdivision, cascade, remesh, full xatlas unwrap, PBR bake, GLB export and mesh preview. Positive random topology biases keep random graphs nonempty; no model output is substituted with a fixture. All nine resolution/precision combinations pass the release worker and production artifact validation. All 85 eligible matrices in the tiny graph are packed in both quantized tiers; all other learned tensors remain FP16. Ten family tests pass, including squared-norm overflow, nested component conversion, direct stored-pack loading and nonzero bias order against an independent scalar result. Ten malformed requests, in-flight cancellation in shape sampling and mesh construction, immediate request handoff and clean unload are checked. The shared host fixture tests additionally check dependency_not_installed for absent/uninstalled preparation models, foreground-path replacement, rejection of opaque or wrong-ID outputs, and intermediate cleanup. These are synthetic host-chain tests; the trained BiRefNet-to-TRELLIS chain awaits the approved weight trial.

Trained output, BF16/FP32-to-FP16 numerical quality, precision quality, production-size topology/memory and visual fidelity remain unqualified until the approved weight trial. No speed or memory figures are claimed. Mesh-conditioned retexturing, arbitrary mesh encoding and animation/video rendering are not exposed by image_to_3d; those require a separate mesh-input operation contract. The default cleanup/remesh can close open surfaces, so the vendor's arbitrary-topology claim is not a guarantee of this exported mesh.

Swift port: `xocialize/mlx-trellis2-swift@f0b8717511fb9573b901118e65d1fd7feb1174ce`. The family vendors only its generation/mesh/xatlas closure on Arsenal's shared MLX pin. See Workers/Geometry/PORTED-FROM.md.

## Usage notes (2 Nov 2026)

TRELLIS.2 4B turns one photo of an object into a textured 3D mesh (GLB with base colour, metallic, roughness and
opacity) plus a preview image. `image_to_3d` takes `image`, `resolution` (512, 1024 or 1536), `texture_size`,
`target_faces`, `steps` and `seed`. Arsenal first removes the background with BiRefNet, so BiRefNet must be added and
downloaded. The seed matters: on a portrait, seed 0 gave a frontal figure and seed 1 a turned one, so try another seed
when the result is off. Results count against the 4 GiB per-job file limit. TRELLIS.2 is MIT; its DINOv3 image
encoder is under Meta's DINOv3 License (no ITAR or other trade-controlled uses); the catalog shows that line.

Parity: no maker reference (Microsoft's code needs CUDA). At seed 0 the 2 Nov mesh matches the 22 Oct one in size
and shape; `results/real-check-2026-11-02.md`, golden in `gate/golden/`.

