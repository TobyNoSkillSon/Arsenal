# Geometry engine

Native depth adapters, behind IntegrationKit's existing framed worker and precision contracts.

- `Sources/MLXMoGe`: `mnmly/mlx-swift-MoGe@2925d7bbf821c8f7e9cb4d7c6cdc5b6061b6bce2`, MIT including its retained third-party notices. DINOv2, dense heads, geometric recovery and sparse refiner only.
- `Sources/MLXDA3`: `mnmly/mlx-swift-da3@496dfb1d1f0969c5dc2518aad59ea21abb1ea88a`, Apache-2.0. DINOv2, DPT/DualDPT, image preprocessing and geometric RayPose only. No SALAD, loop closure, CLI, application, downloader or streaming orchestration.

Each closure contains `LICENSE-upstream` and its detailed `PORTED-FROM.md`. Both resolve the common MLX pin through this package; there is no new engine or runtime package manager. Learned parameters and eligible affine companions remain FP16; integer index maps are computed buffers. Geometric binning, focal/pose solves, host file pixels and output geometry use float arithmetic separately from learned weights.

- `Sources/MLXTrellis`: `xocialize/mlx-trellis2-swift@f0b8717511fb9573b901118e65d1fd7feb1174ce`, generation closure: DINOv3, flow/attention, sparse structure/latent decoders, dual-grid extraction, cascade and PBR baking/export. Replaced raw full-precision dictionaries with named Module banks using shared streamed NativePrecisionTier and installed affine Linear modules. No MLXToolKit, downloader, app, encoder, existing-mesh texturing pipeline or optional provenance-chart path.
- `Sources/MLXMesh`: the same port's vendored mlx-swift-mesh closure (MIT, Hiroaki Yamane); mesh cleanup, remesh, simplify, BVH and geometry kernels. Replaced its Swift/C++ interop wrapper with a narrow pure C ABI for xatlas, preventing C++ mode from leaking into IntegrationKit.
- `Sources/GeometryXatlas`: that pin's xatlas.cpp and xatlas.h, MIT (Jonathan Young), plus the local owned C ABI. Progress cancellation is forwarded; all xatlas stdout is disabled to protect the wire protocol. The original xatlas source itself is unchanged.

TRELLIS pixel preprocessing follows the vendor RGBA crop/composite and DINOv3 LANCZOS resize. Its byte resampler adapts the already qualified BiRefNet Pillow-compatible helper; LICENSE-Pillow is retained. GLB export follows upstream o_voxel `postprocess.to_glb`: vertices and normals are converted from TRELLIS's Z-up frame to glTF's Y-up (`y, z = z, -y`), and the material is OPAQUE while the base-colour texture keeps the sixth opacity channel in alpha. Mesh preview is a depth-buffered raster of the returned mesh.

Upstream third-party notices are retained in each closure. The port pin has no standalone root LICENSE: `xocialize/mlx-trellis2-swift`'s own code (the source of `Sources/MLXTrellis`) carries no licence at that commit or at its current head (checked 2 Nov 2026), so Arsenal's MIT licence does not cover `Sources/MLXTrellis` and its rights stay with its author (see NOTICE); its existing notices cover mesh/xatlas; the original Microsoft model/code is MIT and DINOv3 assets retain their separate terms. No license claim for one component is applied to another.

TRELLIS source pins and no-weight limitations are in models/trellis2-4b. FP16 source conversion and quantization quality remain pending. No RMBG-2.0 code or weights are used.

- `Sources/MLXIdol`: `speridlabs/idol-sdk@4702a7f00850e34b6e83a2170cc123c3e9570744`, Apache-2.0 (`Licenses/LICENSE-idol-sdk`): IDOL's denoiser, octree and Gaussian decoders, sampler, image preparation, Gaussian activation and PLY layout, plus DINOv3 ViT-H+/16 following transformers' DINOv3ViTModel. Its FLUX.2 VAE encoder is Flux3Klein's. IDOL's weights (`speridlabs/idol`) are Apache-2.0; the DINOv3 ViT-H+ weights are Meta's, under the DINOv3 License, downloaded by each user from Meta's gated repository; the VAE weights are FLUX.2 klein 4B's (Apache-2.0). The preview rasterizer is Arsenal's own. Details in `Sources/MLXIdol/PORTED-FROM.md`.
