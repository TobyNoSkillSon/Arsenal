# BiRefNet native engine

Core inference source: https://github.com/mnmly/mlx-swift-BiRefNet at
43c1f83973c6adecf6a4af9c37663f2f86f1f8c3 (MIT; LICENSE-upstream).
Also inspected xocialize/mlx-birefnet-swift at 1198aabe5f56d5fea776215483390ebb60c08cbd;
its core shares this implementation. The smaller upstream was selected.
MLX is pinned to Arsenal's 901941965d82e4a216d4d117231d847d194c563d.

Local adaptations: original checkpoint key/layout mapping on load; complete key/shape/dtype
validation; decoder image-injection channels from the pinned vendor (ipt_cha_opt=1);
finite -100 shifted-window attention masks; restore input dtype after MLX resize;
FP16 image pipeline defaults; half output rendering; independent boundary replay;
ArsenalWire load/remove/unload, cancellation, path receipts and memory bounds.
The upstream Metal source is unchanged. No kernels were written or tuned here.
The local ModelPrecision adapter now streams affine group-64 int8/int4 derivation,
installs stored packs directly, and maps eight eligible 1×1 decoder projections to
Linear over NHWC inputs. Independent saved-input replay records 20 outputs and
internal activation observations; the pipeline writes qualification and table bindings.
No third-party CLI dependency or Python serving path is used.

Vendor reference code and raw captures are retained in the local lab with their MIT notice.
The original 444473596-byte checkpoint stays unchanged on disk (the model.toml hash
is verified by Get); OIHW-to-OHWI transposes occur only in memory. 687 tensors are
FP16 in the original source. The 67 integer buffers are checked and discarded;
relative-position indices are rebuilt from configuration. Derived tiers pack all
105 eligible large matrices; spatial/deformable convolution layouts, small tensors,
norms and other vectors retain FP16. No source checkpoint is changed in place.

Review round 1: image extraction reads straight RGB before resizing, preserving the
PNG colors independent of alpha. ImageBoundary.swift follows Pillow 12.3.0's
Resample.c coefficient geometry and 22-bit fixed-point rounding (LICENSE-Pillow):
antialiased bilinear RGB input; Float16-to-byte quantization before bicubic mask
resize. Internal MLX aligned-corner feature resizes are unchanged. ImageIO PNG
finalization failures now throw. The package resolves ArsenalWire from the canonical
manifest path so the family symlink has the same dependency. Darwin free pages
already include speculative pages, which are no longer added again.
