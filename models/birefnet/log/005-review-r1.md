+++
hypothesis = "The native integration preserves vendor image boundaries and can generate a catalog while local reference tensors remain ignored."
change = "Fix straight RGB extraction, Pillow-compatible resampling, catalog shipping enumeration, package resolution, PNG finalization and memory accounting; correct smoke coverage prose."
measurement = "Exact-byte Pillow fixtures and real partially transparent PNG extraction; model package regressions; complete changed suites and one lab screenshot rerun."
verdict = "kept"
numbers = { image_resize_cases = 5, half_mask_cases = 1 }
links = ["../gate/evidence/native/image-boundary-policy.json", "../runners/swift/Tests/BiRefNetTests/Fixtures/boundaries.json", "../README.md"]
+++

M1: ImageIO's straight RGB/RGBA bytes are extracted before drawing or resizing;
partially transparent and alpha-zero colors remain independent of alpha. Other
CGImage formats are color-converted and explicitly unpremultiplied. Existing
foreground alpha multiplication is retained. The fixed PNG goes through ImageIO,
the real RGB extraction and normalizer, rather than a hand-supplied byte function.

M2 (worker part): CatalogCommand now reads and compares only the shipped model
closure against HEAD. Excluded gate captures are not read or required in Git.
Tests preserve untracked safetensors captures and generated SwiftPM metadata,
reject changes to tracked runtime files, and exclude untracked source files. README, operations, pinned source manifest and native sources are
prepared for catalog generation. Final catalog signing, release assembly, budget
and installation remain coordinator-owned and were not performed here.

S1: The pinned README uses torchvision Resize's Pillow bilinear/antialias input,
then ToPILImage's byte quantization followed by Pillow's default bicubic mask
resize. Swift now matches those external image boundaries. Five checkerboard,
edge, non-square and partial-alpha cases plus a Float16 mask fixture require
exact byte equality; the oracle and equality policy were saved before candidate
execution. Internal feature interpolation remains aligned-corner MLX in FP16.
The original 20 component captures still qualify those unchanged model components.

S2: ArsenalWire resolves from the real Package.swift location for both direct and
Workers/BiRefNetEngine symlink entry paths, matching the existing FLUX recipe.
S3: The shared PNG writer checks CGImageDestinationFinalize and throws on failure;
a real rejecting CGDataConsumer and a directory destination test both error paths.
N1: Reclaimable memory is free plus inactive pages; speculative is already in free.
N2: The README now says MCP discovery/approval/Get/Load and HTTP artifact retrieval.
It makes no MCP inference-result or resource-read coverage claim.

The real unsigned catalog build initially exposed an ignored SwiftPM .build marker
inside the runner. Catalog reads now use the intersection of tracked HEAD paths
and the shipping closure. The expanded regression retains that marker and an
untracked source file while reading only committed shipping bytes. The final
unsigned sequence-26 proof contains 16 models; all three local tensor captures
retain their hashes. The original catalog was restored and never signed.

Final checks: Core 511, API 89, Bench 3, RunnerCheck 24 and App 421 passed through
the requested --changed command. The model package passed 10 tests; bundled wire
checks passed 18 and source-identity checks passed 4. One new native HTTP screenshot
inference passed, with MCP discovery/approval/Add/Get/Load and HTTP artifact retrieval.
The complete receipts are verification-r1.json, catalog-r1.json and api-r1.json
under gate/evidence/native. Lab weights were verified unchanged and deleted;
the three ignored reference captures remain in place.
