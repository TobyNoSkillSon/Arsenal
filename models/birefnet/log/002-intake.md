+++
hypothesis = "Stored group-64 affine Linear weights preserve the vendor model's fixed-input components at int8."
change = "Use the shared intake gate and qualification boundary for the image-removal workload."
measurement = "FP16 and int8 pass 20 saved-input outputs over twelve boundary invocations; int4 fails. Busy-machine quick serving checks completed without added swap."
verdict = "kept"
numbers = { components = 20, component_invocations = 12, added_swap_bytes = 0 }
links = ["004-native.md"]
+++

# Historical Python intake — superseded

This evidence describes the removed Python engine. It is not qualification of the native Swift model. Final native results are in gate/evidence/native and log/004-native.md.



The reference is the pinned vendor implementation on MPS at native FP16. Inputs
are captured once from a fixed synthetic 128-square image at both backbone
scales, each Swin stage, the squeeze block and the decoder. Each candidate
component independently consumes those saved inputs; changed encoder outputs do
not feed the decoder gate. All module outputs inside each component have finite,
shape/count, range and no-clamp traces. Learned tensors/matmuls stay half or
stored affine; no CPU or FP32 model executes.

The selected source is already FP16. The shared A/A format policy qualifies that
path. Int8/int4 use the shared .05 relative-L2 / .999 cosine bounds. Every
eligible large Linear is packed, including all backbone attention/MLP matrices;
convolutions, vectors/buffers and small/incompatible matrices remain half. The
coverage receipt names every tensor and its parameter share. This fixed recipe
uses `searched:false`; it makes no held-out search claim.

Warm checks use the production Load/remove_background path, native 1024-square
preprocessing and the same input image across tiers. Fresh serving workers have
zero measurement_components requests. The balanced ABBA BAAB windows, every
sample, spread, baseline drift, cold request and lifetime physical-footprint
peak remain in the exact completion record. Two resident workers peaked at
10.51 GB under a 12 GB guard; system swap was already 721,616,896 bytes and did
not increase. The Mac was in use. These are quick checks, not controlled
benchmarks or evidence of a speed gain. Int8 is slower on this path because it
unpacks each matrix before its half matmul; native integer GPU compute is not
claimed.

Independent implementation parity is not run: both integration and internal
reference use the pinned vendor engine. No arbitrary external quality score is
claimed. The runner checker and HTTP/MCP smoke are separate integration proof.

The first attempt stopped before model execution because the macOS memory probe
returned zero. Its failure remains under gate/evidence. The corrected vm_stat
headroom calculation has a tiny static regression test. Completion metadata was
reconciled once to the catalog validator after capture; original receipts and the
exact non-numerical correction are retained in that run. Weights, fixture
arrays, bounds and numerical verdicts did not change during reconciliation.
