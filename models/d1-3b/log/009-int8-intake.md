+++
hypothesis = "D1 int8 can preserve BF16 component quality and decision probabilities while storing all eligible matrices as packed weights."
change = "Add streamed affine g64 int8; expose saved-input component intake for both quantized tiers and bind table results to the current implementation."
measurement = "M5 Max, 40 GPU cores, 15 Oct 2026; busy Mac. 93 component outputs and three text probability vectors. Default served workload: one 33-token refund decision, warm ABBA BAAB."
verdict = "kept"
numbers = { int8_max_relative_l2 = 0.01946498516749375, int8_max_dp = 0.0008521080017089844, int8_peak_gb = 3.493153552, int4_max_relative_l2 = 0.24333150765468853, int4_max_dp = 0.003509223461151123, int4_peak_gb = 2.16996468 }
links = ["gate/evidence/quality-binding-2026-10-15.json", "gate/evidence/precision-coverage-2026-10-15.json"]
+++
# Int8 and int4 intake — 15 October 2026

The native d1 worker now converts, loads and reports affine g64 int8 as well as int4. Both derive from the pinned BF16 checkpoint. Conversion reads and releases one tensor at a time; loading consumes stored packed matrices directly, including embedding rows and fused projections. Learned tensors stay BF16 or packed. The small option-logit readout retains FP32 intermediates to preserve near ties; position grids are stored BF16, with unchanged transient interpolation arithmetic.

Fresh intake: `intake-2026-10-15T16-48-03Z-48cf8e71-86cc-44e0-b18f-e222beb40293`. M5 Max, 40 GPU cores, Mac in use. See the run JSON, immutable fixture manifest and `quality-binding-2026-10-15.json` for worker/source/recipe/bounds identities. This qualifies quantization against the existing BF16 implementation; it is not a new independent vendor qualification.

The saved fixture covers a single-question chain and a shared-state two-question tree, every language block (30), every vision block (27), embeddings, projector and option logits: 93 component outputs. Each block independently replays its original BF16 input. Vision uses a deterministic 4×4 patch grid, not a full photograph. Three text-question probability vectors are also evaluated end to end; all chosen answers remain unchanged. This is a component check and a narrow text probability check, not a broad task-accuracy or perceptual claim.

| Tier | Max component relative L2 | Min cosine | Max absolute probability delta | Warm ms/decision | Default-workload peak GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0 | 1 | 0 | 18.98994 | 5.89744 |
| int8 | 0.019465 | 0.999811 | 0.00085211 | 18.13221 | 3.49315 |
| int4 | 0.243332 | 0.970985 | 0.00350922 | 16.85177 | 2.16996 |

Int8 passes relative L2 ≤0.05, cosine ≥0.999, finite outputs/activations, no new clamps and max |dp| ≤0.017. Int4 fails the component bound (largest error: projector); its probabilities alone would pass. Int4 remains offered with its failed-quality warning. The original text-only BF16 catalog peak was 6.032 GB; the fresh paired run above is the timing baseline. All figures are quick checks. Warm four-sample ABBA BAAB medians are reported for each quant against its own paired baseline; baseline drift was −0.116% for int8 and +0.086% for int4. Spreads overlap and the busy Mac does not establish a reliable speed winner.

Default workload: one served refund decision, 33 input tokens. Component-probe workers are discarded before fresh timing workers start, so vision residency and capture buffers cannot inflate the default-workload peak. Peaks include Load/self-tests and the operation. This separation is also a useful procedure for the shared inference skill; updating that System skill is outside this worker's permitted edit scope.

Coverage is identical for int8/int4: 305 matrices / 2,988,652,544 parameters packed. Retained BF16: 374 nonmatrices (664,816 parameters), one position lookup grid (294,912), and 27 vision down-projection matrices (133,871,616) whose K=4304 is not divisible by group 64. These large skips preserve the existing int4 route; padding or changing group arithmetic was not introduced. Every tensor and skip reason is in `precision-coverage-2026-10-15.json`.

The intake ran for 49.29 seconds under `scripts/mlock gpulock`, with a 46.91 GB aggregate worker cap; observed aggregate peak 10.02 GB. Existing system swap was 1.03167 GB and did not grow. Private donor/weight/cache clones were removed on exit. No installed weights or app were changed, no downloads, signing, merging or installation.

The catalog audit covers all 15 shipping manifests, plus the retained chat fixture. All declare their half format, int8 and int4 after adding d1 int8. Breeze's int4 is its existing pinned vendor MXFP4 route, not a missing derive recipe. Other models were not live-run or newly qualified. Unchecked int8 is unavailable; int4 stays offered with its warning. Prior deferred whole-component limitations in other families remain unqualified.
