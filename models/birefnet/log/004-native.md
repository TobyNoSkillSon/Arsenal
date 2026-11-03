+++
hypothesis = "The existing Swift MLX port can run the pinned official FP16 checkpoint without changing its stored weights."
change = "Replace Python serving with the MIT Swift MLX engine in the bundled native worker; adapt the pinned decoder dimensions, checkpoint mapping and resize dtype."
measurement = "12 independent saved-input component invocations, 20 finite FP16 GPU outputs; one lab HTTP screenshot mask; native wire checks. Performance and quantization deferred by the owner."
verdict = "kept"
numbers = { component_calls = 12, component_outputs = 20, maximum_relative_l2 = 0.006828, screenshot_width = 904, screenshot_height = 694 }
links = ["../gate/evidence/native/components.json", "../gate/evidence/native/api.json", "../runners/swift/PORTED-FROM.md", "003-repository-discovery.md"]
+++

Source revision e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4; original F16 checkpoint
SHA256 9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154.
The native candidate is independent of the pinned vendor PyTorch/MPS implementation.
The existing vendor fixture capture was reused, not regenerated to fit the port.
Full/half-resolution patch embedding and each Swin stage, squeeze and decoder were
replayed from their own saved vendor inputs. No candidate component feeds another.
The decoder relative L2 is 0.000435; the largest stage difference is 0.006828.
This is half arithmetic agreement, not a labelled segmentation-accuracy claim.

The first strict load rejected the upstream default's 64-channel injection branch;
the pin actually uses channels/8. A dtype assertion then caught MLX's resize output
promotion to Float32. Both have permanent regression tests. The initial smoke is
superseded; the final saved component tensors are all Float16. Upstream Metal source
is unchanged. No quantized checkpoint or performance figure belongs to this result.

One native default-path HTTP inference produced the transparent foreground and mask.
MCP exercised discovery, local-folder digest approval, Add/Get/Load; HTTP returned
both artifacts, whose byte counts and SHA256 were verified. A separate no-inference
check confirmed operation discovery and invalid-input rejection. HTTP does not attach
MCP resource URIs; the driver's failed attempt to read one did not repeat inference.

The lab was busy with normal Mac activity. Timings are test execution costs only,
not model latency benchmarks. Model allocations were capped below available memory,
with a swap-growth/deadline watchdog. The final serving image was 904×694; its
100% visual sweep is recorded in api.json. No installed profile, app or window changed.

Final verification: ArsenalCoreTests 511/511 (35.240 s), ArsenalRunnerCheckTests
23/23 (19.531 s), model package 5/5 (0.061 s), bundled wire checks 18/18,
quality-identity source closure 4/4. These are test costs on a busy Mac, not
model performance figures. Verification hashes are in ../gate/evidence/native/verification.json.

The generic intake queue previously seeded undeclared quantized precisions; it
now follows model.precisions, with the seed-matrix test checking all 49 declared
repository jobs. Catalog admission required exact log front matter and kept/rejected/
inconclusive verdicts. Static table provenance could not reference the new gate
without a measured intake, so table cells explicitly remain notMeasured.
