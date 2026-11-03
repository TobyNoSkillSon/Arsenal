+++
hypothesis = "Whole-component voice tiers can be qualified without modifying pinned sources or executing FP32 models."
change = "Prepared streamed nested-archive conversion and stored Linear/Embedding loading; recorded exact qualification blockers."
measurement = "Header-only source audit; no real model executed."
verdict = "inconclusive"
numbers = { model_executions = 0, qualified_tiers = 0, blocked_tiers = 3 }
links = ["gate/evidence/intake/intake-2026-10-15T19-40-31Z-fff12f93-100e-4be3-bd79-0bbd2202602c/blocked-result.json"]
+++

# Voice intake preparation — 15 Oct 2026

All three requested tiers are blocked, not numerically rejected. The immutable result is `gate/evidence/intake/intake-2026-10-15T19-40-31Z-fff12f93-100e-4be3-bd79-0bbd2202602c/blocked-result.json`; source hashes and full tensor inventory are under `gate/evidence/intake/intake-2026-10-15T19-40-31Z-fff12f93-100e-4be3-bd79-0bbd2202602c`.

The 221 tensor descriptors contain 65 matrices eligible for the proposed affine-g64 recipe and 156 explicit skips. **No real-weight matrices have been packed or loaded.** Eligibility is a header-only plan, not proof that the production module tree accepts every pack. Large tensors that are not Linear/Embedding weights (including Pocket's standalone `speaker_proj_weight`) still need an adapter; their kept reasons are explicit. Convolutions, norms, buffers, small matrices and incompatible input widths stay half.

The default Pocket path forces FlowLM hidden states and flow noise/time inputs through FP32 into learned flow/EOS projections; preset conditioning is stored FP32. The BF16 checkpoint alone does not establish a half-arithmetic reference for those paths. No retained matched-input half reference was found in this model gate, prior Arsenal smoke location, or inspected Vireo reference locations. A half candidate cannot qualify itself.

The preparation adds an opt-in `intake-v1:<model>` converter for every declared archive, nested codecs and preset conditioning, plus stored Linear/Embedding pack installation. It does not enable a catalog recipe, replace current metadata, issue a passing quality binding, or inherit legacy transformer-only results. Breeze's tied embeddings copy all affine companions with their weight. Legacy checkpoints retain their existing loader path.

Pending implementation after the reference boundary is resolved: architecture component-capture adapters (teacher-forced saved inputs and activation traces), real-weight pack mapping including standalone projections, half selection, quantized component gates, one served-path sanity per tier, interleaved warm timings and default-workload physical peaks, and pipeline-written quality identities. No model was executed; all metric fields and bindings remain null. No CPU model run, download, install, signing, merge or push.

Source pins were hashed before and after inspecting private APFS clones; all match. Lab clones were deleted. Small fixed-tensor conversion/loader tests and root changed-target counts are reported in the worker return, not presented as voice quality evidence.

The original main checkout was also checked for untracked component/PCM artifacts: none remain in this model gate. Its historical v2 cycle JSON remains, but all referenced generated WAVs are absent. Vireo’s bundled diagnose reference retains voice summaries and only Stable Audio mel arrays, not voice component tensors. The hashes and counts are in `gate/evidence/intake/intake-2026-10-15T19-40-31Z-fff12f93-100e-4be3-bd79-0bbd2202602c/reference-audit.json`.
