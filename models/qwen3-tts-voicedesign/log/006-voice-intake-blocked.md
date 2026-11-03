+++
hypothesis = "Whole-component voice tiers can be qualified without modifying pinned sources or executing FP32 models."
change = "Prepared streamed nested-archive conversion and stored Linear/Embedding loading; recorded exact qualification blockers."
measurement = "Header-only source audit; no real model executed."
verdict = "inconclusive"
numbers = { model_executions = 0, qualified_tiers = 0, blocked_tiers = 3 }
links = ["gate/evidence/intake/intake-2026-10-15T19-40-26Z-6942630b-3527-451a-8c5f-95cfcca4a8ba/blocked-result.json"]
+++

# Voice intake preparation — 15 Oct 2026

All three requested tiers are blocked, not numerically rejected. The immutable result is `gate/evidence/intake/intake-2026-10-15T19-40-26Z-6942630b-3527-451a-8c5f-95cfcca4a8ba/blocked-result.json`; source hashes and full tensor inventory are under `gate/evidence/intake/intake-2026-10-15T19-40-26Z-6942630b-3527-451a-8c5f-95cfcca4a8ba`.

The 900 tensor descriptors contain 377 matrices eligible for the proposed affine-g64 recipe and 523 explicit skips. **No real-weight matrices have been packed or loaded.** Eligibility is a header-only plan, not proof that the production module tree accepts every pack. Large tensors that are not Linear/Embedding weights (including Pocket's standalone `speaker_proj_weight`) still need an adapter; their kept reasons are explicit. Convolutions, norms, buffers, small matrices and incompatible input widths stay half.

The pinned speech codec contains FP32 learned parameters. Its original implementation cannot be executed under the no-FP32 rule. No saved original component tensors/PCM with matched inputs were found in this model gate, the prior Arsenal smoke location, or the inspected Vireo reference locations. Casting the codec to either half format and comparing it to itself would not qualify the cast. SAME-L acceptance is specific to that decoder and is not voice-codec acceptance.

The preparation adds an opt-in `intake-v1:<model>` converter for every declared archive, nested codecs and preset conditioning, plus stored Linear/Embedding pack installation. It does not enable a catalog recipe, replace current metadata, issue a passing quality binding, or inherit legacy transformer-only results. Breeze's tied embeddings copy all affine companions with their weight. Legacy checkpoints retain their existing loader path.

Pending implementation after the reference boundary is resolved: architecture component-capture adapters (teacher-forced saved inputs and activation traces), real-weight pack mapping including standalone projections, half selection, quantized component gates, one served-path sanity per tier, interleaved warm timings and default-workload physical peaks, and pipeline-written quality identities. No model was executed; all metric fields and bindings remain null. No CPU model run, download, install, signing, merge or push.

Source pins were hashed before and after inspecting private APFS clones; all match. Lab clones were deleted. Small fixed-tensor conversion/loader tests and root changed-target counts are reported in the worker return, not presented as voice quality evidence.

The original main checkout was also checked for untracked component/PCM artifacts: none remain in this model gate. Its historical v2 cycle JSON remains, but all referenced generated WAVs are absent. Vireo’s bundled diagnose reference retains voice summaries and only Stable Audio mel arrays, not voice component tensors. The hashes and counts are in `gate/evidence/intake/intake-2026-10-15T19-40-26Z-6942630b-3527-451a-8c5f-95cfcca4a8ba/reference-audit.json`.
