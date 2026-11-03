+++
hypothesis = "The repository alone describes the complete model-intake route."
change = "Record the exact gaps encountered while integrating native BiRefNet."
measurement = "Completed component and API checks; generic intake and native-family assembly required source inspection."
verdict = "inconclusive"
numbers = "No performance measurement; ten repository or integration gaps recorded."
links = ["004-native.md", "../gate/evidence/native/components.json"]
+++

# Repository discovery gaps

Final native route, 16 October 2026. Earlier Python intake observations and superseded
patches remain in commits 76aee14a and 94bb5b1a.

1. Runtime order in AGENTS.md now says Swift MLX first, but Lifecycle/Quality still
   unconditionally require three precision tiers and speed/peak-memory checks. That
   conflicts with the owner's current instruction to keep shipped FP16 weights and defer
   optimization/benchmarking. This task records a native static gate and one API smoke;
   it does not pretend to complete the old three-tier measurement procedure.
2. docs/ADDING-A-MODEL.md's generic `arsenal-stats intake --model` path has a fixed
   adapter allowlist in ModelIntakeCommand. Its Images adapter invokes FLUX generation
   with BF16/prompt/steps/seed. It cannot intake a native FP16 image-input segmenter.
   The original checkpoint and independent component/API results therefore live in
   this model's gate/evidence/native, with no invented speed or peak figures.
3. The headless real API server's documented --prepare-real route signs a QA catalog
   and has a fixed model allowlist. An unsigned model can instead be placed in an
   isolated local-models folder over the existing signed catalog in real-resources.
   The README and gate/smoke-http.py record this no-signing route.
4. The native family plumbing is spread across Workers/Native/Package.swift, its
   Main.swift dispatcher, scripts/pkg-lib.sh family list, and quality-identities.py.
   The default family list derives a title-cased path; BiRefNetEngine needs an explicit mapping to avoid SwiftPM identity collision with a worktree named birefnet. None of the model
   intake instructions enumerate all four seams as a single native-family checklist.
5. The chosen existing Swift port's default decoder does not match this HF revision:
   its injection branches use 64 channels, while the pin uses channels/8. Complete
   key/shape checks exposed this immediately. The port's MLX Upsample also promotes
   output to FP32; restoring the input dtype is needed before later convolutions.
   These are upstream integration defects, not missing Arsenal APIs.
6. Creating an empty settings.toml fails validation (requires a section); this model
   needs no settings panel and omits the file. The introductory prose treated that
   file as unconditional while the folder layout called it optional.
7. Many regression tests assume every repository model offers int8/int4 and fixed
   model/tier counts. They now name BiRefNet's task-specific FP16 exception explicitly.
8. HTTP file results contain URLs, while MCP resource URIs are attached only to
   results that pass through MCP. Reading `files[].uri` from an HTTP result failed;
   this was a lab-driver assumption, not an app defect. The single inference was
   already successful and was not repeated for a second transport.
9. RUNNERS says resource bundles belong beside native helpers; the plain debug build
   does not build the native engine. The lab used the model package's debug executable
   under the normal native helper name plus its pinned MLX resource bundle. The full
   native package build verifies the bundled link, without an app release or install.

No sibling runtime/configuration was changed. No signed catalog, release, installed
app or real user profile was touched. The old Python serving and quantization
machinery was removed; only the independent half/GPU oracle remains under gate/.

10. CatalogResultValidation requires a three-tier measured-intake run even for
    availability provenance in schema-2 stats. A static component/API gate cannot
    be represented there without inventing a timing intake. BiRefNet's table record
    therefore explicitly uses notMeasured; correctness remains in its gate evidence.
    Research logs accept kept/rejected/inconclusive, not passed, and require their
    front matter before every heading; those details were found in validation code.
