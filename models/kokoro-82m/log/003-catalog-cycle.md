+++
hypothesis = "Kokoro can complete Arsenal’s signed catalog Get through the native downloader, store and real models service."
change = "Added a bounded service-cycle proof; materialized unchanged voice files and excluded inherited conversion metadata from int8 derivation."
measurement = "Fresh HF Get/cast, signed folder installation, FP16 and int8 load/speak/unload, both Fast and Exact, under gpulock with a 180-second alarm."
verdict = "kept"
numbers = { configurations = 4, failures_after_fix = 0, audio_seconds_each = 3.25, worker_memory_mb_after_unload = 0 }
links = ["gate/catalog-cycle-proof.json", "Workers/Voices/Proof/Sources/voices-get-proof/Main.swift", "docs/notes/voices.md"]
+++

The proof uses CatalogTrust signing/verification, LocalGitCatalogSource, ModelFolderInstaller, RealModelsService, native HuggingFaceClient, WorkerPrecisionConverter and the real voices worker. It does not seed the Get store from sibling weights or replace any production component with a fake. All offered precisions (FP16 and int8) completed in Fast and Exact mode; each produced a 3.25-second WAV with a verified SHA-256 and returned worker residency/memory to zero on unload. This is a functional cycle, not a benchmark or listening/ASR quality gate.

The first actual Get exposed two integration details absent from direct worker smokes: checkpoint derivation’s voice symlinks violate Arsenal’s regular-file store, and the FP16 conversion.json was inherited read-only into int8. The worker now materializes the unchanged voice files and excludes prior conversion metadata; no numerical rule changed. The retry reused verified FP16 assets and completed int8. Evidence and persistent isolated weights live in Arsenal-dev, not the real Arsenal support directory.
