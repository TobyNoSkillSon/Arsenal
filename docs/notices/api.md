# API worker notices

The HTTP multipart parser is ported from Vella `Sources/VellaCore/API/Multipart.swift`, source commit recorded in its file header. Upstream Vella is MIT (relicensed by its author, 2 Oct 2026); its provenance is preserved. No sibling project is mutated or linked. HTTP security/loopback policy follows Vella/Vireo but the streaming transport and dialect adapters are Arsenal code.

Conformance-only dependencies (not runtime requirements of the Swift library): OpenAI Python 2.6.1 — Apache-2.0; Anthropic Python 0.72.0 — MIT; ElevenLabs Python 2.21.0 — MIT; Replicate Python 1.0.7 — Apache-2.0; httpx 0.28.1 — BSD-3-Clause; Pillow 11.3.0 — MIT-CMU. Full transitive package versions and artifact hashes: `tests/conformance/requirements.lock`. Python fixtures use these libraries only against loopback.
