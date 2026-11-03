# Voices

Port from Vireo (Apache-2.0), recorded in `Workers/Voices/PORTED-FROM.md`. Vendored mlx-audio-swift MIT and V-family SmallMGEMM Apache-2.0; texts accompany the worker. Dependencies pinned to sibling revisions: mlx-swift `901941965d82e4a216d4d117231d847d194c563d` (MIT), mlx-swift-lm `bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57` (MIT), swift-transformers `150169bfba0889c229a2ce7494cf8949f18e6906` (Apache-2.0).

Weights: Kokoro Apache-2.0; Pocket CC-BY-4.0 (preset voice terms vary); Qwen3-TTS Apache-2.0; Breeze Research & Non-Commercial v1.1 (commercial=false; bundled licence text). No Stable Audio, enhancement, isolation, MP3 encoder or playback code included.

New Arsenal worker transport, tests, metadata/pinning and dev verification scripts are MIT, with SPDX headers. The copied Vireo engine and vendor files remain under their upstream licences as recorded in `PORTED-FROM.md`; the internal VireoCore module name is retained for source compatibility, not as a relicensing claim. The split Metal build recipe is adapted from Vella `Worker/build-split.sh` (Apache-2.0).
