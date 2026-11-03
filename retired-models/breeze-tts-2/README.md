# Breeze TTS 2

Bilingual English and Chinese speech with voice cloning and voice design.

## Source

`mlx-community/Breeze-TTS-2-mlx` at `3c8829fb7fd335818f085cd2ef49b4100c0e46c8`. Source dtype `bf16`; runtime base `bf16`. Licence: Research & Non-Commercial v1.1. Research and non-commercial use only; restrictions apply to weights and outputs. Port provenance: `Workers/Voices/PORTED-FROM.md`.

## Run

Get → Load → speak through the native voices worker. Fast/Exact retains Vireo’s per-component self-tests and fallbacks. The 16-bit tier casts every learned source tensor, including codecs and preset conditioning. FP16 must pass the BF16-reference format gate; otherwise BF16 remains the runtime base.
Int8 and int4 derive locally with whole-component affine group 64 packs, including the text encoder, codec matrices and split depth output heads. Their quality outcomes belong to this recipe; older transformer-only/MXFP4 figures do not qualify it.

## Measured

Round 5 checks BF16, int8 and int4 on separate selection and held-out sets (69/46 component outputs). Int8 passes after ordinary Load of the final recipe; int4 stays offered with its quality warning wherever its check fails. Per-tier selection/held-out errors, physical peaks, RTF and retained half share are in `log/010-voice-heldout-r5.md`. Figures are debug-worker busy-Mac quick checks; common-prefix spectral distance is diagnostic. Historical schema-2 voice completion bindings qualify only their selection set. Independent vendor and perceptual parity remain unqualified.

## Research

See `log/` for port provenance, accepted/rejected optimization history and real v2 cycles. Current coverage and justified half-only tensors are listed in the model’s round-3 intake log and conversion receipts.

## V2 settings and operations

Saved voice, language, voice description, reference clip and transcript settings apply to the next request. `speak` accepts preset or saved voice IDs; `voices` lists both. `design_voice` saves a description-conditioned voice and returns a preview WAV. `clone_voice` saves a copied reference clip plus transcript; speak with the returned ID to use it. Saved voices persist per model under Arsenal support/voices. Only supported ops appear in this folder.

The deterministic native runner keeps the Vireo component gates and fallback. `optimizations.json` records accepted Exact/Fast components; lab-only candidates are not defaults.

Breeze weights are research/non-commercial (BreezeBlue RNC 1.1), not covered by Arsenal's MIT licence. Commercial use is not permitted. See the bundled weight licence in Workers/Voices/Licenses.
