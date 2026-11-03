# Model information audit, 25 Oct 2026

Read-only audit of all 55 folders in `models/` against the rule in [ADD-A-MODEL.md](../ADD-A-MODEL.md) (full official name, parameters, measured speed and memory, languages and Polish). Cards were fetched at the pinned revision; for converter repos the upstream card at `main`. Gated cards (pocket-tts, sam3.1, sam-audio, stable-audio) could not be read. Parameter totals were compared with the Hugging Face safetensors count. The Qwen3 TTS names and Fish Audio Drama 3's "35B parameters, 3B active per token" were re-checked by hand on their cards; the rest is the auditor's reading.

## Renamed on branch `arsenal/full-model-info` (names and docs only; the signed catalog is re-signed by the night pass)

- `qwen3-tts-base`: "Qwen3 Base 1.7B" to "Qwen3 TTS 12Hz 1.7B Base" (official Qwen3-TTS-12Hz-1.7B-Base).
- `qwen3-tts-voicedesign`: "Qwen3 VoiceDesign 1.7B" to "Qwen3 TTS 12Hz 1.7B VoiceDesign".
- `fish-drama-3`: "Drama 3" to "Fish Audio Drama 3" (card title).
- `ddcolor-modelscope`: "DDColor" to "DDColor ModelScope" (the ModelScope variant).

Left for the owner or a later pass: `parakeet-v3-ultra` (the release is Moondream's "Parakeet Ultra"; "v3" is the base model's version), `stable-audio-3-medium` and `stable-audio-3-small-sfx` (pinned repo is "Stable Audio 3 Optimized").

## Parameter counts that disagree with the weights or the card (not changed)

- Mixture of experts without the active count: `fish-drama-3` (35B total, 3B active per token). It has no `description` yet to carry it.
- Below or above the Hugging Face total: `chandra-ocr-2` 4.5B vs 5.30B, `breeze-tts-2` 3B vs 3.45B, `index-translate-2b` 2.2B vs 2.27B, `index-translate-9b` 9.4B vs 9.65B, `sam-audio-large` 3151M vs 3.04B, `sam-audio-small` 712M vs 602M, `yue2-3b` 3.70B vs 3.63B, `sam3.1` 870M vs about 873M.
- Matches the weights but not the card: `da3-small` (34M; card table 0.08B), `teleocr` (1.4B; card about 1.2B).
- Not verifiable: `audiosr-basic`, `bs-roformer-viperx-1296`, `htdemucs-ft`, `pyannote-community-1`, `seedvr2-3b`, both Stable Audio models, `ace-step-1.5` (5.01B is the whole package).

## Missing measured speed or memory

- No `results/` at all: `qwen3.8-27b` (figures only in `log/003-real-supervisor-smoke.md`: bf16 7.96 tok/s, 50.65 GiB peak), `sam3.1`, `seedvr2-3b`.
- Peak memory recorded but the table's speed cell `notMeasured` (raw per-request seconds only in the quick-check record): `ace-step-1.5`, `audiosr-basic`, `chandra-ocr-2`, `fish-drama-3`, `index-translate-2b`, `index-translate-9b`, `nuextract3`, `qwen-image-2.1`, `qwen-image-2.1-turbo`, `teleocr`, `yue2-3b`.

## Polish

- Confirmed by field and card: `d1-3b`, `fish-drama-3`, `parakeet-v3-ultra`, `qwen3-asr-0.6b`, `qwen3-asr-1.7b`, `harrier-oss-v1-0.6b`, `laya-multilingual`, `chandra-ocr-2` (benchmark row only).
- Claimed without card support: `qwen3.8-27b` lists `pl` but its card says nothing about languages; `ovis-vl-embedding-2b` and `ui-venus-2-9b` are `mul` with no language claim.
- `mul` with Polish not named on the card: `ace-step-1.5`, `index-translate-2b`, `index-translate-9b`, `jina-reranker-v3.5`, `nuextract3`, `qwen3-reranker-8b`.
- No Polish: both Qwen3 TTS models (10 languages, Polish not among them) and every English-only model.
