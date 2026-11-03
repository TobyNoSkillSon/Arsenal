+++
hypothesis = "Copying Vireo’s gated speech path preserves numerical behavior in the Arsenal stdio worker."
change = "Ported speech engines, codec subset, selective quantization and Exact/Fast configuration; removed HTTP and playback."
measurement = "Source audit completed; isolated CLT compilation underway; no Arsenal GPU inference measured yet."
verdict = "inconclusive"
numbers = { models_audited = 5, offered_tiers = 2, real_runs = 0 }
links = ["README.md", "gate/gate.toml", "Workers/Voices/PORTED-FROM.md"]
+++

Vireo source commit `9366da5e14dcafe846fc41565c94f4f145eaf7cc` is copied, not modified. The catalog pins and precision rules were inspected against `Resources/models.json`. The new runner changes framing and PCM/file transport only. Real load, first-audio, RTF, peak RAM and parity remain unmeasured until the worker compiles and runs under `mlock gpulock`.
