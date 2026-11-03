+++
hypothesis = "The operation should offer only the controls the turbo path implements."
change = "Removed include_score, which ACE-Step plans never fill. README now states the language, instrumental, duration and guidance semantics."
measurement = "Code-only review; the central build reruns the family tests and the random-weight smokes."
verdict = "kept"
numbers = { weight_payload_bytes = 0 }
links = ["../README.md"]
+++

Smokes and tests for this change are pending the coordinator's central build; no weights were downloaded.
