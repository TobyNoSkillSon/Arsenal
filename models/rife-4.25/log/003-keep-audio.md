+++
hypothesis = "Interpolated video should keep the source soundtrack."
change = "MediaBridge.keepAudio composes the rendered MP4 with the source audio tracks over the rendered span, passthrough first, re-encode otherwise."
measurement = "Code-only; AudioMuxTests and the existing smokes run in the central build."
verdict = "kept"
numbers = { weight_payload_bytes = 0 }
links = ["../README.md"]
+++

Smokes and tests for this change are pending the coordinator's central build; no weights were downloaded.
