+++
hypothesis = "Upscaled video should keep the source soundtrack."
change = "upscale_video uses the same MediaBridge.keepAudio step as RIFE after the video writer finishes."
measurement = "Code-only; AudioMuxTests and the existing smokes run in the central build."
verdict = "kept"
numbers = { weight_payload_bytes = 0 }
links = ["../README.md"]
+++

Smokes and tests for this change are pending the coordinator's central build; no weights were downloaded.
