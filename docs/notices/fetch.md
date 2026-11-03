# Fetch dependencies and provenance

Manifest, catalog, downloader, store and selection code are implemented in Arsenal. Foundation, Security and CryptoKit are Apple system APIs; no third-party TOML/HF Swift dependency was added.

The streaming downloader design was informed by read-only inspection of Vella `Sources/VellaCore/Catalog/Downloads/NativeModelDownload.swift` and its catalog/derived-store helpers. No Vella implementation source was copied verbatim. Original repository revision at inspection is recorded below. Vireo `Resources/models.json` informed source-cast/assets/vendor-quantization requirements now defined in CONTRACTS §13.
f4c5c34685829704f6492c49f087628d84838d5a
