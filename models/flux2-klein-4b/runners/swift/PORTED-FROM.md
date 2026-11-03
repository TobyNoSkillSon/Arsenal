# FLUX.2 Klein 4B Swift runner: provenance

Arsenal code under MIT. Adapted, not vendored as a dependency:

- **xocialize/flux2-klein-swift** (MIT, Copyright (c) 2026 Xocialize), commit `d8a9fa2b5dc6c3c336bf05d504708377ef26d5c4`:
  the MMDiT (`Transformer.swift`) and the Qwen3 tap encoder and chat template (`TextEncoder.swift`). Its Qwen3 code was
  itself adapted from ml-explore/mlx-swift-lm (MIT). Licence text: `LICENSE-flux2-klein-swift`.
- Written for Arsenal from the vendor's diffusers implementation (Apache-2.0) as the specification: the VAE decoder, the
  scheduler and pipeline, weight loading, PNG helper and the ArsenalWire runner.

Weights: Black Forest Labs FLUX.2 Klein 4B, Apache-2.0, pinned in `model.toml`.

# FLUX 3 Image Klein Swift runner (`Sources/Flux3Klein`, served by FluxWorker for the add-on folder)

Arsenal code under MIT, ported op for op from **mflux** (MIT, Copyright (c) 2026 Filip Strand), commit
`9669af51ff643f1d6d109a29e24f43e61daeb2f1`: `src/mflux/models/flux3` (transformer, flow-map scheduler, prompt encoder,
latent creator, reference images), the shared Qwen3-VL text tower (`models/common_models/qwen3_vl`,
`models/flux2/model/flux2_text_encoder`) and the FLUX.2 VAE encoder (`models/flux2/model/flux2_vae/encoder`). Licence
text: `LICENSE-mflux`. The VAE decoder is FLUX.2 Klein's (`Sources/Flux2Klein/VAE.swift`). The LANCZOS resample is a
reimplementation of Pillow's algorithm (no Pillow code). mflux is the parity reference, not a dependency.

Weights: Black Forest Labs FLUX 3 Image Klein, FLUX Kommunity License v1.0 (not open; text encoder Apache-2.0); a signed
catalog model (`models/flux3-image-klein`) whose weights the user downloads from Hugging Face after accepting the licence there, never bundled.
