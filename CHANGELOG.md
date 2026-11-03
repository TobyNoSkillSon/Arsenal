# Changelog

## 1.0.0

The first public release.

- A menu-bar app that runs AI models on your Mac, on device: transcription and speaker turns, voices and voice cloning, document reading and extraction, translation, image generation and editing, background removal, upscaling and restoration, music, audio separation and enhancement, detection, segmentation and depth, embeddings and reranking, typed decisions, video frame interpolation, image to 3D, forecasting and protein models.
- A signed catalogue of 102 models, 370 GB if you get every one. Each downloads on demand from Hugging Face at a pinned revision, checked file by file, and runs at 16-bit by default (Fish Audio Drama 3 at its maker's 4-bit release). Get more models shows each model's size, licence, limits and any licence conditions before you download it. FLUX 3 Image Klein, SAM 3.1, SAM Audio Large, TRELLIS.2 4B and IDOL are gated: you accept the maker's terms on Hugging Face and Arsenal downloads with your read token.
- Task windows (Speak, Make Music, Make a Sound, Make an Image, Transcribe and more) build their controls from what the chosen model can do.
- Agents get the same models over MCP (fifteen tools, a bundled stdio bridge `arsenal-mcp`) and scripts over a local HTTP API with a bearer token. Downloads need the user's yes.
- Add your own model as a folder, without reinstalling: Arsenal validates it, and code that is not part of the signed app runs only after you approve that exact version.
- Before loading, Arsenal compares a model's peak memory with the RAM free and asks before falling back to swap.
- Updates come from GitHub Releases, signed with the same certificate as the app; the previous version is kept for rollback.
