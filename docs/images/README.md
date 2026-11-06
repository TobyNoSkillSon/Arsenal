# README images

Every model output in these images was made by Arsenal on a Mac (30 Oct and 3 Nov 2026); the app images are screenshots of
Arsenal 1.0 (2 Nov 2026). Inputs:

- `readme-hero*.webp`: an AI-generated street photo, made with FLUX 3 Image Klein (FLUX Kommunity License v1.0;
  Black Forest Labs claims no ownership of outputs, used here as a Qualifying User, reviewed before publishing);
  RF-DETR Nano (Apache-2.0), MoGe-3 ViT-L (MIT) and BiRefNet (MIT) ran on it.
- `readme-more*.webp`: Kokoro 82M speaking the opening of *Moby-Dick*; a song by ACE-Step 1.5 Turbo split into stems
  by HTDemucs fine-tuned; DDColor ModelScope colourising *Mulberry Street, New York, about 1900* (Detroit Publishing
  Co., public domain, Library of Congress 3g04637u); TimesFM 2.5 200M forecasting CO2 at Mauna Loa (NOAA Global
  Monitoring Laboratory data), drawn against what was later measured.
- `readme-app*.webp`: Arsenal 1.0's menu-bar menu and Models table.
- `icon.png`: Arsenal's app icon, rasterised at 256 px from `Resources/Brand/icon.svg` by `scripts/icon.swift`, the
  same renderer the build uses. `install.svg` (no longer in the README) takes its colours from that icon: the tile's silver
  (#f5f5f3 to #e0e0de) and the cannon's ink (#141414).
- `download.svg`, `website.svg` and `support.svg`: the README's top row, copied unchanged from Heimdall's README so
  the family's rows match.
