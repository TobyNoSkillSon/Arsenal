# Real check, 2 Nov 2026

Requests through Arsenal's real service on an isolated profile (`arsenal-testserver --real`, private QA catalog from
the committed folder, Hugging Face client on loopback so Get could only import the local copy). Headless, no app
window. Final code: worktree at d2a5d20c, LCO sources unchanged since the review fixes (ad27d8f3). M5 Max, Mac in use,
on AC. No speed figures are claimed.

- Get: the pinned `LCO-Embedding/LCO-Embedding-Omni-7B` shards imported from the local Hugging Face cache and cast
  from BF16 to FP16, 17.9 GB, nothing downloaded; ready after 195 s. Load returned 200 in 4.5 s.
- `embed` three times: the four model-card texts; an image, an audio clip and a video in one request; one combined
  text-and-image item. All returned 200 with one L2-normalized 3,584-number vector per item. Input tokens 259, 592 and
  147; the first two are the makers' processor counts. Inputs are the committed fixtures in
  `lab/evidence/lco-embedding-omni-7b/fixtures`, hashes as in the golden.
- Against the makers' own code (the card's Transformers path on the GPU in FP16,
  `lab/evidence/lco-embedding-omni-7b/maker-2026-11-02-fp16`), cosine per item: query 0.999997, Everest 0.999994,
  K2 0.999997, Kilimanjaro 0.999997, image 0.99995, audio 0.99981, video 0.99916. The combined item has no maker
  vector.
- Against the first served run: texts, image, audio and the combined item are identical to four decimals; only the
  video vector changed, as the decoder fix intended.
- The card's retrieval example: the query ranks Everest (0.645), Kilimanjaro (0.479), K2 (0.432), as the card does
  (0.6456, 0.4788, 0.4331).
- Memory (measured): worker footprint 17.3 GB after the requests (MLX active 16.8 GB); the whole Mac went from 27.7 GB
  to 45.4 GB in use; swap unchanged at 2.1 GB (in use before the run). No processes left behind.
- Golden: `gate/golden/golden.json`.
- Not checked: PDFs, inputs near the 32,768-token bound, audio near 5 minutes, long videos, colour-tagged video,
  Polish text, speed.

The first run, before the review fixes, measured video 0.983 against the makers and was superseded by this one.
