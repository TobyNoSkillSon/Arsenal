# Real check, 2 Nov 2026

One run through Arsenal's real service on an isolated profile (`arsenal-testserver --real`, private QA catalog from the
committed folder at 8bf6264b, Hugging Face client on loopback so Get could only import the local copy), under the GPU
lock, headless. M5 Max, Mac in use. No speed figures are claimed.

- Get: the pinned `google/embeddinggemma-2` BF16 safetensors cloned from the local Hugging Face cache and cast to FP16
  (`cast:embed-gemma2-v2`, 1.52 GB snapshot); Load at fp16 returned in 5 s.
- `embed` four times, all 200: five queries (`task: query`); three documents (`task: document`, one Polish); a photo,
  a text-page image, a spoken clip, a video and one interleaved text + photo + clip item (no task); one query at
  `dimensions: 256`. Input tokens 69, 80, 1,488 and 15, the same counts as the maker's processor.
- Against the maker's own code (sentence-transformers 6.2 on Transformers 5.22.1, MPS, FP16,
  `lab/evidence/embeddinggemma-2/maker-2026-11-02-fp16`): texts 0.999998 to 1.0, images 1.0, audio 0.999999, the
  interleaved item 0.999999, video 0.99993 (FFmpeg against AVFoundation colour rounding, at most 2 levels); the
  256-number vector 0.9999996 against the maker's vector cut and re-normalized. Against the maker's BF16 run: 0.99988 or
  better.
- Related pairs score highest: the aurora query finds the aurora text (0.87, Polish text 0.68 next); "a rocky ocean
  coastline" the photo (0.76); "a dog barking" the spoken clip (0.78); "how to cook pasta" the recipe page (0.74, the
  pasta text 0.72 next); "a red ball rolling across a field" the video (0.75).
- Memory (measured): worker footprint 2.1 GB after the requests; the whole Mac stayed near 27–30 GB in use; swap unchanged
  at 2.1 GB (in use before the run). No processes left behind.
- Golden: `gate/golden/golden.json`.
- Not checked: PDFs, inputs near 8,192 tokens, audio near 30 s or `truncate`, rotated or colour-tagged video, Polish
  retrieval quality, the app's own Get and Load (headless rule), speed.
