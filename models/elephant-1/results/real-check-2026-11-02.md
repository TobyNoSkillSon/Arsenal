# Real check, 2 Nov 2026

Requests through Arsenal's real service on an isolated profile (`arsenal-testserver --real`, private QA catalog from
the committed folder, Hugging Face client on loopback so Get could only import the local copy). Headless, no app
window. M5 Max, Mac in use. No speed figures are claimed.

- Get: the pinned `baselabs/elephant-1` checkpoint imported from the local Hugging Face cache and cast from BF16 to
  FP16 by the worker's `convert` op (`cast:elephant-1-v1`), 18.8 GB, in 115 s. Load returned in 3.8 s.
- `decide` twice and `judge` once, all 200:
  - the model card's ticket: refund 0.9917, team billing 0.9983, urgency "Today" 0.6454 (maker BF16: 0.9913, 0.9983,
    0.6450);
  - a nested JSON state with structured criteria: refund 0.4773, frustration "angry" 0.4353 (maker BF16: 0.4889,
    0.4357), 294 input tokens;
  - judge over "three red apples" (apples 0.9990, fruit apple), "the box is empty" (0.0160, fruit none) and one item
    with an `image` key, which got its own error while the others answered.
- Against the makers' own code (OneJev qev + serve.patch on PyTorch MPS; `lab/evidence/elephant-1`): nine fixed
  requests, 15 questions, token ids identical, largest probability difference 0.00097 against the makers' FP16 run and
  0.0116 against their BF16 run (their own BF16-to-FP16 difference is 0.0111), no answer changes; per-layer cosine at
  least 0.999992.
- Memory (measured): worker footprint 16.8 GB after the requests (MLX active 15.9 GB); lifetime peak 18.2 GB in the
  direct worker run; the whole Mac went from 29.1 GB to 47.7 GB in use with other jobs running; swap unchanged at
  2.1 GB. No processes left behind.
- Golden: `gate/golden/golden.json`.
- Not checked: requests near the 32,768 and 65,536-token limits, a choice wider than the slots (not reachable: 255
  options fit in 256 slots), cancellation mid-request, speed.

Addendum, after the review fixes (same day, worker run directly, headless, M5 Max, Mac in use). Long inputs with blocked
attention queries and per-layer evaluation: numbered shipment records with one planted "customs seized shipment 99999
in Poznań" line in the middle, asking a yes/no and a 4-way choice.
- 16,039 tokens: seized 0.982, city Poznań 0.999; 7.2 s; MLX peak 18.9 GB, worker footprint peak 19.6 GB; whole Mac
  27.4 → 46.8 GB peak in use; swap unchanged.
- 32,391 tokens: seized 0.986, city Poznań 0.997; 21.7 s; MLX peak 21.9 GB, worker footprint peak 22.5 GB; whole Mac
  27.5 → 49.2 GB peak in use; swap unchanged. So the 32,768-token limit stays.
- Maker parity rerun after the change: unchanged (max |dp| 0.00097 vs FP16, 0.0116 vs BF16, worst cosine 0.999992).
- Not measured: memory before the change (no run at these lengths was made then); quality above 16,384 tokens beyond this
  one check.

Pictures (same day, after the image path landed; the real service on an isolated profile, headless):
- Get imported and cast the pinned checkpoint, now with `processor_config.json`, in 111 s; load 5.2 s.
- The makers' probe screen (`{"screen": "<image:1>", …}`, 320×240): red rectangle yes 0.9994, blue shape circle 1.000,
  clutter "empty" 0.700, 324 input tokens. Two traffic lights in one state (256×256 and 400×300): turned green yes
  0.9992, first light red 0.9999, 353 tokens. These equal the heavy parity run's answers to six places.
- A text-only request in the same session: refund 0.991682, unchanged from the text-only real check.
- Placeholders in the wrong order (`<image:2> … <image:1>`): refused, 422, "media must be referenced once each, in
  order".
- Memory (measured): worker footprint 17.2 GB after the requests (MLX active 16.8 GB, the vision tower adds about
  0.9 GB); whole Mac 27.9 → 46.7 GB in use; swap unchanged. No processes left behind.
- Golden: `gate/golden/golden-images.json`.
- Not checked: large photos near the 16,384-token picture bound, many pictures in one request, JPEG files from
  cameras (EXIF orientation is not applied, as PIL does not apply it), speed.

Pictures, final code after the image review (same day, headless, real service):
- Get 111 s (import and cast), load 5.1 s. Probe screen: red rectangle yes 0.9994, blue shape circle 1.000, clutter
  "empty" 0.698. A real aerial photo (JPEG, 1600×844, lab copy of a macOS wallpaper): water yes 0.9997, scene
  "landscape" 0.9997, 1,485 input tokens, 4.0 s. Traffic lights: green yes 0.9991, first light red 0.9999. Text-only:
  refund 0.991682. Wrong placeholder order: 422.
- Memory (measured): worker footprint 17.3 GB; swap unchanged; no processes left behind. FP16 range on the real photo
  (heavy test): every captured activation finite, largest magnitude 724.
- Golden: `gate/golden/golden-images.json` (replaces the first picture golden).

Videos (frame lists; same day, headless, real service; rerun on the final code after the video review, frames as
{"path": …} objects as the schema now requires, same answers to six places):
- Get 107 s (import and cast), load 3.5 s. A red ball moving right (4 frames, 2 fps): moves right yes 0.9972,
  object "ball" 0.9999, 359 input tokens. A light turning red to green (3 frames, 1 fps):
  green at the end yes 0.9991. These equal the heavy parity run to six places. Text-only refund 0.991682, unchanged. A
  one-frame video is refused by the operation schema (422).
- Memory (measured): worker footprint 17.2 GB; whole Mac 27.8 → 45.3 GB in use; swap unchanged; no processes left.
- Golden: `gate/golden/golden-videos.json`.
- Not checked: long frame lists near 768 frames, large frames, speed.
