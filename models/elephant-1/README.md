<!-- SPDX-License-Identifier: MIT -->
# elephant-1

elephant-1 is Base Labs at Baseten's 9B decision model, a full fine-tune of Qwen3.5-9B on 6.21M decision examples with a Brier calibration term. Given a state (text or any JSON value, with pictures or short videos if needed) and a set of typed questions, it returns a calibrated probability for every option and answers all questions from one shared prefill of the state; it generates no text. The makers report 62.65 on the Jev Decision Index 0.4, the best model under 15B, and 65.2 on their replica of the Artificial Analysis Decision Index *(vendor figures)*.

## Usage notes (2 Nov 2026)

**What it does.** `decide` answers up to 64 named questions about one state; `judge` asks the same questions of up to 256 states. Each question is one of:

| `type` | `criteria` | Answer fields |
|---|---|---|
| `noul` (yes/no) | optional `{"true": …, "false": …}` describing each side | `noul`: P(yes) |
| `choice` | object of option name → description (text, JSON or null), 1 to 255 options, in the order given | `choice`, `probabilities`, `confidence` |
| `score` | list of levels, lowest first, 1 to 10 | `score` (expected level), `probabilities`, `legend`, `confidence` |

`instructions` (text or JSON) is optional; each type has the makers' default wording. Option order matters: options are lettered A, B, C… in the order sent, so the runner reads requests in key order. MCP arguments arrive without their key order: send `questions` as its JSON text, and for an object state also `ordered_json: {"state": "<the state as JSON text>"}` (`items` likewise for `judge`); a string state is always text.

**Inputs.**

| Field | Type | Required | Notes |
|---|---|---|---|
| `state` | any JSON value | `decide`, yes | A string as is, anything else as indented JSON in the key order sent; `null` for no state. |
| `items` | array, 1–256 | `judge`, yes | One state per item. An item that fails (too long, or carrying `image`/`audio`/`video` keys) gets its own `error`; a non-empty `media` list is a 422. |
| `questions` | object, 1–64 | yes | Question name → question. Names come back as answer keys and are never shown to the model. `judge` also takes a choice's options as a plain list of labels. |
| `media` | array, up to 32 items | no (`decide`) | Pictures `{"type": "image", "path": "/absolute/path"}` and videos `{"type": "video", "frames": [{"path": "/abs/f0.png"}, …], "fps": 2}` (2 to 768 stills of one size, each a `{"path": …}` object so Arsenal stages the files; the worker also takes plain path strings as the makers send them; fps defaults to 1), placed where the state says `<image:1>`, `<video:2>`, …: each referenced exactly once, in order, by its own kind, as the makers' requests do. PNG or JPEG, up to 64 megapixels each (refused above, from the header alone); other types and truncated files are refused. |
| `bits` | 0 or 16 | no | 16-bit only; any other value is a 409. |

MCP: `arsenal_run(model:"elephant-1", operation:"decide", input:{"state":"I was charged twice this month, please refund one of them.","questions":{"refund":{"type":"noul","instructions":"Is the customer asking for a refund?"}}})`

With a picture: `{"state":{"screen":"<image:1>","task":"Pay the open invoice."},"media":[{"type":"image","path":"/absolute/path/screen.png"}],"questions":{"button":{"type":"noul","instructions":"Is a Pay button visible?"}}}`

With a video: `{"state":{"clip":"<video:1>"},"media":[{"type":"video","frames":[{"path":"/abs/f0.png"},{"path":"/abs/f1.png"},{"path":"/abs/f2.png"},{"path":"/abs/f3.png"}],"fps":2}],"questions":{"right":{"type":"noul","instructions":"Does the ball move to the right?"}}}`

HTTP: `curl -s "$ARSENAL_URL/arsenal/v1/models/elephant-1/ops/decide" -H 'Content-Type: application/json' -d '{"state":{"ticket":"Order 4471 arrived broken"},"questions":{"team":{"type":"choice","criteria":{"billing":"charges and refunds","shipping":"damaged or lost parcels"}}}}'`

**How it runs.** The makers' decision path (OneJev `qev` at 81ce62f with the checkpoint's `serve.patch`): every question is rendered with the `qev-labels-v2` prompt through the checkpoint's chat template, the longest common token prefix (system turn and state) is prefilled once, and each question's suffix is scored from that shared state. Full-attention layers see their own block bidirectionally, as the checkpoint's `modeling_overrides` declare; the Gated DeltaNet layers stay causal. The answer is the softmax over the LM-head logits of the option letters (A–Z, then two-letter labels), computed in float32. FP16, cast from the BF16 checkpoint at Get. A long prefix is prefilled layer by layer with attention queries in blocks of 1,024 rows, so its memory grows with length but never holds a length-squared score matrix. Pictures follow the makers' multimodal training and evaluation path (`train/common.batch_scores` with the readout, which `train/eval_mm.py` uses): each `<image:N>` becomes the checkpoint's vision block, the image is read as PIL reads it, resized with antialiased bicubic to a multiple of 32 px between 65,536 and 16,777,216 pixels, encoded by the 27-layer vision tower and placed with Qwen3.5's three-axis rotary positions. A video's frames are resized together (Qwen3-VL's per-frame cap: at most 768 tokens per pair of frames, 960 for a 5-frame list), paired in time (an odd count repeats the last frame), and each pair is preceded by its timestamp, "<0.2 seconds>", from the fps; the image tokens sit in the shared prefix, so the block mask covers them too. The makers' `qev/mm_engine.py` is not patched by `serve.patch` and would score images causally; that differs from the trained path by up to 0.16 in probability on the fixed requests, so this runner follows the trained path.

**Limits.**
- Pictures and videos only through `media` with `<image:N>` / `<video:N>` placeholders. Videos are the makers' frame lists: send stills (all one size), not a video file. A picture costs between 64 and 16,384 tokens of the context (one per 32×32-pixel block after resizing); a video up to 768 tokens per pair of frames (960 for a 5-frame list, where the processor's rounding allows 1.25 times its per-frame cap). Pictures are read as PIL's `convert("RGB")` reads them: stored RGB values, no colour management, transparency dropped, grey replicated, 16-bit grey clipped, EXIF orientation not applied; JPEGs go through macOS's decoder and can differ from PIL's by a level or two on average, more at sharp colour edges.
- Trained on sequences up to 16,384 tokens (model card). The runner accepts the makers' serving limits, 32,768 tokens for the state plus its longest question and 65,536 for a whole request, and refuses longer requests rather than truncating them. Above 16,384 quality is unverified: one check found a planted record in a 32,391-token state (below).
- Memory, measured on 2 Nov 2026: about 18 GB at rest and for short requests; a 16,000-token request peaked at 19.6 GB and a 32,391-token request at 22.5 GB (worker footprint), with no swap.
- Trained and evaluated in English. A Polish request answered correctly in a quick check; that is not a measurement.
- The checkpoint ships no calibration file, so probabilities are its raw softmax (temperature 1), as the makers serve it. The makers' option-order debiasing flag is not offered.

**Parity with the makers' code.** Measured on 2 Nov 2026 on an M5 Max against the makers' `DecisionEngine` on PyTorch MPS (transformers 5.19.0), captured once in BF16 and once in FP16, with bounds frozen before this port first ran (every answer's top option unchanged, max probability difference ≤ 0.025). Nine fixed requests, 15 questions: the four Arsenal decision bench cases, the card's ticket example, a nested JSON state with structured criteria, a 30-option choice, a Polish request and questions with no instructions.
- Prompts byte-identical and token ids identical for every prefix and suffix.
- Largest probability difference 0.00097 against the makers' FP16 run and 0.0116 against their BF16 run (their own BF16-to-FP16 difference is 0.0111); no answer changes.
- Activations at the last token of both passes (embedding, 32 layers, final norm): worst cosine 0.999992.

**Pictures, parity.** Measured on 2 Nov 2026 against the makers' trained path on PyTorch MPS in BF16 and FP16 (`lab/evidence/elephant-1/maker-mm-2026-11-02`), bounds frozen before this port ran. Six requests, eleven questions: the makers' own probe screen, two traffic lights in one state, a 50×40 sticker, a textured picture shrunk on both axes, a textured JPEG and a real aerial photo (JPEG, a lab copy, not in the repository). Answers within 0.0063 of the makers' FP16 run and 0.0044 of their BF16 run, no answer changes; decoder states at cosine 0.9999 or better.
- PNG pictures: decoded bytes, resized bytes and normalised pixels identical to the makers' processor; token ids, grids and rotary positions identical.
- JPEG pictures are decoded by macOS ImageIO, not PIL's libjpeg-turbo: on the textured JPEG up to 62 levels apart at sharp colour edges (mean 1.3). The vision features then sit at cosine 0.97 to the makers' (0.965 on the photo), and answers still within 0.005. Fed PIL's decode of the same files, the vision features match at 0.99999 (JPEG) and 0.99996 (photo).
- The vision tower's features on the probe screen reach cosine 0.99611 against the makers' FP16 run, below the 0.999 bound first frozen. A float32 run of the makers' vision tower (a diagnostic) places this port at 0.99867, the makers' own FP16 at 0.99898 and their BF16 at 0.97893: this port's FP16 is as close to float32 as the makers' FP16 is. The vision bound was revised after the first run to the makers' own BF16-versus-FP16 agreement; the revision and its reasons are recorded with the bounds.

**Against d1-3B, a quick check.** On the four labelled Arsenal decision fixtures both answer correctly; elephant-1 puts 0.994 on the right answer on average, d1-3B 0.900 (the gap is the 3-level score case). Four cases cannot rank two models: `results/quick-check-vs-d1-2026-11-02.md`.

**Videos, parity.** Measured on 2 Nov 2026 against the makers' trained path (`lab/evidence/elephant-1/maker-video-2026-11-02`), bounds frozen first. Three requests, four questions: a red ball moving right (4 frames, 2 fps), a light turning red to green (3 frames, 1 fps, so the last frame is repeated) and a picture with a clip in one state. Prompts, grids, normalised pixels, expanded token ids with their timestamps, and rotary positions identical to the makers' processor; answers within 0.00015 of the makers' FP16 and BF16 runs, no answer changes; vision features 0.99995 and 0.99996 on the clips; decoder states 0.9999 or better.

Goldens: `gate/golden/golden.json` (text), `gate/golden/golden-images.json` (pictures) and `gate/golden/golden-videos.json` (videos). Real check: `results/real-check-2026-11-02.md`.
