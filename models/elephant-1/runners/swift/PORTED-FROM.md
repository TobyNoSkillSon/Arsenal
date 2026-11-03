# elephant-1 Swift runner: provenance

- `Sources/ElephantEngine/Prompt.swift`, `Answers.swift`, `Requests.swift`, `Engine.swift` (request path): Swift ports of
  OneJev `qev/prompt.py`, `qev/answers.py`, `qev/calibrate.py` (softmax), `qev/schema.py` and `qev/engine.py`
  (`DecisionEngine`: slot verification, shared-prefix split, sequential branches, chunked wide choices, answer
  assembly) at commit 81ce62f1597c91e46767d4d02d6ac2e18534fe94 (https://github.com/OmniJev/OneJev), with the
  `serve.patch` shipped in baselabs/elephant-1 at revision b3f8a93462fb1051a9de031e94a7ff380e254072; Apache-2.0
  (`LICENSE-OneJev-Apache-2.0`). The confidence formulas in `qev/answers.py` are TypeSafe AI's
  system-one-adapter-python (MIT), as qev credits them. Modified by Arsenal: language, structure, Arsenal's request
  bounds and error codes.
- `Sources/ElephantEngine/Model.swift`: the Qwen3.5 text decoder and the Gated DeltaNet recurrence (including its Metal
  kernel source), adapted from mlx-swift-lm `Libraries/MLXLLM/Models/Qwen35.swift` and
  `Libraries/MLXLMCommon/GatedDelta.swift` at revision bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57 (MIT,
  `LICENSE-mlx-swift-lm-MIT`; ports of mlx-lm). Modified by Arsenal: the checkpoint's block-bidirectional full
  attention, value-type layer states shared by every question branch, blocked attention queries, the float32 slot
  readout and the weight map.
- `Sources/ElephantEngine/Vision.swift`: the Qwen3.5 vision encoder and multimodal rotary positions, ported from
  Arsenal's Ovis port (`Workers/Embed/Sources/EmbedWorker/OvisVision.swift`, from mlx-swift-lm's Qwen3-VL vision tower,
  MIT, `LICENSE-mlx-swift-lm-MIT`) after transformers 5.19.0 (Apache-2.0) `Qwen3_5VisionModel` and
  `Qwen3_5Model.get_rope_index`; deepstack removed, own configuration.
- `Sources/ElephantEngine/Media.swift` and the placeholder rules in `Prompt.swift`: OneJev `qev/media.py` and
  `train/mm.py` (Apache-2.0) and transformers' `Qwen2VLImageProcessor` (smart_resize, patchify), `Qwen3VLVideoProcessor`
  (video smart_resize with the per-frame cap, frame pairing) and `Qwen3VLProcessor` (timestamps, video expansion); the resampler is
  Arsenal's PIL-faithful bicubic from `Workers/Embed/Sources/EmbedWorker/LCOMediaIO.swift` (MIT).
- `Sources/ElephantEngine/PyJSON.swift`: copied from `models/d1-3b/runners/swift/Sources/D1Engine/PyJSON.swift`, whose
  parser comes from Workers/Decisions `OrderedJSON.swift` (from Verdict, Apache-2.0, `LICENSE-Verdict-Apache-2.0`),
  extended with Python's json.dumps, str and repr; Arsenal writes non-finite numbers as json.dumps does.
- `Sources/ElephantWorker/Wire.swift`, `Worker.swift`: d1-3B's ArsenalWire loop and framing (Arsenal, MIT).
- Everything else: Arsenal, MIT.
