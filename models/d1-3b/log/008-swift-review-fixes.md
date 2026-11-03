+++
hypothesis = "The four Swift runner review findings (picture memory, quadratic BPE, Unicode cache keys, 16-bit PNG decoding) can be fixed without changing answers."
change = "Retain capped pictures under 128 MiB retained / 512 MiB working limits; ranked merge heap in BPE; UTF-8 byte cache keys; Pillow-matching 16-bit PNG conversion."
measurement = "M5 Max, 8 Oct 2026, Mac in use. Gate over ArsenalWire against the saved gate/reference-fp32.json: max |dp| 0.003929896861490922, 24/24, unchanged; Polish probe counts unchanged; 256 x 10 picture regression peak growth 149.7 MiB."
verdict = "kept"
numbers = { gate_max_dp = 0.003929896861490922, gate_argmax = 24, questions = 24, picture_regression_peak_mib = 149.7 }
links = ["gate/parity-swift-fixes.json", "runners/swift"]
+++
# Swift review fixes, 8 Oct 2026

All four findings from the Swift runner review are fixed. Scope is the model folder; no app build, install, catalog
signature change, fp32 execution, weight download or new reference capture. The final standalone bf16 runner was
checked once over ArsenalWire against the saved `gate/reference-fp32.json`. The Mac was in use; these are correctness
checks, not performance measurements.

- **Pictures:** worker validation retains only capped pictures. `PictureProcessor.loadCapped` decodes and caps one
  source at a time inside an autorelease pool, with a transactional budget per item. The request has a cumulative
  128 MiB retained RGB limit (including preceding 32-item groups); the header is checked before decoding against a
  512 MiB working reservation, including retained RGB, the decoder/conversion/resampling and coefficient tables.
  Sources past the budget fail as `invalid_image` for decide and a clean per-item error for judge. The pure CPU
  regression retains every successful item of the maximum 256-item × 10-picture batch: 4 items / 40 capped images,
  125,801,280 RGB bytes retained, 252 clean item errors, 156,942,504 bytes (149.7 MiB) increase in lifetime peak physical
  footprint. An additional 256-item batch of valid 178,946,048-pixel PNG sources is rejected before decode. The wire
  check independently returns all 256 clean per-item errors for those sources.
- **BPE:** a linked symbol list and ranked merge heap replace repeated scans and array removals. Queue order is
  rank then original left index; stale entries are discarded and only two changed neighbours are enqueued. This
  bounds the merge work to O(n log n), with O(n) storage. The pure CPU check matches the original scan on 512
  overlapping/tied/recursive merge strings. Against HF tokenizers and the pinned vendor prompt: 3,029 stress
  strings, 139 gate/probe requests (25,013 rendered tokens), and three unbroken 32,768–65,536-scalar strings have
  identical IDs and rendered text. Counts are asserted before comparisons; the script exits nonzero on a mismatch.
- **Unicode:** both verbalizer caches use UTF-8 `Data` keys. A miniature tokenizer gives Å (U+00C5) and Å (U+212B)
  distinct single-token IDs. Warming `[Å, Å]`, then requesting `[Å, Å]` preserves both the exact spelling and the
  reversed readout assignment; it exercises both formerly canonical-equivalent Swift String cache keys.
- **PNG16:** raw integer channel reads clamp grayscale without alpha at 255, matching Pillow's integer mode.
  RGB, gray-alpha and RGBA use the high byte and drop alpha. Four PNG fixtures cover 0, 1, 128, 255, 256, 512,
  32768 and 65535 with independent channel/alpha values; every output byte equals saved Pillow RGB bytes.
  `gate/make_picture_fixtures.py` recreates these fixtures and records the Pillow version.

## Verification

`runners/swift/Tests/Regression/run.sh` compiles just JSON, tokenizer, prompt and pictures plus its regression main
with the CLT compiler; no MLX or model is loaded. Its caller supplies disposable scratch. The existing five Swift
Testing tests also pass. On this updated macOS 27 machine, CLI compilation requires the explicit SDK; SwiftPM's
native test build needs Xcode's developer directory for the Testing framework (the CLT compiler is used).

`gate/tokenizer_parity.py` takes explicit checkpoint, probe, pictures, pinned vendor source and d1-check paths.
`gate/parity.py --memory-fixtures` includes the wire memory rejection check in the same final runner session.

Final evidence: `gate/parity-swift-fixes.json`. Gate: **24/24**, max |Δp| **0.003929896861490922**, exactly equal to
`gate/parity-swift.json`. Every other gate group's max difference and argmax count, and every Polish probe count,
also equals that baseline exactly. In particular, pictures remain 20/20 at 0.021774333871071305; the existing bf16
picture near-tie floor is unchanged. The reference file was reused without modification.
