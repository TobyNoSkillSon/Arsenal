# SAM Audio Large

Isolate a sound by description. Text-prompted sound separation returns both target and residual WAV files. Extract lists the target first; remove lists the residual first. The required T5-base asset is pinned separately; Get copies its encoder into the selected snapshot (`text_encoder/`) at FP16 or with the tier's packed matrices, so Load never reads the raw asset. Temporal anchors (positive/negative time spans) are supported on clips up to ten seconds, matching the source port's single-chunk limit. Visual prompts require a vision encoder absent from this Swift port and are not declared. Long recordings use the source port's ten-second windows and three-second cosine overlaps; the combined two-WAV result is bounded by the host's 64 MiB output limit.

## Source and precision

Checkpoint: Meta's original `facebook/sam-audio-large@5f2cd3a9471a08c7282c06036be6893e18de8b70` (`checkpoint.pt`, FP32, 14.9 GB, and `config.json`). The repository is gated: the user accepts Meta's SAM License on Hugging Face, Meta approves the request, and Get downloads with the user's token. Get's `cast:audio-sam-v2` reads the checkpoint tensor by tensor without executing its pickle, drops the vision encoder and masked-video aligner this port does not run, sums the LSTM bias pairs, moves convolutions to MLX's layout and stores FP16, about 6.3 GB with the T5 encoder; the original is then deleted. The result is the tensor set of `mlx-community/sam-audio-large-fp16@110947e651c2f7837c47d8897cfca1cb22ce9633`, the FP16 re-upload the port was built against, and sampled tensors match it bit for bit (`results/official-checkpoint-2026-11-02.md`). Native port: `Blaizzy/mlx-audio-swift@244458d7b61e54ff135073ddc9ed76782bb20758`. Exact files, SHA-256 digests and byte counts are in model.toml. `gate/header-check.json`, `gate/header-map.json` and `gate/golden/` record checks made against the re-upload, whose layout is the conversion's target.

FP16 is the serving default. int8 and int4 pack every matrix whose input width is divisible by 64 (group size 64); spatial convolutions, norms and narrower matrices stay FP16. Casting and quantized output quality remain unqualified without trained weights. The recurrent LSTM projections are packed with the same rule.

## Verification

The no-weight fixture constructs a shrunk version of the actual graph and uses the ordinary worker operation. This verifies wiring and shapes, not trained output, speed, memory, or numerical fidelity. Native source closure and shared-host prerequisites are recorded in `Workers/Audio/PORTED-FROM.md` in the source repository. Trained-weight qualification is still required.

## Usage notes (2 Nov 2026)

SAM Audio Large pulls one sound out of a recording from a text description. `separate_audio` takes `audio` and a
`prompt` ("speech", "dog barking") and returns two 48 kHz files: `target` (the described sound) and `residual`
(everything else); `mode: remove` swaps which one you want. `anchors` mark time spans where the sound is present;
anchored clips are limited to ten seconds, and longer audio without anchors runs in overlapping windows. `steps` and
`seed` control the sampler. No video prompts in this port. Meta's SAM License applies (no ITAR or other
trade-controlled uses; redistribution under the same licence); the catalog shows that line.

Parity: no maker reference run yet (Meta's repository is gated for this Mac's account). On 2 Nov the speech target from
a speech-over-music clip matched the 22 Oct output at correlation 0.94 (`results/real-check-2026-11-02.md`, golden in
`gate/golden/`); that run used the FP16 re-upload, which sampled tensors show equal to Get's conversion of Meta's file.

