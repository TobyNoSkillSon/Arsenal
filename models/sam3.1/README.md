# SAM 3.1

Select objects and track video.

## Capabilities

Declared native operations: `segment`, `segment_session`, `video_session`, `track_video`. Each operation has a narrowed schema in `ops/`, shared by HTTP, MCP and the native tool window.

## Source

Checkpoint: Meta's `facebook/sam3.1@daa63191845a41281374e725f4c9e51c7a824460`, files `sam3.1_multiplex.pt` (a 3.5 GB FP32 PyTorch checkpoint with Meta's own tensor names) and `config.json`. The repository is gated: you accept Meta's SAM License on Hugging Face, Meta approves each request, and Arsenal downloads with your Hugging Face token. Get downloads 3.5 GB and keeps about 1.7 GB. Exact filenames, sizes and SHA256 hashes are in `model.toml`; how they were confirmed without access to the gated files is in `results/official-checkpoint-2026-11-02.md`.

Get converts the checkpoint once (`cast:sam31-fp16-v3`). It reads the checkpoint tensor by tensor without running any of its code, maps Meta's names to the port's (following the SAM 3.1 converter of mlx-vlm 0.4.3), splits fused q/k/v projections, transposes Conv2d and ConvTranspose2d kernels to MLX's layout, drops the class-token row of the ViT position embedding, and stores every tensor as FP16. It leaves out the RoPE buffers, which the port recomputes, and CLIP's pooled-output projection, which SAM 3.1's forward never reads (the detector projects the CLIP hidden states with its own `detector_model.text_projection`). That projection holds two values of −9.58e18, beyond FP16's range, which made an earlier recipe fail Get. The stored `config.json` names the video model and drops the 0.5x FPN level, for which the checkpoint has no weights. Norms and spatial kernels remain half. The text encoder builds its attention mask in float32, since FP16 turned the padding mask into NaN. Trained-weight quality has not been qualified.

## Limits

Tracks at most 16 objects, with four resident image/video sessions. Tracking returns ordered per-frame, per-object PNG masks, not a re-encoded or alpha-matted video. Calls process at most 128 upright frames and return at most 256 masks. Audio is not copied. Source rotation must be baked into pixels.

## Qualification

Random graph, header-only inventory and precision coverage establish wiring, not trained model quality. See `log/001-integration.md` for the actual checks and remaining gaps.

Swift port: `thesepehrm/sam31-swift@ce4ce6e733566cc52dec14e60c894867e1e434a3`. Adapted files and source terms: `Workers/Perception/licenses/SAM31/PORTED-FROM.md`.

The converted layout matches all 1,960 runtime parameter names and shapes, checked against the real layout of Meta's checkpoint (`SAM31OfficialCheckpointTests`); the RoPE buffers and the unused CLIP projection are the only source tensors left out. The production header checker passes with explicit per-tensor FP32→FP16 Get cast plans; these plans do not establish trained cast quality. The full graph has 707 eligible group-64 matrices (96.42% of learned parameters, derived from the inventory); all eligible matrices in its shrunk graph execute at int8 and int4. See `gate/header-check.json` and `gate/no-weights-results.json`.

Visual exemplar-image prompting is not declared: the current segmentation contract has points, boxes and text, but no exemplar-image input. The port’s interactive/video tracker holds one multiplex bucket of 16 objects. PNG mask sequences deliberately expose frame/object indices in filenames.

A top-level video text prompt seeds all matched instances up to the 16-object bucket limit, in descending score order. A text update for an explicit object ID selects its best matching instance.

When a video text prompt finds no objects, its frames return explicitly named empty masks, matching the image no-match behavior.

Text updates use targeted mask reconditioning: other object masks/pointers are preserved and spatial memory is re-encoded. Prompted and corrected current-frame output is emitted directly; propagation runs only for an unprocessed frame. Two-object and corrected-frame regressions exercise these paths.

Arsenal offers this model at FP16 only: below 1B parameters, 8-bit would save under 1 GB and run no faster.

## Usage notes (2 Nov 2026)

SAM 3.1 cuts objects out of images and follows them through video. `segment` takes an image with a text prompt
("person"), points or boxes and returns one PNG mask per match with its score; `segment_session` keeps an image
encoded for repeated prompts. `track_video` (and `video_session`) follows up to 16 objects through at most 128 upright
frames and returns per-frame, per-object PNG masks. Meta's SAM License applies: no ITAR or other trade-controlled uses,
redistribution under the same licence, and published research credits SAM; the catalog shows that line.

Parity: no maker reference yet. On a test photo the text prompt "person" returned one correct mask at score 0.976
with the earlier mlx-community conversion of the same weights; `results/real-check-2026-11-02.md`, golden in
`gate/golden/`. Tensors converted from Meta's checkpoint equal that conversion's values rounded to FP16
(`results/official-checkpoint-2026-11-02.md`); a full Get from Meta's repository has not run yet.

