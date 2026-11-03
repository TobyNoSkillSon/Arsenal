# V-JEPA 2 ViT-L

Turns a video into numbers: Meta FAIR's V-JEPA 2 video encoder (ViT-L/16, pretrained without labels by predicting masked parts of video in feature space). Clips of the same kind of activity come out close together, so the vectors serve for video search, retrieval, clustering and as features for a classifier of your own.

Input: `video`, a clip (mp4, mov, m4v); or `frames`, image files in time order; or `image`, one picture embedded as a still clip. `num_frames` (default 64, as the card samples; at most 64) are taken evenly spaced over the clip, or over `start` to `end` in seconds. `per_step` (optional) also returns one embedding per pair of frames. Frames are resized so the short side is 292, centre-cropped to 256 x 256 and normalised, as the maker's video processor does. Output: `embedding`, the encoder's patch features averaged over the clip (1,024 numbers), `dimensions`, `frames`, `tokens`, `frame_times` for a video, and `per_step` on request.

Example request (the card's example clip, archery from Kinetics):

```json
{"video": {"path": "/path/to/archery.mp4"}}
```

Limits: embeddings only; this checkpoint has no action classifier, and the JEPA predictor is not served. Only the centre square of each frame is seen, and a long video is sampled sparsely, so pick a window with `start` and `end`. Sound is ignored.

Parity: against transformers VJEPA2Model with VJEPA2VideoProcessor on MPS at float16, on the same weights and the same decoded frames, the clip embedding's cosine was 0.99990 to 0.99998 on three Kinetics clips (16 frames each, downscaled and upscaled frame sizes) and on 64 frames, max difference 0.13 on features reaching 45 (log/003-parity.md).

Source: [mlx-community/V-JEPA2-vitl-fpc64-256](https://huggingface.co/mlx-community/V-JEPA2-vitl-fpc64-256/tree/8eec39e665ffce8ac09174ea3ab2f2d65940ef5f), the FP16 conversion of [facebook/vjepa2-vitl-fpc64-256](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256), MIT. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input: a clip or 64 frames at 256 px, through the model's own video processor. One image: repeat 16 times along time. [model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not relevant.
- Parameters people recommend: 64 frames; latest transformers from GitHub. [model card, undated, first-hand]
- Arsenal call: `POST /arsenal/v1/models/vjepa2-vitl-fpc64-256/ops/embed`, or the same operation through MCP. Vendor example (model card, abridged):

```python
hf_repo = "facebook/vjepa2-vitl-fpc64-256"
model = AutoModel.from_pretrained(hf_repo)
processor = AutoVideoProcessor.from_pretrained(hf_repo)
video = processor(video, return_tensors="pt").to(model.device)
with torch.no_grad():
    video_embeddings = model.get_vision_features(**video)
```

**Good and bad**
- Good at: attentive probes on frozen features score 73.7% on SSv2 and 89.0% on Diving48 (ViT-L). Probe results, not Arsenal output. [vjepa2 README, undated, first-hand]
- Bad at: the action-conditioned robotics predictor is not in this checkpoint. [mlx-community card, undated, first-hand]
- Against alternatives: on EGTEA Gaze+ (Table 2), V-JEPA 2 kept more accuracy under corruptions than VideoMAEv2 (0.857 vs 0.810 retention) and VideoPrism (0.663); VideoPrism's cosine was higher. [arXiv 2605.15618, v1 2026-05-15, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: decord does not support macOS; the README names no official fix. [vjepa2 README, undated, first-hand]
- Fix or workaround: not official. Maintainer mmuckley (2025-06-24) said other users found PR 1's eva-decord helpful and prefers a README note for it; PR 1 closed unmerged 2025-08-18. He declined PR 31's decord2 as unvetted. [vjepa2 PRs 1 and 31, first-hand]

**Evidence**
- Strength: thin. Maker docs, one third-party study, two PR threads. No independent probe test found.
- Sources:
  - Model card, https://huggingface.co/facebook/vjepa2-vitl-fpc64-256, undated
  - vjepa2 README, https://github.com/facebookresearch/vjepa2, undated
  - mlx-community card, https://huggingface.co/mlx-community/V-JEPA2-vitl-fpc64-256, undated
  - arXiv 2605.15618, https://arxiv.org/abs/2605.15618, v1 2026-05-15
  - vjepa2 PR 1, https://github.com/facebookresearch/vjepa2/pull/1, 2025-06-11 to 2025-08-18
  - vjepa2 PR 31, https://github.com/facebookresearch/vjepa2/pull/31, opened 2025-06-23
