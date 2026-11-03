# Real check, 2 Nov 2026

One request through Arsenal's real service on an isolated profile (`arsenal-testserver --real`, private QA catalog from
the committed folder, Hugging Face client on loopback so Get could only import the local copy). M5 Max, Mac in use, on
AC. No speed or memory figures are claimed.

- Get: the pinned `mlx-community/sam3.1-bf16` file and the CLIP BPE asset imported from the local Hugging Face cache and
  cast to FP16 (`cast:sam31-fp16-v2`).
- `segment` with the text prompt "person" on a 1024 x 1280 photo of a man on a pier: HTTP 200, one 1024 x 1280 mask,
  score 0.976. Checked by eye: the mask covers the man's whole figure and nothing else.
- Not checked: a maker reference, point and box prompts, sessions, video tracking.
- Re-pin to `facebook/sam3.1` is parked: Meta's repository is gated and the Hugging Face account on this Mac has not
  accepted its terms. It ships a PyTorch pickle (`sam3.1_multiplex.pt`, 3.5 GB) rather than safetensors, so a re-pin
  also needs a Get recipe that reads it.
