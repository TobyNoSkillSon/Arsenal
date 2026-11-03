# Real check, 2 Nov 2026

One request through Arsenal's real service on an isolated profile (`arsenal-testserver --real`, private QA catalog from
the committed folder, Hugging Face client on loopback so Get could only import the local copy). M5 Max, Mac in use, on
AC. No speed or memory figures are claimed.

- Get: NVIDIA's `model.safetensors` (FP32, PyTorch layout) cloned from the local Hugging Face cache and cast to BF16 by
  `cast:re-use`. The stored `model_mlx.safetensors` has the same 1,416 tensor names and shapes as the earlier
  `faraday/re-use-mlx` file; every value is within BF16 rounding of NVIDIA's FP32 (max relative error 0.0039).
- Load BF16, then `enhance` (denoise) on a fixed 1 s, 16 kHz synthetic voiced signal with noise: HTTP 200, 16,000
  frames out at 16 kHz.
- Parity against NVIDIA's own code (inference.py pipeline on the GPU in FP16, captured once): SNR 39.4 dB, correlation
  0.99996, output RMS 0.0868 against 0.0864. Golden: `gate/golden/golden.json` (tolerance: at least 30 dB).
- Not checked: speech-quality scores on real recordings, the other two modes (restore, upscale), FP16 serving, and
  which kernel path (Metal or stock MLX) served the request.
