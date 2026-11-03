+++
hypothesis = "A Swift runner in the native worker (no Python) keeps the MLX runner's answers and speed."
change = "runners/swift: a Swift port of the vendor prompt/readout, the MLX forward (packed trees, pruned tail, fused short-conv Metal kernel, fp32 option readout) and an MLX bf16 SigLIP2 picture path; linked into Workers/Native as family d1; the Python runner removed."
measurement = "M5 Max, 8 Oct 2026, Mac in use (FLUX port work on CPU and GPU). Gate vs the saved vendor fp32 CPU answers, over ArsenalWire: max |dp| 0.003930, 24/24 (Python MLX runner 0.003930, 24/24); Polish probe counts identical to the Python runner's. Interleaved over the wire: one decision median 61.9 vs 59.0 ms (min 44.4 vs 45.1), 32 ads x 3 q 67.4 vs 107.3 ms/ad, x 1 q 29.0 vs 30.1 ms/ad; in-process forward of one ad x 3 q 47.1 vs 48.2 ms; load 0.52 vs 0.46 s."
verdict = "kept"
numbers = { gate_max_dp = 0.003930, gate_argmax = 24, questions = 24, python_gate_max_dp = 0.003930, picture_cases_max_dp = 0.0218, python_picture_cases_max_dp = 0.0066, single_median_ms = 61.9, python_single_median_ms = 59.0, single_min_ms = 44.4, python_single_min_ms = 45.1, forward_median_ms = 47.1, python_forward_median_ms = 48.2, batch32_3q_ms_per_item = 67.4, python_batch32_3q_ms_per_item = 107.3, batch32_1q_ms_per_item = 29.0, python_batch32_1q_ms_per_item = 30.1, load_s = 0.52, python_load_s = 0.46 }
links = ["runners/swift", "gate/parity-swift.json", "gate/parity-python-mlx.json", "gate/reference-fp32.json", "gate/cases.py", "bench/speed.py"]
+++
# Swift runner

## What runs

`runners/swift` is a Swift package the app's native worker links (`Workers/D1` is a link to it; SwiftPM names a path package after its folder, and every runner folder is called `swift`). The shipped binary is `Helpers/arsenal-worker-native`, started as `arsenal-worker-d1`; `build.sh` builds a standalone `arsenal-worker-d1` for the gate and benchmarks. It runs at the native worker's mlx-swift pin (9019419, MLX core 0.31.x; the Python runner had MLX 0.32.3).

- **Prompt and readout:** `Prompt.swift` ports the vendor's `prompt.py`/`api.py`, including Python's `json.dumps(indent=2)`, `str()`/`repr()` of non-text instructions and descriptions, option aliases and readout ids. `Tokenizer.swift` implements the checkpoint's byte-level BPE (the Split regex hand-written, `ignore_merges`, added tokens) and refuses any other `tokenizer.json`. Checked against `tokenizers` 0.22 and the vendor prompt code: 3,029 stress strings and all 139 gate/probe requests (25,013 tokens) give identical text and ids.
- **Forward:** a direct port of the Python engine (`Model.swift`): the same Metal short-conv kernel through `MLXFast.metalKernel` (load self-test against stock ops), fused `w1|w3` and `q|k|v`, per-tree attention, pruned tail, and the option logits as a fp32 dot product (the vendor's bf16 head rounds near ties into exact ties; this is the readout, not an fp32 model). Weights come through MLX's lazy safetensors loader, so the picture tower is read only on the first picture and load is 0.52 s.
- **Pictures:** decoding takes stored pixel values as Pillow does; the vendor's 1 MP bicubic cap reproduces Pillow's 22-bit fixed-point resampler and the processor's resize torchvision's uint8 antialiased kernel (int16 weights, precision per pass). Both are bit-exact on every PNG fixture (64x64, 640x480, 1600x1000 tiled, 60x300, RGBA, grey). JPEG differs: Apple's decoder and libjpeg disagree by up to 43/255 at sharp colour edges on the synthetic drawing (8 % of values). The SigLIP2 tower and projector run in MLX bf16.
- **Wire:** the Python adapter's checks, codes, statuses and messages: 45 edge and invalid requests sent to both runners gave identical replies except one Pillow-specific decode message (now the same wording). Cancellation stops a 256-item judge within one pass (0.52 s vs 0.45 s). Runner checker: 23/23 in Exact and Fast.

## Fidelity

Reference: `gate/reference-fp32.json`, the vendor code in fp32 on the CPU over every case in `gate/cases.py`, written before any candidate ran. Both runners went through the same wire harness (`gate/parity.py`):

| Group | Python MLX runner max abs dp, argmax | Swift runner max abs dp, argmax |
|---|---|---|
| gate (7 cases, 24 q) | 0.003930, 24/24 | 0.003930, 24/24 |
| pictures (9 cases, 20 q) | 0.0066, 20/20 | 0.0218, 20/20 |
| prompt edge cases (10 q) | 0.0072, 10/10 | 0.0072, 10/10 |
| probe ads x 3 q (96 q) | 0.0122, 94/96 | 0.0122, 94/96 |
| probe ads x 1 q (32 q) | 0.0175, 32/32 | 0.0175, 32/32 |
| probe requirements, full / excerpt (72 q each) | 0.0187 / 0.0106, 72/72 | 0.0187 / 0.0106, 72/72 |

Polish probe counts equal the Python runner's in every row (Polish Noul 24/32; fp32 25/32). Text answers match the Python runner to the fourth decimal: same arithmetic, same kernels.

The picture group's 0.0218 is one three-way near tie (two pictures: green 0.352 / red 0.327 / blue 0.268 in fp32). Projected-feature error against the fp32 tower (relative L2, 7 fixtures): the Python runner's torch MPS bf16 tower 0.025-0.064, this bf16 tower 0.022-0.096, an fp32 tower 0.004 (JPEG 0.046, decoder). The fp32 tower cut the picture group to 0.0068 but cost 27-250 ms per picture (+25-40 %) and changed no answer, so bf16 stays: the dtype is chosen by the gate. A residual stream in fp32 with bf16 products did not help (0.017-0.058).

## Speed

Interleaved over ArsenalWire in one session (`bench/speed.py`, both runners resident, ABBA rounds), client wall time; the Mac was busy with the FLUX port's CPU reference and GPU benchmarks, so medians move by tens of percent between sessions:

| Workload | Python MLX median (min) | Swift median (min) |
|---|---|---|
| one decision, one ad x 3 q (128 calls each) | 59.0 ms (45.1) | 61.9 ms (44.4) |
| 32 ads x 3 q, per ad | 107.3 ms (51.0) | 67.4 ms (53.4) |
| 32 ads x 1 q, per ad | 30.1 ms (25.3) | 29.0 ms (25.8) |
| picture 64x64 / 640x480 / 1600x1000 | 247 / 146 / 1987 ms | 264 / 130 / 891 ms |
| load | 0.46 s | 0.52 s |

The forward itself, in-process on the same one-ad tree, alternating processes (120 runs each): Python 48.2 ms median (42.9 min), Swift 47.1 ms (42.7). The single-decision medians differ by 3-9 ms (two sessions) with equal minima; the gap is inside this session's noise and was not resolved further. Footprint after load 5.84 GB (Python 5.90 GB); after a run with pictures 6.97 GB (Python with its torch tower 10.13 GB).

## Not done

No int8/int4: log/005 stands (int8 slower on batches, int4 changed 6/96 ad answers). No new optimization beyond the Python runner's: by project policy, ports stop at bf16 parity until he approves a precision recommendation.
