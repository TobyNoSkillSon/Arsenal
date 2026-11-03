# d1-3B Swift runner: provenance

- `Sources/D1Engine/Prompt.swift`: a Swift port of LiquidAI d1-3B's `prompt.py` and `api.py` (https://huggingface.co/LiquidAI/d1-3B, revision 051bcc464b01b9f92942b364d9586b0ef5912432), LFM Open License v1.0 (`LICENSE-LFM-Open-1.0`); modified by Arsenal (language, structure). Its rendered prompts and token ids equal the vendor's on every gate and probe request (log/007).
- `Sources/D1Engine/Model.swift`: the vendor's hybrid System One tree (`hybrid.py`) as an MLX forward, ported from Arsenal's earlier Python MLX runner (MIT, git history `models/d1-3b/runners/mlx/engine.py`), including its fused short-convolution Metal kernel.
- `Sources/D1Engine/Vision.swift`, `Pictures.swift`: SigLIP2 NaFlex, the LFM2-VL projector and image processor after transformers 5.14 (Apache-2.0) and Pillow's / torchvision's resamplers; Arsenal-written MIT code.
- `Sources/D1Engine/PyJSON.swift`: the order-preserving parser from Workers/Decisions `OrderedJSON.swift` (from Verdict, Apache-2.0), extended with Python's json.dumps, str and repr.
- Everything else: Arsenal, MIT.
