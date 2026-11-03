# Decode and resize reference for the weight-free MediaTests: PIL's Image.open(f).convert("RGB") bytes (sha256; full
# bytes for the small edge cases) and torchvision's antialiased bicubic resize of those bytes to the processor's
# smart_resize size (the maker processor's resize step, transformers 5.19.0). JPEG: every 7th PIL-decoded byte is stored so
# the ImageIO difference can be measured. usage: python -I -B maker_decode_fixture.py <media dir> <out.json>
import hashlib, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torchvision.transforms.v2 import functional as tvF
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
out = {}
for f in sorted(Path(sys.argv[1]).iterdir()):
    with Image.open(f) as im:
        rgb = np.asarray(im.convert("RGB"))
    h, w = rgb.shape[:2]
    rh, rw = smart_resize(h, w, factor=32, min_pixels=65536, max_pixels=16777216)
    t = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None]
    resized = tvF.resize(t, [rh, rw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)[0].permute(1, 2, 0).numpy()
    entry = {"width": w, "height": h, "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
             "resized": {"width": rw, "height": rh, "sha256": hashlib.sha256(np.ascontiguousarray(resized).tobytes()).hexdigest()}}
    if h * w <= 64:
        entry["rgb"] = rgb.flatten().tolist()
    if f.suffix == ".jpg":
        entry["rgb_sample"] = {"stride": 7, "values": rgb.flatten()[::7].tolist()}
    out[f.name] = entry
json.dump(out, open(sys.argv[2], "w"))
