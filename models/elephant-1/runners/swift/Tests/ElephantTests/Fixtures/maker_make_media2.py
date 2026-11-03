# Second fixture set: textured pictures that shrink on both axes (PNG and a PIL-written JPEG), PIL decode edge cases
# (grey+alpha, RGBA with transparent areas, 16-bit grey, palette with transparency), and a lab-only real photo copy.
import sys, numpy as np
from PIL import Image, ImageDraw
out = sys.argv[1]; photo = sys.argv[2] if len(sys.argv) > 2 else None
rng = np.random.default_rng(7)
y, x = np.mgrid[0:260, 0:300]
base = np.stack([(x * 0.25) % 256, (y * 0.3) % 256, ((x + y) * 0.15) % 256], -1)
tex = np.clip(base + rng.normal(0, 40, base.shape), 0, 255).astype(np.uint8)
im = Image.fromarray(tex); d = ImageDraw.Draw(im)
d.rectangle([30, 30, 120, 90], fill=(240, 240, 20)); d.ellipse([170, 120, 270, 220], fill=(20, 40, 230)); d.text((130, 20), "INVOICE 4471", fill=(255, 255, 255))
im.save(f"{out}/textured-300x260.png")
y2, x2 = np.mgrid[0:300, 0:400]
tex2 = np.clip(np.stack([128 + 100 * np.sin(x2 / 23.0), 128 + 100 * np.cos(y2 / 17.0), (x2 ^ y2) % 256], -1) + rng.normal(0, 25, (300, 400, 3)), 0, 255).astype(np.uint8)
im2 = Image.fromarray(tex2); ImageDraw.Draw(im2).rectangle([130, 90, 270, 200], fill=(200, 30, 30)); im2.save(f"{out}/textured-400x300.jpg", quality=88)
# Decode edge cases (small; their PIL convert("RGB") bytes are the expected values).
la = Image.fromarray(np.stack([np.tile(np.arange(0, 250, 50, dtype=np.uint8), (4, 1)), np.array([[0, 64, 128, 200, 255]] * 4, dtype=np.uint8)], -1), "LA"); la.save(f"{out}/edge-la-5x4.png")
rgba = np.zeros((4, 5, 4), np.uint8); rgba[..., 0] = 200; rgba[..., 1] = np.arange(5) * 40; rgba[..., 2] = 30; rgba[..., 3] = [[0, 0, 128, 255, 255]] * 4
Image.fromarray(rgba, "RGBA").save(f"{out}/edge-rgba-5x4.png")
g16 = Image.fromarray(np.array([[0, 100, 255, 256, 1000], [65535, 300, 7, 128, 255]] * 2, dtype=np.uint16)); g16.save(f"{out}/edge-gray16-5x4.png")
pal = Image.fromarray(rgba[..., :3]).convert("P", palette=Image.ADAPTIVE, colors=8); pal.info["transparency"] = 0; pal.save(f"{out}/edge-palette-5x4.png", transparency=0)
if photo:
    with Image.open(photo) as p:
        p = p.convert("RGB"); p.thumbnail((1600, 1600)); p.save(f"{out}/photo-aerial.jpg", quality=92)
