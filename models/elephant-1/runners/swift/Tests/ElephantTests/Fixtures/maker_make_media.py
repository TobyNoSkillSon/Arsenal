# Fixed images for the image-path parity: the maker's own probe (qev mm_engine _select_fork_mode) and three more.
import sys
from PIL import Image, ImageDraw
out = sys.argv[1]
img = Image.new("RGB", (320, 240), (250, 250, 250)); d = ImageDraw.Draw(img)
d.rectangle([20, 20, 140, 100], fill=(200, 40, 40)); d.ellipse([180, 120, 300, 220], fill=(40, 80, 200)); img.save(f"{out}/probe-320x240.png")
for name, color in (("red", (220, 30, 30)), ("green", (30, 200, 60))):
    w, h = (256, 256) if name == "red" else (400, 300)
    im = Image.new("RGB", (w, h), (30, 30, 35)); d = ImageDraw.Draw(im)
    d.rectangle([w // 2 - 40, 20, w // 2 + 40, h - 20], fill=(15, 15, 15))
    cy = 60 if name == "red" else h - 70
    d.ellipse([w // 2 - 30, cy - 30, w // 2 + 30, cy + 30], fill=color)
    im.save(f"{out}/light-{name}-{w}x{h}.png")
im = Image.new("RGB", (50, 40), (240, 240, 240)); ImageDraw.Draw(im).rectangle([10, 8, 38, 32], fill=(20, 170, 60)); im.save(f"{out}/tiny-50x40.png")
