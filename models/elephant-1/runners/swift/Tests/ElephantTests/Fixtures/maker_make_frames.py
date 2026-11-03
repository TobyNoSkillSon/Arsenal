# Fixed short frame lists for the video path: a ball moving right (4 frames), a light turning red to green (3 frames).
import sys
from PIL import Image, ImageDraw
out = sys.argv[1]
for i in range(4):
    im = Image.new("RGB", (320, 240), (235, 235, 240)); d = ImageDraw.Draw(im)
    d.rectangle([0, 200, 320, 240], fill=(90, 140, 70)); x = 30 + i * 80
    d.ellipse([x - 25, 150, x + 25, 200], fill=(220, 40, 40)); im.save(f"{out}/ball-{i}.png")
for i, color in enumerate([(220, 30, 30), (230, 180, 20), (30, 200, 60)]):
    im = Image.new("RGB", (200, 260), (30, 30, 35)); d = ImageDraw.Draw(im)
    d.rectangle([60, 20, 140, 240], fill=(15, 15, 15)); cy = [60, 130, 200][i]
    d.ellipse([70, cy - 30, 130, cy + 30], fill=color); im.save(f"{out}/light-{i}.png")
