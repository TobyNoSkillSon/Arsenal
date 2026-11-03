"""LCO-Embedding Omni media reference: the makers' processor pixel values (qwen-omni-utils fetch_image/fetch_video +
Qwen2_5OmniProcessor from the pinned snapshot's preprocessor_config.json), no model weights.
    python -I -B lco_media_capture.py <LCO snapshot dir>
Writes the image/video inputs once (if missing) and lco-media-reference.json (every 997th pixel value per input)."""
import json, sys
from pathlib import Path
import numpy as np, torch, av
from PIL import Image, ImageDraw
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from qwen_omni_utils.v2_5 import vision_process as vp

def _read_video_pyav(ele):
    # torchvision.io.read_video (removed in torchvision 0.29) decoded with PyAV to RGB, TCHW.
    with av.open(ele["video"]) as c:
        s = c.streams.video[0]
        frames = [torch.from_numpy(f.to_ndarray(format="rgb24")).permute(2, 0, 1) for f in c.decode(s)]
        fps = float(s.average_rate)
    video = torch.stack(frames); total = video.size(0)
    n = vp.smart_nframes(ele, total_frames=total, video_fps=fps)
    idx = torch.linspace(0, total - 1, n).round().long()
    return video[idx], dict(fps=fps, frames_indices=idx, total_num_frames=total, video_backend="pyav"), n / total * fps
vp.VIDEO_READER_BACKENDS["torchvision"] = _read_video_pyav
vp.get_video_reader_backend = lambda: "torchvision"

snap, here = Path(sys.argv[1]), Path(__file__).parent
if not (here / "lco-resize-1000x750.png").exists():
    im = Image.new("RGB", (1000, 750)); px = im.load()
    for y in range(750):
        for x in range(1000): px[x, y] = (x * 255 // 999, y * 255 // 749, (x * y) % 256)
    d = ImageDraw.Draw(im); d.rectangle((300, 200, 520, 420), fill=(250, 250, 250)); d.line((0, 0, 999, 749), fill=(0, 0, 0), width=3)
    d.ellipse((620, 300, 900, 580), fill=(30, 90, 220))
    im.save(here / "lco-resize-1000x750.png")
if not (here / "lco-alpha-336x252.png").exists():
    im = Image.new("RGBA", (336, 252), (200, 30, 30, 0)); d = ImageDraw.Draw(im)
    d.rectangle((20, 20, 180, 200), fill=(20, 120, 220, 255)); d.ellipse((150, 60, 320, 230), fill=(240, 200, 20, 128))
    im.save(here / "lco-alpha-336x252.png")
if not (here / "lco-video-640x360.mp4").exists():
    c = av.open(str(here / "lco-video-640x360.mp4"), "w"); s = c.add_stream("libx264", rate=4); s.width, s.height, s.pix_fmt = 640, 360, "yuv420p"
    s.options = {"crf": "18"}
    for i in range(8):
        f = Image.new("RGB", (640, 360), (240, 240, 230)); g = ImageDraw.Draw(f)
        g.rectangle((30 + 60 * i, 120, 150 + 60 * i, 240), fill=(200, 30, 60)); g.ellipse((420, 30, 600, 150), fill=(20, 150, 90))
        for p in s.encode(av.VideoFrame.from_image(f)): c.mux(p)
    for p in s.encode(): c.mux(p)
    c.close()
processor = Qwen2_5OmniProcessor.from_pretrained(snap)
out = {"stride": 997}
for name, ele in [("resize", {"type": "image", "image": str(here / "lco-resize-1000x750.png")}),
                  ("alpha", {"type": "image", "image": str(here / "lco-alpha-336x252.png")}),
                  ("video", {"type": "video", "video": str(here / "lco-video-392x308.mp4")}),
                  ("video_resize", {"type": "video", "video": str(here / "lco-video-640x360.mp4")})]:
    messages = [{"role": "user", "content": [ele, {"type": "text", "text": "x"}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(text=text, images=images, videos=videos, return_tensors="pt")
    key, grid = ("pixel_values", "image_grid_thw") if images else ("pixel_values_videos", "video_grid_thw")
    values = inputs[key].float().numpy().reshape(-1)
    out[name] = {"grid": inputs[grid][0].tolist(), "count": int(values.size), "samples": np.round(values[::997], 5).tolist()}
    print(name, out[name]["grid"], values.size)
(here / "lco-media-reference.json").write_text(json.dumps(out))
