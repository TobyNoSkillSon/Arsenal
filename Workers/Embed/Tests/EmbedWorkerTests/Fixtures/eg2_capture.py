"""EmbeddingGemma 2 no-weight reference: Transformers 5.22.1 functions on fixed small inputs (no model weights).
Run once with an environment holding transformers==5.22.1, torch, torchvision, numpy, pillow and jinja2:
    python -I -B eg2_capture.py <embeddinggemma-2 snapshot dir>   (reads its processor and template files only)
Writes eg2-reference.json and the eg2-picture.png fixture next to this script."""
import hashlib, json, sys
from pathlib import Path
import jinja2, numpy as np, torch, transformers
from PIL import Image, ImageOps
from transformers import AutoProcessor
from transformers.models.embedding_gemma2.processing_embedding_gemma2 import EmbeddingGemma2Processor, get_aspect_ratio_preserving_size
from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig, Gemma4VisionConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioAttention, Gemma4AudioModel, Gemma4VisionPooler, Gemma4VisionRotaryEmbedding
from transformers.masking_utils import create_bidirectional_mask
from transformers.models.gemma4.modeling_gemma4 import sliding_window_mask_function
from transformers.video_processing_utils import VideoMetadata


def sscp(mask):
    """The processor's replace_audio_token count: the frame mask through two stride-2 convolutions."""
    for _ in range(2):
        mask = mask[::2][: (len(mask) + 2 - 3) // 2 + 1]
    return int(mask.sum())

snap = Path(sys.argv[1]); here = Path(__file__).parent
proc = AutoProcessor.from_pretrained(snap)
out = {"transformers": transformers.__version__}

# Chat template layouts (the message rendering sentence-transformers uses), with and without manual placeholders.
template = jinja2.Environment().from_string((snap / "chat_template.jinja").read_text())
layouts = [
    [{"type": "text", "text": "hello world"}],
    [{"type": "image"}],
    [{"type": "text", "text": "a photo"}, {"type": "image"}],
    [{"type": "image"}, {"type": "text", "text": " caption"}, {"type": "audio"}, {"type": "video"}],
    [{"type": "text", "text": "Shoes <|image|> mesh <|image|> test: <|video|>"}, {"type": "image"}, {"type": "image"}, {"type": "video"}],
    [{"type": "text", "text": "Rain <|audio|>"}, {"type": "audio"}, {"type": "text", "text": " end"}],
]
out["layouts"] = []
for content in layouts:
    for prompt in [None, "task: search result | query: "]:
        messages = ([{"role": "system", "content": [{"type": "text", "text": prompt}]}] if prompt else []) + [{"role": "user", "content": content}]
        out["layouts"].append({"prompt": prompt, "content": [c.get("text", c["type"]) if c["type"] == "text" else c["type"] for c in content],
                               "kinds": [c["type"] for c in content], "text": template.render(messages=messages)})

# Resize sizes for the image (280) and video (140) budgets.
cases = [(320, 480), (500, 360), (61, 97), (1080, 1920), (30, 4000), (4000, 30), (48, 48), (896, 896), (100, 5000)]
out["sizes"] = [{"h": h, "w": w, "tokens": t, "out": list(get_aspect_ratio_preserving_size(h, w, 16, t * 9, 3))} for h, w in cases for t in (280, 140)]

# A fixed noisy picture through the image and video processors: the resized, rescaled patches as 8-bit levels.
rng = np.random.default_rng(7)
y, x = np.mgrid[0:61, 0:97]
base = np.stack([x * 2.6, y * 4.1, (x + y) * 1.6], -1) + rng.integers(0, 60, (61, 97, 3))
Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(here / "eg2-picture.png")
picture = Image.open(here / "eg2-picture.png").convert("RGB")
def levels(t):
    return np.rint(t.numpy() * 255).astype(np.uint8)
img = proc.image_processor(images=[picture], return_tensors="pt")
real = int((img["image_position_ids"][0, :, 0] >= 0).sum())
image_levels = levels(img["pixel_values"][0, :real])
vid = proc.video_processor(videos=[np.stack([np.asarray(picture)] * 2)], do_sample_frames=False, return_tensors="pt")
vreal = int((vid["video_position_ids"][0, :, 0] >= 0).sum())
video_levels = levels(vid["pixel_values_videos"][0, :vreal])
out["picture"] = {
    "sha256": hashlib.sha256((here / "eg2-picture.png").read_bytes()).hexdigest(),
    "image": {"patches": real, "columns": int(img["image_position_ids"][0, :real, 0].max()) + 1, "soft_tokens": int(img["num_soft_tokens_per_image"][0]),
              "levels_sha256": hashlib.sha256(image_levels.tobytes()).hexdigest(), "first_patch": image_levels[0].tolist(),
              "positions_head": img["image_position_ids"][0, :5].tolist()},
    "video": {"patches": vreal, "columns": int(vid["video_position_ids"][0, :vreal, 0].max()) + 1,
              "levels_sha256": hashlib.sha256(video_levels.tobytes()).hexdigest(), "first_patch": video_levels[0].tolist()},
}

# EXIF orientation 2–8 on a 3×2 picture (Pillow's exif_transpose).
small = Image.fromarray(np.arange(18, dtype=np.uint8).reshape(2, 3, 3) * 10)
out["orientation"] = []
for o in range(2, 9):
    im = small.copy(); exif = im.getexif(); exif[0x0112] = o; im.info["exif"] = exif.tobytes()
    t = ImageOps.exif_transpose(im)
    out["orientation"].append({"orientation": o, "width": t.width, "height": t.height, "pixels": np.asarray(t).reshape(-1).tolist()})

# Frame sampling at 1 fps with the 32-frame uniform cap.
vp = proc.video_processor
out["frames"] = []
for total, fps in [(48, 12.0), (300, 29.97), (10, 2.0), (5000, 25.0), (1, 30.0), (90, 30.0), (3, 1.0)]:
    md = VideoMetadata(total_num_frames=total, fps=fps, duration=total / fps)
    out["frames"].append({"total": total, "fps": fps, "indices": vp.sample_frames(md, fps=vp.fps, max_frames=vp.max_frames, overflow_strategy=vp.overflow_strategy).tolist()})

# Log-mel features of a fixed 1.37 s chirp, the frame mask and the soft-token count.
fe = proc.feature_extractor
t = np.arange(21_937) / 16000
wave = (0.4 * np.sin(2 * np.pi * (300 * t + 900 * t * t)) + 0.1 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
f = fe([wave], padding=True, return_tensors="np")
feats, mask = f["input_features"][0], f["input_features_mask"][0]
out["mel"] = {"samples": len(t), "frames": int(feats.shape[0]), "valid": int(mask.sum()), "every": 7, "rows": np.round(feats[::7], 5).tolist(),
              "chirp": [300, 900, 0.4, 3000, 0.1], "tokens": sscp(mask),
              "filter_column_sums": np.round(fe.mel_filters.sum(0), 7).tolist()}
out["audio_tokens"] = [{"samples": n, "tokens": sscp(fe([np.full(n, 0.1, np.float32)], padding=True)["input_features_mask"][0])}
                       for n in [400, 1600, 16000, 45830, 160000, 480000]]

# Audio tower blocks: the blocked 5-D mask and the relative shift, from the maker's own modules (random init, no weights).
torch.manual_seed(0)
audio_cfg = {k: v for k, v in json.loads((snap / "config.json").read_text())["audio_config"].items() if k not in ("architectures", "_name_or_path", "id2label", "label2id")}
acfg = Gemma4AudioConfig(**{**audio_cfg, "num_hidden_layers": 1, "hidden_size": 64, "num_attention_heads": 2, "output_proj_dims": 32, "subsampling_conv_channels": [8, 16]})
acfg._attn_implementation = "eager"
model = Gemma4AudioModel(acfg)
valid = torch.tensor([[True] * 27 + [False] * 4])
m4 = create_bidirectional_mask(config=acfg, inputs_embeds=torch.zeros(1, 31, 64), attention_mask=valid,
                               and_mask_function=sliding_window_mask_function((acfg.attention_context_left - 1, acfg.attention_context_right)))
m5 = model._convert_4d_mask_to_blocked_5d(m4)
out["audio_mask"] = {"valid": valid[0].tolist(), "allowed": (m5[0, 0] == 0).reshape(-1).tolist(), "shape": list(m5.shape[2:])}
attn = Gemma4AudioAttention(acfg, 0)
rel = torch.randn(1, 2, 3, acfg.attention_chunk_size, attn.max_past_horizon + 1)
out["rel_shift"] = {"input": rel.reshape(-1).tolist(), "shape": list(rel.shape), "output": attn._rel_shift(rel).reshape(-1).tolist(), "context": attn.context_size}

# Vision: axial RoPE tables and the position-ordered pooling.
vcfg = Gemma4VisionConfig(**{k: v for k, v in json.loads((snap / "config.json").read_text())["vision_config"].items() if k not in ("architectures", "_name_or_path", "id2label", "label2id")})
rot = Gemma4VisionRotaryEmbedding(vcfg)
cols, rows = 6, 3
pos = torch.tensor([[[c, r] for r in range(rows) for c in range(cols)]])
cos, sin = rot(torch.zeros(1, dtype=torch.float32), pos)
out["vision_rope"] = {"columns": cols, "rows": rows, "head_dim": vcfg.head_dim, "theta": vcfg.rope_parameters["rope_theta"], "cos": cos[0].reshape(-1).tolist(), "sin": sin[0].reshape(-1).tolist()}
hidden = torch.randn(1, cols * rows, 4)
pooled, _ = Gemma4VisionPooler(vcfg)._avg_pool_by_positions(hidden, pos, cols * rows // 9)
out["vision_pool"] = {"columns": cols, "rows": rows, "hidden": hidden.reshape(-1).tolist(), "pooled": pooled.reshape(-1).tolist()}

(here / "eg2-reference.json").write_text(json.dumps(out))
print(here / "eg2-reference.json", (here / "eg2-reference.json").stat().st_size)
