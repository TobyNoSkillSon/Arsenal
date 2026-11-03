"""LCO-Embedding Omni no-weight reference: Transformers 5.22.1 and qwen-omni-utils functions on fixed small inputs.
Run once with an environment holding transformers==5.22.1, qwen-omni-utils, jinja2, torch and numpy:
    python -I -B lco_capture.py <LCO snapshot dir>   (reads preprocessor_config.json and the ST chat template only)"""
import json, sys, types
from pathlib import Path
import numpy as np, torch, jinja2, transformers
from transformers import WhisperFeatureExtractor
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniPreTrainedModelForConditionalGeneration as Base
from transformers.vision_utils import get_vision_window_index
from qwen_omni_utils.v2_5 import vision_process as vp
snap = Path(sys.argv[1]); dest = Path(__file__).parent / "lco-reference.json"
fx = WhisperFeatureExtractor.from_pretrained(snap)
sr = 16000; t = np.arange(21_937) / sr  # 1.37 s, not a multiple of the hop
wave = (0.4 * np.sin(2 * np.pi * (300 * t + 900 * t * t)) + 0.1 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
f = fx(wave, sampling_rate=sr, padding="max_length", return_attention_mask=True, return_tensors="np")
frames = int(f["attention_mask"].sum()); feats = f["input_features"][0][:, :frames]
mel = {"samples": len(t), "frames": frames, "every": 7, "rows": np.round(feats[:, ::7].T, 5).tolist(),
       "filter_column_sums": np.round(fx.mel_filters.sum(0), 7).tolist(), "chirp": [300, 900, 0.4, 3000, 0.1]}
fake = types.SimpleNamespace(spatial_merge_size=2, config=types.SimpleNamespace(
    image_token_id=151655, video_token_id=151656, audio_token_id=151646, vision_start_token_id=151652,
    audio_start_token_id=151647, position_id_per_seconds=25, seconds_per_chunk=2))
fake.get_llm_pos_ids_for_vision = types.MethodType(Base.get_llm_pos_ids_for_vision, fake)
T, I, V, A, VS, VE, AS, AE = 5, 151655, 151656, 151646, 151652, 151653, 151647, 151648
img, vid, aud_frames = (1, 4, 6), (2, 4, 4), 437
aud_tokens = ((aud_frames - 1) // 2 + 1 - 2) // 2 + 1
ids = [T, T, VS] + [I] * 6 + [VE, T, AS] + [A] * aud_tokens + [AE, T, VS] + [V] * 8 + [VE, T, T]
pos, delta = Base.get_rope_index(fake, torch.tensor([ids]), torch.tensor([img]), torch.tensor([vid]), torch.ones(1, len(ids), dtype=torch.long),
                                 False, torch.tensor([aud_frames]), torch.tensor([1.0]))
rope = {"ids": ids, "image": img, "video": vid, "audio_frames": aud_frames, "audio_tokens": aud_tokens, "positions": pos[:, 0].tolist()}
windows = []
for g in [(1, 8, 12), (2, 4, 6), (1, 16, 16), (1, 18, 10)]:
    index, cu = get_vision_window_index(torch.tensor([g]), 2, 112, 14)
    windows.append({"grid": g, "index": index.tolist(), "cu": cu.tolist()})
resize = []
for h, w, lo, hi in [(252, 336, 3136, 12845056), (100, 3000, 3136, 12845056), (4000, 3000, 3136, 12845056), (30, 41, 3136, 12845056), (308, 392, 100352, 602112), (720, 1280, 100352, 602112), (130, 255, 100352, 602112)]:
    resize.append({"h": h, "w": w, "min": lo, "max": hi, "out": list(vp.smart_resize(h, w, 28, lo, hi))})
nframes = [{"total": n, "fps": r, "frames": vp.smart_nframes({}, total_frames=n, video_fps=r)} for n, r in [(8, 4.0), (300, 30.0), (100000, 25.0), (5, 2.0), (61, 24.0)]]
budget = [{"frames": n, "max": max(min(768 * 784, 128000 * 784 * 0.9 / n * 2), int(128 * 784 * 1.05))} for n in [4, 100, 300, 768]]
st = jinja2.Environment().from_string((snap / "additional_chat_templates" / "sentence_transformers.jinja").read_text())
prompts = []
for content in [[{"type": "text", "text": "hello world"}], [{"type": "image", "image": "x"}], [{"type": "audio", "audio": "x"}],
                [{"type": "video", "video": "x"}], [{"type": "text", "text": "A cooking tutorial"}, {"type": "video", "video": "x"}],
                [{"type": "image", "image": "x"}, {"type": "audio", "audio": "x"}], [{"type": "audio", "audio": "x"}, {"type": "text", "text": "Rain"}, {"type": "image", "image": "y"}]]:
    prompts.append({"content": [c["type"] if c["type"] != "text" else c["text"] for c in content],
                    "kinds": [c["type"] for c in content],
                    "text": st.render(messages=[{"role": "user", "content": content}], add_generation_prompt=True)})
json.dump({"transformers": transformers.__version__, "mel": mel, "rope": rope, "windows": windows, "resize": resize, "nframes": nframes,
           "video_budget": budget, "prompts": prompts}, dest.open("w"))
print(dest, dest.stat().st_size)
