"""DINOv3 ViT-H+ reference with the trained weights: FP32 truth and BF16 (idol-sdk's default), IDOL's token post-processing,
on one gallery picture at 256 px; plus the residual stream's peak per layer (does it fit FP16?)."""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from transformers import DINOv3ViTModel
model_dir, image_path, out = sys.argv[1:4]
im = Image.open(image_path).convert('RGB').resize((256, 256), Image.LANCZOS)
pixels = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255)[None]
mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1); std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
x = (pixels - mean) / std
res = {'pixels': pixels}
model, info = DINOv3ViTModel.from_pretrained(model_dir, torch_dtype=torch.float32, output_loading_info=True)
model = model.eval()
print('loading info', {k: len(v) for k, v in info.items() if hasattr(v, '__len__')}, flush=True)
assert not info['missing_keys'], info['missing_keys'][:5]
layers = model.layer if hasattr(model, 'layer') else model.model.layer
peaks = []
hooks = [layer.register_forward_hook(lambda m, i, o: peaks.append((o[0] if isinstance(o, tuple) else o).abs().max().item())) for layer in layers]
with torch.no_grad():
    t = model(pixel_values=x).last_hidden_state
    res['tokens_fp32'] = F.layer_norm(t, t.shape[-1:])
for h in hooks: h.remove()
print('residual peak per layer:', [round(p) for p in peaks], 'FP16 max 65504', flush=True)
model = model.to(torch.bfloat16)
with torch.no_grad():
    t = model(pixel_values=x.to(torch.bfloat16)).last_hidden_state.float()
    res['tokens_bf16'] = F.layer_norm(t, t.shape[-1:])
rel = lambda a, b: ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()
print('torch bf16 vs fp32 relRMS', rel(res['tokens_bf16'], res['tokens_fp32']), flush=True)
save_file({k: v.contiguous() for k, v in res.items()}, out)
