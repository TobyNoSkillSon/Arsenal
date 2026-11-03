"""Capture IDOL reference tensors once, from Speridlabs' own idol-sdk code on the CPU, at small sizes.

Run: python -I -B Workers/Geometry/scripts/idol_capture.py <idol weights dir> <idol-sdk src dir> <klein snapshot dir> <out dir>

Writes <out>/ref.safetensors (inputs + outputs per component) and <out>/dino_tiny.safetensors / dino_hplus1.safetensors
(random-initialised DINOv3 encoders in the real H+ layout, since the trained H+ weights are gated).
Every module runs in the precision idol-sdk itself uses: transformer blocks FP16, norms/embeddings FP32 as stored,
DINOv3 BF16 (idol-sdk's default dinov3_dtype), the FLUX.2 VAE BF16.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

weights_dir, sdk_dir, klein_dir, out_dir = sys.argv[1:5]
sys.path.insert(0, sdk_dir)
os.environ.setdefault('ATTN_BACKEND', 'sdpa')
from tensordict import TensorDict  # noqa: E402

from idol.models import GaussianFixedlenDecoder, LatentSeqMMFlowModel, OctreeProbabilityFixedlenDecoder  # noqa: E402
from idol.sampler import sample_flow  # noqa: E402

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(0)
config = json.load(open(os.path.join(weights_dir, 'config.json')))
ref = {}


def load(model, name):
    model.load_state_dict(load_file(os.path.join(weights_dir, f'{name}.safetensors')), strict=True)
    return model.eval().requires_grad_(False)


# -- denoiser, on a short latent sequence (the first N fixed Sobol positions) -------------------------------------
N, LC = 256, 69
denoiser = load(LatentSeqMMFlowModel(**config['denoiser']['args']), 'denoiser')
denoiser.pos_pe = denoiser.pos_pe[:, :N].clone()
latent = torch.randn(1, N, 16)
camera = torch.randn(1, 1, 5)
feature1 = F.layer_norm(torch.randn(1, LC, 1280) * 3 + 0.5, (1280,))
feature2 = torch.randn(1, LC, 128)
feature2[:, :5] = 0
x = TensorDict({'latent': latent, 'camera': camera}, batch_size=1)
cond = TensorDict({'feature1': feature1, 'feature2': feature2}, batch_size=1)
neg = cond.apply(torch.zeros_like)
with torch.no_grad():
    t = torch.tensor([637.5])
    ref['den_latent'], ref['den_camera'], ref['den_t'] = latent, camera, t
    ref['den_feature1'], ref['den_feature2'] = feature1, feature2
    out = denoiser(x, t, cond)
    ref['den_v_latent'], ref['den_v_camera'] = out['latent'], out['camera']
    out = denoiser(x, t, neg)
    ref['den_vneg_latent'], ref['den_vneg_camera'] = out['latent'], out['camera']
    # intermediates for localisation
    ref['den_t_emb'] = denoiser.t_embedder(t)
    h = denoiser.input_layer(latent).half() + denoiser.pos_embedder(denoiser.pos_pe).half()
    ref['den_hx0'] = h
    ref['den_repo0'] = torch.view_as_real(denoiser.noise_repo_layers[0](h))
    # three guided Euler steps
    sample = sample_flow(denoiser, x, cond=cond, neg_cond=neg, steps=3, rescale_t=3.0, cfg_strength=3.0, verbose=False)
    ref['den_sample3_latent'], ref['den_sample3_camera'] = sample['latent'], sample['camera']
print('denoiser done', flush=True)
del denoiser

# -- decoders -------------------------------------------------------------------------------------------------------
dec_cond = torch.randn(1, N, 16)
cells = torch.rand(1, 64, 3)
decoder = load(OctreeProbabilityFixedlenDecoder(**config['decoder']['args']), 'decoder')
with torch.no_grad():
    ref['dec_cond'], ref['dec_cells'] = dec_cond, cells
    ref['dec_level'] = torch.tensor([16])
    ref['dec_logits'] = decoder(cells, torch.tensor([16]), dec_cond)
del decoder
decoder_gs = load(GaussianFixedlenDecoder(**config['decoder_gs']['args']), 'decoder_gs')
with torch.no_grad():
    anchors = torch.rand(1, 64, 3)
    features = decoder_gs(anchors, dec_cond)
    ref['gs_anchors'], ref['gs_features'] = anchors, features
    g = decoder_gs.to_representation(anchors, features)[0]
    ref['gs_xyz'], ref['gs_features_dc'], ref['gs_scaling'] = g.xyz, g.features_dc, g.scaling
    ref['gs_rotation'], ref['gs_opacity'] = g.rotation, g.opacity
del decoder_gs
print('decoders done', flush=True)

# -- FLUX.2 VAE encoder (klein weights) on a small image ------------------------------------------------------------
from diffusers.models import AutoencoderKLFlux2  # noqa: E402

vae = AutoencoderKLFlux2.from_pretrained(klein_dir, subfolder='vae', torch_dtype=torch.bfloat16).eval()
with torch.no_grad():
    image = torch.rand(1, 3, 128, 128)
    image[:, :, :20] = 0   # a black border, as the cutout's composite has
    dist = vae.encode(image.to(torch.bfloat16) * 2 - 1).latent_dist
    ref['vae_image'], ref['vae_mean'], ref['vae_std'] = image, dist.mean.float(), dist.std.float()
    latents = dist.mean
    b, c, hh, ww = latents.shape
    latents = latents.view(b, c, hh // 2, 2, ww // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, hh // 2, ww // 2)
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.dtype)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(latents.dtype)
    tokens = ((latents - mean) / std).float().flatten(2).transpose(1, 2)
    ref['vae_tokens_from_mean'] = torch.cat([torch.zeros(1, 5, tokens.shape[2]), tokens], dim=1)
print('vae done', flush=True)
del vae

save_file({k: v.contiguous() for k, v in ref.items()}, os.path.join(out_dir, 'ref.safetensors'))

# -- DINOv3 in the H+ layout, random weights (the trained ones are gated) ----------------------------------------------
from transformers import DINOv3ViTConfig, DINOv3ViTModel  # noqa: E402


def dino(name, layers, hidden, heads, intermediate, size):
    cfg = DINOv3ViTConfig(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
                          intermediate_size=intermediate, use_gated_mlp=True, hidden_act='silu', num_register_tokens=4,
                          rope_theta=100.0, patch_size=16, layerscale_value=1.0e-5, key_bias=False, query_bias=True,
                          value_bias=True, proj_bias=True, mlp_bias=True, layer_norm_eps=1e-5, image_size=224)
    torch.manual_seed(1)
    model = DINOv3ViTModel(cfg).eval()
    with torch.no_grad():   # spread the norms and layer scales so the test sees non-trivial values
        for n, p in model.named_parameters():
            if 'lambda1' in n:
                p.copy_(torch.rand_like(p) * 0.5 + 0.1)
            elif 'norm' in n:
                p.copy_(1 + 0.2 * torch.randn_like(p) if n.endswith('weight') else 0.1 * torch.randn_like(p))
            elif 'register' in n or 'cls' in n or 'mask' in n:
                p.copy_(torch.randn_like(p))
    model = model.to(torch.bfloat16)
    pixels = torch.rand(1, 3, size, size)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    with torch.no_grad():
        tokens = model(pixel_values=((pixels - mean) / std).to(torch.bfloat16)).last_hidden_state.float()
        tokens = F.layer_norm(tokens, tokens.shape[-1:])
    tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
    tensors['ref.pixels'] = pixels
    tensors['ref.tokens'] = tokens
    save_file(tensors, os.path.join(out_dir, f'{name}.safetensors'))
    print(name, tokens.shape, flush=True)


dino('dino_tiny', 2, 128, 4, 160, 48)       # head_dim 32, three registers of RoPE frequencies
dino('dino_hplus1', 1, 1280, 20, 5120, 64)  # one real-size H+ layer
