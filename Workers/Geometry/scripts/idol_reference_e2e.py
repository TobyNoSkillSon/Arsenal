"""IDOL's own pipeline end to end on MPS (optimize off), DINOv3 from the local ViT-H+ snapshot and the FLUX.2 VAE from
klein's folder (bit-identical to dev's after IDOL's BF16 cast). Writes <out>/<name>_ref.ply and the camera."""
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
import torch
import idol.conditioning as conditioning
import idol.pipeline as pipeline_module
from idol.pipeline import IdolPipeline
weights, dino, klein, out = sys.argv[2:6]
images = sys.argv[6:]
conditioning.require_encoder_access = lambda repos: None
pipeline_module.require_encoder_access = lambda repos: None
import json as _json
cfg = _json.load(open(os.path.join(weights, 'config.json')))
cfg['image_encoders'] = {'dinov3': dino, 'flux2_vae': klein}
tmp = os.path.join(out, 'ref-weights'); os.makedirs(tmp, exist_ok=True)
for f in ['denoiser', 'decoder', 'decoder_gs', 'birefnet']:
    link = os.path.join(tmp, f + '.safetensors')
    if not os.path.exists(link): os.symlink(os.path.join(weights, f + '.safetensors'), link)
_json.dump(cfg, open(os.path.join(tmp, 'config.json'), 'w'))
p = IdolPipeline(tmp, device='mps', optimize=False)
for path in images:
    name = os.path.splitext(os.path.basename(path))[0]
    t = time.time()
    captured = {}
    encode = p.image_encoder.__call__
    def spy(images, encode=encode):
        c = encode(images); captured['pixels'] = images.float().cpu(); captured['feature1'] = c['feature1'].float().cpu(); captured['feature2'] = c['feature2'].float().cpu(); return c
    p.image_encoder.__call__ = spy
    class Spy:
        def __init__(self, e): self.e = e
        def __call__(self, images): return spy(images)
    p.image_encoder = Spy(p.image_encoder) if not hasattr(p.image_encoder, 'e') else p.image_encoder
    sample_flow = pipeline_module.sample_flow
    def flow_spy(model, noise, **kw):
        captured['noise_latent'] = noise['latent'].float().cpu(); captured['noise_camera'] = noise['camera'].float().cpu()
        out = sample_flow(model, noise, **kw); captured['camera'] = out['camera'].float().cpu(); return out
    pipeline_module.sample_flow = flow_spy
    decode = p.decode
    def decode_spy(latent, n, decode=decode):
        captured['latent'] = latent.float().cpu(); return decode(latent, n)
    p.decode = decode_spy
    o = p(path, seed=int(os.environ.get("SEED", "42")))
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in captured.items()}, os.path.join(out, name + '_ref.safetensors'))
    torch.mps.synchronize()
    o.gaussian.save_ply(os.path.join(out, name + f"_ref{os.environ.get('SEED','42')}.ply"))
    o.preprocessed_image.save(os.path.join(out, name + '_ref_input.png'))
    print(name, f'{time.time()-t:.1f} s', json.dumps(o.camera), flush=True)
