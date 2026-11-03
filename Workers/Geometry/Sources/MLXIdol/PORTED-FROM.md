# IDOL in MLX

Ported from `speridlabs/idol-sdk@4702a7f00850e34b6e83a2170cc123c3e9570744` (Apache-2.0, `Workers/Geometry/Licenses/LICENSE-idol-sdk`): the
denoiser (`idol/models/dit.py`, itself adapted from diffusers' Wan transformer, Apache-2.0), the layers
(`idol/models/layers.py`), the octree and Gaussian decoders (`idol/models/decoder.py`), the sampler, image
preparation, Gaussian activation, PLY layout and camera decoding. Model weights: `speridlabs/idol` (Apache-2.0).

- DINOv3 ViT-H+/16 follows transformers' `DINOv3ViTModel` (Apache-2.0 code). Its weights are Meta's, under the DINOv3
  License, downloaded by each user from `facebook/dinov3-vith16plus-pretrain-lvd1689m` after accepting Meta's terms.
- The FLUX.2 VAE encoder is `Flux3Klein`'s. IDOL loads FLUX.2-dev's VAE; Arsenal uses FLUX.2 klein 4B's (Apache-2.0),
  whose 251 tensors equal FLUX.2-dev's bit for bit after IDOL's own BF16 cast, with the same configuration.
- Not ported: IDOL's bundled BiRefNet (Arsenal's own BiRefNet removes the background first), torch.compile and the
  CUDA attention backends, the Gradio demo.
- Random numbers come from MLX, so a seed gives a different sample than the PyTorch reference with the same seed.
- The preview rasterizer is Arsenal's own (the 3D Gaussian Splatting projection, simplified); it is not IDOL code.
