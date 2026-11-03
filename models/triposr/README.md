# TripoSR

Turns one photo of an object into a coloured 3D mesh. TripoSR is Stability AI and Tripo AI's feed-forward reconstruction model (2024, LRM design): a DINO ViT-B/16 reads the image, a 16-layer transformer writes a triplane, and a small NeRF decoder gives a density and colour field that marching cubes turns into a mesh. Trained on renders of a curated CC-BY subset of Objaverse.

Input: `image` (PNG or JPEG), and optionally `resolution` (marching-cubes grid per side, 256 to 512, default 256 as in the makers' run.py; the 3D kind allows up to 1536, but 512 is this port's limit and larger values are refused). `seed`, `texture_size` and `simplify` are accepted by the 3D kind but have no effect. Output: `files`, one GLB (`id` `mesh`, name `mesh.glb`, `mime` `model/gltf-binary`, label "3D model") with `path`, `bytes` and `sha256`: one triangle mesh with smooth normals and per-vertex colours.

Example request:

```json
{"image": {"path": "/absolute/path/to/cutout.png"}}
```

How it runs: an image with transparency is prepared as the makers' run.py does after background removal: crop to the visible object, pad to a square with the object at 85% of the side, put it on 50% grey, then resize to 512 with antialiased bilinear filtering. An image with no transparency is used as it is (the makers' `--no-remove-bg` path). The density is sampled on the grid and the surface taken at density 25 with torchmcubes' marching cubes, as the makers do. The GLB is turned upright as in the makers' own GLB export (gradio_app.py), with the object's front toward +Z as glTF expects; faces are wound outward and colours stored linear, as glTF requires.

Limits: needs a cut-out object (a PNG with a transparent background, e.g. from Arsenal's Remove Background); the makers' automatic background removal (rembg) is not included. One object per image; thin parts and the unseen back come out smooth or blobby. Colours are per vertex, not a texture.

Parity: matches the maker's PyTorch TripoSR (MPS) on the triplanes at cosine 0.9999985 and on a 128-cube density grid at cosine 0.99995 or better, with 99.996% of grid points on the same side of the surface, on two of the makers' example images (`log/003-parity.md`).

Source: [stabilityai/TripoSR](https://huggingface.co/stabilityai/TripoSR/tree/4e22858523f295f3c2b3852c39c90f399ea0e7d3) (`model.safetensors` from the Hugging Face convert bot's pull request 11 on the makers' revision 5b52193), MIT; code [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR), tech report [arXiv 2403.02151](https://arxiv.org/abs/2403.02151). Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: One or more image paths, separated by spaces, as `run.py` takes them. [TripoSR GitHub README, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not applicable.
- Parameters people recommend: The README documents `--bake-texture` and `--texture-resolution` (pixels), with no recommended value. Default options take about 6 GB VRAM for one image. [TripoSR GitHub README, undated, first-hand]
- Arsenal call: `POST /arsenal/v1/models/triposr/ops/image_to_3d`, or the same operation through MCP. Vendor example: `python run.py examples/chair.png --output-dir output/`, adding `--bake-texture` for a texture. [TripoSR GitHub README, undated, first-hand]

**Good and bad**
- Good at: The README says under 0.5 seconds per model on an NVIDIA A100 GPU. [TripoSR GitHub README, undated, first-hand]
- Bad at: The model card points to SF3D as newer, with faster generation and more game-ready assets. [TripoSR model card, undated, first-hand]
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: The texture is very low resolution and often buggy, though the mesh is fine. Fix or workaround: Suggested in the thread, none confirmed: UV-unwrap in Blender and apply the source image as the texture, or upscale with ControlNet Tile or SUPIR. [TripoSR issue #22, 2024-03-05, first-hand report; fixes are opinion]
- Complaint or known bug: `bake_texture.py` raises a CPU/CUDA device mismatch in `positions_to_colors` on CUDA. Fix or workaround: Reported by one user, not confirmed: `device=model.device` on the tensor at line 138 removed the crash for the reporter. [TripoSR issue #156, 2026-09-25, first-hand report]

**Evidence**
- Strength: thin. The docs are clear, but most user reports date from 2024 with no maintainer reply. Reddit was not searched again in this audit.
- Sources: model card (huggingface.co/stabilityai/TripoSR); README and issues #22, #156 (github.com/VAST-AI-Research/TripoSR).
