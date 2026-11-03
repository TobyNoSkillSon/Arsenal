+++
hypothesis = "The MLX kit runner matches the maker's TripoSR (tsr/system.py) on the same checkpoint."
change = "None: measurement of the Model.swift / MarchingCubes.swift port on the original FP32 file (cast to FP16 at load)."
measurement = "lab/night2/triposr/ref.py vs the runner's raw float32 dumps (ARSENAL_PARITY_DIR), compare.py, m5-max, 2026-10-31: chair.png and robot.png from the makers' examples, 128-cube density grid."
verdict = "kept"
numbers = { triplane_cosine_min = 0.9999982, density_cosine_min = 0.9999146, inside_agreement_min_pct = 99.9957, input_max_levels_off = 1 }
links = []
+++
# 003 Parity with the maker's TripoSR (2026-10-31)

Weights: `model.safetensors` (sha256 8443343c...) from refs/pr/11 (4e228585, SFconvertbot's "Adding safetensors variant"),
whose parent is the planned revision 5b521936. PR 8 holds the same file. The planned `model.ckpt` was not downloaded, so the
safetensors file was not value-compared with it; the match rests on the bot's conversion and the reference results below.

Method: the maker's code (github.com/VAST-AI-Research/TripoSR @ 107cefd), imported unchanged, built from the makers'
config.yaml (fetched once, 987 bytes) and loaded strictly from the pinned safetensors, run on MPS (PyTorch 2.15.0, M5 Max).
Reference-side adjustments, no model maths changed: transformers 5.22 renamed ViT's modules, so checkpoint keys are mapped to
the new names; ViT position interpolation is replaced by the transformers 4.35 version TripoSR pins (scale factor
(32 + 0.1) / 14, which the runner also uses); packages the forward pass never touches (rembg, trimesh, imageio, torchmcubes)
are stubbed. Preprocessing follows run.py for an RGBA image (its remove_background skips rembg for images with transparency).
float16 reference: torch.autocast("mps", float16) over the FP32 weights (model.half() fails: the image normalisation buffers
mix dtypes). Compared: the 512x512 preprocessed image, the triplanes [3, 64, 64, 40], and density_act on a 128^3 grid.

| image | compared with | input max diff (levels) | triplane cosine | density cosine | grid points on the same side of 25 |
|---|---|---|---|---|---|
| chair | torch fp32 | 1 | 0.9999985 | 0.9999886 | 99.9969% |
| chair | torch fp16 autocast | 1 | 0.9999982 | 0.9999836 | 99.9969% |
| robot | torch fp32 | 1 | 0.9999985 | 0.9999528 | 99.9960% |
| robot | torch fp16 autocast | 1 | 0.9999987 | 0.9999146 | 99.9957% |

torch fp16 autocast vs torch fp32 for reference: triplane 0.9999995/0.9999996, density 0.9999946/0.9999881. Log-density
correlation is 0.999996 or better in every row. The input differs by at most one 8-bit level at the object's edge pixels
(CoreGraphics premultiplies alpha before the grey composite).

Mesh: the runner's marching cubes is torchmcubes' CPU algorithm with its own tables (MCTables.swift, generated from its
source); it was not run against torchmcubes itself (not installed). Default-resolution meshes were rendered from three sides
(lab/night2/triposr/chair-final.png, robot-final.png) and match the input objects, upright and front toward +Z.

Not checked: GLB validity with a third-party loader (read back with lab/night2/triposr/view.py only); opaque photos (no
reference, since the maker would run rembg); resolutions other than 128 against the reference.

Verdict: pass (cosine >= 0.999 on triplanes and density).
