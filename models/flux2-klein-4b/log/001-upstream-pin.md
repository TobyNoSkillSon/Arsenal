+++
hypothesis = "The current mflux release supports a local-only Apache-2.0 FLUX.2 Klein 4B worker."
change = "Pinned mflux 0.20.0, CPython 3.13.12, and the vendor's 16-bit checkpoint; assembled the complete arm64 wheel lock."
measurement = "PyPI metadata, wheel source, and Hugging Face immutable model metadata inspected; hash-enforced wheel-only environment created. GPU correctness is pending."
verdict = "inconclusive"
numbers = { packages_environment_gib_approx = 1.1, transformer_bytes = 7751109744, text_encoder_bytes = 8044981992, vae_bytes = 168120878, image_runs = 0 }
links = ["https://pypi.org/project/mflux/0.20.0/", "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294", "../runners/mflux/requirements.lock"]
+++

The 4B checkpoint is Apache-2.0, not the non-commercial 9B variant. Vendor
diffusers-layout weights include the 4B Qwen3 text encoder and VAE. The root
`flux-2-klein-4b.safetensors` duplicates the transformer and is deliberately not
downloaded. Every inference asset is listed individually in model.toml.

The first cross-platform uv resolution selected an old default macOS target and
rejected MLX wheels. Resolving on this arm64 Mac with macOS 26 and Python 3.13.12
succeeded. This runner is only offered on macOS 26 or newer, matching the app.

Install command: `uv venv --python 3.13.12 /tmp/arsenal-image/env`, then
`uv pip sync --python /tmp/arsenal-image/env/bin/python --require-hashes
--only-binary :all: runners/mflux/requirements.lock`. All heavy CPU work runs
under `scripts/mlock work`. Evidence: `/tmp/arsenal-image/env.log`.

Torch cannot be removed from the unmodified release: `weight_loader.py` imports
Torch and `safetensors.torch` unconditionally, and the Klein variant imports a PiD
decoder that also imports Torch. This worker never enables PiD or any auxiliary
download. OpenCV/matplotlib trimming is not yet proven and the initial lock keeps
the complete upstream closure. No unsupported precision or Fast/Exact claim is
made.
