+++
hypothesis = "A small random checkpoint can prove the real mflux Python/uv load-and-generate machinery without running the reserved 4B model."
change = "Created a labelled 194 MB checkpoint using the upstream Flux2Transformer, Qwen3TextEncoder and full Flux2VAE classes; loaded it through Arsenal UvEnvironment and WorkerProcess."
measurement = "A real 64x64 one-step PNG passed byte-count/dimension/SHA-256 checks, both libraries came from the Arsenal-created environment, no unexpected module imports were found, and unload exited the process."
verdict = "kept"
numbers = { standin_checkpoint_bytes = 194451226, image_width = 64, image_height = 64, steps = 1, official_4b_runs = 0, benchmark_runs = 0 }
links = ["../runners/mflux/standin.py", "../runners/mflux/supervisor/", "/tmp/arsenal-image/support/image-standin-proof.json", "/tmp/arsenal-image/minimal-wire.log"]
+++

This is an explicitly RANDOM, untrained architecture stand-in, not a generated
image sample for FLUX.2 Klein 4B. The text encoder retains 28 layers so the real
engine's hidden-state taps at 9/18/27 execute, but hidden width is 32, intermediate
width 64 and attention heads 2. The transformer has one dual-stream and one
single-stream block, two heads of width 16 and rotary axes (4,4,4,4). The VAE and
the vendor's exact local fast-tokenizer assets are unchanged. Saved BF16
parameters use mflux's own checkpoint format; the wire worker independently
reloads them through mflux's real loader and calls real `generate_image`.

Preparation, load and inference used `scripts/mlock gpulock` only, with bounded
30/45-second request deadlines. No trained transformer/encoder weights were
loaded, no network model download occurred, and no `measure` or benchmark suite
ran. The 64×64 image is noise, as expected from random weights, and was opened
to confirm it was an actual raster rather than an empty/fabricated result.

Arsenal's own `UvEnvironment.prepare` created the environment from this folder's
hash lock, forced wheel-only installation and copied the reviewed protocol shim.
The Swift `WorkerProcess` then validated the hello, load reply, run output, PNG
SHA/size/dimensions, resident status and process exit. Runtime evidence confirms
both `mflux` and `arsenal_worker` were imported from that environment. The helper
does not claim to prove the complete signed-catalog image Get/UI path: it
exercises Get's actual uv setup component plus the real runner supervisor.

The first proof diagnosed a **test classifier** mistake: Torch's synthetic
`torch.ops`/`torch.classes` namespaces have relative pseudo-filenames, which are
not imports from disk. The origin check now inspects only real absolute files;
the same actual engine run then passed. Separately, the real manifest parser
exposed an LF/CRLF grapheme delimiter defect; the fetch owner fixed it in main
72ff2b3 before the successful environment creation. No workaround masks it.

The production vendor tensor-key mapping/full dimensions, model quality,
text-in-image accuracy, first/warm full-model speed, peak RAM, quantization and
app UI are **not** proved by this stand-in. The official row remains Unverified.
