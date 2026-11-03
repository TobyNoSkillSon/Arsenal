+++
hypothesis = "A narrowly configured Klein runner can remain offline without claiming untested model quality."
change = "Added local-only framed runner, CPU fake-engine supervisor fixtures, explicit offline guards and output containment; stopped heavy weight downloads at the owner's scope change."
measurement = "Twelve CPU tests passed, the unchanged mflux entry imported successfully, and the complete hash-locked wheel environment was sized; no trained FLUX model inference or benchmark occurred."
verdict = "kept"
numbers = { cpu_tests = 12, distributions = 56, installed_environment_bytes = 1081523104, official_snapshot_bytes = 15980140843, official_image_runs = 0 }
links = ["../runners/mflux/test_backend.py", "../runners/mflux/test_wire.py", "../runners/mflux/requirements.lock", "https://github.com/mflux-community/mflux"]
+++

The installed Python environment is 1,081,523,104 bytes excluding the managed
interpreter and shared wheel cache. Largest package directories by logical bytes
are Torch 534.76 MB, MLX/Metal 218.94 MB, OpenCV 125.07 MB, Transformers 50.87 MB,
SymPy 26.60 MB, matplotlib 22.99 MB, NumPy 22.37 MB and fontTools 13.48 MB.
Numbers are installation sizes, not download sizes or exclusive APFS storage.

An import-only, gpulock-protected check found `torch` imported, and `cv2`,
`matplotlib`, `sentencepiece` and `hf_transfer` absent. OpenCV/matplotlib may be
removable for this text-to-image path, but import absence alone does not prove
generation safety. The initial lock retains the full upstream dependency
closure; no dishonest minimal-dependency claim is made.

The source shows native BF16 safetensors loading and a local Qwen2 fast tokenizer.
`ModelConfig.precision` is BF16. `WeightApplier` supports affine quantization at
group 64, but that does not establish image quality; int8/int4 remain unoffered.
The source has no operational Fast/Exact switch, so `fast_exact=false`.

The vendor tensors alone total about 15.96 GB (transformer 7.75 GB, encoder
8.04 GB, VAE 0.168 GB), a weight-storage lower bound, **not a peak RAM estimate**.
The 36-layer encoder has hidden width 2560 and intermediate width 9728; transient
activations, weight mapping/transposes, MLX cache and allocator overhead add RAM.
Peak RAM cannot responsibly be inferred as a fixed multiplier and is unmeasured.

The large snapshot download was already running under the original brief when
The owner reserved heavy models for his own trials. It was terminated, not restarted.
Every partial was retained in `~/Library/Application Support/Arsenal-dev/weights/
image/weights/bf16/`, with a compatibility symlink at `/tmp/arsenal-image/weights`.
The environment and uv cache were also moved to the separate persistent dev
store. No partial is marked Ready or hash-verified as a complete weight file.

The official folder is Unverified: “Not tried on this Mac yet.” The manifest
declares a 15,980,140,843-byte download before any shared/existing-blob savings;
the service's remaining-byte calculation owns the live Get size. No menu action
is declared, because a programmatic model should not acquire an unproven UI flow.
