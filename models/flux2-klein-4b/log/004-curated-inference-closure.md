+++
hypothesis = "Removing packages used only by other image modes can reduce installation size while preserving this local text-to-image path."
change = "Excluded OpenCV, matplotlib, fontTools, SentencePiece, protobuf, hf-transfer and hf-xet during uv resolution; retained Torch and all actual inference dependencies."
measurement = "Arsenal UvEnvironment created the reduced hash-enforced wheel-only environment; the real random-architecture stand-in produced a byte-identical PNG, with all imported disk modules inside the approved code/runtime roots."
verdict = "kept"
numbers = { initial_distributions = 56, final_distributions = 43, initial_installed_bytes = 1081523104, final_installed_bytes = 899283315, byte_identical_png = true, trained_4b_runs = 0 }
links = ["../runners/mflux/inference-excludes.txt", "../runners/mflux/requirements.lock", "/tmp/arsenal-image/minimal-summary.json", "/tmp/arsenal-image/minimal-env.log"]
+++

This is a curated inference closure, deliberately narrower than mflux's broad
package metadata. `uv pip compile --generate-hashes --only-binary :all:
--python-version 3.13.12 --excludes inference-excludes.txt requirements.in`
generates the lock; `uv pip sync --require-hashes --only-binary :all:` installs
exactly that lock. Arsenal used those real commands, not an editable install or
sdist. All wheels reused the persistent verified cache; original environments
and partial downloads were kept, not deleted.

The excluded packages cover control-image operations, rendering/plotting,
slow-tokenizer conversions, and Python-side remote downloads. None is used by
this runner: exact Qwen2 fast-tokenizer assets are present, PiD/edit/control/LoRA
modes are not exposed, and all weights are local. The synthetic checkpoint
changes tensor geometry, not the text-to-image classes or execution mode.
The byte-identical output is a dependency-regression check, **not** a production
quality gate. Original dependency licence metadata remains available in the
historical full environment; shipped-lock notices list the retained 43 wheels.

Logical installed bytes include package-generated caches and the copied shim,
but exclude CPython and shared uv cache; APFS exclusive storage differs. The
saved ~182 MB is an installation comparison, not a benchmark or throughput
optimization. Torch's unconditional imports make its roughly 535 MB unavoidable
without changing upstream; that separate port/patch is not undertaken here.
