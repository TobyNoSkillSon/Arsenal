# Arsenal: notes for coding agents and contributors

Arsenal is a native macOS menu-bar app that runs on-device models. `README.md` and `docs/README.md` map the product
and the repository; `docs/CONTRIBUTING.md` covers how to submit a change. These defaults apply to every change.

## Repository

- Everything tracked is the product: source, tests, model folders, docs and the scripts that build them.
  Experiments, probes, raw captures and one-off scripts belong in `lab/`, which git ignores. Move work out of `lab/`
  only when it is ready to ship.
- Every model, kind, engine, tool window and activity card is its own module behind a small declared contract, with
  its own folder or target and tests, so many people can change different parts at once. A change that crosses
  modules goes through the contract rather than edits inside someone else's module. Builds and tests must work per
  worktree without colliding.

## Running models

- Integrate what already runs well; optimise only what earns it. Pick the first that fits:
  1. the bundled native worker, when it already runs the architecture (folder settings only);
  2. an existing MLX implementation in Swift (mlx-swift, mlx-swift-lm, an mlx-audio or mlx-community Swift port);
  3. a Swift port on the pinned mlx-swift, using an MLX Python implementation or the vendor's code as reference;
  4. a mature native engine for the Mac (llama.cpp and the like), wrapped as-is behind the worker wire protocol,
     only when no MLX implementation exists.
- Python never serves a model. It runs the maker's reference code to check an MLX port.
- GPU kernels are Metal Shading Language through `MLXFast.metalKernel`; Swift is the plumbing around them.
- External engines are wrapped, never rewritten.

## Precision and the quality gate

- Models run in FP16 by default, never FP32 (references included). A BF16 release is converted to FP16 and checked
  against the original on fixed tensors; it stays BF16 only if FP16 fails that check. FP16 runs natively on every
  Apple Silicon generation, BF16 does not.
- int8 and int4 tiers are optional. A tier is offered only when its quality check passed or was never run; a tier
  that failed is removed, not offered with a warning. Models whose 16-bit weights are under 1 GB ship 16-bit only.
- Check against an independent reference on the GPU: a different implementation of the same model (the maker's
  code, the established engine, or the maker's published outputs), never the candidate against itself.
- Verify ports per component on small fixed tensors against a reference captured once; at most one end-to-end
  smoke of the final default path. Assert counts and keys before values.
- Keep current gate recipes, results, quality bindings and research logs in the model folder (`log/NNN-slug.md`).
  "Measured" is reserved for controlled measurements; quick checks are labelled as quick checks.

## Menu and windows

- Every tool window (Make an Image, Make a Sound, Remove Background and later tools) is built from one shared kit:
  model picker, options built from the model's schema, run and stop, progress, result area, history.
- Tool windows show exactly what the chosen model declares: its operations and input schemas, nothing it lacks.
- Before any Load, compare the model's peak memory at the chosen precision with free RAM. If it does not fit, say
  so in numbers and offer Cancel or Load Anyway. The API refuses the same way unless the request says force.
- The Models table has the same columns for every kind: Model, Params, Precision, Speed, Peak memory, settings and
  action. Arsenal shows no recommendation; the user picks the precision.
- Panels hosted in a menu draw no background of their own. Controls act in place; only an action that opens a window
  or quits closes the menu. An open menu never changes its structure.
- Motion only when something changed: at rest nothing redraws, and a change moves only what changed.
- Menu interactions use AppKit's own menu structure. Offscreen renders paint the menu material as a flat colour,
  so anything drawn over other content inside a menu needs an opaque ground of its own.
- Every bug found in the live menu becomes a test.

## Builds and tests

- One native build graph: `Workers/Package.swift` links every model family into the one shipped worker and owns
  every family's test target, so MLX compiles once per configuration. Never build or test a family package on its
  own.
- Iterate with `scripts/test.sh --changed`. `scripts/test.sh --native [--changed | --filter <suite>]` runs only
  native suites; `scripts/dev-build.sh` builds without running tests.
- The default suite is unit and contract tests with no model inference. Tests that run a forward pass, generation,
  synthesis or transcription, or evaluate MLX arrays on the GPU, are tagged `.heavy` and run only with
  `scripts/test.sh --heavy [--filter <suite>]`, one target at a time.
- Build a release candidate with `ARSENAL_STRICT=1 scripts/build.sh`.
- Lab and test runs use isolated `ARSENAL_SUPPORT_DIR` and `ARSENAL_CACHE_DIR`, never your real profile. Quit any
  Arsenal instance you launched and stop only the processes you started.
- Never commit secrets, private recordings, weight files or personal paths.
