# Kernels: chip-specific optimisations that live in the model folder

A model folder can carry Metal kernels that replace parts of its runner's computation on a given chip family. They are
plain `.metal` source files, compiled on the user's Mac when the model loads (no Xcode, no toolchain), checked before
they serve anything, and reported in the model's status. This document is the contract for anyone adding one: what
goes in the folder, what the runner does with it, how **Exact** is proven, how **Fast** is gated and how the pull
request records the history.

The library is `Workers/Kernels` (`ArsenalKernels`, plus `ArsenalKernelsCore` for engines without MLX). The reference
integration is Laya English (`models/laya-english/kernels`, runner `Workers/Decisions`).

## Exact and Fast

| Mode | Promise | Proof |
|---|---|---|
| **Exact** | Faster, same numbers: every output bit equals the stock runner's on that chip. | Bitwise comparison of the model's outputs with the kernel on and off (`ExactGate`), at load, persisted. |
| **Fast** | Numbers may change slightly; the model's quality gate still passes. | The model's own gate (`models/<id>/gate/`), run by the contributor and repeated at load where the runner has a load-time check. |

Arsenal selects mode automatically: Fast when the accepted runner or this chip's folder kernels support it, otherwise Exact. Users choose precision; the internal Fast/Exact modes remain kernel and result provenance. **Fast includes Exact**: for a site with no Fast kernel,
Fast mode uses the Exact one. This holds at resolution time only: when a site has a Fast kernel, Fast mode delivers
that one alone, and if it then fails its gate or preflight on this Mac the site runs **stock**, not the site's Exact
kernel (one candidate per site; see Limits).

## Folder layout

```
models/<id>/kernels/manifest.toml
models/<id>/kernels/<chip or any>/<name>.metal        # sources, ASCII only
models/<id>/kernels/<chip or any>/<name>.h.metal      # optional header part (api = "mlx")
models/<id>/log/NNN-<slug>.md                          # one entry per attempt, kept or rejected
```

`kernels/manifest.toml`, one table per kernel:

```toml
[[kernel]]
id = "splitk-exact"                 # [a-z0-9][a-z0-9.-]*
chip = "*"                          # chip family: "m1" … "m9", or "*" for every chip
mode = "exact"                      # exact | fast
replaces = "linear.down"            # the site: a name the runner defines and documents
entry = "splitk_exact"              # MLX kernel name (api mlx) or Metal function name (api metal)
api = "mlx"                         # mlx (default): MLXFast.metalKernel body | metal: a complete MSL library
files = ["any/splitk_exact.h.metal", "any/splitk_exact.metal"]   # relative to kernels/
header = "any/splitk_exact.h.metal" # api mlx: one of `files`, placed before MLX's generated signature; the rest is the body
inputs = ["x", "w", "bias", "res"]  # api mlx: the MLX input names, in call order
outputs = ["out"]                   # api mlx: the output names
```

Rules (`Arsenal catalog validate` checks them): `(replaces, chip, mode)` and `(id, chip, mode)` unique; `.metal` files
inside the folder; `header` one of `files`; `inputs`/`outputs` present for `api = "mlx"`. Sources must be ASCII (MLX
passes them to Metal as ASCII; the library rejects anything else with a clear message).

**Sites** are the runner's vocabulary: `linear.down` in the decisions runner means "the K > N Linears"
(`Workers/Decisions/Sources/VerdictEngine/FolderKernelRoute.swift`). A runner documents each site it accepts, the
shapes and dtypes it routes there and the entries it has host code for. A kernel for a site the runner does not know,
or an entry it has no host code for, reports `stock` with that reason; nothing breaks.

## Resolution and delivery

At load the supervisor resolves the manifest for this Mac's chip family (`Apple M5 Max` → `m5`) and the effective automatic mode:
**one kernel per site**, by priority

- exact mode: `exact@family` > `exact@*`
- fast mode: `fast@family` > `fast@*` > `exact@family` > `exact@*`

and passes the result in `load.config.kernels` as an array of objects with absolute file paths
(`{"id","chip","mode","replaces","entry","api","files":[…],"header":…,"inputs":[…],"outputs":[…]}`). Key absent: the
runner keeps its built-in behaviour (folders without kernels). Empty array: no kernels for this chip and mode.

The runner reports every resolved kernel in its load reply and status:
`"kernels": [{"id","replaces","mode","state":"active"|"stock","reason"}]`; the reason is one sentence (the passed
check, or why the stock path serves the site). The service shows it on the model's row.

## What the runner does (MLX runners)

```swift
import ArsenalKernels

// load
let kernels = try FolderKernels(config: config["kernels"])        // nil key → kernels.configured == false
// gate each site (below), then kernels.activate(site, reason:) or kernels.deactivate(site, reason:)

// at the call site
if let k = kernels.active("linear.down"),
   let y = k([x, w, bias, res], template: [("T", x.dtype), …], grid: …, threadGroup: …,
             outputShapes: [[m, n]], outputDTypes: [x.dtype])?.first {
    return y
}
return stock(x)

// load reply / status
out["kernels"] = kernels.wire
```

`FolderMLXKernel` is `MLXFast.metalKernel` with three additions that keep a bad folder kernel from taking the runner
down or serving unchecked (they are not a sandbox; see "What the checks do not cover" below):

1. **Compile preflight.** MLX JIT-compiles a custom kernel inside `eval`, and a compile error there is `fatalError`.
   Before MLX sees a new instantiation (input dtypes and ranks, template values, output dtypes, threadgroup size), the
   library builds the exact source MLX will build (`MLXKernelSource`, a port of MLX's `write_signature`, prefixed with
   MLX's own preamble read from the linked MLX), compiles it with Metal (`MTLDevice.makeLibrary(source:)`, MLX's options:
   safe math, Metal 4.0 on macOS 26) and builds the pipeline, checking the threadgroup size. A failure makes that call
   return nil, so the caller runs its stock expression, and the site reports the compiler's first error line. A call
   whose arguments do not match the declaration (input/output count, unsupported dtype) is a preflight failure too.
   Any preflight failure is kept in `failure`: the site goes stock for the rest of the process and a gate run that
   hit it cannot pass. The GPU smoke checks the generated source against MLX's own verbose print byte for byte.
2. **Caches.** Compiled libraries are cached in process by (source digest, options). Compile verdicts and gate verdicts
   persist in `<support>/state/kernels.json`. Compile verdicts are keyed by the full generated source (which includes
   MLX's preamble and the signature built from the declared inputs/outputs) and threadgroup size; gate verdicts by the
   **kernel identity** (`LoadedKernel.digest`: sha256 over site, mode, api, entry, the ordered `inputs` and `outputs`,
   header and body; `id` and `chip` are labels and stay out), the caller's key (model, dtype, the weights the check ran
   on, check revision) and the executable. Every key also carries the OS build (`kern.osversion`), the GPU (name +
   architecture) and a schema version (`v2`). So a manifest edit that only reorders `inputs` (MLX binds buffers by
   position, `x - w` becomes `w - x`) is a new kernel to every gate, and verdicts saved before the identity covered
   the declaration are never read. A second load skips the preflight compile (MLX compiles once) and the gate. The
   file is a bounded cache: entries of an older schema are dropped and at most 512 are kept, oldest out first;
   deleting it only costs one recheck per kernel.
3. **Engagement count.** `calls` counts dispatches. A gate that never reached the kernel proves nothing (it happened
   in Vella: an A/B "rejected" a kernel that never ran). The library enforces it: a gate run (`ExactGate`, or
   `KernelVerdicts.gate` for a runner's own Fast gate) in which a kernel was never dispatched or failed its preflight is
   a failure, so a "kernel on" run that silently fell back to stock can never save a pass.

### What the checks do not cover

The compile preflight proves that MLX can build and launch the kernel; it is **not a GPU-code safety boundary**. It
does not catch out-of-bounds reads or writes, a loop that never terminates, or numbers that are wrong away from the
check inputs. Approved folder Metal runs in the worker process's ordinary GPU context: it can read and corrupt any MLX
buffer of that process and can hang a command buffer (the GPU watchdog then aborts it, MLX raises a fatal error, the
worker crashes and the supervisor quarantines it). It cannot reach the app, other workers, or files beyond what the
worker process already has. That is why a kernel in a local (unsigned) folder is code the user approves before it
runs, under the same unsigned-code consent as a local runner (the app warns that the kernels were not reviewed or
signed by Arsenal and run on your GPU), and signed folders' kernels are reviewed in their pull request. Do not describe a passed preflight or gate as sandboxing a shader.

Engines that drive Metal themselves (own engine, C/C++/Rust through a shim) use `api = "metal"` and
`MetalFolderKernel.pipeline(_:)` from `ArsenalKernelsCore` (compile + pipeline + persisted failure), or read the same
JSON and compile with their own Metal code. The contract is the JSON and the reporting, not the Swift library.

## Proving Exact

Exact means bit-identical to the stock runner **on this chip**, not "within rounding". The library checks it; the
contributor has to make it true:

- **Replay the stock arithmetic.** Same products, same accumulation order, same rounding points. In MLX that means
  reading the kernel MLX actually dispatches for each shape (routing in `backend/metal/matmul.cpp`, tile code in
  `mlx-generated/metal/steel/…`) and reproducing it: Laya's `splitk-exact` uses MLX's own fragment layout and
  `matmul2d(16, 32, 16, relaxed, multiply_accumulate)` descriptor, MLX's split-K partition sizes and partition order,
  `0.0f + p0 + p1` for the reduction, MLX's axpby epilogue for the bias and a `T + T` add for the residual. The host
  routes only the shapes MLX itself splits (`FolderKernelRoute.splitKPlan`, same rule); everything else stays stock.
- **Compare bits, not values** (`ExactComparison`): +0 vs −0 and different NaN payloads are differences.
- **Gate at load** with `ExactGate.qualify(kernels:model:dtype:weights:checkRevision:outputs:)`: the runner returns the
  evaluated outputs of fixed check inputs with the kernel on and off; every byte must match and every kernel must have
  been dispatched. `weights` names the weights the check ran on when its outputs depend on them (Laya: snapshot
  revision + checkpoint size/mtime) and is `"-"` for a check on synthetic inputs (Qwen RoPE). Bump `checkRevision`
  when the check changes.
- **The coverage is the check inputs, and choosing them is the contributor's claim.** A kernel can be bit-identical on
  the check set and differ on a shape it also serves; the gate cannot know. Cover every shape class the host routes
  to the kernel (Laya: each built-in item alone, M 16–128, and all rows in one chunk, M ≈ 700); the verdict detail
  lists the output shapes and dtypes it compared, and review checks them against the routing.
- **Gate offline** with the model's own gate before the PR: Laya's Exact gate is Verdict's 202-item fixture regression
  (byte-identical wire answers, `Workers/Decisions/scripts/parity.py`), and the probe compares the kernel arm's 202 wire
  answers with the stock arm's.

A change that cannot be exact (split-K with another order, fp16 accumulation, a fused activation that rounds once
instead of twice) is a Fast kernel.

## Gating Fast

A Fast kernel passes the model's quality gate, defined in its folder (`gate/gate.toml` and the runner's load-time
check). The gate is the model owner's, not the kernel's: Laya's is Verdict's noise ruling (max |Δp| ≤ 0.01, chosen
answer unchanged on ≥ 99.5 % of rows, quick accuracy within 1 point) plus the runner's load-time check (logits relative
RMS ≤ 1e-2 and unchanged answers on the built-in items outside near-ties, `SmallMGate`). Rules that kept earlier gates
honest (skill `optimize-mlx-inference`): build the reference before the first change, calibrate a tolerance against
injected faults *and* the seed/noise floor, run the single-item path when the kernel only serves single items (a
batched quick set never reaches a small-M kernel), and confirm a near-tie pass with a second, equally valid variant.

## Submitting a kernel (pull request)

Real-model probes are opt-in ([AGENTS.md](../AGENTS.md), Running models). Prepare fixtures first; run probes only on a quiet Mac. Independent references follow [AGENTS.md](../AGENTS.md#precision-and-the-quality-gate), including GPU half precision.

1. Add the sources and the manifest entry. One kernel per site per (chip, mode).
2. Prove it: Exact → bitwise as above; Fast → the model's gate. Keep experimental probes and reference captures in
   `lab/`; the historical Laya probe is preserved at `lab/models/laya-english/kernels/probe.py` on the maintainer's Mac, not in a fresh clone. Start with the tracked kernel tests in `Workers/Kernels/Tests/ArsenalKernelsTests/KernelsTests.swift` and
   `Tests/ArsenalCoreTests/ServiceKernelSmokeTests.swift`; the real-model smoke is opt-in. For an
   authorized benchmark, alternate stock and kernel on the same build, A B A B, under `scripts/mlock gpulock`.
3. Write a log entry `models/<id>/log/NNN-<slug>.md` (schema `docs/CONTRIBUTING.md`): hypothesis, change, measurement,
   verdict (kept | rejected | inconclusive), numbers, links. **Rejected attempts get an entry too**: the folder's log is
   the public record of what was tried on which chip and what it gave.
4. Preserve raw probe JSON and historical gate runs in `lab/evidence/<model>/`. Promote only the reviewed current
   gate recipe, compact results, quality bindings and shipped log into the model folder; reusable regression inputs
   belong in `Tests/Fixtures/`. Fresh clones must not depend on lab files.
5. Review accepts it when the numbers are on the same build and suite, the gate passed and the kernel is faster on
   that chip; the next app release ships it. Arsenal never races kernels on the user's Mac: it runs what the folder
   resolves to.

## Worked example: Laya English

`models/laya-english/kernels/manifest.toml` declares two kernels for the site `linear.down` (mlp.Wo 2624→1024 with the
residual, head linear2 4096→1024 with bias and residual):

- `smallm-tile` (fast, m5): the family SmallMGEMM tile kernel (tile-1), moved out of compiled Swift. One threadgroup per
  32×32 output tile, K split across 4–8 simdgroups for small M (9–256 rows). Gate: SmallMGate (load-time) and Verdict's
  fixture noise ruling.
- `splitk-exact` (exact, every NAX chip): MLX's own NAX split-K GEMM plus the accumulate pass and the residual add in
  one dispatch, bit-identical.

Numbers, gate results and the history are in `models/laya-english/log/` (entries 004 onwards).

## Limits

- The preflight replicates MLX 0.32's source generation (mlx-swift `9019419`). A new MLX pin needs
  `MLXKernelSource` re-checked: the GPU smoke's `source-matches-mlx` line does it.
- Exact holds for the build it was proven on: gate verdicts are keyed by the executable (name, size, mtime), OS build
  and GPU, so every new app build or OS update requalifies once at the first load.
- A site failing its preflight in one instantiation is switched to stock for the rest of the process (conservative).
- One candidate per site: in Fast mode a site with a Fast kernel gets only that kernel. If it fails its gate on this
  Mac the site runs stock, not the declared Exact kernel, so Fast can then be slower than Exact for that site. Passing
  both candidates and gating them in order is a possible extension of the delivery format; today the contributor
  should make the Fast kernel pass wherever the Exact one would apply, or not declare it for that chip.
- Model-dependent gates key their verdicts by the weights identity the runner supplies; a runner that passes `"-"`
  asserts that its check does not read the model's weights.
