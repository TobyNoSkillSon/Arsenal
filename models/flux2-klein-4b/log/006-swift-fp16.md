+++
hypothesis = "A half precision Swift FLUX.2 Klein port can preserve the established architecture and remove the Python runtime without deep optimization."
change = "Native bf16 default and gate, half precision GPU component probes, constrained mflux reference capture, vendor overflow guards, bounded wire quick check and interleaved bf16/fp16 trials."
measurement = "M5 Max 40-core GPU, 128 GB, macOS 26A434, 8 Oct 2026, Mac in use. GPU only; component processes below 10 GB and the default combined worker below 20 GB."
verdict = "inconclusive"
numbers = { warm_image_bf16_s = 3.8136, warm_image_fp16_s = 3.8634, warm_step_bf16_s = 0.7165, peak_bf16_gb = 18.6952, quick_512_bf16_s = 1.0685, captured_bf16_image_psnr_db = 27.5221 }
links = ["../gate/evidence/reference-case.json", "../gate/evidence/fp16-fidelity.json", "../gate/evidence/bf16-fidelity.json", "../gate/evidence/op-fidelity.json", "../gate/evidence/precision.json", "../gate/evidence/precision-interleaved.json", "../gate/evidence/default-smoke.json", "../results/m5-max.json", "../gate/evidence/bf16-fp16-interleaved.json", "../gate/evidence/bf16-prerelease-gate.json", "../gate/evidence/prerelease-visual-sweep.json"]
+++

The default runner is now Swift/MLX **bf16 only**. The pre-release decision below
supersedes the original fp16 recommendation; historical trial numbers remain
labelled below. No app build, install or catalog signing was performed.
Checkpoint weights were preserved, with no new download.

The reference is **constrained mflux**, not untouched stock mflux. Upstream
0.20.0 promotes Qwen attention, normalization, RoPE and timestep operations to
fp32 even with bf16 weights. To obey the task's no-fp32 rule, the capture tool
replaces explicit `mx.float32` references with `mx.bfloat16` in process-local
loaded source. It records each affected original source hash and replacement
count. The installed mflux environment is unchanged. MLX's opaque stock kernels
may internally accumulate in a wider type; the application evaluates half
precision arrays. No model was run on the CPU. Host schedule scalars, uint8 PNG
encoding and float64 difference statistics are not model tensors.

## Accepted stock-reference gap (8 Oct 2026)

The owner accepted retaining the 16-bit Swift runner without qualifying against the
old stock-mflux runner: "we are keeping the 16-bit Swift runner… skip the old
runner". This accepts the qualification gap; it does not establish stock-mflux
fidelity. The default remains native bf16 and `gate_passed=false` stays honest.

## Pre-release bf16 decision (8 Oct 2026)

The vendor checkpoint stays in its native bf16 throughout encoder, transformer
and VAE. Pipeline defaults, Gaussian noise and RoPE defaults use bf16; omitted
load dtype resolves to bf16, and explicit fp16/int8/int4 requests fail without
loading. fp16 remains available only in the lab, because the captured first
block and tiny-transformer errors are materially larger and it offers no
established performance advantage. This is a precision selection, not a new
kernel or a deep optimization.

One resident pipeline per arm, one unmeasured 1024² four-step warm-up per arm,
then bf16 → fp16 repeated three times, never simultaneous generation. Full
image wall time includes text encoding, four evaluated denoising steps, VAE,
pixels and PNG writing. Step time is the median of all 12 evaluated denoising
steps per arm, excluding encoding, VAE and PNG. Peak is each arm's lifetime
physical footprint, including load and warm-up, independently monitored with
rusage and bounded to 20 GB per process. The Mac was in use, with two resident
arms; this is a quick timing, not a controlled ranking or energy measurement.

| Arm | Warm image median (s, 1024²/4 steps) | Denoising step median (s, 12 steps) | Lifetime peak physical footprint (GB, per arm) |
|---|---:|---:|---:|
| bf16 (default) | 3.8136 | 0.7165 | 18.6952 |
| fp16 (lab only) | 3.8634 | 0.7532 | 18.7113 |

bf16 image times span 3.7142–3.9458 s;
fp16 3.8495–4.0327 s. The medians
overlap the busy-machine spread; no speed win is claimed. Raw runs, warm-up
records, dtype witnesses and binary SHA-256 are in
`gate/evidence/bf16-fp16-interleaved.json`. All four PNGs per arm are identical
by SHA-256; both unique 1024² images and the new 512² quick fixtures were
inspected at 100%, with the region sweep recorded separately.

Gate v2 freezes the saved bf16-vs-constrained-mflux floor as a **regression
bound**, with a coordinator-chosen 25% margin rounded upward:

| Component | Maximum relative L2 vs constrained mflux bf16 |
|---|---:|
| Full text encoding | 0.015 |
| Short text encoding | 0.001 |
| First transformer block | 0.000000125 |
| Tiny full transformer velocity | 0.009 |
| Small VAE | 0.003 |

Saved image regression bounds are latent relative L2 ≤ 0.27, decoded-image
relative L2 ≤ 0.17, and clipped RGB PSNR ≥ 26 dB. These do not establish a
seed-noise quality tolerance. Replaying host float64 statistics over the
captured half tensors passes all five component bounds and all three image
regression bounds; exact tokenizer IDs, counts/shapes, finiteness and reference
SHA-256 are checked first. Component calculations are unchanged by the default
switch: the original bf16 captures already passed explicit bf16 at those call
sites. The original reference is reused; no CPU or fp32 model run or new
reference capture was made. The comparator rejects wrong dtype metadata,
missing captures and changed reference hashes; a finite zeroed text component
fails the bound. fp16 fails the bf16 dtype and component gate.

**Overall `gate_passed=false` remains honest:** stock mflux fidelity and a
noise-level image-quality tolerance are still unqualified. The human checks
in the fresh quick recipe pass (red ceramic teapot on wood, exactly ARSENAL),
but they cannot promote the constrained capture into stock-reference parity.
The saved comparator result concerns its captured image, not the full quick
fixture suite, and therefore does not claim that suite's human qualification.

The new default-path wire quick check runs one 512² photo warm-up then photo
and text once: 1.0685 s/image median,
15.9158 GB peak physical footprint.
`results/m5-max.json` records `precision=bf16`, `mode=exact`, `profile=quick`,
the actual worker SHA-256 and no measurement lock. Exact matches the table's
single-mode runner filtering (`fast_exact=false`); it does not claim bit-exact
reference parity. Missing GPU/compositor telemetry stays invalid/contended.

Verification: 13 FLUX tests in four suites pass, including native bf16 defaults,
load/generate/unload, rejected dtypes, scheduler/layout and cancellation cleanup.
D1's three formatting-only changes preserve non-whitespace contents and pass
its Unicode, 512-case BPE, PNG16 and bounded-memory regression runner. Strict
repository lint passes all 433 first-party Swift files. Binary, PNGs and raw QA
remain local in `.build/flux-prerelease-20261008/`, outside the signed folder.
No app build, signing, installation or additional worker delegation occurred.

## Fidelity

Capture order: independent mflux text encoder, transformer and VAE, separately
on GPU; then reuse those captures for Swift bf16 and fp16. Tiny transformer
input has 32 text + 16 image tokens; the first block consumes reference
projection/modulation/RoPE inputs, so upstream errors are isolated. The short
text case has 32 tokens; the image case has 512 text + 4096 image tokens, four
steps and seed 24301. Image replay shares the reference's saved bf16 noise
(converted to the tested dtype), while the wire smoke uses default fp16 noise.

| Component | Swift bf16 relative L2 vs constrained mflux | Swift fp16 relative L2 vs constrained mflux |
|---|---:|---:|
| Short text encoding | 0.000723 | 0.009137 |
| Full 512-token encoding | 0.011596 | 0.032275 |
| First transformer block | 0.000000096 | 0.005052 |
| Tiny full transformer velocity | 0.007097 | 0.145200 |
| Small VAE decode | 0.002333 | 0.002804 |
| Final image latents | 0.215818 | 0.406313 |

Small VAE PSNR: bf16 64.65 dB, fp16 62.70 dB. Fixed-image PSNR after clipping
RGB into [0,1]: bf16 27.52 dB, fp16 22.63 dB. Every compared tensor is finite;
shapes, counts and tokenizer IDs match. These image differences are material:
they do not establish a noise-level quality gate or stock mflux fidelity.

Corrections: use mflux's rotate-half Qwen RoPE (bf16 tables because theta
1,000,000 exceeds fp16 range); retain safe stock fused fp16 RMS normalization;
use the vendor's fp16 residual clamps on the double-block context and the
single-block stream. The absence of these clamps produced NaN latents after
later steps. The eager fp16 RMS formula squared valid activations above 256
into infinity and normalized them to zero. A tiny [700,-700,300,-300] GPU
regression now preserves a finite unit-RMS result. A finite-image check refuses
future overflow rather than converting NaN to a black PNG.

Five op witnesses cover 180 values: RMS, large-input RMS, RoPE and timestep
features match the Python GPU witness exactly; SiLU differs at one value by
6.1035e-5 (one bf16 ulp). Swift pins MLX source 901941965d82; Python uses MLX
0.32.3. This establishes the probed operations, not every GEMM/attention shape
or full stock-reference equivalence. Source provenance for the residual guards
was saved before deleting the disposable diffusers/torch refenv.

## Historical fp16 / quantized precision trial

Transformer attention/feed-forward block linears only, MLX affine group 64;
modulation/projection/time linears, Qwen encoder and VAE remain fp16. No kernels,
fusion or tuning were introduced. Each trial loads one model process, warms
one fixed-input step and measures three steps. Order fp16 → int8 → int4 → fp16.
The fp16 bracket medians drifted 0.716 → 0.829 s; quantized timing lies within
this busy-machine bracket. No reliable speed benefit was demonstrated.

| Precision | Warm step median (s, 1024²) | DiT-process peak footprint (GB) | Block relative L2 vs fp16 | Tiny velocity relative L2 vs fp16 | Fixed-image PSNR vs fp16 (dB) |
|---|---:|---:|---:|---:|---:|
| fp16 | 0.777 | 9.87 | 0 | 0 | identical |
| int8 | 0.824 | 8.42 | 0.003255 | 0.006121 | 39.30 |
| int4 | 0.819 | 8.24 | 0.057865 | 0.098122 | 19.36 |

Peak values above are transformer-only processes, including loading/quantizing;
they exclude the resident encoder and VAE of the default worker. The text
encoder and small VAE input/output are exactly unchanged by transformer-only
quantization. Final latent relative L2 vs fp16: int8 0.04862, int4 0.46904;
full decoded-image relative L2: 0.03341 / 0.33183. These are one fixed image,
not a prompt-suite quality claim. Peak memory remains above steady-state
quantized weights because the loader starts from the vendor half checkpoint.

### Roofline derivation

1024² gives 4096 image + 512 text = 4608 tokens, hidden dimension 3072.
Five double and twenty single blocks each perform the equivalent of 13·d²
linear weights per token: `25 × 2 × 4608 × 13 × 3072² = 28.2663 TFLOPs`.
Joint attention: `25 × 4 × 4608² × 3072 = 6.5230 TFLOPs`.
Input/text/output projections add 0.0306 TFLOPs, so dominant work is
**34.8198 TFLOPs/step**, unchanged by weight-only quantization. Modulation,
normalization and elementwise FLOPs are omitted from this dominant estimate.

HF tensor-header counts: 3,875,544,576 total transformer parameters;
3,680,501,760 quantized block parameters. Half weights need 7.751 GB.
Group-64 affine storage adds two half parameters per group:
`(total − quantized) × 2 + quantized × (bits/8 + 4/64)` bytes.
Int8 needs 4.301 GB, int4 2.460 GB. Linear activation input/output traffic is
`(5 × 19 + 20 × 15) × 4608 × 3072 × 2 = 11.183 GB`;
minimal attention Q/K/V reads + output writes add 2.831 GB.

| Precision | Derived logical bytes/step (GB) | Bandwidth floor at 614 GB/s (ms) | Compute floor at empirical 61 TFLOPs/s (ms) |
|---|---:|---:|---:|
| fp16 | 21.765 | 35.45 | 570.82 |
| int8 | 18.315 | 29.83 | 570.82 |
| int4 | 16.475 | 26.83 | 570.82 |

614 GB/s is Apple's 40-GPU-core M5 Max specification:
https://www.apple.com/macbook-pro/specs/
61 TFLOPs/s is a reused historical empirical fp16 NAX GEMM ceiling from
`System/skills/optimize-mlx-inference/SKILL.md` / Verdict, not an Apple-published
ceiling or a measurement in this run. This is an optimistic arithmetic roofline:
logical bytes are not measured DRAM traffic, cache/fusion can reduce them, and
norm/RoPE/residual/SwiGLU traffic and quantization overhead are omitted.
Attention and dequantization kernels need not reach the GEMM ceiling. The large
compute floor explains why fewer weight bytes alone do not ensure faster steps.

**Original recommendation (superseded by the bf16 decision above): retain fp16;
defer int8 to a demonstrated memory requirement, and reject int4 as a default
because its image delta is large without a proven speed gain.** These quantized
figures use fp16 as their baseline; they do not qualify bf16-based int8/int4.

## Historical fp16 default worker and mflux baseline

One 1024² default-path wire smoke: load 1.6098 s, image 8.2642 s including
first-use compilation, steps 2–4 median 0.7300 s, peak 18,693.65 MB physical
footprint. A warm 1024² full image was intentionally not repeated. The original three 512²
images used the explicitly required frozen quick recipe (photo warm-up then
photo/text once), not extra 1024² smokes: case median 1.0663 s, peak 15,915.82 MB.
The quick result has `profile=quick`, `precision=fp16`, `mode=exact`, no
measurement lock, missing GPU telemetry marked invalid and `gate_passed=false`.

The preserved old stock-mflux ABBA wire run in
A private bench record (not distributed) measured warm 1024² images
3.9442, 4.4827, 4.5148, 4.5442 s (median 4.4987), first images 8.5721 / 7.1160 s,
load 6.8553 / 3.4894 s, and coarse sampled peak footprint ~39 GB. Its callback
fires before the work it labels; subsequent warm intervals have median 0.9488 s
and do not include the final step. It is historical stock bf16 with fp32
promotions, a different numerical path, not a fresh same-path baseline. No
speedup ratio is claimed; rerunning it would violate the precision/RAM limits.

## Original implementation inspection and remaining checks

The 100% region sweep and PNG identities are saved in gate/evidence. Raw
captures and measured runner/metallib/test logs are retained under
`.build/flux-finish-20261008/`, outside the signed model folder (the catalog
enumerates ignored files too). The torch refenv was removed: 837,513,216
allocated bytes. Checkpoint weights and the existing mflux lab environment
were preserved. The root dependency lock was restored after SwiftPM changed
it during the prebuilt-test invocation. Worker
wire defaults, cancellation cleanup, seed wrapping, scheduler, tensor layout,
RoPE and the RMS overflow regression pass 11 tests in four suites. Four targeted
prebuilt Core tests for quick-result presentation/schema also pass. The changed
FLUX Swift files and native dispatcher pass strict formatting/lint. Original model-folder
validation accepted the fp16 quick result. The all-repository lint preflight
stops on existing D1 formatting in its regression/tokenizer/pictures sources;
those sources were preserved at that stage and fixed by the pre-release pass above.

The full native aggregate binary/app was neither built nor installed. Installed
app behavior, default supervisor dispatch and packaged metallib loading remain
unverified; the standalone package includes its own tested metallib. Signed
catalog unchanged and stale for this new model implementation until the
coordinator regenerates/signs at release. No independent review was spawned,
per the explicit sub-agent assignment. Stock fidelity and full image-quality
noise tolerance remain unqualified; next decision belongs to the coordinator
and the owner, not a silent precision expansion.
