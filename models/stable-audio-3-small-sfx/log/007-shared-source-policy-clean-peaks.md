+++
hypothesis = "One shared BF16-source rounding policy can qualify FP16 without probe-contaminated serving peaks."
change = "Implement coordinator R3 source-rounding policy in IntakeComponentGate and separate all intake kinds' capture workers from ordinary serving workers."
measurement = "Busy M5 Max quick checks, 15 October 2026; default Load, five-second/eight-step sound, balanced warm ABBA BAAB."
numbers = { format_decoder_relative_l2 = 0.10813467428123456, format_decoder_bound = 0.2448793673994303, added_swap_bytes = 0 }
verdict = "kept"
links = ["../results/intake/intake-2026-10-15T21-28-58Z-744e716d-c984-44a6-a386-2962793b4114.json", "../gate/evidence/quality-binding-2026-10-15.json"]
+++

FP16 is selected under shared policy fp32-source-bf16-nearest-truncation-fp16-ulp-v1. Int8 passes against FP16; int4 fails and remains offered with its warning. Prior runs, policies, bounds, failures and outcomes remain historical evidence. This is a new qualification and intake run.

The method follows DECISIONS-fp32-method-r3.md. Fresh GPU BF16 nearest and BF16 source-weight truncation captures measure rounding sensitivity; a separate third BF16 nearest A/A capture is exact and never calibrates that floor. Bounds freeze before executing the fresh FP16 candidate: twice the perturbation relative L2 plus one FP16 output ULP, matching cosine, finite outputs/activations, no new clamps and candidate FP16 range 65,504. No model-specific rewrite follows freezing. The pre-candidate-freeze receipt records the bounds hash and absent candidate directory before execution. No FP32 model executes.

| FP32-source component | Rounding floor relative L2 | FP16 ULP relative L2 | Frozen relative L2 max | Frozen cosine min |
|---|---:|---:|---:|---:|
| decoder | 0.122081236008 | 0.000716895383 | 0.244879367399 | 0.985526401810 |
| seconds | 0.006330033253 | 0.000711600934 | 0.013371667440 | 0.999968148253 |
| padding | 0.003974698758 | 0.000706434009 | 0.008655831525 | 0.999991380966 |

| Component | FP16 zero where BF16 normal (values) | FP16 subnormal where BF16 normal (values) |
|---|---:|---:|
| decoder | 1490 | 115906 |
| seconds | 0 | 0 |
| padding | 0 | 17 |

Output underflow counts are diagnostics, not proof that every intermediate avoided underflow. All three source-format components pass. SAME-S's relatively large rounding floor makes this a conservative internal policy; independent vendor half-GPU/component qualification and listening/perceptual equivalence remain unavailable.

| Tier | Verdict | Max component relative L2 | Min cosine | Log-mel diagnostic dB vs FP16 | Warm median s / 5-s clip | Ordinary serving lifetime peak GB |
|---|---|---:|---:|---:|---:|---:|
| fp16 | Pass; selected | 0.108134674 | 0.994175881 | 0.000000 | 0.137589 | 2.602748 |
| int8 | Pass | 0.026991469 | 0.999646469 | 0.552379 | 0.164088 | 1.605961 |
| int4 | Fail; warning | 0.416823009 | 0.913101198 | 14.901255 | 0.153780 | 1.067599 |

FP16 errors include source-format comparison with BF16; its served A/A outputs are exact. Integers and log-mel compare with selected FP16. Log-mel never decides a precision. The saved production fixture/reference are reused from the hash-identical pinned FP16 snapshot; all source-format calibration/candidate captures are fresh. Current served decoder/duration outputs equal the new candidate exactly.

ModelIntakeCommand now unloads capture workers and starts fresh ordinary default Load/generation workers for every kind, baseline and candidates. WorkerProcess counts dispatched measurement_components requests and the peak reader rejects any contaminated worker. All three result rows record serving PIDs and zero capture requests. A regression exercises the real worker transition and contamination rejection for decisions, sound, images and voices. Ordinary generation may establish a higher peak than the prior probe worker: isolation establishes the boundary, not a promised reduction.

Warm medians are busy-Mac quick checks; paired samples and drift remain in the result, with no speed ranking. Per-worker physical peaks cover Load/self-tests and generation, excluding probes and the other arm. Intake took 28.718 seconds under a 31.528-GB hard worker cap; existing swap 1,023,279,104 bytes did not grow. Source-format captures had a 4-GB cap and 590-second ceiling: 18.405/5.912 seconds including wrappers. Native GPU test times were 2.092/0.337 seconds, one test each. Calibration A/A relative L2 is zero for every component.

Medium was not rerun or rebound: Sound runtime source identity is unchanged and every retained peak equals its original ordinary Load-reply lifetime peak (see medium-retained-evidence.json). This new Small SFX run binds its own exact worker binary; Medium keeps its original exact-run binding.

Coverage remains 281 packed matrices per integer tier: DiT/conditioning 171, T5Gemma 85, SAME-S 25. Existing explicit skip reasons and half-only stored parameters remain unchanged. Final test counts, source hashes and cleanup receipts accompany this run. No downloads, installed app/store changes, merge, push, signing or installation.
