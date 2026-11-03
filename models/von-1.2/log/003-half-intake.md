+++
hypothesis = "The pinned FP32-only release can serve without FP32 learned parameters and expose all three precision tiers."
change = "Qualify the half format, stream all eligible matrices, and bind default Load figures through intake."
measurement = "M5 Max, 40 GPU cores, 15 October night / 16 October 2026 local; busy-machine quick check."
numbers = { int8_max_relative_l2 = 0.010798241518485402, int4_max_relative_l2 = 0.1145965684602558, int8_peak_gb = 0.719963264, added_swap_bytes = 0 }
verdict = "kept"
links = ["../results/intake/intake-2026-10-15T22-12-44Z-571bdecc-ad47-4de2-afc5-63cd4db8ef74.json", "../gate/evidence/quality-binding-2026-10-15.json"]
+++

Get downloaded exactly 1,584,908,301 bytes at the pinned revision, including sidecars. Source hashes were rechecked after intake. The isolated lab store and downloaded weights are removed at closing; the installed app/profile was not used.

FP16 failed the pre-frozen shared FP32-source policy, so BF16 is the native tier. The GPU BF16 nearest/truncation source casts, exact A/A repeat, failed FP16 captures and frozen bounds remain in this run’s format-qualification directory. No FP32 learned tensor or matmul executed. Independent vendor qualification remains open.

395,247,616 of 395,310,081 learned parameters are packed at both int8 and int4 (99.98420%). Quality-driven 16-bit fallbacks: 0 parameters, 0%. Int8 passed every saved-input component; no component needed an exemption. Int4 uses maximum supported compression, fails the policy, and remains offered with the orange quality warning.

The row-streamed token embedding, every encoder matrix and scorer dense matrix are packed. Norms, biases and the 512-parameter output matrix stay BF16. Both English-only probes capture the embedding, all 28 encoder layers and scorer (60 outputs), plus three calibrated served probability vectors. All three chosen answers are unchanged at int8.

The older converter wrote its view metadata to the store-reserved arsenal-derived.json name. The current recipe uses arsenal-decisions-view.json, preserving coverage and declared half fallbacks. Only the current exact-run binding qualifies the table; older runs remain historical.

| Tier | Verdict against BF16 | Max rel L2 | Min cosine | Max probability delta | Warm ms/decision | Peak GB |
|---|---|---:|---:|---:|---:|---:|
| bf16 | Pass | 0.000000000 | 1.000000000 | 0.000000000 | 7.163625 | 1.172310 |
| int8 | Pass | 0.010798242 | 0.999941713 | 0.005583972 | 7.668812 | 0.719963 |
| int4 | Fail; offered with warning | 0.114596568 | 0.993436331 | 0.027073145 | 6.906062 | 0.529630 |

Each component independently consumes its saved reference input; exact keys, shapes and counts precede values. Quant bounds are relative L2 ≤0.05, cosine ≥0.999, finite, no new clamps and BF16 activation range. Decision end-to-end bounds are max probability delta ≤0.017 and unchanged answers. Audio log-mel/waveform distances are diagnostics, not precision selectors.

Warm samples in balanced ABBA BAAB windows (conversion excluded; baseline and candidate resident):
- int8: baseline seconds [0.007692666724324226, 0.007528791669756174, 0.007643874967470765, 0.006950875045731664]; candidate seconds [0.007811583345755935, 0.0075630833161994815, 0.007556041586212814, 0.00777454162016511]; baseline spread 0.000742s, candidate spread 0.000256s; early-to-late baseline drift -4.117%.
- int4: baseline seconds [0.0073763750260695815, 0.006881749955937266, 0.006784041645005345, 0.006888333358801901]; candidate seconds [0.007144249975681305, 0.006940291612409055, 0.006769583327695727, 0.0068718333495780826]; baseline spread 0.000592s, candidate spread 0.000375s; early-to-late baseline drift -4.108%.

Overlapping/noisy busy-Mac samples do not establish a speed gain. Every peak is a fresh serving worker’s lifetime physical footprint including Load and ordinary operation, with zero measurement_components requests. The raw run retains cold request/load separately. Packing reduces learned-weight bytes read by matmuls but leaves the operation count and most activation/scan costs; RE-USE’s multi-GB peak is dominated by stock scan intermediates, not its roughly 19-MB BF16 learned weights.

The real run completed in 21.955s under a 44.715-GB aggregate worker cap. Existing system swap 1,023,279,104 bytes did not grow. Runs enforce a 590-second ceiling and terminate only their own children. No merge, push, signing or installation was performed.
