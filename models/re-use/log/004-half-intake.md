+++
hypothesis = "The pinned FP32-only release can serve without FP32 learned parameters and expose all three precision tiers."
change = "Qualify the half format, stream all eligible matrices, and bind default Load figures through intake."
measurement = "M5 Max, 40 GPU cores, 15 October night / 16 October 2026 local; busy-machine quick check."
numbers = { int8_max_relative_l2 = 0.013236952048142321, int4_max_relative_l2 = 0.10968726658131488, int8_peak_gb = 2.536588536, added_swap_bytes = 0 }
verdict = "kept"
links = ["../results/intake/intake-2026-10-15T22-18-28Z-eb5a33d6-dbed-43fc-b36d-e86baf57ae2c-rescore-8bbf4359-e31b-4ea1-bd2b-11628594cb5d.json", "../gate/evidence/quality-binding-2026-10-15.json"]
+++

Get downloaded exactly 38,588,118 bytes at the pinned revision, including sidecars. Source hashes were rechecked after intake. The isolated lab store and downloaded weights are removed at closing; the installed app/profile was not used.

FP16 failed the pre-frozen shared FP32-source policy, so BF16 is the native tier. The GPU BF16 nearest/truncation source casts, exact A/A repeat, failed FP16 captures and frozen bounds remain in this run’s format-qualification directory. No FP32 learned tensor or matmul executed. Independent vendor qualification remains open.

7,495,680 of 9,607,747 learned parameters are packed at both int8 and int4 (78.01704%). Quality-driven 16-bit fallbacks: 0 parameters, 0%. Int8 passed every saved-input component; no component needed an exemption. Int4 uses maximum supported compression, fails the policy, and remains offered with the orange quality warning.

All group-64-compatible matrices are included, notably the SSM xProj matrices; width-4 dtProj, convolutions, norms, biases and SSM vectors stay BF16 because their layouts are not compatible with this affine Linear recipe. Captures cover encoder, each of 30 time/frequency blocks and both decoders (63 outputs). The shared default workload is a three-second 24-kHz mono tone, default five-second chunk setting, denoise mode, and usable WAV delivery.

The BF16/int8/int4 kernel self-tests fail the retained 40 dB SNR criterion (38.2/38.5/20.3 dB), so ordinary Load serves stock MLX. It is slower than real time on this workload. This intake preserves that gate and records the actual fallback; kernel changes require separate work. The final default path is verified, while speech quality beyond the synthetic diagnostic remains unqualified.

Numeric captures and measurements are from `intake-2026-10-15T22-18-28Z-eb5a33d6-dbed-43fc-b36d-e86baf57ae2c`. A host-only pipeline rescore re-emitted RE-USE/BF16-specific policy and binding wording; inputs, outputs, all warm samples, source/worker identity and physical peaks are unchanged. This is not a new measurement. The original run retains its historical generic Sound wording.

| Tier | Verdict against BF16 | Max rel L2 | Min cosine | Log-mel diagnostic dB | Warm RTF s/s | Peak GB |
|---|---|---:|---:|---:|---:|---:|
| bf16 | Pass | 0.000000000 | 1.000000000 | 0.000000000 | 2.473202 | 2.560165 |
| int8 | Pass | 0.013236952 | 0.999913381 | 0.335602638 | 2.428791 | 2.536589 |
| int4 | Fail; offered with warning | 0.109687267 | 0.993971363 | 1.801591667 | 2.716430 | 2.509506 |

Each component independently consumes its saved reference input; exact keys, shapes and counts precede values. Quant bounds are relative L2 ≤0.05, cosine ≥0.999, finite, no new clamps and BF16 activation range. Decision end-to-end bounds are max probability delta ≤0.017 and unchanged answers. Audio log-mel/waveform distances are diagnostics, not precision selectors.

Warm samples in balanced ABBA BAAB windows (conversion excluded; baseline and candidate resident):
- int8: baseline seconds [7.406582291587256, 7.400039333384484, 7.234831416630186, 7.287488416652195]; candidate seconds [7.346708666649647, 7.269095291616395, 7.262002041679807, 7.303650416666642]; baseline spread 0.171751s, candidate spread 0.084707s; early-to-late baseline drift -1.920%.
- int4: baseline seconds [9.98796145839151, 8.028886000043713, 7.432630791678093, 7.494028791668825]; candidate seconds [10.440140333259478, 8.510445041581988, 7.788133125053719, 7.430058458354324]; baseline spread 2.555331s, candidate spread 3.010082s; early-to-late baseline drift -17.152%.

Overlapping/noisy busy-Mac samples do not establish a speed gain. Every peak is a fresh serving worker’s lifetime physical footprint including Load and ordinary operation, with zero measurement_components requests. The raw run retains cold request/load separately. Packing reduces learned-weight bytes read by matmuls but leaves the operation count and most activation/scan costs; RE-USE’s multi-GB peak is dominated by stock scan intermediates, not its roughly 19-MB BF16 learned weights.

The real run completed in 216.356s under a 48.541-GB aggregate worker cap. Existing system swap 1,023,279,104 bytes did not grow. Runs enforce a 590-second ceiling and terminate only their own children. No merge, push, signing or installation was performed.
