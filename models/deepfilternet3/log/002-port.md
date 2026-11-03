+++
hypothesis = "The mlx-audio-swift DeepFilterNet port, with libdf's exact DSP and its GRUs moved to the GPU, reproduces the maker's DeepFilterNet3 in the Sound worker at FP16."
change = "Vendored the port's network layers; wrote libdf analysis, synthesis and running normalisation, a Metal GRU kernel and a Metal running-mean kernel, the maker's post filter and attenuation limit, an FP16 cast and the Sound-family adapter."
measurement = "Weights check against the maker checkpoint; weight-free goldens on CPU; component parity against the maker's FP16 MPS capture; one real Get, Load, enhance and Unload."
verdict = "kept"
numbers = { pinned_tensors_identical = 118, mask_rel_l2 = 0.0027, spec_enhanced_rel_l2 = 0.00064, wave_rel_l2 = 0.00052, real_file_rel_l2 = 0.0007 }
links = ["README.md", "gate/golden/golden.json"]
+++

FP16 is selected: the maker's FP16 network runs finite on the reference clip. Only its post filter fails in FP16 (eps 1e-12 underflows to 0 and 0/0 gives NaN), so Arsenal computes the post filter in float32 and the reference applies the maker's formula in float32 too.

Bugs found and fixed:

- The deep filter's two accumulators were one array: `var outR = zeros, outI = outR`, and MLX's `+=` updates its left side in place, so real and imaginary parts summed into the same buffer (relative L2 1.24 against the maker). Covered by the default golden test.
- The port's closed-form running mean multiplies by 0.99^-t, which overflows float32 after about 8,800 frames (88 s); replaced by an exact sequential Metal kernel. The default test checks libdf's formula over 12,002 frames (in plain Swift on the CPU); the kernel itself is checked against the maker in the opt-in heavy test.
- The port's dense transposed-convolution weights and folded batch norm were float32 and promoted the whole network to float32; they now keep the weights' half dtype (checked in the heavy test).

The first maker capture was wrong, not the port: libdf's `synthesis` overwrites the spectrum array it is given, and the capture saved a view of it afterwards. The capture now passes a copy.

Real files: on the maker's `assets/noisy_snr0.wav` (10.6 s, 48 kHz) Arsenal's output is 0.0007 relative L2 from the maker's own; both raise SNR against the clean original from -2.98 dB to 0.78 dB. The no-weights smoke passes with 118 FP16 tensors.

Review (fresh reviewer, 2 Nov 2026) found no serious bug. Fixed from it: clips shorter than the two-frame lookahead now get all-zero features as the maker's negative pad gives (test `deepFilterNet3ShortClipsAndOddLengths`); odd lengths and analysis-synthesis round trips are tested without weights; the heavy test also cleans an odd-length clip; the random-weight path no longer reseeds the global generator; the unused LSNR output stays half; an fp16 `[[format]]` is declared; README and this log no longer overstate coverage or say nothing runs on the CPU (file reading, resampling and channel handling do).
