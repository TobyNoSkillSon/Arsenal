+++
hypothesis = "The mlx-audio-swift MossFormer2 SE port, with the maker's Kaldi front end and windowed decode, reproduces ClearerVoice-Studio's MossFormer2_SE_48K in the Sound worker at half precision."
change = "Vendored the mlx-audio-swift network (dtype-safe constants, float32 positions), wrote Kaldi fbank, STFT/iSTFT and the maker's decode, a non-executing cast from the maker checkpoint to BF16, and the Sound-family adapter."
measurement = "Weight-free goldens against the maker on CPU; component parity against the maker's BF16 MPS capture; one real Get, Load, enhance and Unload in an isolated profile."
verdict = "kept"
numbers = { features_rel_l2 = 1.94e-07, post_rel_l2 = 2.59e-07, segmented_rel_l2 = 1.93e-07, mask_rel_l2 = 0.00775, mask_cosine = 0.99997, wave_rel_l2 = 0.0132 }
links = ["README.md", "gate/golden/golden.json"]
+++

FP16 was tried first and rejected: the maker's own half run turns to NaN in block 1, where the gated FSMN product reaches about 1.9e7 (fsmn memory output 3.7e4 times the gate's 500). BF16 keeps every activation finite, so the cast is BF16, as the precision rule allows when FP16 fails.

The mlx-audio-swift port's front end used an HTK mel bank interpolated in Hz and a ceil frame count; the maker uses torchaudio's Kaldi fbank (triangles in mel space, 2048-point power spectrum, floor frame count). Arsenal's DSP follows the maker and matches it to about 2e-7 relative L2.

Maker deviations kept on purpose: fbank dither is 0 (the maker adds random dither of one 16-bit step), and a clip whose length fits the window stride exactly gets one more stride of padding so its last 0.5 s is enhanced (the maker leaves it silent).

The first parity run failed its frozen bounds (mask 0.044, computation block 0.13). Three port faults, each now covered by a default test (`mossFormer2NormsMatchTheMakerDefinitions`) and the heavy parity:

- The computation block's `intra_norm` is the maker's `select_norm("ln", dim, 3)`, which is `nn.GroupNorm(1, dim, eps=1e-8)` over channels and time; the mlx-audio-swift port used a per-frame LayerNorm.
- GlobalLayerNorm rounded its normalized tensor to BF16 before the affine step: 1.1% relative error at the encoder, where the maker matches exact math to 7.5e-5. It now runs in float32 and rounds once.
- Epsilons: MossFormerM's LayerNorm is 1e-6, ScaleNorm and the FSMN channel norm 1e-5 (the port used 1e-8).

Floors for scale, from the maker's own code: BF16 rounding versus truncation of the weights moves the mask by 0.022; the same BF16 network on CPU and MPS differs by 0.0056. After the fixes: encoder 1.4e-4, mask 0.0078 (cosine 0.99997), end to end 0.013.

Real cycle (isolated profile, 2 Nov 2026): Get from the local Hugging Face cache with no network, Load, enhance of the maker's 18.4 s stereo 44.1 kHz sample through the windowed path, Unload; same rate, channels and length back; quietest tenth of frames −53.8 dB in, −87.6 dB out. The no-weights smoke (`scripts/model-check.py smoke`) passes with 904 BF16 parameters.

## Review fix, 2 Nov 2026

The windowed decode (clips longer than `one_time_decode_length`) left a silent tail in two short-clip padding branches:
a clip shorter than one window, or shorter than a window plus a stride, could end inside the last window's dropped
0.5 s edge (at the default 4 s window). Padding and window starts now come from `MossFormer2DSP.windowPlan`: it keeps
the maker's padding and every maker window, then adds one stride wherever the kept ranges stop before the clip does (the
exact fit, as before, and those two branches). The maker returns silence in those cases; Arsenal enhances them.
Tests: `mossFormer2WindowPlanCoversEverySampleOnce` (both branches, the exact fit and the maker's uneven padding; kept
ranges tile the clip once and the maker's windows survive) and `mossFormer2WindowedDecodeLeavesNoSilentTail` (the
real decode with an all-pass mask on the CPU device, no weights). Parity results are unchanged: the golden's fixture is
one second and decodes in one pass.
