+++
hypothesis = "A Swift MLX port of the maker's UniPASE (DeWavLM-Omni, Vocos adapter and vocoder, TF-GridNet PostNet) with the maker's torchaudio resampling and inference scripts reproduces Xiaobin-Rong/unipase in the Sound worker at FP16."
change = "Wrote the four networks and their DSP on the app's mlx-swift, a non-executing cast from the maker's four checkpoints to one FP16 safetensors file, the Sound-family adapter (rates, PLC, one-pass limit, long-form, peak matching), weight-free goldens and heavy component parity."
measurement = "Weight-free goldens against the maker's code on the CPU; component parity against the maker's FP16 MPS capture on each component's own maker input; one real Get, Load, enhance and Unload in an isolated profile."
verdict = "kept"
numbers = { resample_rel_l2_max = 1.73e-07, postnet_dsp_rel_l2_max = 2.97e-07, long_form_rel_l2_max = 5.97e-08, vocos_istft_rel_l2 = 9.12e-06, L24_rel_l2 = 0.00679, adapter_rel_l2 = 0.00174, vocoder_wav_rel_l2 = 0.00687, postnet_out_rel_l2 = 0.000404, end_to_end_stft_magnitude_rel_l2 = 0.0216 }
links = ["README.md", "gate/golden/golden.json"]
+++

FP16 was tried first and kept. In the maker's own FP16 run on MPS every captured activation is finite; the largest at a component boundary is 1,053 (the PostNet's band-split input), and the largest inside WavLM is 761 (layer 23), far inside FP16's 65,504. The cast also checks every weight against that range. No BF16 fallback was needed.

Reference: the maker's modules, called component by component on a fixed 3 s, 48 kHz clip (the maker's own noisy sample fileid_295.wav with a 100 ms gap of digital silence, so the packet-loss path runs), networks in FP16 on MPS, DSP in float32 on the CPU. The walk was checked against the maker's own module calls (WavLM_feat.forward, VocosAdapter.forward): identical. The PostNet imports two espnet2 helpers; they were stubbed with exactly what they are (an nn.Module base class and nn.PReLU), so espnet was not installed.

Bounds were frozen before any port code ran (lab/evidence/unipase/bounds-frozen-2026-11-02.json). For scale, the maker's own BF16 run differs from its FP16 run by 0.053 at L24, 0.32 at the vocoder waveform and 0.116 in STFT magnitude at the output: the vocoder turns small input differences into phase drift, so the end-to-end bound is on STFT magnitude, not on the waveform.

Result, each component on the maker's own input: WavLM CNN 8.3e-4, layer 1 5.9e-4 (isolated 4.0e-4), layer 24 0.0068 (isolated 0.0065), L1 6.0e-4, L24 0.0068 (cosine 0.99998), PLC off 9.6e-4 and 0.0080, adapter 0.0017, vocoder hidden 0.0058, vocoder waveform 0.0069, PostNet network 0.0047, PostNet output 4.0e-4. End to end: STFT magnitude 0.022 (bound 0.116), waveform 0.063 (cosine 0.998). The first parity run passed; no port fault was found.

Maker faults found and handled (each covered by a default test):

- inference_long.py crashes on clips its loop leaves with one segment (6 to 10 s: `.cpu()` on a NumPy array) or none (under 6 s: an empty list). Arsenal enhances such a clip in one piece.
- inference_long.py slices the enhanced segments and writes the file at the input rate even when `--sr_out` differs, so its long-form output is wrong whenever the output rate is not the input rate. Arsenal slices at the output rate.
- `--enable_plc` is an argparse `type=bool`, so any value on the command line (even "False") turns PLC on; Arsenal's `enable_plc` is a real boolean.

Deliberate differences: stereo is enhanced channel by channel (the maker's scripts take mono, inference_long.py only the first channel); peaks are matched per channel.
