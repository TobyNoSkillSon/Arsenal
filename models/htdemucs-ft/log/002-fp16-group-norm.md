+++
hypothesis = "The FP16 separation returns NaN because a half-precision reduction overflows."
change = "Group-norm statistics in Workers/Audio/Sources/AudioWorker/Demucs compute in float32 and return the input dtype, as the fused Metal kernels already did."
measurement = "Trace of every encoder/transformer stage on a 3.2 s speech + 220 Hz stereo clip with the pinned trained weights; op-level float32 reference tests (DemucsGroupNormTests)."
verdict = "kept"
numbers = { group_elements = 515970, separation_seconds_quick_check = 1.15, stems_sum_snr_db = 21.2 }
links = ["../README.md"]
+++

# FP16 group-norm overflow

22 October 2026. `separate_audio` at FP16 returned HTTP 422 "Audio output is invalid or exceeds the output bound." for every input: every clip is padded to the 7.8 s training segment, and the time branch's first DConv normalizes one group of 6 × 85,995 values. Above 32,768 elements the fused GroupNorm+GELU/GLU calls fell back to pure MLX `mean`/`variance` in FP16; the FP16 sum overflowed, the mean was infinite, and NaN reached every stem. The spectral branch's groups are smaller and stayed finite.

Statistics, affine and activation now run in float32 and return the input dtype, matching the fused kernels. With the trained weights, every traced stage is finite and the whole graph stays FP16 (it previously drifted to float32 after the first normalization, because a float32 epsilon promoted it). Quick check on the M5 Max while in use: the 3.2 s clip separated in 1.15 s; the four stems sum back to the mix at 21.2 dB SNR, and speech lands in vocals. No vendor (PyTorch Demucs) parity was run; trained-weight quality remains unqualified.
