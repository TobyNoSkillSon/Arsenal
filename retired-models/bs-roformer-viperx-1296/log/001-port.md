+++
hypothesis = "A native Swift MLX port of BS-RoFormer matches python-audio-separator at FP16."
change = "Added the audio-family adapter, a non-executing PyTorch checkpoint reader, the Get-time FP16 conversion and this folder."
measurement = "Fixed-tensor component parity and one 4 s end-to-end clip against python-audio-separator 0.45.0 on MPS with FP16 weights (22 Oct, M5 Max, machine in use)."
verdict = "kept"
numbers = { chunk_output_rel_l2 = 0.00056, mask_rel_l2 = 0.0015, vocals_wave_rel_l2 = 0.0005, vocals_logmel_rel_l2 = 0.0014, instrumental_wave_rel_l2 = 0.0008 }
links = ["gate/evidence/native/parity.json"]
+++

Reference and candidate both ran the network in FP16 on the GPU, with STFT and iSTFT in float32. Warm run of the 4 s clip: 0.63 s, MLX peak 1.3 GB (quick check, not a measurement).
