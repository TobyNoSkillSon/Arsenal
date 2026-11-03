+++
hypothesis = "A native Swift MLX port of AudioSR basic matches audiosr 0.0.7 at FP16."
change = "Added the audio-family adapter (VAE, UNet, HiFi-GAN, DDIM, the reference's DSP), the Get-time FP16 conversion and this folder."
measurement = "DSP test vectors against SciPy/librosa; component parity on the reference's inputs and one 5.12 s clip replaying the reference's noise, against audiosr on MPS under FP16 autocast (22 Oct, M5 Max, machine in use)."
verdict = "kept"
numbers = { unet_rel_l2 = 0.0013, vae_decode_rel_l2 = 0.001, vocoder_rel_l2 = 0.0019, final_wave_rel_l2 = 0.022, final_logmel_rel_l2 = 0.0057 }
links = ["gate/evidence/native/parity.json"]
+++

The generated high band depends strongly on the noise: the reference itself puts 3.8% of the energy above 8 kHz with seed 42 and 55% with seed 43 on the same clip. That is the noise-like band measured on 21 Oct, not a port error. Warm run of the 5.12 s clip: 3.5 s, MLX peak 3.3 GB (quick check, not a measurement).
