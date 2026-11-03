+++
hypothesis = "KimberleyJSN/melbandroformer runs in the audio worker as an MLX port of the maker's MelBandRoformer, matching the maker's own inference."
change = "Added the Mel-Band RoFormer network (sharing BS-RoFormer's layers), the mel band layout, the maker's demix_track, a .ckpt Get conversion, separate_audio with the maker's overlap option, the golden and tests."
measurement = "The maker's code (Mel-Band-Roformer-Vocal-Model 25f44ffb, torch 2.15.0, librosa 1.0.0) on MPS with the network in FP16, captured once on a 12 s fixture; MLX port compared per component and end to end; one real run through the isolated test server."
verdict = "kept"
numbers = { features_relative_l2 = 0.00268, layer0_relative_l2 = 0.00082, trunk_relative_l2 = 0.00495, mask_relative_l2 = 0.0003, render_from_maker_mask_relative_l2 = 3.8e-7, chunk_relative_l2 = 0.00088, demix_relative_l2 = 0.00086, maker_sdr_db = 28.79, arsenal_sdr_db = 28.79 }
links = ["README.md", "gate/golden/golden.json"]
+++

# Final picks, 2 Nov 2026

Runtime choice (AGENTS.md order): the audio worker already ran BS-RoFormer, whose attention, feed-forward, band input and band output layers are the same upstream code; no Swift Mel-Band RoFormer existed in the app. The new network reuses those layers and adds what differs: 60 overlapping mel bands (librosa's Slaney filterbank membership with the maker's forced first and last corners), an RMSNorm closing every time and frequency transformer (BS-RoFormer's port has a final norm instead), a mask MLP of three linears (`MLP(depth=2)` upstream), and the scatter-and-average of overlapping band masks.

Reference choice: the maker of this checkpoint is Kimberley Jensen; her repository's `inference.py`/`utils.demix_track` is the documented way to run it. ZFTurbo's current training repository has a newer MelBandRoformer that zeroes the DC bin by default (`zero_dc=True`); Kim's code does not, and the port follows Kim's.

Weights: the release is a bare FP32 state dict without configuration. The configuration is built in and Get refuses any checkpoint whose 672 learned tensor names and shapes differ from it (the band widths come from the computed mel layout, so this also confirms the layout); the 12 rotary frequency buffers are checked against theta 10 000 and not stored.

Bugs found while porting (tests left behind): index tensors stored on the module were counted as parameters and blocked the conversion; they now live outside the module tree (the heavy test converts the real checkpoint). The parameter-count test uses a formula so the default suite never builds MLX arrays.

Not checked: real recordings, a benchmark, speed and memory figures (none claimed).
