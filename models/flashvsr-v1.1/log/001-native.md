+++
hypothesis = "FlashVSR v1.1 runs in the restoration worker from the xocialize MLX-Swift port, at FP16 from the BF16 release."
change = "FlashVSRNative imports the port's core; FlashVSRAdapter streams the whole clip through FlashVSRStream with one seed and keeps the source audio; Get casts BF16 to FP16."
measurement = "FlashVSRTests.fp16MatchesBF16Release on 25 real frames (384×384 padded, chunk 0, seed 0), each FP16 component on the BF16 run's inputs: LQ projector rel L2 0.0030, cosine 0.999996; DiT 0.0225, 0.99975; decoder 0.0302, 0.99954; all finite. End to end (diagnostic) 0.0695, 0.9976. Bounds fixed before the run: rel L2 ≤ 0.05, cosine ≥ 0.999."
verdict = "kept"
numbers = { lq_proj_rel_l2 = 0.0030, dit_rel_l2 = 0.0225, decoder_rel_l2 = 0.0302, end_to_end_rel_l2 = 0.0695 }
links = ["../README.md", "../../../Workers/Restoration/PORTED-FROM.md"]
+++

FP16 is selected: every component passes against the BF16 release. The makers' PyTorch code was not run as a reference; the port's author reports CPU FP32 parity with it.
