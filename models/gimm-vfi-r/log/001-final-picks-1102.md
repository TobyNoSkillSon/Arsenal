+++
hypothesis = "GIMM-VFI-R (GSean/GIMM-VFI gimmvfi_r_arb.pt) runs in the restoration worker as a Swift MLX port that matches the makers' own code, through the existing interpolate_video contract."
change = "Added GIMMNative (RAFT, correlation pyramids, softsplat, implicit motion model, AMT decoders), a torch-checkpoint Get conversion, the GIMM adapter, a shared interpolation driver used by RIFE and GIMM, the model folder, golden and tests."
measurement = "Makers' code on MPS (FP16 weights under autocast; CuPy softsplat replaced by a PyTorch scatter-add proved against the kernel formula) captured once on a 281x240 pair at three timesteps; MLX port compared per component on the reference's inputs and end to end; one real run through the shipped worker on an isolated profile."
verdict = "kept"
numbers = { softsplat_proof_max_abs = 2.4e-7, raft_flow_relative_l2 = 0.00047, max_component_relative_l2 = 0.0014, synthesis_psnr_db_min = 82.1, end_to_end_psnr_db_min = 67.37, end_to_end_max_abs = 0.032, downscaled_half_psnr_db = 62.28, downscaled_quarter_psnr_db = 64.36, worker_png_psnr_db_min = 61.8, weights_downloaded_by_worker = 0 }
links = ["README.md", "gate/golden/golden.json", "results/real-check-2026-11-02.md"]
+++

# Final picks, 2 Nov 2026

Runtime choice (AGENTS.md order): no bundled runner or existing MLX implementation runs GIMM-VFI, so this is Arsenal's
own Swift port on the pinned mlx-swift with the makers' code as the reference. The restoration family's RIFE port was
not reusable (different network); its NHWC pixel-shuffle convention was followed.

Weights: only `gimmvfi_r_arb.pt` is needed. It holds its own fine-tuned RAFT (the 179 `flow_estimator` tensors differ
from `raft-things.pth`, which the makers' constructor loads and the checkpoint then overwrites). The `_lpips` variant
(GIMM-VFI-R-P, perceptual loss) and the FlowFormer variant (-F) are not used.

Reference: the makers' code needs CUDA for CuPy's softsplat. `lab/evidence/gimm-vfi-r/torch_softsplat.py` replaces the
forward with a float32 scatter-add (the makers cast its inputs to float32) and proves it against a literal scalar
transcription of the CUDA kernel. The makers' code also needs Python 3.10 or older (dataclass defaults); CuPy, scipy
(an unused RAFT helper) and tensorboard (logging) are stubbed, nothing else changed. Precision: `model.half()` with
`torch.autocast(mps, float16)`, the maker's autocast mixed-precision mode; RAFT's `.float()` feature maps make a pure-half run impossible.

Decisions recorded once: the local flow-deviation term (E[f²] - E[f]² under a 3x3 Gaussian) is computed in float32, an
accumulator; the maker's autocast runs it in FP16, where the difference loses most of its digits and flow² overflows past
255 px. Interpolation positions within 1e-4 of 0 or 1 return the source frame (NTSC rates put output times microseconds off
the source frames). Frames under 64 px a side are padded to 64 so RAFT's
four correlation levels stay at least one cell wide. Above about a megapixel the flow stages run at half or quarter size
(the makers' ds_factor), chosen from the frame size. The frames/video driver that RIFE had inline is now shared
(`InterpolationDriver`), unchanged for RIFE except that a pair is prepared only when an in-between frame is needed.

Licence: S-Lab License 1.0, non-commercial, accepted on 2 Nov under the zero-income licence rule: `commercial = false` with the
amber conditions line. The Swift port derives from S-Lab-licensed code, so GIMMNative's sources carry
`LicenseRef-S-Lab-1.0` and `LICENSE-upstream-GIMMNative`; a public export of Arsenal must treat that folder as
non-commercial code.

Review fixes, 2 Nov: endpoint snapping (29.97 fps clip checked through the worker), float32 flow deviation, a refusal
for correlation volumes past Int32 indexing, Get refusing BF16 (Load is FP16 only), the PyTorch checkpoint reader moved
into IntegrationKit as one copy for Audio, Sound and Restoration, driver and adapter tests for RIFE and GIMM, the
reduced-size motion path compared with the makers at ds_factor 0.5 and 0.25, and end-to-end bounds tightened to just
under the measured values. Random-weight smokes now use fan-in scaled weights and the makers' initial splat weights;
the old 0.05 scale overflowed FP16 in the deep convolution stacks.

Not checked: benchmark accuracy, real footage above a megapixel, speed, the app-level Get/Load path.
