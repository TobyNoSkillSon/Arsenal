# GIMM-VFI-R

GIMM-VFI-R is the RAFT variant of GIMM-VFI, "Generalizable Implicit Motion Modeling for Video Frame Interpolation"
(Guo et al., NeurIPS 2024, S-Lab, Nanyang Technological University). It makes the frame at any moment between two
frames. RAFT estimates the optical flow both ways between the pair; a generalizable implicit motion model, a small
coordinate network modulated by motion latents that are forward-splatted to the requested time, predicts the flow at
that time; AMT-style decoders then refine the flow, warp both frames three ways and blend them into the frame. This
checkpoint (`gimmvfi_r_arb.pt`, 19.8M parameters) is trained on Vimeo-90K septuplets for arbitrary-timestep
interpolation.

## What it is for

- In-between frames at any moment, not only the midpoint: up to sixteen positions per pair.
- Slow motion and frame-rate conversion of video.
- Large or non-linear motion, where the motion model's per-time flow helps over a linear blend.

Compared with RIFE 4.25 in Arsenal it is slower (RAFT runs 20 refinement steps each way per frame pair) and aims at
quality on larger motion. Flow and pair features are computed once per pair, so extra timesteps of the same pair are
cheap.

## Inputs and outputs

`interpolate_video` uses the same input schema and outputs as RIFE 4.25 (`ops/interpolate_video.input.json`):

- `frames`: exactly two images of equal size (PNG or JPEG), with optional `t`: 1 to 16 positions from 0 to 1, default
  `[0.5]`. 0 and 1 return the source images without running the model. Returns one PNG per `t`, ids `frame-0`, ...
- `video` with `factor` (2 to 16, default 2) or `target_fps` (1 to 240): returns `interpolated.mp4`, H.264 at the
  source size, with the source's audio and rotation kept.

## How Arsenal runs it

A Swift MLX port in the restoration worker (`Workers/Restoration/Sources/GIMMNative`), checked against the makers' code.
Get reads the makers' PyTorch checkpoint without executing it, drops BatchNorm step counters and RAFT's duplicate
`norm3` entries, transposes convolution kernels and stores FP16 (`cast:restoration-gimm-vfi-r-v1`, 39.6 MB). The network
runs as the maker's autocast mixed-precision mode does: convolutions and matmuls in FP16; normalisation, warping, correlation
lookups, flows and the forward splat in float32. One deliberate difference: the local flow-deviation term
(E[f²] - E[f]² under a 3x3 Gaussian) is computed in float32, where the maker's FP16 loses most of its digits and
overflows on motion beyond 255 px. Positions within 1e-4 of 0 or 1 return the source frame, so NTSC rates (29.97 fps)
do not run the model for frames that coincide with source frames.

Frames are padded, centred and edge-replicated, to multiples of 32, as the makers' demo does. Motion is estimated at
full size up to about a megapixel (1280x736 after padding); larger frames estimate it at half size (1080p, 1440p) or a
quarter (4K) and synthesise the frame at full size, as the makers run their 2K demo at half.

## Usage notes (2 Nov 2026)

What it does: in-between frames at any moment between two frames, and frame-rate conversion of video.
Inputs: two same-size images with up to sixteen `t` values, or a video with `factor` or `target_fps`; frames up to 16
megapixels, at least 16 pixels a side.
Limits: occlusions, motion blur and scene cuts between the frames give ghosting; above about a megapixel the motion is
estimated at half or quarter size; it is slower than RIFE; FP16 only. Memory grows quickly with frame size: the
real check's 844x720 clip peaked at about 6 GB in the worker (one run, Mac in use, not a benchmark).

Parity: checked against the makers' own code (GSeanCDAT/GIMM-VFI) run once on the Mac's GPU with FP16 weights under
its autocast mixed-precision mode, on two 281x240 frames at t = 0.25, 0.5 and 0.75. The makers' splatting kernel needs CUDA
(CuPy); it was replaced in the reference by a PyTorch equivalent, proved against the kernel's own formula to 2.4e-7.
On the reference's own inputs every part matches: RAFT flow within 0.05% (relative error), the other components within
0.14%, the synthesised frames at 82 to 85 dB PSNR. From the frames alone the whole model reaches 67.4 to 69.8 dB PSNR
(average difference 0.00014 of full scale; worst pixel 0.032). Through the shipped worker, the 8-bit frames match the
makers' at 62 dB, an average of 0.03 of one level. The reduced-size motion path for large frames, on the makers' own
844x720 demo pair against their code at the same scale, gives 62.3 dB at half size and 64.4 dB at a quarter. Component
bounds were set before comparing (components 1%, RAFT 2%); the end-to-end bounds sit just under the measured values
(67 dB, 62 dB at reduced size) so a regression fails them. Golden: `gate/golden/golden.json`; real run: `results/real-check-2026-11-02.md`.

Not checked: benchmark accuracy (Vimeo-90K, SNU-FILM-arb), real footage above a megapixel (the reduced-size path was
compared with the makers on an 844x720 pair run at half and quarter scale), speed.

## Licence and credit

S-Lab License 1.0: non-commercial use only. Redistribution in source or binary form must keep S-Lab's copyright
notice, conditions and disclaimer; commercial use needs the authors' permission. Arsenal shows this line in amber. The
Swift port is a derivative of the makers' code and carries the same licence (`Workers/Restoration/LICENSE-upstream-GIMMNative`);
its RAFT parts come from princeton-vl/RAFT (BSD-3-Clause). Cite Guo, Li, Lin, Hu, Wang and Loy, "Generalizable Implicit
Motion Modeling for Video Frame Interpolation", NeurIPS 2024.
