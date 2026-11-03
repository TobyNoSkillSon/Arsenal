+++
hypothesis = "The makers' video frame lists, through Qwen3-VL's video processor and the same vision tower, reproduce the makers' trained multimodal scoring at FP16."
change = "VideoInput (video smart_resize with the per-frame cap, frame pairing, timestamps, patching), <video:N> placeholders, video expansion with '<t seconds>' blocks, per-pair rotary grids, video features into the video pads; media items type video with frames and fps."
measurement = "Weight-free prompt, grid, pixel, id and position checks against the makers' processor; answers, vision and decoder states against the makers' trained path on MPS in BF16 and FP16, bounds frozen first; headless real check."
verdict = "kept"
numbers = { max_abs_dp_vs_maker_fp16 = 0.000084, max_abs_dp_vs_maker_bf16 = 0.000148, maker_bf16_vs_fp16_floor = 0.000163, argmax_flips = 0, branches = 4, worst_decoder_cosine = 0.999905, vision_cosine_ball = 0.99995, vision_cosine_light = 0.99996, max_abs_activation = 691 }
links = ["README.md", "gate/gate.toml"]
+++

# 004 Video frame lists (2 Nov 2026)

The makers' requests carry videos as pre-extracted stills: `{"type": "video", "frames": [...], "fps": f}` (qev/media.py),
processed by Qwen3VLVideoProcessor with `do_sample_frames=False` and `cap_pixels_per_frame=True`. Ported: every frame is
read as PIL reads it and resized together to the video smart_resize size (per-frame pixels at most 768 tokens' worth and
the longest edge over the frame count, at least 1.05 x the shortest edge; frames smaller than 32 px scaled up first); an
odd count repeats the last frame; frames are paired in time for the patch embedding. The template's video block is
expanded per frame pair into "<t seconds>" (the pair's mean frame time, one decimal), vision start, that pair's video
pads and vision end; get_rope_index places each pair like a picture. Video features go into the video pads, picture
features into the image pads, each run through the vision tower on its own as HF does.

Parity: prompts, grids, normalised pixels, expanded ids and rotary positions identical on all three fixed requests;
answers within 0.00015 of the makers (their own BF16 and FP16 runs differ by 0.00016), no answer changes; vision 0.99995
and 0.99996 on the clips; decoder 0.9999 or better.

Not ported: decoding video files (the makers' runtime takes frames, not files). A one-frame video is refused, as the
makers' processor refuses it.
