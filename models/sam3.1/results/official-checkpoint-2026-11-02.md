# Meta's official checkpoint (2 Nov 2026)

Quick check, no model inference. Source: `facebook/sam3.1@daa63191845a41281374e725f4c9e51c7a824460`, recipe `cast:sam31-fp16-v3`.

## Pins

The repository is gated and this Mac's Hugging Face account is not approved, so the files were not read from it directly.

- `sam3.1_multiplex.pt`, 3,502,755,717 bytes. Ten public ungated copies share one LFS SHA-256 and this size. The Hugging Face API hides the gated file's SHA-256, but publishes the git id of its LFS pointer. The pointer rebuilt from the copies' hash and size has exactly that git id (88589b6b0bee708b57010d3ed6f413476e4be255), so the pinned hash is the official file's.
- `config.json`, 25,843 bytes, downloaded from a public copy; its git blob id equals the official one (3e600e765b0f1b4c52680aed9ad242c9a90f7ed1). The pinned SHA-256 is of those bytes.

## Layout

The checkpoint's ZIP directory and pickle (0.43 MB, by HTTP range requests) list 1,623 tensors: 1,591 FP32 and 32 complex64 RoPE buffers. The conversion plan maps them onto exactly the 1,960 tensors the port loads, with equal names and shapes; it leaves out the RoPE buffers and CLIP's unused pooled-output projection. Test: `SAM31OfficialCheckpointTests` (fixture `sam31-official-layout.json`).

## Values

Ten real tensors (5.8 MB, range-read from a public copy) went through the built worker's `convert` and were compared with the same tensors of `mlx-community/sam3.1-bf16@a992e302ea9b0f03f41dfd93414a4fd0e818f65b` (FP32, made from the same weights by mlx-vlm). They covered a Conv2d kernel, a ConvTranspose2d kernel, a fused q/k/v attention weight and bias split in three, the ViT position embedding without its class-token row, the four point embeddings stacked, and the Gaussian matrix stored twice. All 12 outputs are FP16 and equal to the reference rounded to FP16, element for element (largest absolute difference 0.00086, within half an FP16 step everywhere). The converted `config.json` parses identical to mlx-community's.

Not checked: a full Get from Meta's repository with an approved token, and the remaining 1,948 tensors' values.
