# Official checkpoint, 2 Nov 2026

Source: `facebook/sam-audio-large@5f2cd3a9471a08c7282c06036be6893e18de8b70`, files `checkpoint.pt`
(14,861,356,211 bytes, an FP32 `torch.save` archive) and `config.json` (2,251 bytes). Get converts it with
`cast:audio-sam-v2` to the tensor set the port already loads (the layout of `mlx-community/sam-audio-large-fp16`).
The full checkpoint was not downloaded for this check; no model ran.

Pins. Meta's repository is gated, and the Hugging Face API masks the LFS SHA-256 of gated files. Its git object ID for
`checkpoint.pt` (`856a9f4b…`) is the SHA-1 of the LFS pointer, which holds the file's SHA-256 and size. Rebuilt from the
SHA-256 shared by eight public ungated copies and the size, the pointer gives that same object ID, so the pinned SHA-256
is Meta's file. The `-tv` copies have the same size but a different hash, and are a different model. Four of the
eight (mrfakename, jetjodh, ml-resources and oulianov, each `<name>/sam-audio-large`) were also checked directly: their
`checkpoint.pt` carries Meta's object ID. `config.json` has the same git blob as the re-upload's, so its
pinned SHA-256 is unchanged.

Layout, read from the checkpoint's `data.pkl` by HTTP Range (362 KB), never unpickled with code:

- 1,345 tensors, all FP32, 3,715 M parameters. 606 are dropped: the vision encoder (601, 671 M parameters) and the
  masked-video aligner (5). There are no `text_encoder` tensors, so T5-base stays the separately pinned asset.
- The rest maps to exactly the port's 735 tensors, with the same names and shapes. mlx-audio's `sam_audio` gives the
  renames (mlx-audio 0.2.10 `sanitize` and `_convert_weight_name`, read as text). The 4 LSTM bias pairs are summed.
  142 convolution tensors go from (out, in, k) to (out, k, in), and 12 ConvTranspose1d tensors from (in, out, k) to
  (out, k, in). Weight-normalized convolutions keep `weight_g` and `weight_v`, with no fold.
- `Workers/Audio/Tests/AudioWorkerTests/Fixtures/sam-audio-large-layout.json` holds both layouts.
  `SAMOfficialCheckpointTests` asserts the plan against them.

Values. Nine tensors (or leading rows of one) were range-read (536 KB), converted by the same rule, and compared with
the re-upload's FP16 shards. All 131,840 elements are bit-identical, maximum difference 0. The nine cover a square
residual Conv1d, a weight_g, a ConvTranspose1d weight_v and weight_g, the watermark ConvTranspose1d, a Snake alpha, a
summed LSTM bias, two output rows of the square transformer Conv1d, and the anchor embedding. On the square Conv1d,
the other permutation matches only 448 of 28,672 elements, so the check can tell the layouts apart. A BF16 round
trip before the cast would match only about one element in eight. The re-upload was cast from FP32 to FP16 directly, and FP16 Get output
should equal it.

Get cost, derived. Download 14.9 GB. Disk actually used peaks at about 22 GB: the original, the 6.08 GB of FP16
weights, and the 0.9 GB T5 asset with its 0.2 GB FP16 encoder. Get's shared preflight asks for twice the source size
free after the download, so about 45 GB must be free when Get starts. The original and the raw asset are deleted
afterwards, leaving about 6.3 GB. Each tensor is read, cast and written before the next. The largest kept tensor is 190 MB in FP32, so
conversion memory stays under about 1 GB.

Not checked here: a full Get of the 14.9 GB file, and output from converted weights. Bit-identical samples imply the
earlier live checks carry over, but this was not run end to end.
