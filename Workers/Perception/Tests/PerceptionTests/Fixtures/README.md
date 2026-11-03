`rf-detr-nano-config.json` is the unmodified configuration from
`Roboflow/rf-detr-nano@9f201577d9415f0378084748d27e68586e6b3600/config.json`.
It separates 1-based feature extraction indices from 0-based window attention indices.
The attention regression computes the reference merge/softmax using scalar arithmetic;
the positional regression follows `roboflow/rf-detr@5b22dfc/src/rfdetr/models/transformer.py`.
No weights or tensor payloads are included.

`tokenizers/` holds byte-identical copies of the non-weight tokenizer files that the model folders pin: `tipsv2/tokenizer.model` (google/tipsv2-b14), `locateanything/` (mlx-community/LocateAnything-3B-bf16) and `clip-bpe/` (`vocab.json`, `merges.txt` of openai/clip-vit-base-patch32, the SAM 3.1 `clip-bpe` asset). Production loads read the verified model folder, never these copies. Random-weight lab checks name them through `assets` in `models/<id>/gate/no-weights.json` (`$PERCEPTION_FIXTURES`), and the worker has no resource bundle for them.

`sam31-official-layout.json` lists, without values, every tensor of Meta's `facebook/sam3.1@daa63191845a41281374e725f4c9e51c7a824460/sam3.1_multiplex.pt` (name, shape, storage dtype, read from the checkpoint's pickle by HTTP range requests) and the tensor names and shapes the SAM 3.1 port loads (the `mlx-community/sam3.1-bf16@a992e302ea9b0f03f41dfd93414a4fd0e818f65b` safetensors header without the unused CLIP projection). `SAM31OfficialCheckpointTests` checks the Get-time mapping between them.
