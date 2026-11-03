+++
hypothesis = "The Arsenal kit runner can serve all of ESM3 open's documented tracks at FP16: the 1.4B transformer with every input track and head, the structure-token encoder and decoder and the function-token decoder."
change = "New folder models/esm3-sm-open-v1 with a kit runner ported from esm 3.4.2 (commit 8f0834c): Network.swift (transformer, geometric attention, VQ-VAE encoder and decoder, function decoder), Tokens.swift (the five tokenizers, the TF-IDF and LSH function tokenizer, the unmasking schedule), Protein.swift (PDB read and write, backbone frames, kNN), Sampling.swift and Model.swift (eight ops). Two kit changes: the runner's convert op also reads PyTorch archives and NumPy arrays (kit/runner/Sources/Runner/Checkpoints.swift) and kit try loads from that converted copy; folder_check inspects a pinned .pth without unpickling it."
measurement = "Isolated scratch profile, 2026-11-02, m5-max: convert of the ten pinned files, Load from the converted copy, every op once plus ten refusals, through scripts/mlock gpulock."
verdict = "kept"
numbers = { params_measured = 2701421969, fp16_bytes = 5402951146, load_s = 0.64, peak_footprint_mb = 5452, ops_implemented = 8, ops_refusing = 0 }
links = ["002-parity.md", "../gate/golden/golden.json", "../results/real-check-2026-11-02.md"]
+++
# 001 Port

## What is in the folder

One Swift module per maker module, property names following the checkpoint. The maker's `nn.Sequential` positions
("layernorm_qkv.0", "ffn.3", "sequence_head.0") are renamed at load (`CheckpointKeys`), because MLX reads a numeric key
segment as a list index.

- `Network.swift`: `EncodeInputs` (all eight input embeddings, including the pLDDT radial-basis projections and the
  summed residue-annotation bag), 48 `UnifiedTransformerBlock`s with rotary attention and q/k LayerNorms, geometric
  attention in block 0, the SwiGLU feed-forward with the 1/sqrt(48/36) residue scaling, the final LayerNorm, the six
  regression heads; `StructureTokenEncoder` (2 fully geometric blocks over each residue's 16 nearest neighbours, the
  relative-position embedding, the 4,096-code book); `StructureTokenDecoder` (30 blocks, the 6D-rotation backbone head,
  the pLDDT head, the pairwise head's PAE bins with pTM); `FunctionTokenDecoder` (LSH bits, 3 GELU blocks, mean pooling,
  the InterPro and keyword heads).
- `Tokens.swift`: the sequence, SS8 and SASA tokenizers; the function tokenizer (TF-IDF over the 58,641 keywords with
  the sublinear term frequency and L2 normalisation, the element-wise maximum over an InterPro entry's keywords, then
  the eight 8-bit locality-sensitive hashes); the vocabularies read from the pinned CSV, list and text files; the
  cosine and linear schedules and the annealed temperature.
- `Protein.swift`: PDB reading (first model, amino-acid ATOM records, one chain, first alternate location, MSE's SE as
  SD) and writing (N, CA, C and the inferred O, B-factor = pLDDT x 100), the Gram-Schmidt backbone frames with the
  black-hole frame for residues without coordinates, and the kNN graph with the sequence-distance fallback.
- `Sampling.swift`: top-p, temperature and argmax sampling, the function track's `<none>` ensemble rule, categorical
  entropy, and the choice of which masked positions to fill at each step.
- `Model.swift`: the prompt builder (sequence, length, PDB segment placed anywhere in a longer protein, SS8, SASA,
  function annotations) and the eight ops.

## Ops

The eight ops live in ESM3's own kind, `local.protein` (`kinds/local.protein.toml`). The planned ESM-2, Evo 2 and other
science folders share `local.science`, whose tree must stay identical between folders that install side by side, so
ESM3 does not extend it; its `embed_sequence` keeps ESM-2's fields and meaning.

| Op | What it does |
|---|---|
| `embed_sequence` | the same op ESM-2 declares: pooled and per-residue embeddings of the last block, before the final LayerNorm |
| `predict_masked` | one forward pass: the filled sequence with per-position candidates, predicted SS8 and SASA, residue annotations, optional raw logits of any track |
| `score_variants` | masked-marginal (default) or wild-type-marginal log-ratio scores for single and multiple substitutions |
| `generate` | iterative unmasking of one track or several in order (chain of thought), with temperature, annealing, top-p, num_steps, schedule, strategy and seed; returns whatever the prompt now knows, including a PDB backbone when the structure track is complete |
| `fold` | sequence to structure tokens to a backbone, with pLDDT, pTM, optional PAE, written as a PDB file in the request's out_dir |
| `inverse_fold` | a PDB backbone to a designed sequence, with the recovery against the file's own sequence |
| `predict_function` | function tokens from sequence, structure or both, decoded to InterPro entries and keywords over residue ranges |
| `encode_structure` | a PDB chain to its 4,096-way structure tokens, optionally decoded back (round trip) |

Every op is implemented; none refuses as unsupported. What the maker documents and this pass does not cover is in
`model.toml` `limits` and the README: multi-chain complexes, batching, prompting residue annotations, the Forge/Biohub
service API, guided and constrained generation, and cRMSD against a reference.

## Precision

FP16 throughout, chosen because every component passed its FP16 check and the largest activation anywhere in the
reference capture is 11,136, far inside FP16's 65,504 (log/002-parity.md). Float32 is kept only where the maker keeps
it or where the values demand it: the backbone frames and the distance and rotation logits of geometric attention, the
structure head's rotation and translation, the codebook's squared distances, the pLDDT, PAE and pTM expectations, the
rotary angle table, and every logit that is returned or sampled. Sampling, the schedule and the annotation merging run
on the host in Double.

Two numerical choices came out of the parity work, both recorded in `gate/golden/golden.json`:

- Geometric attention and the codebook use elementwise products and sums, not `matmul`, for their 3-wide contractions.
  Metal's float32 matmul on these Angstrom-scale vectors measured about 1e-3 relative error, which moved the softmax by
  10% and the encoder's tokens by 72%. With exact sums the encoder reproduces the maker's structure tokens exactly.
- The structure head places the backbone the same way.

## Weights

The repository ships four `torch.save` archives and two NumPy files, so Get converts them: the runner's `convert` op
reads each archive's pickle opcode by opcode (nothing is imported or called; only `OrderedDict`,
`_rebuild_tensor_v2` and `_rebuild_parameter` are interpreted) and writes FP16 safetensors beside the unchanged CSV,
list and vocabulary files. `kit/folder_check.py` reads the same pickles the same way to count 2,701,421,969 BF16
values and to refuse an archive that would need code. The kit's tests cover both readers, including a pickle holding a
function call and a compressed archive.
