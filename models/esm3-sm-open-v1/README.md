# ESM3 Open Small (esm3-sm-open-v1)

ESM3 reads and writes a protein on three tracks at once: amino-acid sequence, 3D structure as 4,096 learned structure
tokens, and function as InterPro entries and keywords, with secondary structure and solvent accessibility alongside.
Mask any part of any track and the model fills it in. This folder is the open 1.4B model from Chan Zuckerberg Biohub
(released as EvolutionaryScale ESM3), with its structure-token encoder and decoder and its function-token decoder, as
a kit runner on MLX.

## Usage notes (2 Nov 2026)

### Ops

| Op | In | Out |
|---|---|---|
| `embed_sequence` | `sequence` | pooled 1,536-wide embedding, optionally per residue |
| `predict_masked` | a prompt (below) | the filled sequence with per-position candidates, predicted SS8 and SASA, residue annotations, optionally raw logits of any track |
| `score_variants` | `sequence`, `variants` like `A23G` (join a multi-mutant with `:`), `method` | one log-ratio score per variant |
| `generate` | a prompt plus `track` or `tracks` | the prompt as it now stands: sequence, SS8, SASA, function, structure tokens, and a PDB backbone with pLDDT and pTM once the structure track is complete |
| `fold` | `sequence` | a PDB file in the request's folder, per-residue pLDDT, pTM, structure tokens, optionally the PAE matrix |
| `inverse_fold` | `structure` (a PDB file) | a designed sequence and its recovery against the file's own sequence |
| `predict_function` | a prompt | InterPro entries and keywords over residue ranges |
| `encode_structure` | `structure` | the structure tokens of that chain, optionally decoded back to coordinates |

A **prompt** is any mix of: `sequence` (uppercase one-letter codes, `_` for a masked residue), `length` instead of a
sequence, `structure` (a PDB file, with `chain`, `structure_residues` to take a segment and `structure_at` to place
that segment inside a longer protein), `secondary_structure` (the 8 DSSP classes, `_` for unknown), `sasa` (square
angstrom per residue, `null` for unknown) and `function` (labelled residue ranges: a function keyword, an InterPro id
the model knows, or a residue-annotation label).

Generation takes the maker's controls: `num_steps`, `temperature`, `temperature_annealing`, `top_p`, `schedule`
(cosine or linear), `strategy` (random or entropy), `seed`, and `condition_on_coordinates_only`. With several `tracks`
the model generates them in the order given, which is the maker's chain of thought (function, then secondary
structure, then structure, then sequence). Temperature 0 takes the most likely token at every position. The seed is
returned, so a run can be repeated.

### Limits

- At most 1,024 residues, one chain per request. Multi-chain complexes (`|` chain breaks) are refused.
- A PDB file is read as: the first model, amino-acid `ATOM` records, one chain, the first alternate location of each
  atom, N, CA and C. mmCIF is not read. Residues missing a backbone atom are listed by `encode_structure` and are
  treated as having no frame, as the maker does. A file in which no residue has all three is refused, as is a gzip
  file; text that is not UTF-8 is read as Latin-1.
- A reply carries at most 640,000 numbers (16 MiB as JSON), checked before the model runs: `per_residue` embeddings up to
  415 residues, `return_pae` up to 800 residues, and `return_logits` up to 640,000 numbers in all (structure logits are
  4,096 per residue, so about 156 residues). Ask for less or a shorter protein above that.
- `generate` runs at most 1,024 model passes per request (`num_steps` per track, up to its masked positions, summed
  over tracks), so it ends inside the 15-minute limit; send fewer steps or tracks above that.
- A request body is at most 1 MiB of JSON over HTTP and MCP.
- Residue annotations can be read out of the model's head but cannot be put into a prompt: the maker's tokenizer needs
  the per-residue reference letters of each annotation. Such labels are accepted and leave that track empty.
- One protein per request: no batching.
- Not ported: the maker's Forge/Biohub service API, batch generation, guided and constrained generation, cRMSD against
  a reference, SASA computed from a structure (the maker computes that with biotite, not the model), and the other
  models in the same repository (ESMFold2, ESMC, the SAEs).
- Designs are predictions. The maker removed viruses and select agents from the training data and filtered the keyword
  vocabulary, and asks users to follow its acceptable-use policy.

### Parity

FP16, matched component by component against the maker's own code (esm 3.4.2, commit 8f0834c) run once on the Apple
GPU with bf16 weights and bf16 autocast, on a 60-residue sequence with six masks and on the first 64 residues of the
repository's own `data/1utn.pdb` with secondary structure, SASA and three function annotations. Of 57 checks, 51
passed and 6 missed bounds frozen beforehand (every row in `log/002-parity.md`, generated from the capture).

Passed: input tokens identical on every track; the input embedding and blocks 0, 1, 23 and 47 within relative L2
0.0115; every head's logits within 0.0058 at cosine 0.99998 or better; the sequence argmax identical; the structure
encoder's tokens identical; the structure decoder's pLDDT within 0.0023, pTM within 0.0012 and PAE within 0.12
angstrom; the function decoder's logits within 0.0020 and its 21 decoded annotations identical; one argmax generation
step giving the same sequence.

Missed: `A.argmax.structure` and the four fold checks, which come from the 4,096-way structure head, where the bf16
reference's top two logits are tied (median gap 0.0); and `D.backbone`, the structure decoder on identical tokens,
0.269 angstrom at most against a bound of 0.25 (RMSD 0.160). A float32 run of the maker's code, a diagnostic only and
not the reference, agrees with the port on 93.3% of structure tokens where the maker's own bf16 run agrees on 56.7%,
and puts the port's decoder 0.041 angstrom from float32 where the maker's bf16 decoder is 0.299 from it. The misses
stay recorded against the bf16 reference. The maker's benchmark figures (ProteinGym zero-shot average Spearman 0.466)
are not reproduced here.

The embeddings `embed_sequence` returns are the last block's residue states before the final LayerNorm, the maker's
`ESMOutput.embeddings`, averaged over residues without the start and end tokens.

### Real check

One isolated profile, 2 Nov 2026: Get-time conversion of the pinned PyTorch and NumPy files to FP16 safetensors, Load,
every op once, ten bad inputs refused. `results/real-check-2026-11-02.md` has the answers, including the folded
backbone (mean pLDDT 0.69, pTM 0.35) and inverse folding of the 1utn crop at 59.4% recovery.

Source: [biohub/esm3-sm-open-v1](https://huggingface.co/biohub/esm3-sm-open-v1/tree/47f0545b2b6daf26a93439a3cd610f4f7f3d5478),
MIT. Runs as FP16 through the Arsenal kit runner (MLX).
