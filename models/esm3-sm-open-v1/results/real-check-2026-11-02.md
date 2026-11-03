# Real check, 2 Nov 2026

One isolated scratch profile (its own `ARSENAL_SUPPORT_DIR` and
`ARSENAL_CACHE_DIR`), the ten pinned files APFS-cloned out of the Hugging Face cache, every request through the kit
runner over Arsenal's wire protocol under `scripts/mlock gpulock`, headless. M5 Max, Mac in use, on AC. No speed or
memory figures are claimed; the request times below are quick checks the kit recorded.

- **Get-time conversion**: the runner's `convert` op read the four `torch.save` archives and the two NumPy files
  without unpickling them and wrote FP16 safetensors; the CSV, list and keyword files were copied unchanged. 5.45 GB
  in ten files (`esm3_sm_open_v1.safetensors` 2,803,532,516 bytes). No tensor left FP16's range, so none stayed
  float32.
- **Load** from that converted copy: every tensor of the four checkpoints matched by name and shape; 0.64 s, peak
  footprint 5,452 MiB (whole Mac stayed near 35 GB in use, zero swap added).
- **Every op once** (12 requests, all HTTP-equivalent `ok`):
  - `embed_sequence`: 1,536 numbers for 60 residues; per-residue variant returned 12 rows.
  - `predict_masked`: the six masked residues filled (`MSKGLELFTGVGPILVELDGDVDGHKFSVSGEGTGDATYGKLTLTFICTTGKLPVGWPTL`),
    three candidates each with probabilities, a full SS8 string, per-residue SASA, and the sequence logits on request.
  - `score_variants`: S2A +0.41, G4W -1.37, the double S2A:G4W -0.96 (masked marginal); wild-type marginal gave S2A
    +0.40.
  - `generate`: inpainting three steps at temperature 0.7 from the masked prompt; a chain of thought over
    secondary structure, structure and sequence for a 24-residue protein (mean pLDDT 0.94, pTM 0.38, a 24 x 24 PAE
    matrix and a PDB backbone); and a motif prompt, 20 residues of 1utn placed at position 30 of an 80-residue protein,
    whose designed sequence carries that motif unchanged.
  - `fold`: the 60-residue GFP prefix to a PDB file (18,717 bytes, `chemical/x-pdb`, with the inferred O and pLDDT in
    the B-factor column), mean pLDDT 0.69, pTM 0.35.
  - `inverse_fold`: the 64-residue 1utn crop to `IIGGSECGAHNWPWQVSLDFGGHFCGGTIINENWVLTAAHCYRDGITVVAGAHNLNVDEGNEQR`,
    59.4% identical to the file's own sequence (`IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQF`).
  - `predict_function` on that chain's sequence: 27 keywords, among them peptidase, serine, proteolysis and
    chymotrypsin, and InterPro IPR014780; with the structure as well (parity case G) the maker's own 21 annotations
    come back identically, including Peptidase S1C (IPR001940).
  - `encode_structure`: 64 structure tokens, no missing backbone atoms, and the round trip back to coordinates with
    mean pLDDT 0.95 and pTM 0.76.
- **Ten bad inputs refused** with `invalid_request` and a plain message: 1,025 residues, the letter J, a `|` chain
  break, masks in `fold`, a prompt with nothing masked, a variant naming the wrong wild-type residue, an unknown
  function label, a secondary-structure string of the wrong length, an unknown schedule, and `inverse_fold` without a
  structure.
- **Not checked**: anything above 1,024 residues, multi-chain complexes, long multi-step sampling at temperature,
  prompting residue annotations (not supported), int8 or int4, and the maker's benchmark figures. The app's own
  service was not used: the folder is not installed and `~/Applications/Arsenal.app` was left alone.

## Rerun after the review fixes, 2 Nov 2026

The same isolated setup, rebuilt runner (hardened checkpoint reader, nonstandard PDB residues kept as X, sanitized
capture names, its own `local.protein` kind), fresh APFS clones of the ten pinned files, under `scripts/mlock gpulock`
in about 12 s of requests after a one-minute conversion, Mac at about 30 GB in use with no swap added.

- Conversion and Load: identical file sizes; Load matched every tensor.
- All 22 requests gave the same answers as the first run, field for field, except `predict_function`'s seed, which
  that request leaves unset so a new one is drawn (its annotations are unchanged).
- The parity capture was rerun on the same fixtures: every one of the 57 components gives the same number as before
  (`lab/evidence/esm3-sm-open-v1/compare-output-rerun.txt`), 51 passed and 6 missed, as in `log/002-parity.md`.

