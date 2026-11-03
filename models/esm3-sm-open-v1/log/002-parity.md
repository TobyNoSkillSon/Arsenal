+++
hypothesis = "The MLX kit runner reproduces the maker's own esm code for ESM3 open, component by component, on fixed inputs."
change = "None to the port after the measurement; three fixes came out of it (exact float32 contractions in geometric attention, the codebook and the structure head), each remeasured."
measurement = "esm 3.4.2 (commit 8f0834c) in an isolated venv, ESM3_sm_open_v0 cast to bfloat16 on MPS with bf16 autocast in logits(), captured once; the port driven over the wire with ARSENAL_ESM3_CAPTURE_DIR. Bounds frozen in lab/evidence/esm3-sm-open-v1/bounds.json before any comparison. m5-max, 2026-11-02."
verdict = "kept"
numbers = { components_passed = 51, components_missed = 6, worst_block_rel_l2 = 0.0115, logits_worst_rel_l2 = 0.0058, decoder_backbone_max_abs_a = 0.269, encoder_token_agreement = 1.0, decoder_plddt_max_abs = 0.0023, decoder_ptm_abs = 0.0012, function_annotations = "identical", largest_activation = 11136 }
links = ["../gate/golden/golden.json", "001-port.md"]
+++
# 002 Parity with the maker's code

Reference: the maker's own package at the pinned commit, run once on the GPU at half precision (bf16 weights, bf16
autocast, which is what `ESM3.from_pretrained` does on a GPU). Fixtures: a 60-residue GFP prefix with six masked
residues, and the first 64 residues of chain A of the repository's own `data/1utn.pdb` with secondary structure, SASA
values and three function annotations (a keyword, an InterPro entry and a residue-annotation label). Raw tensors stay
in `lab/evidence/esm3-sm-open-v1`.

## Per component

Generated from `lab/evidence/esm3-sm-open-v1/parity.json` by `tables.py`; every row is a component of
`gate/golden/golden.json`, in capture order. A miss is a value outside the bound frozen before the comparison.

| Component | Verdict | Port vs maker (bf16 GPU reference) | Frozen bound |
|---|---|---|---|
| `A.tokens.sequence` | pass | 0 mismatches over [62] | identical |
| `B.tokens.sequence` | pass | 0 mismatches over [66] | identical |
| `B.tokens.structure` | pass | 0 mismatches over [66] | identical |
| `B.tokens.secondary_structure` | pass | 0 mismatches over [66] | identical |
| `B.tokens.sasa` | pass | 0 mismatches over [66] | identical |
| `B.tokens.function` | pass | 0 mismatches over [66, 8] | identical |
| `A.embed_in` | pass | relative L2 0.002572, cosine 0.9999968 | 0.01, cosine 0.9999 |
| `A.block.0` | pass | relative L2 0.001369, cosine 0.9999991 | 0.02, cosine 0.999 |
| `A.block.1` | pass | relative L2 0.002163, cosine 0.9999977 | 0.02, cosine 0.999 |
| `A.block.23` | pass | relative L2 0.011471, cosine 0.9999589 | 0.03, cosine 0.999 |
| `A.block.47` | pass | relative L2 0.005739, cosine 0.9999835 | 0.05, cosine 0.999 |
| `A.embeddings` | pass | relative L2 0.005739, cosine 0.9999835 | 0.05, cosine 0.998 |
| `A.logits.sequence` | pass | relative L2 0.002244, cosine 0.9999975 | 0.05, cosine 0.998 |
| `A.argmax.sequence` | pass | agreement 1.0000 | 0.95 |
| `A.logits.structure` | pass | relative L2 0.002084, cosine 0.9999979 | 0.05, cosine 0.998 |
| `A.argmax.structure` | miss | agreement 0.6167 | 0.8 |
| `A.logits.secondary_structure` | pass | relative L2 0.002407, cosine 0.9999971 | 0.05, cosine 0.998 |
| `A.argmax.secondary_structure` | pass | agreement 1.0000 | 0.95 |
| `A.logits.sasa` | pass | relative L2 0.002330, cosine 0.9999974 | 0.05, cosine 0.998 |
| `A.argmax.sasa` | pass | agreement 0.9833 | 0.9 |
| `A.logits.function` | pass | relative L2 0.002305, cosine 0.9999983 | 0.05, cosine 0.998 |
| `A.argmax.function` | pass | agreement 0.9958 | 0.9 |
| `A.logits.residue_annotations` | pass | relative L2 0.003049, cosine 0.9999957 | 0.05, cosine 0.998 |
| `A.argmax.residue_annotations` | pass | agreement 0.9833 | 0.9 |
| `B.embed_in` | pass | relative L2 0.003389, cosine 0.9999943 | 0.01, cosine 0.9999 |
| `B.block.0` | pass | relative L2 0.003219, cosine 0.9999948 | 0.02, cosine 0.999 |
| `B.block.1` | pass | relative L2 0.003743, cosine 0.9999931 | 0.02, cosine 0.999 |
| `B.block.23` | pass | relative L2 0.010939, cosine 0.9999410 | 0.03, cosine 0.999 |
| `B.block.47` | pass | relative L2 0.009438, cosine 0.9999569 | 0.05, cosine 0.999 |
| `B.embeddings` | pass | relative L2 0.009438, cosine 0.9999569 | 0.05, cosine 0.998 |
| `B.logits.sequence` | pass | relative L2 0.002850, cosine 0.9999962 | 0.05, cosine 0.998 |
| `B.argmax.sequence` | pass | agreement 1.0000 | 0.95 |
| `B.logits.structure` | pass | relative L2 0.003719, cosine 0.9999931 | 0.05, cosine 0.998 |
| `B.argmax.structure` | pass | agreement 0.9531 | 0.8 |
| `B.logits.secondary_structure` | pass | relative L2 0.004367, cosine 0.9999905 | 0.05, cosine 0.998 |
| `B.argmax.secondary_structure` | pass | agreement 1.0000 | 0.95 |
| `B.logits.sasa` | pass | relative L2 0.004903, cosine 0.9999886 | 0.05, cosine 0.998 |
| `B.argmax.sasa` | pass | agreement 1.0000 | 0.9 |
| `B.logits.function` | pass | relative L2 0.005804, cosine 0.9999879 | 0.05, cosine 0.998 |
| `B.argmax.function` | pass | agreement 0.9941 | 0.9 |
| `B.logits.residue_annotations` | pass | relative L2 0.002971, cosine 0.9999958 | 0.05, cosine 0.998 |
| `B.argmax.residue_annotations` | pass | agreement 1.0000 | 0.9 |
| `C.encoder.latent` | pass | relative L2 0.000395, cosine 0.9999999 | 0.02 |
| `C.encoder.tokens` | pass | agreement 1.0000 (0 mismatches) | agreement 0.95 |
| `D.backbone` | miss | max 0.269 A, RMSD 0.160 A | max 0.25 A |
| `D.plddt` | pass | max 0.0023 | max 0.01 |
| `D.ptm` | pass | 0.7569 vs 0.7557 | 0.01 |
| `D.pae` | pass | max 0.117 A | max 0.25 A |
| `E.interpro_logits` | pass | relative L2 0.001994, cosine 0.9999981 | 0.03 |
| `E.keyword_logits` | pass | relative L2 0.002009, cosine 0.9999980 | 0.03 |
| `F1.sequence` | pass | identical | identical |
| `F2.structure_tokens` | miss | agreement 0.5667 | agreement 0.9 |
| `F2.ca_rmsd` | miss | CA RMSD 2.491 A | 1.0 A |
| `F2.plddt` | miss | mean 0.0288 | mean 0.02 |
| `F2.ptm` | miss | 0.4018 vs 0.3593 | 0.02 |
| `G.function_tokens` | pass | rows identical 1.0000 | rows 0.95 |
| `G.annotations` | pass | 21 vs 21 annotations, identical | identical |

## The six misses

Six checks missed their frozen bound. Five come from the 4,096-way structure head: `A.argmax.structure`, and the four
fold checks (`F2.structure_tokens`, `F2.ca_rmsd`, `F2.plddt`, `F2.ptm`), which follow from the structure tokens the
first fold step picks. In the bf16 reference the median gap between its top two structure logits is 0.0, so its argmax
is decided by rounding; in float32 that gap is 0.078. The sixth, `D.backbone`, is the structure decoder on identical
tokens: its largest coordinate difference is 0.269 A against a frozen bound of 0.25 A (RMSD 0.160 A), a miss; pLDDT,
pTM and PAE from the same decoder pass. A float32 probe of the same code explains the structure-head misses (`lab/evidence/esm3-sm-open-v1/probe-logits.json`; MPS refuses this model in
float32, so the probe ran on the CPU, on the 60-residue input only). The CPU float32 run is a diagnostic only, used to
explain the six misses: it is not the reference, qualifies nothing and changes no bound. The reference of record stays
the maker's bf16 run on the GPU, and the six misses stay recorded as misses against it:

| Track | port FP16 vs maker FP32 | maker BF16 vs maker FP32 |
|---|---|---|
| sequence argmax | 1.000 | 1.000 |
| structure argmax | 0.933 | 0.567 |
| SASA argmax | 1.000 | 0.983 |
| fold step's structure tokens | 0.967 | 0.533 |

In this diagnostic the port is closer to the maker's float32 answer than the maker's own half-precision run is. The
fold checks follow from those flipped tokens: the folded backbone is 2.49 A CA-RMSD from the bf16 reference's backbone,
and pLDDT and pTM are 0.029 and 0.043 apart, while on identical tokens they are 0.0023 and 0.0012 apart. For
`D.backbone`, a float32 run of the maker's decoder on MPS (`probe-decoder.txt`, also a diagnostic only) puts its own
bf16 output 0.299 A from float32 and the port 0.041 A from float32; the miss against the bf16 reference stands.

What this does not establish: parity for proteins longer than 64 residues, for multi-step sampling with a temperature
above 0 (the two runtimes draw from their own generators), or for prompts with residue annotations in the input.
Nothing here reproduces the maker's benchmark figures.

## Precision

FP16 is selected. The largest activation anywhere in the capture is 11,136 (block 47 of the 60-residue case), so
nothing approaches FP16's 65,504, and no component needed BF16. The float32 islands are listed in `001-port.md` and in
the golden. The three fixes the measurement produced are also recorded there; each left the port strictly closer to
the reference.
