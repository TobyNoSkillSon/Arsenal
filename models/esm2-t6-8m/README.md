# ESM-2 8M

Turns a protein sequence into numbers: Meta AI's ESM-2 protein language model, 8M size (6 layers, 320-wide embeddings). Similar proteins get similar vectors, so they serve for similarity search, clustering and as features for your own predictors.

Input: `sequence`, one-letter amino-acid codes in uppercase (whitespace and line breaks are ignored), at most 1,022 residues; `per_residue` (optional, default false). Output: `embedding`, the final-layer embedding averaged over residues (320 numbers), `dimensions`, `residues`, and with `per_residue` true, one 320-number embedding per residue.

Example request (the first 60 residues of human preproinsulin):

```json
{"sequence": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAED"}
```

Limits: no more than 1,022 residues per request; lowercase letters, digits and other symbols give invalid_request. Proteins only, not DNA or RNA. The embedding predicts nothing by itself; structure prediction and masked-residue filling are not served. It is the weakest ESM-2 size: the card says larger checkpoints are more accurate. Use ESM-2 650M when quality matters more than speed.

Parity: against transformers EsmModel on MPS at float16, the sequence embedding's cosine was 1.000000 on human insulin 1-60 and GFP 1-60 (0.999998 on a single residue), with the final hidden state within 0.008 (log/003-parity.md).

Source: [facebook/esm2_t6_8M_UR50D](https://huggingface.co/facebook/esm2_t6_8M_UR50D/tree/c731040fcd8d73dceaa04b0a8e6329b345b0f5df), MIT. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: The card says the model is suitable for fine-tuning on tasks that take protein sequences as input; no usage limits are stated. [HF card facebook/esm2_t6_8M_UR50D, undated (read 2026-10-31), first-hand]
- Prompts, reference audio or voice settings, if relevant: Not applicable.
- Parameters people recommend: Not found in sources checked.
- Arsenal call: `POST /arsenal/v1/models/esm2-t6-8m/ops/embed_sequence`, or the same operation through MCP. Vendor example (the card's masked-residue loader, which Arsenal does not serve): `AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")` and `AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D", device_map="auto")`.

**Good and bad**
- Good at: Not found in sources checked.
- Bad at: Not found in sources checked.
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: Issue #643 (opened 2023-12-18, still open) reports that requesting layer 33 from esm2_t6_8M returns an empty representation, with a KeyError. Fix or workaround: Reported by one user, not confirmed: a reply (2023-12-29) says the README's "33" should read "6", because this model has only 6 layers; the thread does not say the change fixed the error. [facebookresearch/esm issue #643, 2023-12-18, opinion]
- Complaint or known bug: The upstream repo was archived by its owner on 1 Aug 2024 and is read-only. Fix or workaround: Not found in sources checked. [facebookresearch/esm repository, archived 1 Aug 2024, first-hand]

**Evidence**
- Strength: none. The card documents the model, but no outside accuracy or usage test was found.
- Sources:
  - facebook/esm2_t6_8M_UR50D model card, https://huggingface.co/facebook/esm2_t6_8M_UR50D, undated (read 2026-10-31), first-hand.
  - facebookresearch/esm issue #643, https://github.com/facebookresearch/esm/issues/643, 2023-12-18, opinion.
  - facebookresearch/esm repository, https://github.com/facebookresearch/esm, archived 1 Aug 2024, first-hand.
