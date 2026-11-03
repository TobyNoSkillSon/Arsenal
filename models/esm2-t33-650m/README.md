# ESM-2 650M

Turns a protein sequence into numbers: Meta AI's ESM-2 protein language model, 650M size (33 layers, 1280-wide embeddings). Similar proteins get similar vectors, so they serve for similarity search, clustering and as features for your own predictors.

Input: `sequence`, one-letter amino-acid codes in uppercase (whitespace and line breaks are ignored), at most 1,022 residues; `per_residue` (optional, default false). Output: `embedding`, the final-layer embedding averaged over residues (1280 numbers), `dimensions`, `residues`, and with `per_residue` true, one 1280-number embedding per residue.

Example request (the first 60 residues of human preproinsulin):

```json
{"sequence": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAED"}
```

Limits: no more than 1,022 residues per request; lowercase letters, digits and other symbols give invalid_request. Proteins only, not DNA or RNA. The embedding predicts nothing by itself; structure prediction and masked-residue filling are not served. Slower and heavier than ESM-2 8M; the 3B and 15B sizes are larger still but are not part of Arsenal.

Parity: against transformers EsmModel on MPS at float16, the sequence embedding's cosine was 0.999999 or higher on human insulin 1-60 and GFP 1-60 (0.99998 on a single residue), with the final hidden state within 0.031 (log/003-parity.md).

Source: [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D/tree/08e4846e537177426273712802403f7ba8261b6c), MIT. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: The card says the model is suitable for fine-tuning on tasks that take protein sequences as input; no input limit is stated. [HF card facebook/esm2_t33_650M_UR50D, undated (read 2026-10-31), first-hand]
- Prompts, reference audio or voice settings, if relevant: Not applicable.
- Parameters people recommend: Not found in sources checked.
- Arsenal call: `POST /arsenal/v1/models/esm2-t33-650m/ops/embed_sequence`, or the same operation through MCP. Vendor example (the card's masked-residue loader, which Arsenal does not serve): `AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")` and `AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", device_map="auto")`.

**Good and bad**
- Good at: The FANTASIA v3.0.0 changelog says its switch to ESM2-650M is "improving embedding accuracy"; no measurement is given. [FANTASIA v3.0.0 release, undated (28 Jul, year not shown), opinion]
- Bad at: Not found in sources checked.
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: The card's own fill-mask widget shows "No mask token found for this model." Fix or workaround: Not found in sources checked. [HF card facebook/esm2_t33_650M_UR50D, undated (read 2026-10-31), first-hand]
- Complaint or known bug: The upstream repo was archived by its owner on 1 Aug 2024 and is read-only. Fix or workaround: Not found in sources checked. [facebookresearch/esm repository, archived 1 Aug 2024, first-hand]

**Evidence**
- Strength: thin. The maker's card and one project's changelog are the only sources; no outside accuracy test was found.
- Sources:
  - facebook/esm2_t33_650M_UR50D model card, https://huggingface.co/facebook/esm2_t33_650M_UR50D, undated (read 2026-10-31), first-hand.
  - FANTASIA v3.0.0 release, https://github.com/CBBIO/FANTASIA/releases/tag/v3.0.0, undated (28 Jul, year not shown), opinion.
  - facebookresearch/esm repository, https://github.com/facebookresearch/esm, archived 1 Aug 2024, first-hand.
