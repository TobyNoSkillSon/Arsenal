# MoLFormer-XL Both 10%

Turns a molecule into numbers: IBM Research's MoLFormer-XL chemical language model, the checkpoint pretrained on 10% of ZINC15 and 10% of PubChem (12 layers, 768-wide embeddings). Related molecules get similar vectors, so they serve for similarity search, clustering and as features for your own property predictors.

Input: `smiles`, one molecule as a SMILES string, at most 200 tokens. Output: `embedding`, the final-layer token embeddings averaged (768 numbers, the card's `pooler_output`), `dimensions`, `tokens` (SMILES tokens read), and `unknown_tokens` when some tokens are missing from the vocabulary.

Example request (caffeine, the card's first example):

```json
{"smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O"}
```

Limits: SMILES only; a string with characters outside SMILES gives invalid_request, and so does a molecule over 200 tokens. Training used RDKit-canonical SMILES without stereo marks, so non-canonical or invalid SMILES give weaker embeddings. The embedding predicts nothing by itself; masked-token filling and molecule generation are not served.

Parity: against the maker's modeling_molformer.py on MPS (float32, because its float16 path overflows to NaN in the linear-attention normaliser), the embedding's cosine was 1.000000 on caffeine, aspirin and theobromine, 0.999999 on ibuprofen and 0.999986 on methane, max difference 0.019 (log/003-parity.md).

Source: [ibm-research/MoLFormer-XL-both-10pct](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct/tree/361063d0ad524ef77cf39b08469f6be770dc550f), Apache-2.0. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: The card's example calls `tokenizer(smiles, padding=True, return_tensors="pt")`. [HF card ibm-research/MoLFormer-XL-both-10pct, undated (read 2026-10-31), first-hand]
- Prompts, reference audio or voice settings, if relevant: Not applicable.
- Parameters people recommend: Not a stated recommendation. The card's example code uses `deterministic_eval=True` and runs the forward pass under `torch.no_grad()`. [HF card, undated (read 2026-10-31), first-hand]
- Arsenal call: `POST /arsenal/v1/models/molformer-xl-both-10pct/ops/embed_molecule`, or the same operation through MCP. Vendor example: `AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)`, `tokenizer(smiles, padding=True, return_tensors="pt")`, reading `outputs.pooler_output`. The card uses `ibm/`, not the README's `ibm-research/`.

**Good and bad**
- Good at: The card says the model is mainly intended as a feature extractor or to be fine-tuned for a prediction task. [HF card, undated (read 2026-10-31), first-hand]
- Bad at: The card says it is not tested for molecules larger than about 200 atoms (macromolecules). [HF card, undated (read 2026-10-31), first-hand]
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: Discussion #2 (opened 2024-04-27) reports an AttributeError on padded SMILES with `trust_remote_code=True`. Fix or workaround: Reported by one user, not confirmed: a reply (2024-08-13) suggested upgrading transformers; no one confirmed it. The card says v4 users should pass `revision="compat-v4"`. [HF discussion #2, 2024-04-27, opinion]
- Complaint or known bug: Not found in sources checked. Fix or workaround: Not found in sources checked.

**Evidence**
- Strength: thin. The maker's card is the only first-hand source; the forum threads show errors and questions, with no test results.
- Sources:
  - ibm-research/MoLFormer-XL-both-10pct model card, https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct, undated (read 2026-10-31), first-hand.
  - Discussion #2, https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct/discussions/2, 2024-04-27, opinion.
  - Discussions index, https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct/discussions, read 2026-10-31, opinion. Discussion #3 ("Replicating QM9 Benchmark Results") is open; its body was not visible.
