# Mitra-v2 Classifier

Labels the rows of a numeric table from rows you have already labelled, in one call and with no training. AutoGluon's Mitra-v2 (75.7M parameters, a 2D transformer attending across rows and columns) reads the labelled rows as context and returns a label and class probabilities for each new row.

Input: `train_rows` (labelled rows, each a list of numbers, 2 to 2048 rows), `train_labels` (one text label per labelled row, 2 to 10 different labels) and `rows` (the rows to label, same columns, at most 100 columns).

Example request:

```json
{"train_rows": [[1, 0.5], [2, 0.1], [9, 0.9], [8, 0.8]], "train_labels": ["low", "low", "high", "high"], "rows": [[8.5, 0.7], [1.5, 0.2]]}
```

Output: `labels` (the most likely label of each row), `classes` (every label, sorted) and `probabilities` (one list per row, in the order of `classes`).

On the card's wine example (sklearn's 178 wines, 13 chemical measurements, an 80/20 split) it labels all 36 held-out wines with the right cultivar from the 142 labelled ones.

Limits: zero-shot only (the maker's reported accuracy uses fine-tuning and bagging); numeric columns only, no missing values; at most 10 classes, 2048 labelled rows and 100 columns; one deterministic estimator.

Parity: matches the maker's AutoGluon Mitra code on Apple GPU in fp16: the same labels on all 36 wine rows (probability cosine 0.9999999, largest gap 0.0016), and 699 of 700 labels on a 2048-row synthetic table, where the one difference is an exact tie in the reference and the maker's fp32 run agrees with the runner. Details in `log/003-parity.md`.

Source: [autogluon/mitra-classifier-2](https://huggingface.co/autogluon/mitra-classifier-2/tree/edada0d20759c58ada8c8605c25f22f6e98ea5f0), Apache-2.0. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: `fit` takes labelled rows and their labels; `predict_proba` takes new rows and returns class probabilities. Each predict call runs one bagged fine-tune, not a single forward pass. [Mitra-v2 model card, 2026-09 inferred, first-hand]
- Prompts, reference audio or voice settings: Not applicable, tabular only.
- Parameters people recommend: The reported recipe is 50 fine-tune steps with 8-fold bagging, and it needs a CUDA GPU. AutoGluon's stock defaults differ from that recipe. [Mitra-v2 model card, 2026-09 inferred, first-hand]
- Arsenal call: `POST /arsenal/v1/models/mitra-classifier-2/ops/classify_rows`, or the same operation through MCP. Vendor example [Mitra-v2 model card, 2026-09 inferred, first-hand]:
```python
from mitra_finetune import MitraFinetune
from huggingface_hub import snapshot_download
ckpt_dir = snapshot_download("autogluon/mitra-classifier-2")
model = MitraFinetune(checkpoint_dir=ckpt_dir, problem_type="classification")
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)
```

**Good and bad**
- Good at: The maker reports state-of-the-art results on TabArena and TALENT, level with TabFM and EXAONE Tabular. These are self-reported; no third-party benchmark was found. [Mitra-v2 model card, 2026-09 inferred, first-hand]
- Bad at: The maker's recipe requires a CUDA GPU. AutoGluon's docs table gives its Mitra row limits of 10 classes, 10,000 rows and 500 features; that row is dated 2025-07, while v2 is a 2026 release. [Mitra-v2 model card, 2026-09 inferred; AutoGluon tabular foundation model docs, undated, first-hand]
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: Not found in sources checked. The repo's HF discussions list is empty.

**Evidence**
- Strength: thin. Only the maker's card and AutoGluon's docs; no independent test or user report was found.
- Sources:
  - Mitra-v2 model card, huggingface.co/autogluon/mitra-classifier-2, 2026 (arXiv 2609.04540; 2026-09 inferred), first-hand.
  - AutoGluon tabular foundation model docs, auto.gluon.ai/stable/tutorials/tabular/tabular-foundational-models.html, undated, first-hand.
