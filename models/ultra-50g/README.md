# ULTRA 50g

Fills in missing links in a knowledge graph you bring. ULTRA is a graph foundation model that learns nothing per entity or relation, so one checkpoint works on any graph with any names; this one was pre-trained on 50 knowledge graphs. You send the known facts as (head, relation, tail) triples and a question such as (Italy, located_in, ?); it ranks every entity in the graph as the missing end.

Input: `triples`, the graph as `[head, relation, tail]` lists of names (strings or numbers; every fact is also read backwards), at most 1,000,000 triples with at most 1,000,000 distinct entities and 4,096 distinct relations; `relation`, the query relation; `head` to find missing tails, or `tail` to find missing heads; `top_k` (default 10); `exclude_known` (default true: leave out answers already in the triples). Output: `predictions`, best first, each with `entity`, `score` (the model's logit), `probability` (its sigmoid) and `known`; `predicting` (tail or head); and the counts of entities, relations and triples read.

Example request (a small geography graph; ULTRA ranks Europe first, probability 0.96, then Asia at 0.05):

```json
{"triples": [["Paris", "capital_of", "France"], ["Rome", "capital_of", "Italy"], ["France", "located_in", "Europe"],
             ["Paris", "located_in", "Europe"], ["Rome", "located_in", "Europe"], ["Tokyo", "capital_of", "Japan"],
             ["Japan", "located_in", "Asia"], ["Tokyo", "located_in", "Asia"], ["Italy", "borders", "France"]],
 "relation": "located_in", "head": "Italy", "top_k": 3}
```

Limits: the model reads only the graph's structure, never the names, so sparse entities and tiny graphs give rough answers. The query entity and relation must appear in the triples. The whole graph travels with each request. Probabilities rank candidates for one query and are not calibrated across graphs.

Parity: against the maker's ultra package on MPS (its fused rspmm path, the one its training used, with the C++ kernel's sum written in torch; float32, because the maker's code builds float32 tensors that cannot meet float16 weights), the same top answer on all five queries, score cosine at least 0.999995 and max difference 0.012, on a 16-entity geography graph and a random 300-entity graph (log/003-parity.md).

Source: [mgalkin/ultra_50g](https://huggingface.co/mgalkin/ultra_50g/tree/d46107d8415549e7c7386d26903ad11c9a8de4fd), MIT. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: A query (head, relation, ?) ranks every node as a possible tail; your own graph loads as TSV or CSV. [ULTRA 50g model card, undated, first-hand; ULTRA GitHub README, undated, first-hand]
- Prompts, reference audio or voice settings: Not applicable. Fine-tuning one graph adds about 10% relative MRR and Hits@10, per the maker. [ULTRA 50g model card, undated, first-hand]
- Parameters people recommend: Not found in sources checked.
- Arsenal call: `POST /arsenal/v1/models/ultra-50g/ops/predict_links`, or the same operation through MCP. Vendor example [ULTRA 50g model card, undated, first-hand]:
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("mgalkin/ultra_50g", trust_remote_code=True)
```

**Good and bad**
- Good at: Averages 0.389 MRR and 0.549 Hits@10 over 16 transductive graphs, against 0.329 and 0.479 for ultra_3g, per the README. [ULTRA GitHub README, 2023-12 news entry, first-hand]
- Bad at: The maker says its zero-shot test does not apply to ultra_50g, pre-trained on 50 graphs. [ULTRA 50g model card, undated, first-hand]
- Against the obvious alternatives: Beats the maker's per-graph supervised models (0.371 MRR, 0.511 Hits@10). [ULTRA 50g model card, undated, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: Issue #31 (Nov 2024): a user with eight A10G GPUs got no GPU run through the Hugging Face setup. Fix or workaround: the maintainer advised gpus=[0] first; the reporter saw no process run, and the issue closed with no confirmed fix. [ULTRA GitHub issue #31, 2024-11, first-hand]

**Evidence**
- Strength: thin. Maker card and README, one closed issue; no independent evaluation found.
- Sources:
  - ULTRA GitHub README, github.com/DeepGraphLearning/ULTRA, 2023-12 news entries, first-hand.
  - ULTRA 50g model card, huggingface.co/mgalkin/ultra_50g, undated, first-hand.
  - ULTRA issue #31, github.com/DeepGraphLearning/ULTRA/issues/31, opened 21 Nov 2024, first-hand.
