+++
hypothesis = "The MLX kit runner matches the maker's ultra package (Ultra, fused rspmm message passing) on ultra_50g."
change = "Port fixed twice during this check: message direction (rspmm gathers from edge_index[1] into edge_index[0]) and the relation-graph query for head questions (Ultra.forward uses the relation before inversion)."
measurement = "lab/night2/ultra-50g/ref_parity.py vs runjson.py replies on m5-max, 2026-10-31: 5 queries, scores of every entity."
verdict = "kept"
numbers = { top1_match = "5/5", score_cosine_min = 0.999995, score_max_abs_diff = 0.0120 }
links = []
+++
# 003 Parity with the maker's ultra package (2026-10-31)

Method: the repository's own ultra/models.py, layers.py, base_nbfnet.py and tasks.py (copied as text from the pinned
revision), weights from model.safetensors, PyTorch 2.15.0 on MPS, torch_geometric 2.4.0 (the API the maker's code
calls), against the kit runner (MLX, FP16 weights) on the same checkpoint, M5 Max. The graph is built as the maker's
datasets build it (inverse edges with relation + R; tasks.build_relation_graph), and the query goes through
Ultra.forward with every entity as a candidate (the maker's tasks.all_negative layout).

Two stand-ins, because the maker's compiled pieces do not build on macOS: torch_scatter's scatter/scatter_add in plain
torch (refpkg/torch_scatter), and ultra/rspmm's generalized_rspmm written in torch from rspmm.cpp's
rspmm_forward_out_cpu (refpkg/ultra/rspmm/rspmm.py: output[edge_index[0]] += weight * relation[type] * input[edge_index[1]]).
The maker's GeneralizedRelationalConv.propagate itself is unchanged and takes its fused path, as in training.
PyG's generic message/aggregate path (the maker's gradient-visualisation fallback) sums in the opposite direction:
the port first followed it and ranked capitals for (Italy, located_in, ?); against the fused path it ranks Europe.

Reference precision: float32. The maker's bellmanford builds its query and boundary tensors as float32 regardless of
the model's dtype, so with float16 weights MPS aborts on a mixed-dtype matmul. The MLX port keeps weights, node
states and messages in float16 and runs the neighbour sums and each layer's update before LayerNorm in float32.

Inputs (lab/night2/ultra-50g/requests.json): geo.json (16 entities, 3 relations, 25 triples) and random.json
(300 entities, 8 relations, 2,000 random triples, seed 7); every entity scored, exclude_known off.

| query | entities | top answer (both) | score cosine | max abs diff | rank swaps beyond 0.01 |
|---|---|---|---|---|---|
| (Italy, located_in, ?) | 16 | Europe | 1.000000 | 0.0116 | 0 |
| (South Korea, located_in, ?) | 16 | Asia | 1.000000 | 0.0095 | 0 |
| (?, capital_of, Italy) | 16 | Berlin | 0.999999 | 0.0120 | 0 |
| (e5, r3, ?) random | 300 | e19 | 0.999997 | 0.0060 | 0 |
| (?, r1, e17) random | 300 | e147 | 0.999995 | 0.0069 | 0 |

Scores span about -11 to 4.5. Order differences occur only between candidates the reference separates by less than
0.01 (float16 ties). Note (?, capital_of, Italy) puts the known answer Rome below other capitals on both sides:
the model was trained with the queried edge removed, so it scores alternatives by structure.

Verdict: pass (identical top answer on every query, cosine >= 0.999 on the score vectors).
