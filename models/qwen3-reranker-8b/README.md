# Qwen3 Reranker 8B

Qwen3-Reranker-8B is the largest text reranker in Qwen's Qwen3 Embedding series: an 8.19B-parameter, 36-layer Qwen3 language model fine-tuned to read a query and one candidate passage together and answer "yes" or "no" to whether the passage meets the query. The probability of "yes" is the relevance score. Because it reads each pair together instead of comparing two separately computed vectors, it needs one model pass per candidate.

## What it is for

Qwen built it to reorder the candidates a fast embedding search returns, across more than 100 languages and code, and to follow a task instruction; they report that a suitable instruction usually adds 1% to 5%. Their scores for this model: MTEB-R 69.02, CMTEB-R 77.45, MMTEB-R 72.94, MLDR 70.19, MTEB-Code 81.22, FollowIR 8.05, each reranking the top 100 results of Qwen3-Embedding-0.6B.

- Refine semantic search: take the top 20 to 100 hits from an embedding model and put the truly relevant ones first.
- Pick the passages to feed a language model for question answering over documents.
- Rank code snippets or documentation against a programming question.
- Filter candidates by a custom criterion written as an instruction.

## Inputs and outputs

### `rerank` (default)

Score every document against the query and return them best first.

| Field | Type | Required | Notes |
|---|---|---|---|
| `query` | string | yes | 1 to 32768 characters. |
| `documents` | array of strings | yes | 1 to 1024 items, each up to 1000000 characters. |
| `top_n` | integer | no | 1 to 1024. Arsenal's default returns every document. |
| `max_tokens_per_doc` | integer | no | 1 to 8192. Arsenal's default is 8192. Tokens of each document kept, counted from its start. |
| `instruction` | string | no | Up to 32768 characters. Arsenal's default is `Given a web search query, retrieve relevant passages that answer the query`. |

Returns `results`: up to `top_n` objects `{"index": <position in documents, from 0>, "relevance_score": <0 to 1>}`, sorted by score from high to low, ties kept in input order.

Each document is scored on its own, in the maker's template: a system message telling the model to judge whether the Document meets the requirements of the Query and the Instruct with only "yes" or "no", then `<Instruct>: <instruction>\n<Query>: <query>\n<Document>: <document>`, then an empty thinking block in the assistant turn. The score is the softmax of the final `yes` and `no` logits, taken for `yes`. The whole prompt is at most 8,192 tokens: a long document is cut from its end to fit, and a query plus instruction that leave no room for the document are refused. Write instructions in English even for other languages, as Qwen recommends; most of its training instructions were English.

Where it fits in search by meaning: an embedding model (for example `harrier-oss-v1-0.6b`) finds the nearest few dozen passages by cosine similarity; this model then reads each of them with the query and its `relevance_score` decides the final order.

## Run it

MCP: `arsenal_run(model:"qwen3-reranker-8b", input:{"query":"how do I reset my router","documents":["Unplug the router for thirty seconds.","Our office opens at nine."],"top_n":1})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/qwen3-reranker-8b/ops/rerank" -H 'Content-Type: application/json' -d '{"query":"how do I reset my router","documents":["Unplug the router for thirty seconds.","Our office opens at nine."]}'`

Precision (FP16, int8, int4) is chosen in Arsenal's model table. A warm request with a query and four short documents took 0.14 to 0.15 s on an M5 Max in use at every precision (quick checks, measured). Documents run one pass each, so time grows with their number and length; the request times out after 180 s.

## Limits

- Text only.
- 8,192 tokens per query-document pair. The model card lists a 32k context, but its own usage code caps each pair at 8,192, and Arsenal follows the code.
- Not an embedding model: it produces no vectors and cannot search a collection by itself; rerank a shortlist.
- Large: 16.4 GB of weights at FP16, 8.7 GB at int8 and 4.6 GB at int4.
- Int4 ran without errors but its ranking quality has not been judged.

## Licence and credit

Apache-2.0. Commercial use is allowed; keep the licence and notices, and state changes if you redistribute modified weights. Made by the Qwen team. Cite: Zhang, Y. et al., "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models", arXiv:2506.05176, 2025.

## Links

- Model card: https://huggingface.co/Qwen/Qwen3-Reranker-8B/tree/77d193c791ed757ca307ee72715aa132723da912
- Code: https://github.com/QwenLM/Qwen3-Embedding
- Paper: https://arxiv.org/abs/2506.05176
- Blog: https://qwenlm.github.io/blog/qwen3-embedding/

## How Arsenal runs it

Source: `Qwen/Qwen3-Reranker-8B@77d193c791ed757ca307ee72715aa132723da912`. Immutable filenames, byte sizes and SHA-256 hashes are in `model.toml` and `gate/source.json`. All source tensors are BF16. The serving default is FP16: Get streams each BF16 tensor into the FP16 base snapshot (`cast:embed-qwen3-v2`) before execution. Int8 and int4 derive every eligible 2-D Linear/Embedding matrix with MLX affine group size 64; learned norms stay half.

Runner: the native Swift MLX embed worker, sequential requests only, macOS 26 or later, ported from `ml-explore/mlx-swift-lm@bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57` and adapted inside `Workers/Embed` (see its `PORTED-FROM.md` and `LICENSE-upstream`). Pairwise reranking uses Qwen's documented instruction/query/document template with no thinking and a two-class softmax of the final `no`/`yes` logits. As in the card's `process_inputs`, each prompt (template, pair and answer suffix) is at most `max_length` 8,192 tokens, the formatted pair is tokenized as one string and cut from the document's end, and an overlong query or instruction fails. Original indices are kept stable.

Checked live on 20 Oct 2026 at FP16 and int8: the relevant document ranked first. Int4 ran without errors; its ranking was not judged. Quick checks, 22 Oct 2026, M5 Max, Mac in use, a query and four short documents per request (`results/`): FP16 load 3.89 s, warm 0.15 s, peak 18.3 GB; int8 load 2.47 s, warm 0.15 s, peak 9.69 GB; int4 load 1.96 s, warm 0.14 s, peak 5.27 GB, peaks summed over Arsenal's worker processes.

Gate (`gate/gate.toml`): exact checkpoint inventory and every declared operation on synthetic weights pass; trained-weight reference and precision qualification are pending. Log: `log/001-port.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: The card's Transformers example needs transformers 4.51.0 or later; older versions may raise `KeyError: 'qwen3'`. [model card, paper Jun 5 2025, first-hand (maker)]
- Prompts, reference audio or voice settings, if relevant: The card recommends customizing the `instruct` for each scenario, task and language. [model card, paper Jun 5 2025, first-hand (maker)]
- Parameters people recommend: Not found in sources checked.
- Arsenal call: `arsenal_run(model:"qwen3-reranker-8b", input:{"query":"how do I reset my router","documents":["Unplug the router for thirty seconds.","Our office opens at nine."],"top_n":1})` [this README, 2026-10-31, first-hand]

**Good and bad**
- Good at: Maker-reported MTEB-R 69.02 and FollowIR 8.05 for this 8B; no independent test found. [model card, first-hand (maker)]
- Bad at: The card says Qwen3-Reranker-4B scores higher on MTEB-R (69.76) and FollowIR (14.84). [model card, first-hand (maker)]
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: A Q4_K_S GGUF of this model returned near-zero scores and a wrong top result in llama.cpp (not Arsenal's MLX path); the reporter used a q4_0 K/V cache. Fix or workaround: No 8B fix confirmed. The maintainer said the file was likely badly converted and uploaded a corrected Qwen3-Reranker-0.6B Q8_0 GGUF, not the 8B; he later advised against q4_0 K/V cache. The reporter closed the issue on 2025-10-09 with no closing comment. Later comments report mismatched rankings for other models, not this 8B. [ggml-org/llama.cpp #16407, 2025-10-03, first-hand]

**Evidence**
- Strength: thin. One maker benchmark table and no independent test of the 8B.
- Sources:
  - Qwen3-Reranker-8B model card, huggingface.co/Qwen/Qwen3-Reranker-8B, paper Jun 5 2025, first-hand (maker).
  - llama.cpp issue #16407, github.com/ggml-org/llama.cpp/issues/16407, 2025-10-03, first-hand.
