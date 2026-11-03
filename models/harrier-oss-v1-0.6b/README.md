# Harrier OSS v1 0.6B

Harrier OSS v1 0.6B is Microsoft's multilingual text embedding model: a 596M-parameter decoder-only Transformer (Qwen3 architecture, 28 layers) that reads up to 32,768 tokens and turns a text into one 1024-number vector by taking the last token's hidden state and L2-normalising it. Texts with similar meaning get vectors that point the same way, so it is the piece that makes "search by meaning" work across languages.

## What it is for

Microsoft trained it with contrastive learning on a large multilingual mix of tasks, with knowledge distillation from larger embedding models, and reports 69.0 on Multilingual MTEB v2, where it says the family was state of the art at release. Queries carry a one-sentence task instruction; documents carry none.

- Semantic search over notes, documents or chat history: find passages that mean the same as the question, in any of dozens of languages.
- Deduplicate or group near-identical texts by similarity.
- Cluster a collection by topic, or classify texts by their nearest labelled examples.
- Bitext mining: match sentences with their translations.

## Inputs and outputs

### `embed` (default)

Turn each text into one vector.

| Field | Type | Required | Notes |
|---|---|---|---|
| `input` | array | yes | 1 to 2048 items. Each item is a string (at least 1 character) or `{"type":"text","text":"..."}` (text at least 1 character). |
| `task` | string | no | `query`, `document`, `similarity`, `classification` or `clustering`. Arsenal's default is `document`. |
| `instruction` | string | no | Up to 32768 characters. Replaces the task's built-in instruction; ignored when `task` is `document`. |
| `dimensions` | integer | no | Minimum 1024, maximum 1024: only the native size. |
| `normalize` | boolean | no | Arsenal's default is `true` (unit-length vectors). |
| `truncate` | boolean | no | Arsenal's default is `false`: an input over 32,768 tokens is refused. `true` keeps the first 32,768 tokens. |

Returns `embeddings`: one array of 1024 numbers per input, in input order, and `input_tokens`: the total tokens read, prompts included.

How the text is wrapped before encoding, by `task`:

- `document`: the text as is. Use this for everything you store and search over.
- `query`: `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: <text>`.
- `similarity`: `Instruct: Retrieve semantically similar text\nQuery: <text>`.
- `classification`: `Instruct: Classify the text\nQuery: <text>`.
- `clustering`: `Instruct: Identify the topic of the text\nQuery: <text>`.

With `instruction`, the line becomes `Instruct: <instruction>\nQuery: <text>`. The `query` and `similarity` wordings are the maker's own (`web_search_query` and `sts_query` in its Sentence Transformers configuration); the `classification` and `clustering` wordings are Arsenal's generic defaults, while the maker evaluated with a task-specific instruction per dataset, so pass your own for those.

To search by meaning: embed every document once with `task:"document"` and store the vectors. At search time embed the question with `task:"query"` (or a one-sentence `instruction` describing the search). Score each document by cosine similarity with the query vector; with the default `normalize:true` that is the plain dot product. Take the top results; to refine their order, send the query and the top candidates' texts to a reranker such as `qwen3-reranker-8b`. Vectors from this model are comparable only with vectors from this model.

## Run it

MCP: `arsenal_run(model:"harrier-oss-v1-0.6b", input:{"input":["how do I reset my router"],"task":"query"})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/harrier-oss-v1-0.6b/ops/embed" -H 'Content-Type: application/json' -d '{"input":["Unplug the router for thirty seconds, then plug it back in."],"task":"document"}'`

There are no settings. A warm request with three short sentences took 0.08 s on an M5 Max in use (quick check, measured). Inputs in one request run one after another; the request times out after 180 s.

## Limits

- Text only.
- 32,768 tokens per input; longer text needs `truncate:true` or splitting.
- No smaller vector sizes: the maker documents no Matryoshka reduction, so `dimensions` only accepts 1024.
- Queries without an instruction retrieve worse; the maker's FAQ says performance degrades without one.
- The maker lists dozens of languages and calls the list not exhaustive; quality per language is not stated.

## Licence and credit

MIT. Commercial use is allowed; keep the copyright and licence notice. Made by Microsoft; the model card gives no paper or citation.

## Links

- Model card: https://huggingface.co/microsoft/harrier-oss-v1-0.6b/tree/f9b9dc8d367d443f2479d27aa5d8d2850c0774ee

## How Arsenal runs it

Source: `microsoft/harrier-oss-v1-0.6b@f9b9dc8d367d443f2479d27aa5d8d2850c0774ee`. Immutable filenames, byte sizes and SHA-256 hashes are in `model.toml` and `gate/source.json`. All source tensors are BF16. The serving default is FP16: Get streams each BF16 source tensor into the FP16 base snapshot (`cast:embed-qwen3-v2`) before execution. Arsenal offers this model at FP16 only: below 1B parameters, 8-bit would save under 1 GB and run no faster.

Runner: the native Swift MLX embed worker, sequential requests only, macOS 26 or later, ported from `ml-explore/mlx-swift-lm@bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57` and adapted inside `Workers/Embed` (see its `PORTED-FROM.md` and `LICENSE-upstream`). Causal Qwen3, last-token pooling, then optional L2 normalisation in float32.

Tokenisation: Arsenal encodes the prompt without special tokens. The pinned tokenizer's post-processor appends `<|endoftext|>` when special tokens are on, which is the Transformers default in the card's example, so the upstream reference pools that end token while Arsenal pools the last text token. The effect on retrieval quality has not been measured.

Checked live at FP16 on 19 Oct 2026: a cat/kitten pair scored 0.82, unrelated texts 0.40 to 0.42. Quick check, 22 Oct 2026, M5 Max, Mac in use: load 2.64 s; warm `embed` on three short sentences 0.08 s; peak memory 1.97 GB summed over Arsenal's worker processes (`results/`).

Gate (`gate/gate.toml`): exact checkpoint inventory and every declared operation on synthetic weights at FP16, int8 and int4 pass (`gate/smoke-check.json`); trained-weight reference and precision qualification against the vendor implementation are pending. Benchmark deferred (`bench/bench.toml`). Log: `log/001-port.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Text only. The card lists 42 languages, "including but not limited to". [Hugging Face model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Queries need a one-sentence instruction; documents need none. Without it, queries degrade. [model card FAQ, undated, first-hand]
- Parameters people recommend: Not found in sources checked.
- Arsenal call: `arsenal_run(model:"harrier-oss-v1-0.6b", operation:"embed", input:{"input":["how do I reset my router"],"task":"query"})`

**Good and bad**
- Good at: Multilingual retrieval. The card reports 69.0 on Multilingual MTEB v2. A leaderboard page gives 70.75 multilingual retrieval, within 0.13 of Qwen3-Embedding-8B. [model card, undated, first-hand; supermemory.ai, scores read 2026-10-16, opinion]
- Bad at: Not found in sources checked. The same leaderboard marks its English retrieval score "Not run". [supermemory.ai, scores read 2026-10-16, opinion]
- Against the obvious alternatives: About 6 points ahead of Qwen3-Embedding-0.6B (64.65, Workers AI row, by subtraction). [supermemory.ai, opinion]

**Complaints, bugs and fixes**
- Complaint or known bug: An Unsloth port zeroed the `<|endoftext|>` embedding the model pools, so it produced wrong embeddings (an Unsloth bug, not the model). Fix or workaround: Maintainer says fixed in Unsloth PR #13628; upgrade `unsloth` and `unsloth_zoo`. [unslothai/unsloth issue #4984, opened 2026-04-12, maintainer comment 2026-10-14, first-hand]
- Complaint or known bug: No 0.6B reranker; one user asked for one. Fix or workaround: None; no replies shown. [Hugging Face discussion #7, "Apr 16" (year inferred), opinion]

**Evidence**
- Strength: thin; maker card, leaderboard numbers, user threads; no independent test.
- Sources:
  - https://huggingface.co/microsoft/harrier-oss-v1-0.6b, undated, first-hand
  - https://huggingface.co/microsoft/harrier-oss-v1-0.6b/discussions/7, "Apr 16" (year inferred), opinion
  - https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/, byline 2025-06-27, scores read 2026-10-16, opinion
  - https://github.com/unslothai/unsloth/issues/4984, opened 2026-04-12, maintainer comment 2026-10-14, first-hand
