# Qwen3 Embedding 0.6B

Turns text into one 1024-number vector so texts with similar meaning point the same way. The smallest of Qwen's Qwen3 Embedding models (595M parameters, Qwen3 architecture, 28 layers), for over 100 languages, Polish included, and code. Its card reports 64.33 mean on MTEB multilingual.

Input: `input`, a list of texts, and optionally `task` (`document`, the default, or `query`, `similarity`, `classification`, `clustering`, which add an instruction in front of the text), `instruction` (replaces that instruction; ignored for `document`), `dimensions` (keep the first 32 to 1024 numbers), `normalize` (default true) and `truncate` (cut texts over 32,768 tokens instead of refusing them). Output: `embeddings`, one vector per text, and `input_tokens`.

Example request:

```json
{"input": ["What is the capital of China?"], "task": "query"}
```

Limits: vectors match Qwen's own code; they are not interchangeable with vectors from Arsenal builds before 1.0, which left out the tokenizer's end token, so an index built with those should be rebuilt. Text only.

Parity: Qwen's model-card code (transformers on MPS, FP16, the tokenizer's `<|endoftext|>` and last-token pooling) against Arsenal's real service at FP16 on four documents and two queries: cosine 0.999994 to 0.999997 per text, and the query-to-document scores agree to 0.001. Before the end token was added the same texts gave 0.44 to 0.83 (`log/003-parity.md`). Details in `results/real-check-2026-11-02.md`.

Runner: Arsenal's bundled native embed worker (no folder code). Source: [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3), Apache-2.0. Stored as FP16 (cast at Get, `cast:embed-qwen3-v2`).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: left padding, last-token pooling, L2 normalisation. Queries take `Instruct: {task}\nQuery:{query}`; documents take none. [model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: an English task instruction for queries; the card says it typically adds 1% to 5%. [model card, undated, first-hand]
- Parameters people recommend: up to 1024 dimensions; transformers 4.51.0 or newer. [model card, undated, first-hand]
- Arsenal call: `POST /arsenal/v1/models/qwen3-embedding-0.6b/ops/embed`, or the same operation through MCP. Vendor example (model card, abridged; `last_token_pool` is the card's helper):

```python
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
batch_dict = tokenizer(input_texts, padding=True, truncation=True, max_length=8192, return_tensors="pt")
outputs = model(**batch_dict)
embeddings = F.normalize(last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']), p=2, dim=1)
```

**Good and bad**
- Good at: one user's post says an INT8 ONNX build separated related entries better than sentence-transformers models; its cold load took about 3 s. [reddit post via a third-party mirror, snapshot 2026-03-20, opinion]
- Bad at: Not found in sources checked.
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: the text-embeddings-inference ONNX backend's last-token pooling had issues, and it set pad_token_id to 0 instead of reading config.json. [TEI PR 705, opened 2025-08-21, first-hand]
- Fix or workaround: PR 705, merged 2025-09-01, reads padding_side (default right) and pad_token_id. Shipping release not checked. [TEI PR 705, first-hand]

**Evidence**
- Strength: thin. Maker docs and paper, one merged TEI fix, one mirrored user post. Both Reddit threads were unopenable (login wall); their claims are removed.
- Sources:
  - Model card, https://huggingface.co/Qwen/Qwen3-Embedding-0.6B, undated
  - Qwen3 Embedding paper, https://arxiv.org/abs/2506.05176, v1 2025-06-05
  - Reddit post, https://reddit.sentinel-team.org/posts/1ryd3by/snapshots/2026-03-20T18%3A55%3A41.336186Z, via a third-party mirror, snapshot 2026-03-20, post date inferred about 2025-08-07, opinion
  - TEI PR 705, https://github.com/huggingface/text-embeddings-inference/pull/705, opened 2025-08-21, merged 2025-09-01
