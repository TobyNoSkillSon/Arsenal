# astroBERT

Turns astronomy and astrophysics text into vectors of 768 numbers. Passages on the same topic land close together, so you can group, cluster or classify abstracts. astroBERT is a cased BERT-base model that the NASA Astrophysics Data System team pretrained on astrophysics papers. Following the maker's Tutorial 0, each vector is the model's last hidden state averaged over the text's tokens.

Input: `input`, a list of texts (plain strings or `{"type": "text", "text": ...}`). Options: `normalize` (default true, unit-length vectors) and `truncate` (default false; when true, a text longer than 512 tokens is cut instead of refused). `task` is accepted and has no effect. Output: `embeddings`, one list of 768 numbers per text, and `input_tokens`.

Example request:

```json
{"input": ["M67 is one of the most studied open clusters.", "M67 is a well-studied old open star cluster in Cancer."]}
```

Limits: it was not trained for sentence similarity or search. Unrelated texts still score about 0.5 to 0.7 cosine, so compare scores against each other rather than reading them as absolutes. In the quick check, two sentences about M67 scored 0.88 and a baking sentence scored 0.68 against them. English astronomy text only; it does not handle Polish. 512 tokens per text. No instruction or shorter dimensions, and no images or audio. The card's masked-word, NER and SciX classifier heads are not served.

Parity: matches the maker's transformers code (BertModel, MPS fp16) on four fixed texts at cosine 0.99999 or better, max abs diff 0.008 (log/003-parity.md).

Source: [adsabs/astroBERT](https://huggingface.co/adsabs/astroBERT/tree/6b9c25bb799152c5849a5b263ea71025fce548f0), MIT. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: The card says the model is cased, so "ads" and "ADS" are treated differently. The NER-DEAL variant loads only with `revision="NER-DEAL"`. [HF card adsabs/astroBERT, undated (read 2026-10-31; the linked paper 2112.00590 shows "Published Dec 1, 2021"), first-hand]
- Prompts, reference audio or voice settings, if relevant: Not applicable.
- Parameters people recommend: Not found in sources checked.
- Arsenal call: `POST /arsenal/v1/models/astrobert/ops/embed`, or the same operation through MCP. Vendor example: `AutoTokenizer.from_pretrained("adsabs/astroBERT")` and `AutoModelForPreTraining.from_pretrained("adsabs/astroBERT", device_map="auto")`.

**Good and bad**
- Good at: The abstract says the work sets out to "show how astroBERT improves over existing public language models on astrophysics specific tasks"; it gives no figures. [arXiv 2212.00744, submitted 2022-11-29, first-hand]
- Bad at: Not found in sources checked. The abstract gives no sentence-similarity results, and its only listed future work is using the citation graph and citation context to improve astroBERT. [arXiv 2212.00744, submitted 2022-11-29, first-hand]
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: Not found in sources checked. Fix or workaround: Not found in sources checked.

**Evidence**
- Strength: thin. The maker's paper and card are the only sources; no outside test of its embeddings was found.
- Sources:
  - adsabs/astroBERT model card, https://huggingface.co/adsabs/astroBERT, undated (read 2026-10-31), first-hand.
  - "Improving astroBERT using Semantic Textual Similarity", https://arxiv.org/abs/2212.00744, 2022-11-29, first-hand.
