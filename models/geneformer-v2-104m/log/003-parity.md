+++
hypothesis = "The MLX kit runner matches transformers BertForMaskedLM on the Geneformer-V2-104M checkpoint."
change = "None: measurement of the Model.swift port on the original FP32 files (the runner casts to FP16 at load)."
measurement = "lab/night2/geneformer-v2-104m/ref_parity.py vs runjson.py replies on m5-max, 2026-10-31: 5 requests, <cls> and mean embeddings."
verdict = "kept"
numbers = { cls_cosine_min = 0.999999, mean_cosine_min = 0.999999, max_abs_diff = 0.0078 }
links = []
+++
# 003 Parity with transformers BertForMaskedLM (2026-10-31)

Method: transformers 5.22.0 `BertForMaskedLM` (as the maker's `pu.load_model` loads a "Pretrained" model; sdpa
attention) on MPS at float16, PyTorch 2.15.0, against the kit runner (MLX, FP16) on the same checkpoint, M5 Max.
Tokens as the maker's TranscriptomeTokenizer builds them for V2: <cls>, the gene tokens cut to 4,094, <eos>, mapped
through geneformer/token_dictionary_gc104M.pkl (read with Python's unpickler on the reference side, the runner's own
pickle reader on the MLX side). Embeddings as the maker's EmbExtractor defaults: hidden_states[12 + emb_layer] with
emb_layer -1, emb_mode "cls" (token 0) and emb_mode "cell" (mean over the gene tokens, <cls> and <eos> excluded).

Inputs (lab/night2/geneformer-v2-104m/inputs.json and requests.json): three cells written as marker genes in rank
order (a T cell, a reordered T cell, a hepatocyte; symbols mapped to Ensembl IDs with the maker's gene_name_id_dict),
a full-length cell (gene tokens 4-4199, cut to 4,094, so 4,096 tokens), and the T cell at the last layer.

| input | tokens | cls cosine | cls max abs diff | mean cosine | mean max abs diff |
|---|---|---|---|---|---|
| T cell | 30 | 1.000000 | 0.0010 | 0.999999 | 0.0015 |
| T cell, reordered | 30 | 1.000000 | 0.0039 | 1.000000 | 0.0016 |
| hepatocyte | 32 | 1.000000 | 0.0010 | 0.999999 | 0.0012 |
| full length | 4,096 | 1.000000 | 0.0010 | 1.000000 | 0.0004 |
| T cell, last layer | 30 | 0.999999 | 0.0078 | 1.000000 | 0.0024 |

Hidden values reach 6 to 17 in magnitude, so these are one or two float16 steps.
Sense check (both sides agree): <cls> cosine T cell vs reordered T cell 0.9963, T cell vs hepatocyte 0.9599.

Verdict: pass (cosine >= 0.999 on every input).
