+++
hypothesis = "Arsenal's bundled embed worker serves Qwen/Qwen3-Embedding-0.6B with the maker's vectors at FP16."
change = "None; comparison only. Folder from Arsenal catalog add-local into an isolated support folder."
measurement = "arsenal-worker-embed from the installed Arsenal (build of 30 Oct 08:34), driven over its wire outside the app: convert with cast:embed-qwen3-v2, load FP16, embed; against transformers AutoModel on MPS float16 with last-token pooling, 2026-10-30, M5 Max."
verdict = "inconclusive"
numbers = { same_tokens_min_cosine = 0.999997, same_tokens_max_abs = 0.0005, maker_path_cosine_min = 0.440, maker_path_cosine_max = 0.827, load_s = 0.99, first_embed_4_texts_s = 0.41, warm_s = 0.011, peak_mb = 1586 }
links = ["https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"]
+++
# 003 Parity with the maker's code

Inputs: four documents (an English fact, a longer English paragraph, a Polish sentence with diacritics, a sentence with
a degree sign and an em dash) and two queries with the default retrieval instruction. Scripts and vectors:
lab/night1/qwen3-embedding-0.6b/{drive.py,ref.py,inputs.json,mlx.json,parity.json}.

| text | cosine, same tokens | cosine, maker's path with end token |
|---|---|---|
| The capital of China is Beijing. | 0.999998 | 0.792 |
| Gravity paragraph | 0.999997 | 0.714 |
| Kraków sentence (Polish) | 0.999997 | 0.807 |
| Bread sentence | 0.999997 | 0.827 |
| query: What is the capital of China? | 0.999998 | 0.708 |
| query: Explain gravity | 0.999997 | 0.440 |

The network is a match: on the same token ids the worker and transformers agree to cosine 0.999997 or better. The full
path does not: the pinned tokenizer appends `<|endoftext|>` and the maker pools that token, while the worker encodes
without special tokens and pools the last text token (the same choice documented for Harrier). Query-to-document
scores, worker then maker: capital query 0.871 vs 0.765 on its document (next best 0.498 vs 0.351), gravity query 0.505
vs 0.601 (next best 0.194 vs 0.136). Ranking held on this small set; retrieval quality was not measured.

Awkward input: an empty string gives invalid_request "Embedding text must be nonempty.".
Measured once: load 0.99 s, first embed of four texts 0.41 s, warm single text 0.011 s, footprint 1586 MB.
