# Real check, 2 Nov 2026

One request through Arsenal's real service on an isolated profile (`arsenal-testserver --real`, private QA catalog from
the committed folder, Hugging Face client on loopback so Get could only import the local copy), after the embed worker
began adding the tokenizer's `<|endoftext|>` and pooling it. M5 Max, Mac in use, on AC. No speed figures are claimed.

- Get: the pinned `Qwen/Qwen3-Embedding-0.6B` files imported from the local cache and cast to FP16 (`cast:embed-qwen3-v2`).
- `embed` on the four documents and two queries of `log/003-parity.md` (queries with `task = "query"`), compared with
  Qwen's model-card code (transformers, MPS, FP16, last-token pooling) on the same texts.
- Cosine per text against the card's path: 0.999997 on all four documents, 0.999997 and 0.999994 on the queries
  (0.44 to 0.83 before the fix). Query-to-document scores, Arsenal then Qwen: capital query 0.764 / 0.765 on its
  document, gravity query 0.600 / 0.601; every other pair within 0.001.
- Not checked: inputs near the 32k context, `truncate`, reduced `dimensions`, int8 and int4.
