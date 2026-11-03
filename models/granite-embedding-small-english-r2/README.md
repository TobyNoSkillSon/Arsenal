# Granite Embedding Small English R2

Turns English text into 384-number vectors for search by meaning and similarity: IBM's 47M-parameter Granite embedding model, with an 8192-token context and CLS pooling.

Input: `input`, a list of text strings, with optional `normalize`, `truncate` and `dimensions` (only 384). Output: `embeddings`, one vector per input, and `input_tokens`.

Example request:

```json
{"input": ["Who made the song My achy breaky heart?", "Achy Breaky Heart is a country song written by Don Von Tress."]}
```

Limits: English only and text only; at most 8192 tokens per input. Parity with the maker's code has not been run, and the card's retrieval figures were not reproduced here.

Source: [ibm-granite/granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2/tree/2ab6fa8ea2d674564defd37171ae19079b864b33), Apache-2.0. Runs as FP16 through the Arsenal kit runner (MLX).
