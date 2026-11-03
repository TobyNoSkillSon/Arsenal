# Qwen3-Embedding-8B

Turn text into vectors for semantic search and similarity.

Source: `Qwen/Qwen3-Embedding-8B@1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`. Immutable filenames, byte sizes and SHA256 hashes are in `model.toml` and `gate/source.json`. All source tensors inspected are BF16. The serving default is FP16; Get streams each BF16 source tensor into the FP16 base snapshot before execution. Int8 and int4 derive every eligible 2-D Linear/Embedding matrix with MLX affine group size 64; learned norms and spatial kernels stay half. No quality or speed claims are made.

Text embeddings use causal Qwen3, last-token pooling and optional L2 normalization after truncating the vector. Query instructions use `Instruct: ...\nQuery: ...`; documents remain unprefixed. Qwen’s Matryoshka output supports dimensions from 32 through the native width. Similarity, classification and clustering use task instructions; callers can supply a task-specific instruction.

Port: `ml-explore/mlx-swift-lm@bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`, adapted inside `Workers/Embed`; see its `PORTED-FROM.md` and `LICENSE-upstream`.

No trained weights have been run. Header/shape and synthetic operation evidence cannot establish tokenizer/prompt fidelity, retrieval relevance, FP16 activation safety, quantized quality, speed or peak RAM on the trained model. Those require the separately approved live trial.

The pinned snapshot is BF16. Get streams it into the default FP16 base snapshot with the family-owned cast recipe before deriving int8/int4. Numerical cast quality still awaits trained-weight qualification.

## Usage notes (2 Nov 2026)

Qwen3-Embedding-8B turns text into 4,096-number vectors for search and similarity. `embed` takes `input` (up to 2,048
texts) and a `task`: `document` (default) embeds text as is; `query`, `similarity`, `classification` and
`clustering` prefix Qwen's instruction ("Instruct: ...\nQuery:"), which `instruction` can replace. `dimensions`
(32 to 4,096) shortens the vector; vectors are L2-normalized unless `normalize` is false. Context is 32,768 tokens;
longer input fails unless `truncate` is true. Apache-2.0.

Parity: on 2 Nov, after fixing the end token and query format (see `results/real-check-2026-11-02.md`), Arsenal's FP16
vectors match Qwen's own code run on the GPU in FP16 at cosine 0.999996 or better; golden in `gate/golden/`.

