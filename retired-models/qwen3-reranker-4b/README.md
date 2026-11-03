# Qwen3-Reranker-4B

Rank candidate passages against a search query.

Source: `Qwen/Qwen3-Reranker-4B@22e683669bc0f0bd69640a1354a6d0aebcfeede5`. Immutable filenames, byte sizes and SHA256 hashes are in `model.toml` and `gate/source.json`. All source tensors inspected are BF16. The serving default is FP16; Get streams each BF16 source tensor into the FP16 base snapshot before execution. Int8 and int4 derive every eligible 2-D Linear/Embedding matrix with MLX affine group size 64; learned norms and spatial kernels stay half. No quality or speed claims are made.

Pairwise reranking uses Qwen’s documented instruction/query/document template, no thinking, and a two-class softmax of the final `no`/`yes` logits. Supports a custom instruction, per-document token bound, top_n and stable original indices. Documents truncate to the remaining model context; an overlong query/instruction fails.

Port: `ml-explore/mlx-swift-lm@bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`, adapted inside `Workers/Embed`; see its `PORTED-FROM.md` and `LICENSE-upstream`.

No trained weights have been run. Header/shape and synthetic operation evidence cannot establish tokenizer/prompt fidelity, retrieval relevance, FP16 activation safety, quantized quality, speed or peak RAM on the trained model. Those require the separately approved live trial.

The pinned snapshot is BF16. Get streams it into the default FP16 base snapshot with the family-owned cast recipe before deriving int8/int4. Numerical cast quality still awaits trained-weight qualification.
