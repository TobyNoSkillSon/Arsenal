# Qwen3-Reranker-0.6B

Rank candidate passages against a search query.

Source: `Qwen/Qwen3-Reranker-0.6B@e61197ed45024b0ed8a2d74b80b4d909f1255473`. Immutable filenames, byte sizes and SHA256 hashes are in `model.toml` and `gate/source.json`. All source tensors inspected are BF16. The serving default is FP16; Get streams each BF16 source tensor into the FP16 base snapshot before execution. No quality or speed claims are made.

Pairwise reranking uses Qwen’s documented instruction/query/document template, no thinking, and a two-class softmax of the final `no`/`yes` logits. Supports a custom instruction, per-document token bound, top_n and stable original indices. Documents truncate to the remaining model context; an overlong query/instruction fails.

Port: `ml-explore/mlx-swift-lm@bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`, adapted inside `Workers/Embed`; see its `PORTED-FROM.md` and `LICENSE-upstream`.

No trained weights have been run. Header/shape and synthetic operation evidence cannot establish tokenizer/prompt fidelity, retrieval relevance, FP16 activation safety, quantized quality, speed or peak RAM on the trained model. Those require the separately approved live trial.

The pinned snapshot is BF16. Get streams it into the default FP16 base snapshot with the family-owned cast recipe. Numerical cast quality still awaits trained-weight qualification.

Arsenal offers this model at FP16 only: below 1B parameters, 8-bit would save under 1 GB and run no faster.
