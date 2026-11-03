# jina-reranker-v3.5

Rank candidate passages against a search query.

Source: `jinaai/jina-reranker-v3.5-mlx@3dd4ac901ccdcac85abe3815df0a0aaaf44e4a21`. Immutable filenames, byte sizes and SHA256 hashes are in `model.toml` and `gate/source.json`. All source tensors inspected are BF16. The serving default is FP16; Get streams each BF16 source tensor into the FP16 base snapshot before execution. No quality or speed claims are made.

Listwise reranking uses the official v3.5 template, two query markers (scores use the late marker), per-layer sliding/full causal attention, a two-layer ReLU projector, 125-document blocks and weighted query fusion. Native cosine scores are mapped as `(1 + cosine) / 2` to the shared API’s [0, 1] range; order and original indices are retained. The projector is verified from both checkpoint headers. `return_embeddings: true` includes each ranked document’s 512-dimensional projected representation through the canonical operation, MCP and schema-driven tool. The provider rerank aliases currently do not forward this model-specific option.

Port: `ml-explore/mlx-swift-lm@48114eedf3c3d4a682a6b1d644cc5af5c2945d10`, adapted inside `Workers/Embed`; see its `PORTED-FROM.md` and `LICENSE-upstream`.

Checked live at FP16 on 19 Oct: the relevant document ranked first; scores within one request sit close together.

The pinned snapshot is BF16. Get streams it into the default FP16 base snapshot with the family-owned cast recipe.

Arsenal offers this model at FP16 only: below 1B parameters, 8-bit would save under 1 GB and run no faster.
