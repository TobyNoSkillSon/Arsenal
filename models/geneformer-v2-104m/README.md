# Geneformer V2 104M

Turns one human cell into numbers: the Theodoris lab's Geneformer, the V2 model with 104M parameters, pretrained on about 104 million non-cancer human single-cell transcriptomes. A cell goes in as its genes in rank order; cells of the same type come out close together, so the vectors serve for comparing, clustering and classifying cells.

Input: `genes`, the cell's human Ensembl gene IDs, highest rank first (Geneformer's rank value encoding: expression divided by the cell's total counts and by the gene's median across the training corpus), at most 4,094 used; or `gene_tokens`, the same cell as token ids from the maker's tokenizer; `last_layer` (optional, default false: the second-to-last layer, the maker's default). Output: `embedding`, the cell's `<cls>` embedding (768 numbers, the maker's default), `mean_embedding`, the gene embeddings averaged, `dimensions`, and counts of genes used, skipped (not in the dictionary) and truncated.

Example request (a T cell's top marker genes, CD3E, CD3D, CD2, IL7R, ...):

```json
{"genes": ["ENSG00000198851", "ENSG00000167286", "ENSG00000116824", "ENSG00000168685"]}
```

Limits: Arsenal does not rank raw counts; use the maker's TranscriptomeTokenizer, or rank by normalised expression yourself, which is only an approximation. Human protein-coding genes only; gene symbols and other species are skipped. Trained on non-cancer cells. Raw `<cls>` cosines run high even across cell types, so compare cells relative to each other. Classification and in silico perturbation need a model or code on top.

Parity: against transformers BertForMaskedLM on MPS at float16, the `<cls>` embedding's cosine was 1.000000 on three marker-gene cells and on a full 4,096-token cell, and the mean embedding's at least 0.999999; max difference 0.008 (log/003-parity.md).

Source: [ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer/tree/1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5), folder Geneformer-V2-104M and the token dictionary geneformer/token_dictionary_gc104M.pkl, Apache-2.0. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: a folder of tokenised `.dataset` inputs for EmbExtractor; up to 4096 tokens over about 20K protein-coding genes. [EmbExtractor docs and model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not relevant.
- Parameters people recommend: name the 104M folder (the repo default is V2-316M); `emb_layer` -1 is second-to-last. [model card and EmbExtractor docs, undated, first-hand]
- Arsenal call: `POST /arsenal/v1/models/geneformer-v2-104m/ops/embed_cells`, or the same operation through MCP. Vendor example (EmbExtractor docs, abridged, CellClassifier example):

```python
from geneformer import EmbExtractor
embex = EmbExtractor(model_type="CellClassifier", num_classes=3, emb_mode="cell", max_ncells=1000, emb_layer=-1)
embs = embex.extract_embs("path/to/model", "path/to/input_data", "path/to/output_directory", "output_prefix")
```

**Good and bad**
- Good at: NVIDIA's TransformerEngine port claims weights and outputs match the original "within numerical precision"; no measurement. [NVIDIA BioNeMo readme, undated, first-hand]
- Bad at: the maker says GPU resources are required and hyperparameters need tuning. [model card, undated, first-hand]
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: EmbExtractor's summary stat dropped the last dimension (reported 2025-03-11). Fix or workaround: owner pushed a change 2025-03-12, then updated it the same day after the reporter's follow-up. [HF discussion 501, first-hand]
- Complaint or known bug: `get_state_embs` failed with `emb_label` set (reported 2024-10-17). Fix or workaround: dropping `emb_label` worked for the reporter on 2024-10-19. The maintainer (2024-10-23) says `emb_label` labels per-cell output, which a one-embedding-per-condition dict cannot carry. [HF discussion 434, first-hand]

**Evidence**
- Strength: thin. Maker docs, one port page, two bug threads; no independent test found.
- Sources:
  - Model card, https://huggingface.co/ctheodoris/Geneformer, undated
  - EmbExtractor docs, https://geneformer.readthedocs.io/en/latest/geneformer.emb_extractor.html, undated
  - NVIDIA BioNeMo readme, https://docs.nvidia.com/bionemo-recipes/latest/main/recipes/models/geneformer/model_readme/, undated
  - HF discussion 501, https://huggingface.co/ctheodoris/Geneformer/discussions/501, 2025-03-11 to 2025-03-12
  - HF discussion 434, https://huggingface.co/ctheodoris/Geneformer/discussions/434, 2024-10-17 to 2024-10-23
