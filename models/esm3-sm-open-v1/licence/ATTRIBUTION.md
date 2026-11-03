# Attribution: ESM3 open (esm3-sm-open-v1)

- Maker: Chan Zuckerberg Biohub (the ESM team; the model was released as EvolutionaryScale ESM3).
- Weights and tokenizer data: https://huggingface.co/biohub/esm3-sm-open-v1 at revision
  47f0545b2b6daf26a93439a3cd610f4f7f3d5478
- Reference code (read, never shipped): https://github.com/Biohub/esm at commit
  8f0834cbc350aeaf507f755475873f7c61fa44a1 (esm 3.4.2)
- Licence: MIT. The model card says "This repository is under a MIT license" and links the esm repository's
  LICENSE.md, copied here as `licence/LICENSE.md` (Copyright 2026 Chan Zuckerberg Biohub, Inc.).
- The esm repository's THIRD_PARTY_NOTICE.md lists only the Python libraries that repository's own code depends on
  (PyTorch, flash-attn, xformers, jaxtyping, einops, omegaconf, attrs, scipy, torchmetrics). None of them is part of
  this runner, which is Swift on MLX; the notice adds no condition to the weights.
- The maker's rotary-embedding file keeps its upstream Apache-2.0 header (EleutherAI and Hugging Face, after LLaMA 2);
  the port follows the same maths and the same convention, written from the reference.

Citation, as the ESM3 paper is cited by the maker:

```bibtex
@article{hayes2024simulating,
  author  = {Hayes, Thomas and Rao, Roshan and Akin, Halil and Sofroniew, Nicholas J. and Oktay, Deniz and Lin, Zeming
             and Verkuil, Robert and Tran, Vincent Q. and Deaton, Jonathan and Wiggert, Marius and Badkundri, Rohil
             and Shafkat, Irhum and Gong, Jun and Derry, Alexander and Molina, Raul S. and Thomas, Neil and Khan,
             Yousuf A. and Mishra, Chetan and Kim, Carolyn and Bartie, Liam J. and Nemeth, Matthew and Hsu, Patrick D.
             and Sercu, Tom and Candido, Salvatore and Rives, Alexander},
  title   = {Simulating 500 million years of evolution with a language model},
  journal = {Science},
  year    = {2025},
  doi     = {10.1126/science.ads0018}
}
```

Safety, from the model card: the training data had virus-related data and some select-agent organisms removed, and the
function decoder's keyword vocabulary is filtered. The maker asks users to follow its acceptable-use policy.
