+++
hypothesis = "A Swift MLX port of the maker's diffusers MiniMax Music 3 pipeline in the Music family reproduces the maker's components at FP16 and makes intelligible songs."
change = "Ported the global Qwen3 language model (input embeddings, KV cache), RVQ depth decoder, condition encoder, flow-matching transformer and Flow-VAE vocoder, the guided autoregressive loop, windowed Euler flow matching with overlap carry and cropped stitching, plus the Get layout recipe cast:minimax-music3-v1."
measurement = "Weight-free contract fixture from the maker's diffusers functions; heavy tiny random graph; maker component capture on PyTorch MPS pending the 57 GB download."
verdict = "inconclusive"
numbers = { contract_cases = 5, scheduler_cases = 4 }
links = ["README.md", "gate/golden/contract-fixture.json"]
+++

Started 2 Nov 2026 from code while the checkpoint downloaded. The maker's own GitHub repository holds documentation and a caption-rewriting skill, not inference code; the model card names diffusers (merged as huggingface/diffusers#14456) and SGLang-Omni as the inference paths, so the diffusers modules are the parity reference. The heavy tiny-graph smoke found one bug before any weights arrived: the first decode step emits no frame, and an unconditional force-unwrap of the last frame crashed every sampled generation.
