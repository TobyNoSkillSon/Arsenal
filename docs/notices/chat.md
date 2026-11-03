# Chat dependencies and model

- **mlx-swift** revision `901941965d82e4a216d4d117231d847d194c563d`: MIT, Apple Inc. https://github.com/ml-explore/mlx-swift/blob/901941965d82e4a216d4d117231d847d194c563d/LICENSE
- **mlx-swift-lm** revision `bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`: MIT, Apple Inc. https://github.com/ml-explore/mlx-swift-lm/blob/bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57/LICENSE
- **swift-transformers** revision `150169bfba0889c229a2ce7494cf8949f18e6906`: Apache-2.0, Hugging Face. https://github.com/huggingface/swift-transformers/blob/150169bfba0889c229a2ce7494cf8949f18e6906/LICENSE

- **Qwen3.8-27B**: Apache-2.0, Alibaba Qwen, vendor bf16 revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE

Dependency licence files remain in the pinned SwiftPM source checkouts. No sibling binaries or private model weights are bundled in the repository.

Original Arsenal chat worker, test harnesses and tokenizer adapter are **MIT**.
The adapted build script records the source recipe in `Workers/Chat/PORTED-FROM.md`; pinned upstream
libraries retain their own licence notices, and Qwen weights remain Apache-2.0.
