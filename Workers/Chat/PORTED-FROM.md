# Source provenance

No sibling engine implementation is copied. Chat uses the public APIs of:
- mlx-swift `901941965d82e4a216d4d117231d847d194c563d`, MIT.
- mlx-swift-lm `bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`, MIT.
- swift-transformers `150169bfba0889c229a2ce7494cf8949f18e6906`, Apache-2.0.

The split compiler/Metal build recipe adapts the read-only
`Projects/Vella/Worker/build-split.sh` from the family. The dependency selection was inspected at
`Projects/Vireo/Package.swift`, Vireo commit `9366da5e14dcafe846fc41565c94f4f145eaf7cc`.
The tokenizer adapter and runner are original Arsenal code. No sibling app or sources are changed.
