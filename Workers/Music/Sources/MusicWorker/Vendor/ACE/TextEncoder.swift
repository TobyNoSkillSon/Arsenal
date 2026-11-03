//
//  TextEncoder.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  Qwen3-Embedding-0.6B as ACE-Step uses it: the caption prompt runs
//  through the whole model (last hidden state, after the final norm); the
//  lyrics only through its token-embedding table. Arsenal supplies the selected
//  half or packed parameters, including the lyric embedding.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

final class TextEncoder {
    let model: Qwen3Model
    let tokenizer: any Tokenizers.Tokenizer

    init(model: Qwen3Model, tokenizer: any Tokenizers.Tokenizer) {
        self.model = model; self.tokenizer = tokenizer
    }

    // Token ids, truncated as upstream truncates (`max_length`).
    func tokens(_ text: String, maxLength: Int) -> [Int] {
        Array(tokenizer.encode(text: text).prefix(maxLength))
    }

    // `[1, L, 1024]` hidden states for a prompt's tokens.
    func encode(_ ids: [Int]) -> MLXArray {
        model.model(MLXArray(ids.map(Int32.init)).reshaped(1, -1), cache: nil)
    }

    // `[1, L, 1024]` token embeddings (the lyric path).
    func embed(_ ids: [Int]) -> MLXArray {
        // swiftlint:disable:next force_cast
        (model.namedModules().first(where: { $0.0 == "model.embed_tokens" })!.1 as! Embedding)(MLXArray(ids.map(Int32.init))).expandedDimensions(axis: 0)
    }
}
