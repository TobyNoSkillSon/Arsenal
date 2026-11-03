// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM

/// One generation's view of a language model that hands the prefill's `LMOutput.State` to the decode steps, and
/// prefills long text-only Qwen prompts in chunks.
///
/// State: the pinned mlx-swift-lm `TokenIterator.prepare` keeps no state from a `.logits` prepare, so the first decode
/// step runs with `state: nil`. Every Qwen-VL family graph here keeps its multimodal RoPE offsets in that state: TeleOCR
/// (Qwen2.5-VL) fell back to 1D RoPE at the cache offset, about 1,300 positions too far after a layout image; Qwen3-VL
/// and Qwen3.5 recompute positions from the single decode token, so the first generated token sat at position 0 and
/// later ones at the cache offset without the image's (negative) delta, text-only prompts included. The model's own
/// prefill state (rope deltas, precomputed positions) is what Transformers carries as `rope_deltas`; this wrapper passes
/// it to every decode step that arrives without state. Later steps carry the state each step returns.
///
/// Chunks: the pinned Qwen3.5 and Qwen3-VL graphs prefill a whole prompt in one call with an explicit causal mask,
/// which MLX attends unfused for Qwen3.5's 256-wide heads, and project every prompt position to the vocabulary: both
/// grow with the square or the full length of the prompt (a 32,768-token Index-Translate prompt would hold tens of
/// gigabytes of attention scores). Text-only prompts longer than `chunk` tokens prefill `chunk` tokens at a time
/// through the cache, positions continuing from the cache offset exactly as decode steps do.
///
/// Created per generation, never registered as a module child: parameters and weights stay on the wrapped model.
final class PrefillState: Module, LanguageModel {
    static let defaultChunk = 1024
    private let inner: any LanguageModel
    private let chunk: Int
    private var prefill: LMOutput.State?

    init(_ inner: any LanguageModel, chunk: Int = PrefillState.defaultChunk) {
        precondition(chunk > 0)
        self.inner = inner
        self.chunk = chunk
        super.init()
    }

    /// Graphs whose decode path continues positions from the cache offset plus the prefill's rope deltas, so a text
    /// prompt may be fed through the cache in pieces.
    private var chunks: Bool { inner is Qwen35 || inner is Qwen3VL }

    func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        prefill = nil
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        let length = tokens.dim(1)
        guard chunks, input.image == nil, input.video == nil, input.audio == nil, tokens.dim(0) == 1, length > chunk else {
            let result = try inner.prepare(input, cache: cache, windowSize: windowSize)
            if case .logits(let output) = result { prefill = output.state }
            return result
        }
        let first = LMInput(text: .init(tokens: tokens[0..., ..<chunk], mask: input.text.mask.map { $0[0..., ..<chunk] }))
        guard case .logits(var output) = try inner.prepare(first, cache: cache, windowSize: windowSize) else {
            throw VLMError.processing("Chunked prefill expects prefill logits.")
        }
        var start = chunk
        while start < length {
            eval([output.logits] + cache.flatMap { $0.innerState() })
            let end = min(start + chunk, length)
            output = inner(LMInput.Text(tokens: tokens[0..., start..<end]), cache: cache, state: output.state)
            start = end
        }
        prefill = output.state
        return .logits(output)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?) -> LMOutput {
        inner(input, cache: cache, state: state ?? prefill)
    }

    func newCache(parameters: GenerateParameters?) -> [any KVCache] { inner.newCache(parameters: parameters) }
}
