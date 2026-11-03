// SPDX-License-Identifier: MIT
import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXNN
import ArsenalWire
@testable import LanguageWorker

/// A decode step must see the same positions as a prefill of the whole sequence: after an image, MROPE puts the next
/// text token at max(position) + 1, not at the cache offset, and after a text-only prompt at the prompt length, not 0.
/// The pinned `TokenIterator` drops the prefill's state, so these run the worker's real path (TokenIterator over
/// `PrefillState`), on each Qwen-VL family architecture's shrunk random graph from its model folder.
struct DecodePositionTests {
    init() { MLXReady.ensure() }

    private static let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()

    private func model(_ id: String) throws -> (any LanguageModel, JSONObject) {
        let fixture = try WireJSON.decode(Data(contentsOf: Self.root.appendingPathComponent("models/\(id)/gate/no-weights.json")))
        let config = try #require(fixture["config"]?.object)
        let graph = try LanguageGraph.make(try WireJSON.encode(config), type: try #require(config["model_type"]?.string))
        // Module initializers give random weights; serve them in FP16 like the worker.
        let half = graph.parameters().flattened().map { ($0.0, $0.1.dtype.isFloatingPoint ? $0.1.asType(.float16) : $0.1) }
        try graph.update(parameters: ModuleParameters.unflattened(Dictionary(uniqueKeysWithValues: half)), verify: .none)
        return (graph, config)
    }

    /// Records the logits TokenIterator hands its processor at each step.
    final class Recorder: @unchecked Sendable { var logits: [MLXArray] = [] }
    struct Recording: LogitProcessor {
        let recorder: Recorder
        mutating func prompt(_ prompt: MLXArray) {}
        func process(logits: MLXArray) -> MLXArray { recorder.logits.append(logits.asType(.float32)); return logits }
        mutating func didSample(token: MLXArray) {}
    }
    /// Always samples `token`, so the decode step sees a known input.
    struct Fixed: LogitSampler {
        let token: Int
        func sample(logits: MLXArray) -> MLXArray { MLXArray([Int32(token)]) }
    }

    /// Largest |Δlogit| between the decode step after `next` and a prefill of prompt + next, and the reference's scale.
    private func decodeDifference(
        _ graph: any LanguageModel, prompt: [Int], next: Int, image: LMInput.ProcessedImage?, wrapped: Bool, chunk: Int = PrefillState.defaultChunk
    ) throws -> (Float, Float) {
        func input(_ tokens: [Int]) -> LMInput {
            let ids = MLXArray(tokens.map(Int32.init)).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: ids, mask: ones(like: ids).asType(.int8)), image: image)
        }
        guard case .logits(let whole) = try graph.prepare(input(prompt + [next]), cache: graph.newCache(parameters: nil), windowSize: nil) else {
            Issue.record("Expected prefill logits"); return (.infinity, 0)
        }
        let reference = whole.logits[0, -1].asType(.float32)
        let recorder = Recorder()
        var iterator = try TokenIterator(
            input: input(prompt), model: wrapped ? PrefillState(graph, chunk: chunk) : graph, processor: Recording(recorder: recorder), sampler: Fixed(token: next),
            maxTokens: 4)
        #expect(iterator.next() == next)
        #expect(recorder.logits.count == 2)
        let decoded = recorder.logits[1].reshaped(-1)
        #expect(reference.shape == decoded.shape)
        return (abs(reference - decoded).max().item(Float.self), abs(reference).max().item(Float.self))
    }

    /// 8×8 patches merged 2×2: 16 image tokens whose MROPE positions span only 4 rows and columns.
    private func imagePrompt(_ config: JSONObject, patch: Int) throws -> ([Int], LMInput.ProcessedImage) {
        let image = Int(try #require(config["image_token_id"]?.number)), start = Int(try #require(config["vision_start_token_id"]?.number))
        let end = Int(try #require(config["vision_end_token_id"]?.number))
        let pixels = MLXRandom.normal([64, 3 * 2 * patch * patch], key: MLXRandom.key(7)).asType(.float16)
        return ([11, 12, start] + Array(repeating: image, count: 16) + [end, 21, 22, 23, 24], .init(pixels: pixels, frames: [THW(1, 8, 8)]))
    }

    /// FP16 rounding between a prefill and a one-token step stays far below the shift a wrong position causes; random
    /// logits are near-tied, so argmax is not compared. The unwrapped graph shows the shift the wrapper removes.
    private func check(_ id: String, patch: Int, image: Bool) throws {
        let (graph, config) = try model(id)
        var prompt = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24]
        var processed: LMInput.ProcessedImage?
        if image { (prompt, processed) = try imagePrompt(config, patch: patch) }
        let (fixed, scale) = try decodeDifference(graph, prompt: prompt, next: 31, image: processed, wrapped: true)
        let (broken, _) = try decodeDifference(graph, prompt: prompt, next: 31, image: processed, wrapped: false)
        #expect(fixed <= 0.02 * max(1, scale), "\(id) image \(image): max |Δlogit| \(fixed) of \(scale)")
        #expect(broken > 10 * max(fixed, 0.001), "\(id) image \(image): the unwrapped decode should be off (\(broken) vs \(fixed))")
    }

    @Test(.heavy) func teleOCRQwen25VLDecodeMatchesPrefillAfterAnImage() throws { try check("teleocr", patch: 14, image: true) }

    @Test(.heavy, arguments: ["chandra-ocr-2", "index-translate-2b", "nuextract3", "ui-venus-2-9b"])
    func qwen35DecodeMatchesPrefill(id: String) throws {
        try check(id, patch: 16, image: true)
        try check(id, patch: 16, image: false)
    }

    /// A long text prompt prefilled five tokens at a time through the cache, then one decode step, matches a single
    /// prefill of prompt + token: positions continue from the cache offset and DeltaNet state carries across chunks.
    @Test(.heavy, arguments: ["index-translate-2b", "ui-venus-1.5-2b"]) func chunkedTextPrefillMatchesOnePrefill(id: String) throws {
        let (graph, _) = try model(id)
        let prompt = (0..<23).map { 40 + ($0 * 37) % 900 }
        let (difference, scale) = try decodeDifference(graph, prompt: prompt, next: 31, image: nil, wrapped: true, chunk: 5)
        #expect(difference <= 0.02 * max(1, scale), "\(id): max |Δlogit| \(difference) of \(scale)")
    }

    @Test(.heavy) func qwen3VLDecodeMatchesPrefill() throws {
        try check("ui-venus-1.5-2b", patch: 16, image: true)
        try check("ui-venus-1.5-2b", patch: 16, image: false)
    }
}
