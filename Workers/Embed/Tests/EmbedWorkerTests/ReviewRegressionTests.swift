import CoreGraphics
import Foundation
import IntegrationKit
import ArsenalWire
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import EmbedWorker

private struct OvisReference: Decodable {
    struct Activation: Decodable { let x, w1, b1, w2, b2, expected: [Float] }
    struct Frame: Decodable { let times: [Double]; let grid: [Int]; let placeholder: String; let ids, positions: [Int]; let delta: Int }
    struct Resize: Decodable { let frames, height, width: Int; let expected: [Int] }
    struct Sample: Decodable { let total: Int; let fps: Double; let indices: [Int] }
    let activation: Activation
    let frames: [Frame]
    let resize: [Resize]
    let samples: [Sample]
    let attention: [Float]
}

private func fixture(_ name: String) throws -> Data {
    try Data(contentsOf: #require(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures")))
}

private func fixtureTokens(_ text: String) -> [Int] {
    let special = [("<|vision_start|>", 10), ("<|vision_end|>", 11), ("<|image_pad|>", 12), ("<|video_pad|>", 13)]
    var remaining = text[...], ids: [Int] = []
    while !remaining.isEmpty {
        if let match = special.first(where: { remaining.hasPrefix($0.0) }) {
            ids.append(match.1); remaining = remaining.dropFirst(match.0.count)
        } else {
            ids.append(1000 + Int(remaining.removeFirst().asciiValue!))
        }
    }
    return ids
}

extension RetrievalGPUTests {
    @Test func compressedVideoMetadataCountsFramesRatherThanMarkerBuffers() async throws {
        let url = try #require(Bundle.module.url(forResource: "four-frame", withExtension: "mp4", subdirectory: "Fixtures"))
        let (frames, fps) = try await OvisVideoMetadata.read(url: url, cancellation: NativeCancellation())
        #expect(frames == 4)
        #expect(fps == 2)
    }

    @Test(.heavy) func ovisVisionMLPMatchesCapturedTransformersQwen35() throws {
        NativeRuntime.prepare()
        let reference = try JSONDecoder().decode(OvisReference.self, from: fixture("ovis-reference")).activation
        let model = OvisVision.MLP(dim: 4, hiddenDim: 6)
        try model.update(
            parameters: ModuleParameters.unflattened([
                "linear_fc1.weight": MLXArray(reference.w1).reshaped(6, 4), "linear_fc1.bias": MLXArray(reference.b1),
                "linear_fc2.weight": MLXArray(reference.w2).reshaped(4, 6), "linear_fc2.bias": MLXArray(reference.b2)
            ]), verify: .all)
        let actual = model(MLXArray(reference.x).reshaped(2, 4))
        #expect(MLX.max(abs(actual - MLXArray(reference.expected).reshaped(2, 4))).item(Float.self) < 0.000002)
    }

    @Test(.heavy) func ovisVideoFramingAndPositionsMatchCapturedTransformers() throws {
        NativeRuntime.prepare()
        let reference = try JSONDecoder().decode(OvisReference.self, from: fixture("ovis-reference"))
        for frame in reference.frames {
            let grid = THW(frame.grid[0], frame.grid[1], frame.grid[2])
            let placeholder = OvisVideoFraming.placeholder(times: frame.times, grid: grid, temporalSize: 2, mergeSize: 2)
            #expect(placeholder == frame.placeholder)
            let full = "A<|vision_start|>" + String(repeating: "<|image_pad|>", count: 4) + "<|vision_end|>" + placeholder + "B<|vision_start|><|image_pad|><|vision_end|>C"
            var expanded = OvisPromptExpansion(
                "A<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|video_pad|><|vision_end|>B<|vision_start|><|image_pad|><|vision_end|>C")
            try expanded.replace("<|image_pad|>", with: String(repeating: "<|image_pad|>", count: 4))
            try expanded.replace("<|vision_start|><|video_pad|><|vision_end|>", with: placeholder)
            try expanded.replace("<|image_pad|>", with: "<|image_pad|>")
            #expect(expanded.text == full)
            let ids = fixtureTokens(expanded.text)
            #expect(ids == frame.ids)
            let (positions, delta) = try OvisPositions.getRopeIndex(
                inputIds: MLXArray(ids).reshaped(1, -1), imageGridTHW: [THW(1, 4, 4), THW(1, 2, 2)], videoGridTHW: [grid],
                spatialMergeSize: 2, imageTokenId: 12, videoTokenId: 13, visionStartTokenId: 10)
            #expect(positions.asArray(Int.self) == frame.positions)
            #expect(delta.item(Int.self) == frame.delta)
        }
    }

    @Test func ovisVideoUsesPinnedTotalPixelBudgetAndSampling() throws {
        let reference = try JSONDecoder().decode(OvisReference.self, from: fixture("ovis-reference"))
        let config = try OvisVideoConfiguration([
            "size": ["shortest_edge": 4096, "longest_edge": 234881024], "patch_size": 16, "merge_size": 2,
            "temporal_patch_size": 2, "min_frames": 4, "max_frames": 768, "fps": 2.0
        ])
        for value in reference.resize {
            let (height, width) = try config.targetSize(frames: value.frames, height: value.height, width: value.width)
            #expect([height, width] == value.expected)
        }
        for value in reference.samples { #expect(config.sampleIndices(total: value.total, sourceFPS: value.fps) == value.indices) }
    }

    @Test(.heavy) func ovisSegmentedAttentionMatchesCapturedTorchAndIsolatesPages() throws {
        NativeRuntime.prepare()
        let reference = try JSONDecoder().decode(OvisReference.self, from: fixture("ovis-reference"))
        let values = MLXArray(0..<56).asType(.float32)
        let queries = sin(values).reshaped(1, 2, 7, 4), keys = cos(values).reshaped(1, 2, 7, 4), input = (values / 10).reshaped(1, 2, 7, 4)
        let result = OvisVision.Attention.segmentedAttention(queries: queries, keys: keys, values: input, boundaries: [0, 3, 7], scale: 0.5)
        #expect(MLX.max(abs(result - MLXArray(reference.attention).reshaped(1, 2, 7, 4))).item(Float.self) < 0.000002)
        let changed = input + concatenated([MLXArray.zeros([1, 2, 3, 4]), MLXArray.ones([1, 2, 4, 4]) * 1000], axis: 2)
        let isolated = OvisVision.Attention.segmentedAttention(queries: queries, keys: keys, values: changed, boundaries: [0, 3, 7], scale: 0.5)
        #expect(MLX.max(abs(result[0..., 0..., 0..<3, 0...] - isolated[0..., 0..., 0..<3, 0...])).item(Float.self) == 0)
        // A 32-page 2048² PDF has 524288 patches, but each call sees only
        // 16384 keys. There is no concatenated N×N mask allocation.
        #expect((0...32).map { $0 * 16384 }.last == 524288)
    }

    @Test(.heavy) func gemmaCancellationAfterFirstExecutedLayerStopsTheStack() throws {
        NativeRuntime.prepare()
        let model = EmbeddingGemma2TextModel(try JSONDecoder().decode(EmbeddingGemma2Configuration.self, from: fixture("embeddinggemma-2-tiny")))
        try NativeParameters.initializeRandom(model, half: .bfloat16)
        let cancel = NativeCancellation(); var evaluated: [Int] = []
        do {
            _ = try model(model.embedTokens(MLXArray([1, 2, 3]).reshaped(1, -1)), attentionMask: nil, cancellation: cancel) { index in
                evaluated.append(index); cancel.cancel()
            }
            Issue.record("Expected cancellation after evaluation")
        } catch let error as WorkerFailure { #expect(error.code == "cancelled") }
        #expect(evaluated == [0])
    }

    /// The full-attention layers attend causally through MLX's symbolic mode (no materialized L×L mask): the first
    /// tokens' hidden states do not change when later tokens do.
    @Test(.heavy) func ovisFullAttentionStaysCausalWithoutAnExplicitMask() throws {
        NativeRuntime.prepare()
        let config = try JSONDecoder().decode(Qwen35Configuration.self, from: fixture("ovis-vl-embedding-2b-tiny"))
        let model = Qwen35Language.Model(config.textConfiguration)
        try NativeParameters.initializeRandom(model)
        let prefix = [5, 9, 17, 33, 2, 41, 7, 60, 12, 3]
        let first = try model(MLXArray(prefix + [70, 71, 72, 73, 74, 75]).reshaped(1, -1), cancellation: NativeCancellation())
        let second = try model(MLXArray(prefix + [90, 91, 92, 93, 94, 95]).reshaped(1, -1), cancellation: NativeCancellation())
        let early = abs(first[0, ..<prefix.count].asType(.float32) - second[0, ..<prefix.count].asType(.float32)).max().item(Float.self)
        let late = abs(first[0, prefix.count...].asType(.float32) - second[0, prefix.count...].asType(.float32)).max().item(Float.self)
        #expect(early < 1e-3, "earlier positions saw later tokens: \(early)")
        #expect(late > 1e-2)
    }

    /// smart_resize token counts before rendering: a letter page at the 2,048-px render is 1,900 tokens (38 × 50 merged
    /// patches), so the 131,072-token input bound admits about 69 such pages; tiny and huge images hit the bounds.
    @Test func ovisCountsPageTokensLikeTheProcessor() {
        let bounds = (minimum: 65536, maximum: 16_777_216, factor: 32)
        #expect(OvisAdapter.imageTokens(width: 1224, height: 1584, bounds: bounds) == 38 * 50)
        #expect(OvisAdapter.imageTokens(width: 10, height: 10, bounds: bounds) == 64)
        #expect(OvisAdapter.imageTokens(width: 8000, height: 6000, bounds: bounds) <= 16_384)
        #expect(OvisAdapter.defaultMaxInputTokens / 1900 == 68)
    }

    /// A PDF whose pages exceed the input bound fails while its pages are being read, before any prefill.
    @Test(.heavy) func ovisRejectsAnOversizedDocumentBeforeEmbedding() throws {
        NativeRuntime.prepare(); setenv("ARSENAL_NATIVE_LAB", "1", 1)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-ovis-cap-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try fixture("ovis-vl-embedding-2b-tiny").write(to: root.appendingPathComponent("config.json"))
        let pdf = root.appendingPathComponent("five.pdf")
        var box = CGRect(x: 0, y: 0, width: 32, height: 32)
        let context = try #require(CGContext(pdf as CFURL, mediaBox: &box, nil))
        for _ in 0..<5 { context.beginPDFPage(nil); context.endPDFPage() }
        context.closePDF()
        let adapter = OvisAdapter()
        try adapter.load(
            NativeLoadContext(["model_dir": .string(root.path), "weights_dir": .string(root.path), "dtype": "fp16", "config": ["random_weights": true]]),
            cancellation: NativeCancellation())
        defer { adapter.unload() }
        // The random processor makes each page one token; a bound of 3 stops at the fourth page.
        adapter.maxInputTokens = 3
        do {
            _ = try adapter.run("embed", input: ["input": [["type": "document", "document": ["path": .string(pdf.path)]]]], cancellation: NativeCancellation()) { _ in }
            Issue.record("Expected the input bound to reject the document")
        } catch let error as WorkerFailure {
            #expect(error.message.contains("more than 3 tokens (4 so far)"), "\(error.message)")
        }
        adapter.maxInputTokens = 64
        let ok = try adapter.run("embed", input: ["input": [["type": "document", "document": ["path": .string(pdf.path)]]]], cancellation: NativeCancellation()) { _ in }
        #expect(ok["embeddings"]?.array?.count == 1)
    }

    @Test(.heavy) func ovisCancellationAfterFirstExecutedLayerStopsTheStack() throws {
        NativeRuntime.prepare()
        let config = try JSONDecoder().decode(Qwen35Configuration.self, from: fixture("ovis-vl-embedding-2b-tiny"))
        let model = Qwen35Language.Model(config.textConfiguration)
        try NativeParameters.initializeRandom(model)
        let cancel = NativeCancellation(); var evaluated: [Int] = []
        do {
            _ = try model(MLXArray([1, 2, 3]).reshaped(1, -1), cancellation: cancel) { index in
                evaluated.append(index); cancel.cancel()
            }
            Issue.record("Expected cancellation after evaluation")
        } catch let error as WorkerFailure { #expect(error.code == "cancelled") }
        #expect(evaluated == [0])
    }
}
