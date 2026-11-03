import Foundation
import CoreGraphics
import Testing
import MLX
import MLXNN
import IntegrationKit
import ArsenalWire
@testable import SAM31
@testable import MLXRFDETR
@testable import MLXLocateAnything

@Suite(.serialized) struct ReviewRegressionTests {
    init() { if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { GPU.metallib = URL(fileURLWithPath: path) } }

    private func sam() throws -> (SAM31Model, FrameFeatures) {
        let config = NativeSAM31.smallConfig()
        let root = SAM31Root(config)
        try NativeParameters.initializeRandom(root)
        let model = SAM31Model(nativeConfig: config, root: root)
        return (model, FrameFeatures(backbone: MLXArray.ones([1, 8, 8, 64], dtype: .float16), sourceSize: CGSize(width: 112, height: 112)))
    }

    @Test(.heavy) func maskCorrectionTargetsOneOfTwoObjectsAndReencodesMemory() throws {
        let (model, features) = try sam()
        let state = SessionState()
        let empty = try Mask(logits: [-1], width: 1, height: 1)
        let filled = try Mask(logits: [1], width: 1, height: 1)
        try model.sessionAddObject(state, id: 7, frameIndex: 0, features: features, prompt: .mask(empty))
        try model.sessionAddObject(state, id: 8, frameIndex: 0, features: features, prompt: .mask(filled))
        let tracker = try #require(state.tracker)
        let before = try #require(tracker.condFrameOutputs[0])
        let otherMask = before.predMasksHighRes[1].asArray(Float.self)
        let otherScore = before.objectScoreLogits[1].asArray(Float.self)
        let otherPointer = tracker.multiplexState.demux(before.objPtr!)[1].asArray(Float.self)
        let memory = before.maskmemFeatures!.asArray(Float.self)
        try model.sessionRefineMask(state, id: 7, frameIndex: 0, features: features, mask: filled)
        let output = try #require(tracker.condFrameOutputs[0])
        #expect(output.predMasksHighRes.shape == [2, 1, 112, 112])
        #expect(output.predMasksHighRes[0].asArray(Float.self).allSatisfy { $0 == 10 })
        #expect(output.predMasksHighRes[1].asArray(Float.self) == otherMask)
        #expect(output.objectScoreLogits[1].asArray(Float.self) == otherScore)
        #expect(tracker.multiplexState.demux(output.objPtr!)[1].asArray(Float.self) == otherPointer)
        #expect(output.maskmemFeatures!.asArray(Float.self) != memory)
        #expect(MLX.all(MLX.isFinite(output.maskmemFeatures!)).item(Bool.self))
        #expect(tracker.multiplexState.objectIDs == [7, 8])
        _ = try model.sessionFrame(state, features: features, frameIndex: 1)
    }

    @Test(.heavy) func correctedFrameIsEmittedWithoutAnotherPropagation() throws {
        let (model, features) = try sam()
        let state = SessionState()
        let mask = try Mask(logits: [-1], width: 1, height: 1)
        try model.sessionAddObject(state, id: 7, frameIndex: 0, features: features, prompt: .mask(mask))
        _ = try model.sessionRefine(state, id: 7, frameIndex: 0, features: features, prompt: .points([PromptPoint(x: 24, y: 70, label: .positive)]))
        let tracker = try #require(state.tracker)
        let corrected = try #require(tracker.nonCondFrameOutputs[0] ?? tracker.condFrameOutputs[0])
        // The final adapter emission uses sessionFrame, for both interactive and text corrections.
        let emitted = try model.sessionFrame(state, features: features, frameIndex: 0)
        let actual = try #require(emitted.objects[ObjectID(rawValue: 7)])
        #expect(actual.mask.logits.asArray(Float.self) == corrected.predMasks[0, 0].asArray(Float.self))
        let filled = try Mask(logits: [1], width: 1, height: 1)
        try model.sessionRefineMask(state, id: 7, frameIndex: 0, features: features, mask: filled)
        let correctedMask = try model.sessionFrame(state, features: features, frameIndex: 0)
        #expect(try #require(correctedMask.objects[ObjectID(rawValue: 7)]).mask.logits.asArray(Float.self).allSatisfy { abs($0 - 10) < 0.001 })
        let next = try model.sessionFrame(state, features: features, frameIndex: 1)
        #expect(next.frameIndex == 1 && tracker.nonCondFrameOutputs[1] != nil)
    }

    @Test(.heavy) func nanoWindowPlacementAndFeatureExtractionFollowPinnedConfig() throws {
        let url = try #require(Bundle.module.url(forResource: "rf-detr-nano-config", withExtension: "json", subdirectory: "Fixtures"))
        let config = try JSONDecoder().decode(JSONObject.self, from: Data(contentsOf: url))
        let layout = NativeRFDETR.backboneLayout(config)
        let backbone = DINOv2Backbone(imgSize: 32, patchSize: 16, embedDim: 4, depth: 12, numHeads: 1, featureIndices: layout.features, windowBlockIndices: layout.windows)
        #expect(backbone.featureIndices == [2, 5, 8, 11])
        #expect(backbone.fullAttnLayers == [3, 6, 9])
        // Reference dinov2_with_windowed_attn: global blocks merge the four windows;
        // fixed inputs and identity projections make its softmax attention independently calculable.
        let input: [Float] = [1, 2, -1, 0, 3, 0, 1, -2, 0, 4, 2, 1, -1, 2, 0, 3]
        let array = MLXArray(input, [4, 1, 4])
        for index in [2, 3, 5, 6, 8, 9, 11] {
            let block = backbone.blocks[index]
            var parameters: [String: MLXArray] = [:]
            for (name, value) in block.parameters().flattened() {
                if ["attn.q.weight", "attn.k.weight", "attn.v.weight", "attn.out.weight"].contains(name) {
                    parameters[name] = MLXArray((0..<16).map { Float($0 / 4 == $0 % 4 ? 1 : 0) }, [4, 4])
                } else {
                    parameters[name] = name.hasSuffix(".weight") || name == "ls1.gamma" ? MLXArray.ones(value.shape) : MLXArray.zeros(value.shape)
                }
            }
            try block.update(parameters: ModuleParameters.unflattened(parameters), verify: .all)
            let global = backbone.fullAttnLayers.contains(index)
            let output = Device.withDefaultDevice(.cpu) { block(array, runFullAttention: global).asArray(Float.self) }
            let rows = stride(from: 0, to: input.count, by: 4).map { Array(input[$0..<($0 + 4)]).map(Double.init) }
            let normed = rows.map { row -> [Double] in
                let mean = row.reduce(0, +) / 4
                let variance = row.map { pow($0 - mean, 2) }.reduce(0, +) / 4
                return row.map { ($0 - mean) / sqrt(variance + 1e-6) }
            }
            let reference = rows.indices.flatMap { row -> [Float] in
                let indices = global ? Array(rows.indices) : [row]
                let scores = indices.map { key in zip(normed[row], normed[key]).map(*).reduce(0, +) / 2 }
                let exps = scores.map { exp($0 - scores.max()!) }
                let sum = exps.reduce(0, +)
                return (0..<4).map { column in Float(rows[row][column] + zip(indices, exps).map { normed[$0.0][column] * $0.1 / sum }.reduce(0, +)) }
            }
            #expect(zip(output, reference).allSatisfy { abs($0 - $1) < 0.00001 })
        }
    }

    @Test(.heavy) func boxSineEmbeddingMatchesReferenceWidthBeforeHeight() {
        // rf-detr@5b22dfc src/rfdetr/models/transformer.py::gen_sineembed_for_position.
        let coordinates: [Float] = [0.13, 0.29, 0.41, 0.73]
        let actual = genSineembedForPosition(MLXArray(coordinates, [1, 4]), dModel: 8).asArray(Float.self)
        let reference: [Float] = [1, 0, 2, 3].flatMap { coordinate in
            (0..<8).map { frequency in
                let angle = Double(coordinates[coordinate]) * 2 * Double.pi / pow(10000, Double(2 * (frequency / 2)) / 8)
                return Float(frequency.isMultiple(of: 2) ? sin(angle) : cos(angle))
            }
        }
        #expect(zip(actual, reference).allSatisfy { abs($0 - $1) < 0.000001 })
    }

    @Test func screenInstructionsChooseGuiAndPointTemplates() throws {
        #expect(
            try NativeLocateAnything.question(operation: "ground_screen", input: ["instruction": "the save button"])
                == "Locate the region that matches the following description: the save button.")
        #expect(try NativeLocateAnything.question(operation: "ground_screen", input: ["instruction": "the save button", "output_mode": "points"]) == "Point to: the save button.")
        #expect(try NativeLocateAnything.question(operation: "detect", input: ["phrase": "red car", "output_mode": "points"]) == "Point to: red car.")
        #expect(try NativeLocateAnything.question(operation: "detect", input: ["instruction": "Detect all objects."]) == "Detect all objects.")
    }

    /// The same box written again and again stops at the 256-token span; 300 distinct boxes (2,100 tokens) never do.
    @Test func repeatedBoxesStopWhileDistinctBoxesContinue() throws {
        let config = try JSONDecoder().decode(LocateAnythingConfiguration.self, from: Data("{\"text_config\":{},\"vision_config\":{}}".utf8))
        func box(_ values: [Int]) -> [Int] { [config.boxStartTokenId] + values.map { config.coordStartTokenId + $0 } + [config.boxEndTokenId] }
        var looping = LocateAnythingRepetitionGuard(), stoppedAt: Int?
        for index in 0..<200 where looping.append(box([20, 30, 700, 800])) { stoppedAt = (index + 1) * 6; break }
        #expect(stoppedAt == 258, "stopped after \(stoppedAt ?? -1) tokens")
        var distinct = LocateAnythingRepetitionGuard()
        let boxes = (0..<300).map { box([$0 % 997, ($0 * 7) % 997, ($0 * 13 + 5) % 997, ($0 * 31 + 9) % 997]) + [77] }
        #expect(!boxes.contains { distinct.append($0) })
    }

    @Test func cappedAndMalformedOutputCannotBecomeNoMatch() throws {
        let config = try JSONDecoder().decode(LocateAnythingConfiguration.self, from: Data("{\"text_config\":{},\"vision_config\":{}}".utf8))
        let start = config.boxStartTokenId, end = config.boxEndTokenId, eos = config.eosTokenIds[0]
        let coordinates = [20, 30, 700, 800].map { config.coordStartTokenId + $0 }
        let box = [start] + coordinates + [end]
        #expect(throws: LocateAnythingOutputError.incompleteGeneration) { try LocateAnythingBoxes.completed(box, config: config) }
        let body = try LocateAnythingBoxes.completed(box + [eos], config: config)
        #expect(try LocateAnythingBoxes.parse(body, config: config, imageWidth: 100, imageHeight: 200) { _ in "" }.count == 1)
        let negative = [start, config.noneTokenId, end]
        #expect(
            try LocateAnythingBoxes.parse(LocateAnythingBoxes.completed(negative + [eos], config: config), config: config, imageWidth: 100, imageHeight: 200) { _ in "" }.isEmpty)
        for invalid in [[start] + coordinates, [start, config.noneTokenId], [start] + coordinates + [coordinates[0], end], [start, 77, end], [config.refStartTokenId, 77], [], [77]]
        {
            #expect(throws: LocateAnythingOutputError.malformedGrammar) { try LocateAnythingBoxes.parse(invalid, config: config, imageWidth: 100, imageHeight: 200) { _ in "" } }
        }
    }
}
