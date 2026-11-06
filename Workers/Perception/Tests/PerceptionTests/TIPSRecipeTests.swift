import Foundation
import Testing
import MLX
import MLXRandom
import IntegrationKit
@testable import MLXTIPS

/// TIPSv2's zero-shot segmentation follows `pytorch/TIPS_zeroshot_segmentation.ipynb`: value-attention patch features,
/// a nine-template label ensemble, bilinear upsampling of the cosine scores, argmax per pixel.
@Suite(.serialized) struct TIPSRecipeTests {
    init() { if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) } }

    /// `F.interpolate(mode="bilinear", align_corners=False)` written out per pixel, then argmax.
    private func reference(_ grid: [Float], rows: Int, columns: Int, count: Int, height: Int, width: Int) -> [Int32] {
        func source(_ index: Int, _ input: Int, _ output: Int) -> (Int, Int, Float) {
            let value = max(0, (Float(index) + 0.5) * Float(input) / Float(output) - 0.5)
            let low = min(Int(value), input - 1)
            return (low, min(low + 1, input - 1), value - Float(low))
        }
        var labels: [Int32] = []
        for y in 0..<height {
            let (y0, y1, fy) = source(y, rows, height)
            for x in 0..<width {
                let (x0, x1, fx) = source(x, columns, width)
                var best = 0, bestScore = -Float.infinity
                for label in 0..<count {
                    func at(_ r: Int, _ c: Int) -> Float { grid[(r * columns + c) * count + label] }
                    let score = (1 - fy) * ((1 - fx) * at(y0, x0) + fx * at(y0, x1)) + fy * ((1 - fx) * at(y1, x0) + fx * at(y1, x1))
                    if score > bestScore { best = label; bestScore = score }
                }
                labels.append(Int32(best))
            }
        }
        return labels
    }

    @Test(.heavy) func upsampledArgMaxMatchesTorchBilinearAtEveryTiling() throws {
        var state: UInt64 = 0x2545_F491_4F6C_DD1D
        let rows = 3, columns = 4, count = 5
        let values = (0..<(rows * columns * count)).map { _ -> Float in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(state >> 40) / Float(1 << 24) * 2 - 1
        }
        let grid = MLXArray(values, [rows, columns, count])
        for (height, width) in [(7, 10), (3, 4), (2, 3), (31, 17)] {
            let expected = reference(values, rows: rows, columns: columns, count: count, height: height, width: width)
            // The default budget is one tile here; a tiny budget forces one-row bands.
            for budget in [1 << 22, 1] {
                let actual = try NativeTIPS.upsampledArgMax(grid, height: height, width: width, budget: budget, cancellation: NativeCancellation())
                #expect(actual.count == height * width)
                #expect(actual == expected, "\(height)x\(width) budget \(budget)")
            }
        }
        // Same size is the identity: every pixel takes its own patch's best label.
        let identity = NativeTIPS.bilinearWeights(output: 4, input: 4)
        #expect(MLX.max(abs(identity - MLXArray.eye(4))).item(Float.self) == 0)
    }

    @Test(.heavy) func segmentUsesValueAttentionFeaturesAndTheNineTemplateEnsemble() throws {
        let model = NativeTIPS.makeModel(large: false, shrunk: true)
        MLXRandom.seed(11)
        try NativeParameters.initializeRandom(model)
        let tokenizer = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/tokenizers/tipsv2/tokenizer.model")
        model.tokenizer = try TIPSTokenizer(modelPath: tokenizer.path)
        #expect(NativeTIPS.segmentTemplates.count == 9)
        #expect(NativeTIPS.segmentTemplates.first == "itap of a {}." && NativeTIPS.segmentTemplates.last == "a photo of {}s.")
        let labels = ["background", "car", "a red bus"]
        // The batched ensemble equals the notebook's per-template computation.
        var batched: [MLXArray] = []
        for label in labels {
            let each = try NativeTIPS.segmentTemplates.map { template -> MLXArray in
                l2Normalize(try model.encodeText(template.replacingOccurrences(of: "{}", with: label)).asType(.float32))
            }
            let vector = l2Normalize(mean(concatenated(each, axis: 0), axis: 0, keepDims: true))
            let served = try NativeTIPS.segmentLabelVector(model, label)
            #expect(served.shape == [1, 64])
            #expect(MLX.max(abs(vector - served)).item(Float.self) < 2e-3)
            batched.append(served)
        }
        let pixels = (MLXRandom.uniform(low: 0, high: 1, [1, 28, 28, 3])).asType(.float16)
        let values = model.visionEncoder.encodeValueAttention(pixels)[0]
        let final = model.encodeImage(pixels).patchTokens[0]
        #expect(values.shape == [2, 2, 64])
        #expect(MLX.max(abs(values.reshaped(4, 64) - final)).item(Float.self) > 1e-2, "value features are not the final patch tokens")
        let similarity = matmul(l2Normalize(values.reshaped(4, 64).asType(.float32)), concatenated(batched, axis: 0).T).reshaped(2, 2, 3)
        let expected = try NativeTIPS.upsampledArgMax(similarity, height: 9, width: 6, cancellation: NativeCancellation())
        let actual = try NativeTIPS.segmentClasses(model, pixels: pixels, labels: labels, height: 9, width: 6, cancellation: NativeCancellation())
        #expect(actual == expected)
    }
}
