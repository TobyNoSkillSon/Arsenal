// SPDX-License-Identifier: MIT
import Foundation
import CoreImage
import Testing
import MLX
import ArsenalWire
import IntegrationKit
@testable import LanguageWorker

/// Gemma 4 image path against Transformers image_processing_gemma4.py / modeling_gemma4.py. Sizes were evaluated once
/// with get_aspect_ratio_preserving_size (patch 16, pooling 3, 2,520 patches unless noted).
struct GemmaProcessorTests {
    init() { MLXReady.ensure() }
    private func processor(maxSoftTokens: Int = 280) throws -> GemmaVisionProcessor {
        let config: JSONObject = ["patch_size": 16, "pooling_kernel_size": 3, "max_soft_tokens": .number(Double(maxSoftTokens))]
        return GemmaVisionProcessor(
            try JSONDecoder().decode(GemmaVisionConfiguration.self, from: WireJSON.encode(config)),
            tokenizer: RandomTokenizer(imageID: 258_880, startID: 261, endID: 262, imageToken: "<|image_pad|>"), imageTokenID: 258_880,
            boiTokenID: 255_999, eoiTokenID: 258_882)
    }

    @Test func aspectPreservingSizesMatchTransformers() throws {
        // (height, width) → (height, width, soft tokens)
        let cases: [([Int], [Int])] = [
            ([800, 800], [768, 768, 256]), ([1080, 1920], [576, 1056, 264]), ([3024, 4032], [672, 912, 266]), ([100, 60], [1008, 576, 252]),
            ([20, 2000], [48, 8016, 167]), ([1170, 2532], [528, 1152, 264]), ([48, 48], [768, 768, 256])
        ]
        for (input, expected) in cases {
            let size = try GemmaVisionProcessor.targetSize(height: input[0], width: input[1], patchSize: 16, maxPatches: 2520, poolingKernelSize: 3)
            #expect([size.height, size.width, (size.height / 16) * (size.width / 16) / 9] == expected, "\(input)")
        }
        // A 140-token budget (1,260 patches), as a folder max_pixels would select.
        let small = try GemmaVisionProcessor.targetSize(height: 1080, width: 1920, patchSize: 16, maxPatches: 1260, poolingKernelSize: 3)
        #expect(small.height == 384 && small.width == 720)
        var limited: JSONObject = ["patch_size": 16, "pooling_kernel_size": 3, "max_soft_tokens": 280]
        try LanguageNative.applyPixelLimits(&limited, spec: ["max_pixels": 400_000], architecture: "gemma4")
        #expect(limited["max_soft_tokens"]?.number == 140)
    }

    @Test(.heavy) func encoderInputAndPromptExpansion() throws {
        NativeRuntime.prepare()
        let gemma = try processor()
        let image = CIImage(color: CIColor(red: 1, green: 0, blue: 1)).cropped(to: CGRect(x: 0, y: 0, width: 1920, height: 1080))
        let input = try gemma.encoderInput(image)
        #expect(input.rows == 36 && input.columns == 66)
        #expect(input.patches.shape == [36 * 66, 768] && input.positions.shape == [36 * 66, 2])
        // Row-major (x, y): the second patch is x=1, the first of the second row is (0, 1).
        let positions = input.positions.asArray(Int32.self)
        #expect(Array(positions.prefix(4)) == [0, 0, 1, 0] && positions[66 * 2] == 0 && positions[66 * 2 + 1] == 1)
        // Pixels are rescaled to [0, 1]: magenta is (1, 0, 1) in every patch (c fastest).
        let first = input.patches[0].asArray(Float.self)
        #expect(abs(first[0] - 1) < 0.01 && abs(first[1]) < 0.01 && abs(first[2] - 1) < 0.01)
        let expanded = try gemma.expand([5, 258_880, 6], softTokens: [264])
        #expect(expanded.count == 2 + 2 + 264)
        #expect(expanded[1] == 255_999 && expanded[2] == 258_880 && expanded[expanded.count - 2] == 258_882 && expanded.last == 6)
        #expect(throws: (any Error).self) { try gemma.expand([258_880, 258_880], softTokens: [264]) }
    }

    /// HF _avg_pool_by_positions: one-hot kernel index x//k + (columns//k)·(y//k), mean of k², then ×sqrt(hidden) in float32.
    @Test(.heavy) func poolerMatchesPositionAveragingInFloat32() {
        NativeRuntime.prepare()
        let (rows, columns, width, k) = (6, 9, 2, 3)
        var values = [Float]()
        for index in 0..<(rows * columns) { values += [Float(index), Float(index % 7) - 3] }
        let pooled = ArsenalGemma4Pooling.pool(MLXArray(values, [1, rows * columns, width]), columns: columns, rows: rows, kernel: k, scale: 2)
        var expected = [Float](repeating: 0, count: (rows / k) * (columns / k) * width)
        for y in 0..<rows {
            for x in 0..<columns {
                let target = x / k + (columns / k) * (y / k)
                for c in 0..<width { expected[target * width + c] += values[(y * columns + x) * width + c] / Float(k * k) * 2 }
            }
        }
        #expect(pooled.shape == [1, 6, 2])
        #expect(zip(pooled.asArray(Float.self), expected).allSatisfy { abs($0 - $1) < 1e-4 })
        // The sqrt(768) scaling of a 3,000 activation is 83,138: beyond float16, kept finite in float32.
        let large = ArsenalGemma4Pooling.pool(
            MLX.full([1, 9, 1], values: MLXArray(Float(3000))).asType(.float16), columns: 3, rows: 3, kernel: 3, scale: Float(768).squareRoot())
        #expect(large.dtype == .float32 && abs(large.item(Float.self) - 83_138.4) < 1)
    }

    @Test(.heavy) func clippingBoundsAreClampedNotRefused() {
        NativeRuntime.prepare()
        let bound = LanguageNative.boundedClipping(
            "model.vision_tower.encoder.layers.0.mlp.up_proj.output_max", MLXArray(Float.infinity).asType(.bfloat16)
        ).asType(.float16)
        #expect(bound.asType(.float32).item(Float.self) == 65504)
        let lower = LanguageNative.boundedClipping("vision_tower.encoder.layers.0.self_attn.q_proj.input_min", MLXArray(-Float.infinity))
        #expect(lower.asType(.float16).asType(.float32).item(Float.self) == -65504)
        // Ordinary weights pass through untouched (a non-finite weight still fails the loader's check).
        let weight = LanguageNative.boundedClipping("vision_tower.encoder.layers.0.mlp.up_proj.linear.weight", MLXArray(Float.infinity))
        #expect(weight.item(Float.self).isInfinite)
    }
}
