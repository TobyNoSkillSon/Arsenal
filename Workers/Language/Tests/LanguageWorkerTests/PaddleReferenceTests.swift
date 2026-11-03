// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import Testing
import MLX
import MLXNN
import IntegrationKit
@testable import LanguageWorker

/// Small fixed tensors exercise the section/axis and patch-merge math from the pinned Paddle reference.
/// These compare graph components, not trained-model accuracy or half-precision quality.
struct PaddleReferenceTests {
    init() { MLXReady.ensure() }
    @Test(.heavy) func fixedTensorReference() throws {
        NativeRuntime.prepare()
        let values: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let rotated = PaddlePositions.apply(MLXArray(values).asType(.float16).reshaped(1, 1, 1, 8), positions: MLXArray([2, 3, 4]).reshaped(3, 1), base: 10000, sections: [1, 1, 2])
        // modeling_paddleocr_vl.py apply_multimodal_rotary_pos_emb: t/h/w sections,
        // inverse frequencies [1,.1,.01,.001], followed by NeoX rotate_half.
        let angles: [Float] = [2, 0.3, 0.04, 0.004, 2, 0.3, 0.04, 0.004]
        let expected: [Float] = (0..<8).map { index in
            let turned: Float = index < 4 ? -values[index + 4] : values[index - 4]
            return values[index] * Darwin.cosf(angles[index]) + turned * Darwin.sinf(angles[index])
        }
        let actual = rotated.asType(.float32).asArray(Float.self)
        #expect(actual.count == expected.count)
        #expect(zip(actual, expected).allSatisfy { abs($0 - $1) < 0.008 })
        let vision = PaddleOCRVLVisionConfig(hiddenSize: 2, numHiddenLayers: 1, numAttentionHeads: 1, imageSize: 28, intermediateSize: 4)
        let embeddings = PaddleVisionEmbeddings(vision)
        try embeddings.update(
            parameters: ModuleParameters.unflattened([
                "patch_embedding.weight": MLX.zeros([2, 14, 14, 3], dtype: .float16), "patch_embedding.bias": MLX.zeros([2], dtype: .float16),
                "position_embedding.weight": MLXArray([Float(1), 10, 2, 20, 3, 30, 4, 40]).reshaped(4, 2).asType(.float16),
                "packing_position_embedding.weight": MLX.zeros([32768, 2], dtype: .float16)
            ]), verify: .all)
        let interpolated = embeddings(MLX.zeros([1, 28, 56, 3], dtype: .float16)).asType(.float32).asArray(Float.self)
        // torch.interpolate bilinear with align_corners=False, exact output 2x4.
        let interpolationReference: [Float] = [1, 10, 1.25, 12.5, 1.75, 17.5, 2, 20, 3, 30, 3.25, 32.5, 3.75, 37.5, 4, 40]
        #expect(interpolated == interpolationReference)
        // Projector's reshape/transposition must merge adjacent 2x2 patches, not four sequential row patches.
        let config = PaddleOCRVLConfig(
            visionConfig: PaddleOCRVLVisionConfig(hiddenSize: 2, numHiddenLayers: 1, numAttentionHeads: 1, imageSize: 28, intermediateSize: 4),
            textConfig: PaddleOCRVLTextConfig(hiddenSize: 8))
        let projector = MultiModalProjector(config: config)
        let identity = MLX.eye(8).asType(.float16)
        try projector.update(
            parameters: ModuleParameters.unflattened([
                "pre_norm.weight": MLX.ones([2], dtype: .float16), "pre_norm.bias": MLX.zeros([2], dtype: .float16),
                "linear_1.weight": identity, "linear_1.bias": MLX.zeros([8], dtype: .float16),
                "linear_2.weight": identity, "linear_2.bias": MLX.zeros([8], dtype: .float16)
            ]), verify: .all)
        // Row-major 2x4 grid; alternate [-1,1] and [1,-1] patch channels.
        let input: [Float] = [-1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1]
        let output = projector(MLXArray(input).asType(.float16).reshaped(1, 2, 4, 2)).asType(.float32).asArray(Float.self)
        let indices = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
        let reference = indices.map { index -> Float in
            let normalized = input[index] / sqrt(1 + 1e-5)
            return 0.5 * normalized * (1 + Float(erf(Double(normalized) / sqrt(2))))
        }
        #expect(output.count == reference.count)
        #expect(zip(output, reference).allSatisfy { abs($0 - $1) < 0.003 })
    }
}
