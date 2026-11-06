// SPDX-License-Identifier: MIT
import Foundation
import Testing
import MLX
import IntegrationKit
import ModelPrecision
import ArsenalWire
import PerceptionSupport
import PerceptionWorker
@testable import SAM31

/// 22 Oct: Get of sam3.1 at FP16 failed with "FP16 non-finite tensor: detector_model.text_encoder.text_projection.weight".
/// The FP32 weights hold -9.58e18 at [6, 0] and [10, 0] of that unused CLIP projection.
@Suite(.serialized) struct SAM31ConversionTests {
    init() { if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) } }
    static let unused = "detector_model.text_encoder.text_projection.weight"

    @Test func theUnusedProjectionIsOutsideTheGraph() throws {
        #expect(NativeSAM31.unusedTensors == [Self.unused])
        #expect(try NativeSAM31.map(Self.unused, MLXArray.zeros([512, 1024])) == nil)
        let parameters = SAM31Root(NativeSAM31.smallConfig()).parameters().flattened().map(\.0)
        #expect(!parameters.contains { $0.hasPrefix("detector_model.text_encoder.text_projection") })
        #expect(parameters.contains("detector_model.text_projection.weight"))
        let registration = try #require(PerceptionWorker.registrations.first { $0.modelTypes.contains("sam3.1_video") })
        #expect(registration.conversion?.recipe == "cast:sam31-fp16-v3" && registration.conversion?.half == .float16)
    }

    /// 22 Oct: at FP16 every text prompt found nothing. The padding mask's -1e9 is -inf in FP16, and a kept
    /// position's 0 × -inf made every text embedding NaN.
    @Test(.heavy) func fp16TextEmbeddingsWithPaddingAreFinite() throws {
        var config = NativeSAM31.smallConfig().detectorConfig.textConfig
        config.vocabSize = 64
        let model = CLIPTextModel(config)
        try NativeParameters.initializeRandom(model, half: .float16)
        let ids = MLXArray((0..<32).map { Int32($0 % 64) }, [1, 32])
        let mask = MLXArray((0..<32).map { Int32($0 < 5 ? 1 : 0) }, [1, 32])
        let hidden = model(ids, attentionMask: mask)
        #expect(hidden.dtype == .float16 && hidden.shape == [1, 32, config.hiddenSize])
        #expect(MLX.all(MLX.isFinite(hidden)).item(Bool.self))
    }
    // Get dropping the out-of-range projection from Meta's checkpoint: SAM31OfficialCheckpointTests.
}
