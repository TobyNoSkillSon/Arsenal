// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire
import IntegrationKit
import MLX
import MLXNN
import MLXLMCommon

final class OvisLanguageContainer: Module {
    @ModuleInfo var model: Qwen35Language.Model
    init(_ config: Qwen35Configuration) { self.model = Qwen35Language.Model(config.textConfiguration) }
}
final class OvisGraph: Module {
    @ModuleInfo(key: "language_model") var language: OvisLanguageContainer
    @ModuleInfo(key: "vision_tower") var vision: OvisVision.VisionModel
    let config: Qwen35Configuration
    init(_ config: Qwen35Configuration) {
        self.config = config
        _language.wrappedValue = OvisLanguageContainer(config)
        _vision.wrappedValue = OvisVision.VisionModel(config.visionConfiguration)
    }
    func embed(ids: [Int], pixels: MLXArray?, frames: [THW], imageFrames: [THW], videoFrames: [THW], cancellation: NativeCancellation) throws -> MLXArray {
        try cancellation.check()
        let tokens = MLXArray(ids).reshaped(1, -1)
        let embeds = language.model.embedTokens(tokens)
        if let pixels {
            let (features, _) = try vision(pixels.asType(embeds.dtype), gridTHW: frames, cancellation: cancellation)
            let positions = ids.indices.filter { ids[$0] == config.videoTokenId || ids[$0] == config.imageTokenId }
            guard positions.count == features.dim(0) else { throw WorkerFailure(code: "invalid_request", message: "Ovis visual token count mismatch.") }
            embeds[0, MLXArray(positions)] = features.asType(embeds.dtype)
        }
        let (positions, _) = try OvisPositions.getRopeIndex(
            inputIds: tokens, imageGridTHW: imageFrames.isEmpty ? nil : imageFrames, videoGridTHW: videoFrames.isEmpty ? nil : videoFrames,
            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
            imageTokenId: config.imageTokenId, videoTokenId: config.videoTokenId, visionStartTokenId: config.visionStartTokenId)
        let hidden = try language.model(tokens, inputsEmbeds: embeds, positionIds: positions, cancellation: cancellation)
        return hidden[0, ids.count - 1]
    }
}
