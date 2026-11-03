import Foundation
import MLX
import MLXNN

class PaddleOCRVLModel: Module {
    @ModuleInfo(key: "visual") var visionModel: NaViTVisionEncoder
    @ModuleInfo(key: "mlp_AR") var projector: MultiModalProjector
    @ModuleInfo(key: "model") var languageModel: ERNIEModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let config: PaddleOCRVLConfig

    init(config: PaddleOCRVLConfig) {
        self.config = config

        self._visionModel.wrappedValue = NaViTVisionEncoder(config: config.visionConfig)
        self._projector.wrappedValue = MultiModalProjector(config: config)
        self._languageModel.wrappedValue = ERNIEModelInner(config: config.textConfig)
        self._lmHead.wrappedValue = Linear(
            config.textConfig.hiddenSize,
            config.textConfig.vocabSize,
            bias: false
        )

        super.init()
    }

    func getImageFeatures(_ pixelValues: MLXArray) -> MLXArray {
        let visionFeatures = visionModel.getImageFeatures(pixelValues)
        return projector(visionFeatures)
    }

    func mergeInputIdsWithImageFeatures(
        inputIds: MLXArray,
        imageFeatures: MLXArray
    ) -> MLXArray {
        let inputsEmbeds = languageModel.getEmbedding(inputIds)
        let imageTokenMask = inputIds .== config.visionTokenId

        return mergeEmbeddings(
            inputsEmbeds: inputsEmbeds,
            imageFeatures: imageFeatures,
            mask: imageTokenMask
        )
    }

    private func mergeEmbeddings(
        inputsEmbeds: MLXArray,
        imageFeatures: MLXArray,
        mask: MLXArray
    ) -> MLXArray {
        let seqLen = inputsEmbeds.dim(1)
        let numImageTokens = imageFeatures.dim(1)

        let maskInt = mask.asType(.int32)
        let firstTrueIdxArray = argMax(maskInt, axis: 1)
        let firstTrueIdx = firstTrueIdxArray[0].item(Int.self)

        let prePad = firstTrueIdx
        let postPad = seqLen - firstTrueIdx - numImageTokens

        var alignedImageFeatures = imageFeatures
        if prePad > 0 || postPad > 0 {
            let paddingWidths: [IntOrPair] = [[0, 0], [prePad, max(0, postPad)], [0, 0]]
            alignedImageFeatures = padded(imageFeatures, widths: paddingWidths)
        }

        if alignedImageFeatures.dim(1) > seqLen {
            alignedImageFeatures = alignedImageFeatures[0..., 0..<seqLen, 0...]
        } else if alignedImageFeatures.dim(1) < seqLen {
            let extraPad = seqLen - alignedImageFeatures.dim(1)
            let extraPadWidths: [IntOrPair] = [[0, 0], [0, extraPad], [0, 0]]
            alignedImageFeatures = padded(alignedImageFeatures, widths: extraPadWidths)
        }

        let expandedMask = mask.expandedDimensions(axis: -1)

        eval(expandedMask)
        eval(alignedImageFeatures)
        eval(inputsEmbeds)

        return MLX.which(expandedMask, alignedImageFeatures, inputsEmbeds)
    }

    func forward(
        inputIds: MLXArray,
        pixelValues: MLXArray?,
        cache: [PaddleKVCache]?,
        positions: MLXArray? = nil
    ) -> MLXArray {
        var inputsEmbeds: MLXArray

        if let pixelValues = pixelValues {
            let imageFeatures = getImageFeatures(pixelValues)
            eval(imageFeatures)
            inputsEmbeds = mergeInputIdsWithImageFeatures(
                inputIds: inputIds,
                imageFeatures: imageFeatures
            )
        } else {
            inputsEmbeds = languageModel.getEmbedding(inputIds)
        }

        let hiddenStates = languageModel.forward(inputsEmbeds, cache: cache, positions: positions)
        return lmHead(hiddenStates)
    }

    func forwardGeneration(
        inputIds: MLXArray,
        cache: [PaddleKVCache]?,
        positions: MLXArray? = nil
    ) -> MLXArray {
        let embeds = languageModel.getEmbedding(inputIds)
        let hiddenStates = languageModel.forward(embeds, cache: cache, positions: positions)
        return lmHead(hiddenStates)
    }

    func newCache() -> [PaddleKVCache] {
        (0..<config.textConfig.numHiddenLayers).map { _ in PaddleKVCache() }
    }
}

extension PaddleOCRVLModel {
    func mapCheckpoint(_ name: String, _ tensor: MLXArray) -> (String, MLXArray)? {
        // PyTorch Conv2d OIHW -> MLX OHWI. Every other 1.6 key is native.
        if name == "visual.vision_model.embeddings.patch_embedding.weight" {
            return (name, tensor.transposed(0, 2, 3, 1))
        }
        return (name, tensor)
    }
}
