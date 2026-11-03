// SPDX-License-Identifier: MIT
import MLX
import MLXNN

final class MultiModalProjector: Module {
    @ModuleInfo(key: "pre_norm") var norm: LayerNorm
    @ModuleInfo(key: "linear_1") var first: Linear
    @ModuleInfo(key: "linear_2") var second: Linear
    init(config: PaddleOCRVLConfig) {
        let size = config.visionConfig.hiddenSize
        _norm.wrappedValue = LayerNorm(dimensions: size, eps: 1e-5)
        _first.wrappedValue = Linear(4 * size, 4 * size)
        _second.wrappedValue = Linear(4 * size, config.textConfig.hiddenSize)
    }
    func callAsFunction(_ features: MLXArray) -> MLXArray {
        let height = features.dim(1), width = features.dim(2), size = features.dim(3)
        let merged = norm(features).reshaped(1, height / 2, 2, width / 2, 2, size).transposed(0, 1, 3, 2, 4, 5).reshaped(1, height * width / 4, 4 * size)
        return second(gelu(first(merged)))
    }
}
