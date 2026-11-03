// SPDX-License-Identifier: MIT
// VoxCPM2 local encoder (LocEnc): each latent patch becomes one LM embedding through a CLS token.
// From mlx-audio-swift e8450402 (MIT), Sources/MLXAudioTTS/Models/VoxCPM2/VoxCPMEncoder.swift; maker: modules/locenc.

import Foundation
import MLX
import MLXNN

final class VoxCPMLocEnc: Module {
    let hiddenSize: Int
    @ParameterInfo(key: "special_token") var specialToken: MLXArray
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo var encoder: VoxMiniCPMModel

    init(_ config: VoxCPM2LMConfig, inputDim: Int = 64) {
        hiddenSize = config.hiddenSize
        _specialToken.wrappedValue = MLXArray.zeros([1, 1, 1, config.hiddenSize])
        _inProj.wrappedValue = Linear(inputDim, config.hiddenSize, bias: true)
        _encoder.wrappedValue = VoxMiniCPMModel(config)
        super.init()
    }

    /// x: (B, T, P, D) latent patches → (B, T, H) CLS outputs.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), t = x.dim(1), p = x.dim(2)
        let projected = inProj(x.asType(inProj.weight.dtype))
        let cls = MLX.broadcast(specialToken.asType(projected.dtype), to: [b, t, 1, hiddenSize])
        let flat = MLX.concatenated([cls, projected], axis: 2).reshaped(b * t, p + 1, hiddenSize)
        let (out, _) = encoder(flat, causal: false)
        return out[0..., 0, 0...].reshaped(b, t, hiddenSize)
    }
}
