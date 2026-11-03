// SPDX-License-Identifier: MIT
// VoxCPM2 local diffusion transformer (LocDiT v2) and its Euler flow-matching sampler with CFG-Zero*.
// From mlx-audio-swift e8450402 (MIT), Sources/MLXAudioTTS/Models/VoxCPM2/VoxCPMDiT.swift; checked against the maker's
// modules/locdit/local_dit_v2.py and unified_cfm.py. Changes: activations stay in the weight dtype (the sinusoidal
// time embedding is computed in float32 and cast, as the maker casts it to x.dtype), the noise can be injected for
// fixed-noise parity, and the sway-scheduled time grid is built on the host in float32.

import Foundation
import MLX
import MLXNN
import MLXRandom

final class VoxTimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ dim: Int) {
        _linear1.wrappedValue = Linear(dim, dim)
        _linear2.wrappedValue = Linear(dim, dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

/// sin/cos time embedding of t (N,), scale 1000, as the maker's SinusoidalPosEmb (float32 here).
func voxTimeEmbedding(_ t: MLXArray, dim: Int, scale: Float = 1000) -> MLXArray {
    let half = dim / 2
    let step = Foundation.log(Float(10000)) / Float(half - 1)
    let freqs = MLX.exp(MLXArray(0..<half).asType(.float32) * -step)
    let args = scale * t.asType(.float32).reshaped(-1, 1) * freqs.expandedDimensions(axis: 0)
    return MLX.concatenated([MLX.sin(args), MLX.cos(args)], axis: -1)
}

final class VoxCPMLocDiT: Module {
    let hiddenSize: Int
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "cond_proj") var condProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "time_mlp") var timeMlp: VoxTimestepEmbedding
    @ModuleInfo(key: "delta_time_mlp") var deltaTimeMlp: VoxTimestepEmbedding
    @ModuleInfo var decoder: VoxMiniCPMModel

    init(_ config: VoxCPM2LMConfig, inChannels: Int = 64) {
        hiddenSize = config.hiddenSize
        _inProj.wrappedValue = Linear(inChannels, config.hiddenSize)
        _condProj.wrappedValue = Linear(inChannels, config.hiddenSize)
        _outProj.wrappedValue = Linear(config.hiddenSize, inChannels)
        _timeMlp.wrappedValue = VoxTimestepEmbedding(config.hiddenSize)
        _deltaTimeMlp.wrappedValue = VoxTimestepEmbedding(config.hiddenSize)
        _decoder.wrappedValue = VoxMiniCPMModel(config)
        super.init()
    }

    /// x (N, C, T) noisy patch, mu (N, 2H) LM conditioning, t and dt (N,), cond (N, C, T') prefix patch → velocity (N, C, T).
    func callAsFunction(_ x: MLXArray, mu: MLXArray, t: MLXArray, cond: MLXArray, dt: MLXArray) -> MLXArray {
        let dtype = inProj.weight.dtype
        let xp = inProj(x.asType(dtype).transposed(0, 2, 1))
        let cp = condProj(cond.asType(dtype).transposed(0, 2, 1))
        let time = timeMlp(voxTimeEmbedding(t, dim: hiddenSize).asType(dtype)) + deltaTimeMlp(voxTimeEmbedding(dt, dim: hiddenSize).asType(dtype))
        let muTokens = mu.asType(dtype).reshaped(x.dim(0), -1, hiddenSize)
        let hidden = MLX.concatenated([muTokens, time.expandedDimensions(axis: 1), cp, xp], axis: 1)
        let (decoded, _) = decoder(hidden, causal: false)
        let skip = muTokens.dim(1) + 1 + cp.dim(1)
        return outProj(decoded[0..., skip..., 0...]).transposed(0, 2, 1)
    }
}

final class VoxUnifiedCFM: Module {
    let inChannels: Int
    let meanMode: Bool
    @ModuleInfo(key: "estimator") var estimator: VoxCPMLocDiT

    init(inChannels: Int, estimator: VoxCPMLocDiT, meanMode: Bool = false) {
        self.inChannels = inChannels
        self.meanMode = meanMode
        _estimator.wrappedValue = estimator
        super.init()
    }

    /// The maker's time grid: linspace(1, 0, n + 1) with sway sampling (coefficient 1).
    static func timeSpan(steps: Int) -> [Float] {
        (0...steps).map { i in
            let t = 1 - Float(i) / Float(steps)
            return t + (Foundation.cos(Float.pi / 2 * t) - 1 + t)
        }
    }

    /// Euler solve from noise `z` (B, C, P). CFG-Zero*: the first max(1, ⌊0.04·(n+1)⌋) steps take zero velocity.
    func solveEuler(_ z: MLXArray, span: [Float], mu: MLXArray, cond: MLXArray, cfgValue: Float) -> MLXArray {
        var x = z
        var t = span[0], dt = span[0] - span[1]
        let zeroInit = max(1, Int(Float(span.count) * 0.04))
        let b = x.dim(0)
        let muIn = MLX.concatenated([mu, MLXArray.zeros(like: mu)], axis: 0)
        let condIn = MLX.concatenated([cond, cond], axis: 0)
        for step in 1..<span.count {
            if step > zeroInit {
                let xIn = MLX.concatenated([x, x], axis: 0)
                let tIn = MLXArray([t, t])
                let dtIn = meanMode ? MLXArray([dt, dt]) : MLXArray.zeros([2])
                let out = estimator(xIn, mu: muIn, t: tIn, cond: condIn, dt: dtIn)
                let pos = out[..<b], neg = out[b...]
                let posFlat = pos.reshaped(b, -1).asType(.float32), negFlat = neg.reshaped(b, -1).asType(.float32)
                let st = ((posFlat * negFlat).sum(axis: 1, keepDims: true) / ((negFlat * negFlat).sum(axis: 1, keepDims: true) + 1e-8))
                    .reshaped(b, 1, 1).asType(pos.dtype)
                let velocity = neg * st + cfgValue * (pos - neg * st)
                x -= dt * velocity
            }
            t -= dt
            if step < span.count - 1 { dt = t - span[step + 1] }
        }
        return x
    }

    /// One patch: noise drawn here unless `noise` is given (fixed-noise parity).
    func sample(mu: MLXArray, steps: Int, patchSize: Int, cond: MLXArray, cfgValue: Float, noise: MLXArray? = nil) -> MLXArray {
        let dtype = estimator.inProj.weight.dtype
        let z = (noise ?? MLXRandom.normal([mu.dim(0), inChannels, patchSize])).asType(dtype)
        return solveEuler(z, span: Self.timeSpan(steps: steps), mu: mu.asType(dtype), cond: cond.asType(dtype), cfgValue: cfgValue)
    }
}
