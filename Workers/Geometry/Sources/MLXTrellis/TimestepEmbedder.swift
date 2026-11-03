// swiftlint:disable all
// Vendored inference/geometry source; see PORTED-FROM.md.
import Foundation
import MLX
import MLXNN

/// Sinusoidal timestep embedding + MLP — matches trellis2 TimestepEmbedder.
/// freq_dim=256, half=128, max_period=1e4; emb = cat([cos(args), sin(args)]);
/// then Linear(256→C) → SiLU → Linear(C→C). Input t is already 1000*t_norm.
public struct TimestepEmbedder {
    public let b0, b2: MLXArray
    public let w0, w2: Linear
    public let freqDim: Int

    public init(w0: Linear, b0: MLXArray, w2: Linear, b2: MLXArray, freqDim: Int = 256) {
        self.w0 = w0; self.b0 = b0; self.w2 = w2; self.b2 = b2; self.freqDim = freqDim
    }

    private func sinusoid(_ t: MLXArray) -> MLXArray {
        let half = freqDim / 2
        let idx = MLXArray(Array(0..<half).map { Float($0) }) // [half]
        let freqs = MLX.exp(idx * Float(-log(10000.0) / Double(half)))
        let args = t.reshaped([-1, 1]) * freqs.reshaped([1, -1]) // [B, half]
        return MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1) // [B, freqDim]
    }

    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        let e = sinusoid(t)
        var h = w0(e.asType(.float16)) + b0
        h = h * MLX.sigmoid(h) // SiLU
        return w2(h.asType(.float16)) + b2
    }
}

// swiftlint:enable all
