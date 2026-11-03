import Foundation
import MLX

// Two Fourier-feature variants, each mirroring Stability's reference op order
// exactly (fp32 rounding of arguments up to ~6e4 rad is visible otherwise):
//  * DiT timestep: `ExpoFourierFeatures` in dit_mlx_medium.py — linspace ramp,
//    freqs pre-multiplied by 2π, `t[:, None] * freqs`.
//  * seconds conditioner: `expo_fourier_features` in sa3_pipeline.py — arange/(half-1)
//    ramp, `t * freqs * 2 * π` evaluated left to right.
// The log constants are Python doubles rounded to fp32 when applied; π is
// `Float(Double.pi)` (nearest), not Swift's `Float.pi`, which rounds toward zero.

/// DiT timestep features ([B] → [B, dim]).
func expoFourierFeatures(_ t: MLXArray, dim: Int, minFreq: Float, maxFreq: Float) -> MLXArray {
    let half = dim / 2
    let logMin = log(Double(minFreq)), logMax = log(Double(maxFreq))
    let ramp = MLX.linspace(Float(0), Float(1), count: half)
    let freqs = MLX.exp(ramp * Float(logMax - logMin) + Float(logMin)) * Float(2) * Float(Double.pi)
    let args = t.asType(.float32).reshaped([t.size, 1]) * freqs.reshaped([1, half])
    return MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
}

/// Seconds-conditioner features ([B] → [B, dim]).
func expoFourierFeaturesSeconds(_ t: MLXArray, dim: Int, minFreq: Float = 0.5, maxFreq: Float = 10_000) -> MLXArray {
    let half = dim / 2
    let logMin = log(Double(minFreq)), logMax = log(Double(maxFreq))
    let ramp = MLXArray(0 ..< half).asType(.float32) / Float(max(half - 1, 1))
    let freqs = MLX.exp(ramp * Float(logMax - logMin) + Float(logMin))
    let args = t.asType(.float32).reshaped([-1, 1]) * freqs.reshaped([1, half]) * Float(2) * Float(Double.pi)
    return MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
}

/// `NumberConditioner({min_val:0, max_val:384, fourier_features_type:"expo"})`
/// → 768-d embedding: ExpoFourierFeatures(256) → Linear(256, 768) + bias.
public final class SecondsTotalEmbedder {
    public let weight: MLXArray    // (768, 256)
    public let bias: MLXArray      // (768,)
    public let scales: MLXArray?
    public let affineBiases: MLXArray?
    public let quantBits: Int?
    public let halfOperands: Bool
    public let minVal: Float
    public let maxVal: Float
    public let fourierDim: Int

    public init(weight: MLXArray, bias: MLXArray,
                minVal: Float = 0.0, maxVal: Float = 384.0, fourierDim: Int = 256,
                scales: MLXArray? = nil, affineBiases: MLXArray? = nil, quantBits: Int? = nil, halfOperands: Bool = false) {
        self.weight = weight
        self.bias = bias
        self.scales = scales; self.affineBiases = affineBiases; self.quantBits = quantBits; self.halfOperands = halfOperands
        self.minVal = minVal
        self.maxVal = maxVal
        self.fourierDim = fourierDim
    }

    public func callAsFunction(_ seconds: Float) -> MLXArray {
        let clipped = max(minVal, min(maxVal, seconds))
        let norm = (clipped - minVal) / (maxVal - minVal)
        let t = MLXArray([norm])
        let ff = expoFourierFeaturesSeconds(t, dim: fourierDim)
        let out: MLXArray
        if let scales, let affineBiases, let quantBits {
            out = MLX.quantizedMatmul(ff.asType(scales.dtype), weight, scales: scales, biases: affineBiases, transpose: true, groupSize: 64, bits: quantBits) + bias
        } else if halfOperands {
            out = MLX.matmul(ff.asType(weight.dtype), weight.transposed()) + bias
        } else {
            // Preserve Medium's original arithmetic exactly; the half/packed conditioner is Small SFX only.
            out = MLX.matmul(ff, weight.transposed()) + bias
        }
        return out.expandedDimensions(axis: 1)                                  // [1, 1, 768]
    }
}

/// Replace padded T5Gemma positions with the learned padding embedding.
/// `embeds` [B,S,768] · `mask` [B,S] int32 · `paddingEmbedding` [768].
public func applyPromptPadding(embeds: MLXArray, mask: MLXArray, paddingEmbedding: MLXArray) -> MLXArray {
    let m = mask.asType(embeds.dtype).expandedDimensions(axis: -1)              // [B,S,1]
    let pe = paddingEmbedding.asType(embeds.dtype).reshaped([1, 1, -1])
    return embeds * m + pe * (MLXArray(Float(1.0)).asType(embeds.dtype) - m)
}
