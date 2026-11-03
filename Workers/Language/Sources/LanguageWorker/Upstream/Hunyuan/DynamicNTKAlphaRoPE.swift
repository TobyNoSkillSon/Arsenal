import Foundation
import MLX
import MLXNN
public class DynamicNTKAlphaRoPE: Module, OffsetLayer, ArrayOffsetLayer {
    let dimensions: Int
    let traditional: Bool
    private let _freqs: MLXArray

    public init(
        dimensions: Int,
        base: Float = 10000,
        scalingAlpha: Float = 1.0,
        traditional: Bool = false
    ) {
        precondition(dimensions % 2 == 0, "Dimensions must be even")
        self.dimensions = dimensions
        self.traditional = traditional

        let adjustedBase = base * pow(scalingAlpha, Float(dimensions) / Float(dimensions - 2))
        self._freqs = pow(
            adjustedBase,
            MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / Float(dimensions))
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: dimensions, traditional: traditional, base: nil, scale: 1.0,
            offset: offset, freqs: _freqs)
    }

    public func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: dimensions, traditional: traditional, base: nil, scale: 1.0,
            offset: offset, freqs: _freqs)
    }
}
