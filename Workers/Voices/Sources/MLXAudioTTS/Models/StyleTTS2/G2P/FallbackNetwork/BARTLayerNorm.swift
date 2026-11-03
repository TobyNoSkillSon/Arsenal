import Foundation
import MLX
import MLXNN

final class BARTLayerNorm : LayerNorm {
  public init(dimensions: Int, weight: MLXArray, bias: MLXArray) {
    super.init(dimensions: dimensions)
    
    self.weight?._updateInternal(weight)
    self.bias?._updateInternal(bias)
  }
}
