import Foundation
import MLX
import MLXNN

nonisolated final class FeedForward: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    
  init(linear1: Linear, linear2: Linear) {
    self._linear1 = ModuleInfo(wrappedValue: linear1)
    self._linear2 = ModuleInfo(wrappedValue: linear2)
    super.init()
  }
    
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var output = linear1(x)
    output = gelu(output)
    output = linear2(output)
    return output
  }
}
