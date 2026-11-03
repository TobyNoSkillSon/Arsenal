import Foundation
import MLX
import MLXNN

nonisolated final class BARTDecoderLayer: Module {
  @ModuleInfo var selfAttn: BARTMultiHeadAttention
  @ModuleInfo var selfAttnNorm: LayerNorm
  @ModuleInfo var crossAttn: BARTMultiHeadAttention
  @ModuleInfo var crossAttnNorm: LayerNorm
  @ModuleInfo var ffn: FeedForward
  @ModuleInfo var ffnNorm: LayerNorm

  init(dModel: Int, numHeads: Int, dFF: Int, modelKey: String, weights: BARTWeightView) {
    self._selfAttn = ModuleInfo(wrappedValue: BARTMultiHeadAttention(dModel: dModel, numHeads: numHeads, modelKey: modelKey + ".self_attn", weights: weights))
    self._selfAttnNorm = ModuleInfo(wrappedValue: BARTLayerNorm(
      dimensions: dModel,
      weight: weights[modelKey + ".self_attn_layer_norm.weight"]!,
      bias: weights[modelKey + ".self_attn_layer_norm.bias"]!))
    
    self._crossAttn = ModuleInfo(wrappedValue: BARTMultiHeadAttention(dModel: dModel, numHeads: numHeads, modelKey: modelKey + ".encoder_attn", weights: weights))
    self._crossAttnNorm = ModuleInfo(wrappedValue: BARTLayerNorm(
      dimensions: dModel,
      weight: weights[modelKey + ".encoder_attn_layer_norm.weight"]!,
      bias: weights[modelKey + ".encoder_attn_layer_norm.bias"]!))
    
    self._ffn = ModuleInfo(wrappedValue: FeedForward(linear1: weights.linear(modelKey + ".fc1"), linear2: weights.linear(modelKey + ".fc2")))
    
    self._ffnNorm = ModuleInfo(wrappedValue: BARTLayerNorm(
      dimensions: dModel,
      weight: weights[modelKey + ".final_layer_norm.weight"]!,
      bias: weights[modelKey + ".final_layer_norm.bias"]!))
    
    super.init()
  }
    
    func callAsFunction(_ x: MLXArray, encoderOutput: MLXArray, selfMask: MLXArray? = nil, crossMask: MLXArray? = nil) -> MLXArray {
      // Self-attention with residual
      let attnOutput = selfAttn(x, mask: selfMask)
      var output = selfAttnNorm(x + attnOutput)
            
      // Cross-attention with residual
      let crossOutput = crossAttn(output, key: encoderOutput, value: encoderOutput, mask: crossMask)
      output = crossAttnNorm(output + crossOutput)
                  
      // Feed-forward with residual
      let ffnOutput = ffn(output)
      output = ffnNorm(output + ffnOutput)
      
      return output
    }
}
