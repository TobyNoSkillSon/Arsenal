import Foundation
import MLX
import MLXNN

nonisolated final class BARTMultiHeadAttention: Module {
    let numHeads: Int
    let dModel: Int
    let headDim: Int
    
    @ModuleInfo var qProj: Linear
    @ModuleInfo var kProj: Linear
    @ModuleInfo var vProj: Linear
    @ModuleInfo var outProj: Linear
    
    init(dModel: Int, numHeads: Int, modelKey: String, weights: BARTWeightView) {
      self.dModel = dModel
      self.numHeads = numHeads
      self.headDim = dModel / numHeads
      
      self._qProj = ModuleInfo(wrappedValue: weights.linear(modelKey + ".q_proj"))
      self._kProj = ModuleInfo(wrappedValue: weights.linear(modelKey + ".k_proj"))
      self._vProj = ModuleInfo(wrappedValue: weights.linear(modelKey + ".v_proj"))
      self._outProj = ModuleInfo(wrappedValue: weights.linear(modelKey + ".out_proj"))
        
      super.init()
    }
    
    func callAsFunction(_ query: MLXArray, key: MLXArray? = nil, value: MLXArray? = nil, mask: MLXArray? = nil) -> MLXArray {
      let key = key ?? query
      let value = value ?? query
            
      let batchSize = query.shape[0]
      let seqLen = query.shape[1]
        
      // Project and reshape
      let q = qProj(query).reshaped([batchSize, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)
      let k = kProj(key).reshaped([batchSize, -1, numHeads, headDim]).transposed(0, 2, 1, 3)
      let v = vProj(value).reshaped([batchSize, -1, numHeads, headDim]).transposed(0, 2, 1, 3)
        
      // Scaled dot-product attention
      let scale = Float(1.0 / sqrt(Double(headDim)))
      var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        
      if let mask {
          scores = scores + mask
      }
        
      let attnWeights = softmax(scores, axis: -1)
      let attnOutput = matmul(attnWeights, v)
      
      // Reshape and project
      let output = attnOutput.transposed(0, 2, 1, 3).reshaped([batchSize, seqLen, dModel])
      return outProj(output)
    }
}
