// SPDX-License-Identifier: MIT
import MLX
import MLXNN

/// NLC 1x1, stride-one, ungrouped convolution is exactly a per-position Linear.
/// Keep the Conv1d type so speaker-encoder ModuleInfo setters retain their declared contract.
public final class StoredVoicePointwise: Conv1d {
    public let scales: MLXArray
    public let biases: MLXArray
    public let bits: Int
    public let groupSize: Int
    public init(weight: MLXArray, bias: MLXArray?, scales: MLXArray, biases: MLXArray, inputs: Int, group: Int, bits: Int) {
        self.scales = scales; self.biases = biases; self.bits = bits; self.groupSize = group
        super.init(inputChannels: inputs, outputChannels: weight.dim(0), kernelSize: 1, bias: bias != nil)
        self.weight._updateInternal(weight)
        if let bias { self.bias?._updateInternal(bias) }
    }
    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = quantizedMatmul(x, weight, scales: scales, biases: biases, transpose: true, groupSize: groupSize, bits: bits)
        if let bias { result = result + bias }
        return result
    }
}
