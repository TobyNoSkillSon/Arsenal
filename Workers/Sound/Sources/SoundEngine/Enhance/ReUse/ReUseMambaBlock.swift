// SPDX-License-Identifier: Apache-2.0
// Vendored from faraday/SpeechLens Sources/Inference (Apache-2.0)
// @ d20e11bcafbc32435ef18a17ed56caf22fca6833; Vireo changes listed in CHANGES.md.

import Foundation
import MLX
import MLXNN

class ReUseBidirectionalMamba: Module, UnaryLayer {
    let forward: ReUseMamba
    let backward: ReUseMamba
    @ModuleInfo var outputProj: Linear   // Vireo: @ModuleInfo for converted 8/4-bit weights
    let norm: LayerNorm
    
    init(dModel: Int, mode: ReUseScanMode = .standard, kernels: ReUseKernels = .metal) {
        self.forward = ReUseMamba(dModel: dModel, mode: mode, kernels: kernels)
        self.backward = ReUseMamba(dModel: dModel, mode: mode, kernels: kernels)
        self._outputProj.wrappedValue = Linear(2 * dModel, dModel)
        self.norm = LayerNorm(dimensions: dModel)
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // PyTorch MambaBlock.forward:
        //   out_fw = forward_blocks(x) + x
        //   out_bw = backward_blocks(flip(x)) + flip(x)
        //   out_bw = flip(out_bw)
        //   return norm(output_proj(cat([out_fw, out_bw], -1)))

        // Reverse along axis 1 using a negative-stride view (zero-copy).
        let reversedX = x[0..., .stride(by: -1), 0...]

        let outFw = forward(x) + x
        let outBw = backward(reversedX) + reversedX
        let outBwReversed = outBw[0..., .stride(by: -1), 0...]

        let combined = MLX.concatenated([outFw, outBwReversed], axis: -1)
        return norm(outputProj(combined))
    }
}

class ReUseTFMambaBlock: Module, UnaryLayer {
    let timeMamba: ReUseBidirectionalMamba
    let freqMamba: ReUseBidirectionalMamba
    
    init(hidFeature: Int, mode: ReUseScanMode = .standard, kernels: ReUseKernels = .metal) {
        self.timeMamba = ReUseBidirectionalMamba(dModel: hidFeature, mode: mode, kernels: kernels)
        self.freqMamba = ReUseBidirectionalMamba(dModel: hidFeature, mode: mode, kernels: kernels)
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        intakeForward(x, session: nil, prefix: "")
    }

    func intakeForward(_ x: MLXArray, session: ReUseIntakeSession?, prefix: String) -> MLXArray {
        // x: [B, T, F, C] (MLX convention)
        let (b, t, f, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        
        // Treat each frequency bin as an independent time sequence.
        var y = x.transposed(0, 2, 1, 3).reshaped(b * f, t, c)
        y = session?.input(prefix + "time", y) ?? y
        y = timeMamba(y) + y
        session?.output(prefix + "time", y)
        
        // Treat each time step as an independent frequency sequence.
        y = y.reshaped(b, f, t, c).transposed(0, 2, 1, 3).reshaped(b * t, f, c)
        y = session?.input(prefix + "frequency", y) ?? y
        y = freqMamba(y) + y
        session?.output(prefix + "frequency", y)
        
        // Back to [B, T, F, C]
        y = y.reshaped(b, t, f, c)
        return y
    }
}
