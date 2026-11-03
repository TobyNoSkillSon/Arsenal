// SPDX-License-Identifier: Apache-2.0
// Vendored from faraday/SpeechLens Sources/Inference (Apache-2.0)
// @ d20e11bcafbc32435ef18a17ed56caf22fca6833; Vireo changes listed in CHANGES.md.

import Foundation
import MLX
import MLXNN

let reUseMambaStateSize = 16

class ReUseMamba: Module, UnaryLayer {
    // Vireo: inProj/outProj are @ModuleInfo so converted 8/4-bit weights can install QuantizedLinear layers
    // (QuantRules.reUse); keys are unchanged (the property names).
    @ModuleInfo var inProj: Linear
    let conv1d: Conv1d
    @ModuleInfo var xProj: Linear
    let dtProj: Linear
    
    // These stored-property names are part of the converted weight-key contract.
    var A_log: MLXArray
    var D: MLXArray
    @ModuleInfo var outProj: Linear
    let mode: ReUseScanMode
    let kernels: ReUseKernels

    init(dModel: Int, dConv: Int = 4, expand: Int = 4, mode: ReUseScanMode = .standard, kernels: ReUseKernels = .metal) {
        self.mode = mode
        self.kernels = kernels
        let dInner = Int(expand * dModel)
        let dtRank = Int(ceil(Double(dModel) / 16.0))
        
        self._inProj.wrappedValue = Linear(dModel, dInner * 2, bias: false)
        self.conv1d = Conv1d(inputChannels: dInner, outputChannels: dInner, kernelSize: dConv, padding: dConv - 1, groups: dInner, bias: true)
        self._xProj.wrappedValue = Linear(dInner, dtRank + reUseMambaStateSize * 2, bias: false)
        self.dtProj = Linear(dtRank, dInner, bias: true)
        
        self.A_log = MLX.zeros([dInner, reUseMambaStateSize])
        self.D = MLX.ones([dInner])
        self._outProj.wrappedValue = Linear(dInner, dModel, bias: false)
        
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let seqLen = x.shape[1]
        
        let xz = inProj(x)
        let split = xz.split(indices: [xz.shape[2] / 2], axis: -1)
        var innerX = split[0]
        let innerZ = split[1]
        
        // Conv1d — custom Metal kernel for the causal depthwise shape (one dispatch /
        // one graph node). Falls back to the stock path on failure or when
        // SPEECHLENS_DISABLE_METAL_CONV1D=1. See docs/kernel-optimization.md.
        if kernels == .metal, ReUseMetalCausalConv1d.isEnabled, let convBias = conv1d.bias {
            do {
                innerX = try ReUseMetalCausalConv1d.run(
                    x: innerX,
                    weight: conv1d.weight,
                    bias: convBias
                )
            } catch {
                ReUseLog.mamba.warning(
                    "ReUseMetalCausalConv1d failed, falling back to stock MLX: \(String(describing: error), privacy: .private)"
                )
                innerX = Self.stockCausalConv(innerX, weight: conv1d.weight, bias: conv1d.bias)
            }
        } else {
            innerX = Self.stockCausalConv(innerX, weight: conv1d.weight, bias: conv1d.bias)
        }
        // MLXNN.silu fuses `x * sigmoid(x)` into one dispatch, avoiding a
        // separate graph node at each of the 120 ReUseMamba call sites per chunk.
        // Both paths hand it the convolution output in the activation dtype, so
        // SiLU, xProj and the scan's `u` see identical values on Metal and stock.
        innerX = MLXNN.silu(innerX)
        
        let x_dbl = xProj(innerX.asType(x.dtype))
        let nState = A_log.shape[1]
        let dtRank = x_dbl.shape[2] - (nState * 2)
        
        let splitSSM = x_dbl.split(indices: [dtRank, dtRank + nState], axis: -1)
        let dt = splitSSM[0]
        let B = splitSSM[1]
        let C = splitSSM[2]
        
        // Raw outputs of dtProj; softplus and -exp(A_log) are fused into
        // the fixed-16 Metal kernel.
        let deltaRaw = dtProj(dt)
        let y: MLXArray
        switch kernels {
        case .metal:
            y = ReUseMetalSelectiveScan.run(
                deltaRaw: deltaRaw,
                A_log: A_log,
                B: B,
                C: C,
                u: innerX,
                D: D,
                z: innerZ,
                mode: mode
            )
        case .stock:
            y = StockSelectiveScan.run(
                deltaRaw: deltaRaw, A_log: A_log, B: B, C: C, u: innerX, D: D, z: innerZ
            )
        }
        
        return outProj(y.asType(x.dtype))
    }

    /// Stock causal depthwise convolution: explicit left padding, then MLX's depthwise conv1d
    /// (FP32 accumulation, one rounding to the activation dtype), then the bias in that dtype.
    /// Same values as `conv1d(x)[.., ..<L]`, but the padded Conv1d call routes a depthwise
    /// half convolution through MLX's grouped GEMM, which rounds its products more coarsely
    /// (16 Oct 2026, M5 Max: BF16 relative L2 0.00236 vs the exact convolution, against
    /// 0.00166 for a single rounding). The Metal kernel follows this single-rounding contract.
    static func stockCausalConv(_ x: MLXArray, weight: MLXArray, bias: MLXArray?) -> MLXArray {
        let taps = weight.shape[1]
        let padded = MLX.padded(x, widths: [.init((0, 0)), .init((taps - 1, 0)), .init((0, 0))])
        let y = MLX.conv1d(padded, weight, groups: x.shape[2])
        return bias.map { y + $0 } ?? y
    }

}
