// SPDX-License-Identifier: Apache-2.0
// Vendored from faraday/SpeechLens Sources/Inference (Apache-2.0)
// @ d20e11bcafbc32435ef18a17ed56caf22fca6833; Vireo changes listed in CHANGES.md.

import Foundation
import MLX
import MLXFast

enum ReUseCausalConv1dError: LocalizedError {
    case invalidShape(String)

    var errorDescription: String? {
        switch self {
        case .invalidShape(let message):
            return message
        }
    }
}

/// Specialized causal depthwise 1D convolution for the ReUseMamba block.
///
/// Equivalent to the following MLX eager graph (`ReUseMamba.stockCausalConv`):
/// ```
/// let padded = MLX.padded(x, widths: [0, (3, 0), 0])
/// innerX = MLX.conv1d(padded, weight, groups: D) + bias
/// ```
///
/// Why a custom kernel?
///   One Metal dispatch / one graph node regardless of L. The MLXNN path and
///   a rejected multi-node MLX rewrite are documented in
///   `docs/kernel-optimization.md`.
///
/// Assumptions enforced by `run(...)`:
///   * kernelSize = 4
///   * groups = inputChannels = outputChannels = D  (depthwise)
///   * padding = 3 (kernelSize - 1, i.e. left-causal after trim)
///   * input `[B, L, D]`, weight `[D, 4, 1]`, bias `[D]`
///
/// These all hold for ReUseMamba's `conv1d`; if they're violated the wrapper
/// throws and ReUseMamba falls back to MLXNN.Conv1d on that call.
enum ReUseMetalCausalConv1d {
    static var isEnabled: Bool {
        let env = ProcessInfo.processInfo.environment
        if env["SPEECHLENS_DISABLE_METAL_CONV1D"] == "1" { return false }
        return true
    }

    // Thread layout:
    //   One thread per (batch, channel). Each thread walks the L dimension
    //   in order, shifting a 3-register window of past inputs (x_m3..x_m1)
    //   and reading one fresh x_curr per step. Weights and bias are hoisted
    //   to registers at init — 4 weights is well under any spill threshold.
    //
    // Memory layout:
    //   input/output are row-major `[B, L, D]`, so stepping along L means
    //   stride = D elements. Adjacent d-threads read adjacent memory,
    //   giving fully coalesced loads and stores.
    //
    // Causal boundary:
    //   Left-pad via register init to 0.0. For t < 3 the register window
    //   still contains zeros from init, matching the `padded = zeros(3) ++ x`
    //   semantics of MLXNN.Conv1d + trim-to-L.
    //
    // Rounding contract (Arsenal): `T` is the activation dtype. The four taps
    // accumulate in FP32 (as MLX's depthwise conv1d kernel does), round once to
    // `T`, and the bias is then added in `T`, exactly like the stock path
    // `ReUseMamba.stockCausalConv`. The output is `T`, so the following SiLU runs
    // on the same values and in the same dtype on both the Metal and stock paths.
    private static let source = """
    #pragma clang fp contract(off)

    uint tid = thread_position_in_grid.x;

    int B_size = input_shape[0];
    int L_size = input_shape[1];
    int D_size = input_shape[2];

    uint total_bd = B_size * D_size;
    if (tid >= total_bd) {
        return;
    }

    uint b = tid / D_size;
    uint d = tid % D_size;

    // Weight layout [D, 4, 1] -> flat index = d*4 + k.
    uint w_base = d * 4;
    float w0 = weight[w_base + 0];
    float w1 = weight[w_base + 1];
    float w2 = weight[w_base + 2];
    float w3 = weight[w_base + 3];
    T bias_d = bias_in[d];

    // Rolling window of the three most recent past inputs; seeded with
    // zero to emulate left causal padding.
    float x_m3 = 0.0f;
    float x_m2 = 0.0f;
    float x_m1 = 0.0f;

    uint base = b * uint(L_size * D_size) + d;
    uint stride_l = uint(D_size);

    for (uint t = 0; t < uint(L_size); ++t) {
        uint offset = base + t * stride_l;
        float x_curr = input[offset];

        // y_t = T(w0*x_{t-3} + w1*x_{t-2} + w2*x_{t-1} + w3*x_t) + bias
        float acc = 0.0f;
        acc += x_m3 * w0;
        acc += x_m2 * w1;
        acc += x_m1 * w2;
        acc += x_curr * w3;
        T conv = static_cast<T>(acc);
        output[offset] = static_cast<T>(conv + bias_d);

        x_m3 = x_m2;
        x_m2 = x_m1;
        x_m1 = x_curr;
    }
    """

    private static let kernel = MLXFast.metalKernel(
        name: "speechlens_causal_depthwise_conv1d_k4_typed",
        inputNames: ["input", "weight", "bias_in"],
        outputNames: ["output"],
        source: source,
        ensureRowContiguous: true
    )

    /// Runs the fused causal depthwise Conv1d.
    /// - Parameters:
    ///   - x: input `[B, L, D]`
    ///   - weight: Conv1d.weight `[D, 4, 1]`
    ///   - bias: Conv1d.bias `[D]`
    /// - Returns: output `[B, L, D]` in `x.dtype`
    static func run(
        x: MLXArray,
        weight: MLXArray,
        bias: MLXArray
    ) throws -> MLXArray {
        guard x.shape.count == 3 else {
            throw ReUseCausalConv1dError.invalidShape(
                "input must be rank 3 [B, L, D], got shape \(x.shape)"
            )
        }
        let batch = x.shape[0]
        let seqLen = x.shape[1]
        let dInner = x.shape[2]

        guard weight.shape.count == 3,
              weight.shape[0] == dInner,
              weight.shape[1] == 4,
              weight.shape[2] == 1
        else {
            throw ReUseCausalConv1dError.invalidShape(
                "weight must be [D=\(dInner), 4, 1], got \(weight.shape)"
            )
        }
        guard bias.shape == [dInner] else {
            throw ReUseCausalConv1dError.invalidShape(
                "bias must be [D=\(dInner)], got \(bias.shape)"
            )
        }

        let totalThreads = max(1, batch * dInner)
        let threadGroup = min(256, totalThreads)
        guard weight.dtype == x.dtype, bias.dtype == x.dtype else {
            throw ReUseCausalConv1dError.invalidShape(
                "input, weight and bias must share one dtype, got \(x.dtype), \(weight.dtype), \(bias.dtype)"
            )
        }
        let outputs = kernel(
            [x, weight, bias],
            template: [("T", x.dtype)],
            grid: (totalThreads, 1, 1),
            threadGroup: (threadGroup, 1, 1),
            outputShapes: [[batch, seqLen, dInner]],
            outputDTypes: [x.dtype]
        )
        return outputs[0]
    }
}
