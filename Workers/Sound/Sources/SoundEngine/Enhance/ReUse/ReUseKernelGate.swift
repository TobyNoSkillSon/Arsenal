// SPDX-License-Identifier: MIT
// Vireo engine gate for RE-USE's custom Metal kernels (family rule: a kernel
// serves only after an on-device self-test agrees with the stock MLX path;
// failure → stock, sticky per (model, kernel version, GPU family, macOS, app
// version) — persistence belongs to the helper's gate, not here).
// GPU WORK: run only when the GPU is free (never during Vella measurements).

import Foundation
import MLX
import MLXNN
import MLXRandom

public enum ReUseKernelGate {
    /// Bump when either kernel's source changes.
    public static let kernelVersion = "speechlens-d20e11b-scan16+conv4-v2-typed"

    public struct Result: Sendable, CustomStringConvertible {
        public let passed: Bool
        public let scanMaxAbsError: Float
        public let convMaxAbsError: Float
        public let reason: String
        public var description: String {
            "\(passed ? "pass" : "fail") scan \(scanMaxAbsError) conv \(convMaxAbsError)\(reason.isEmpty ? "" : " · \(reason)")"
        }
    }

    /// Compares the Metal selective scan and causal conv against plain MLX on fixed-seed inputs
    /// shaped like the model's (D = 256, N = 16), including large-magnitude FP32 inputs, in FP32 and
    /// in both half activation dtypes the model serves (each half case rounds the stock output to
    /// that dtype, as the model does). Errors are the max absolute difference relative to the
    /// stock output's max magnitude (at least 1). FP32 must stay within `tolerance`; a half
    /// result may differ by at most one unit in the last place of its dtype, from FP32
    /// accumulation order deciding a rounding tie (BF16 2^-7, FP16 2^-10 relative to the max).
    public static func selfTest(tolerance: Float = 1e-3) -> Result {
        Device.withDefaultDevice(.gpu) {
            var scanErr: Float = 0
            var convErr: Float = 0
            var failure = ""
            let key = MLXRandom.key(20260926)
            let keys = MLXRandom.split(key: key, into: 9)
            for dtype in [DType.float32, .float16, .bfloat16] {
                let bound: Float = switch dtype {
                case .bfloat16: 0x1p-7
                case .float16: 0x1p-10
                default: tolerance
                }
                for (batch, length, fullScale) in [(2, 64, Float(1)), (3, 257, Float(1)), (1, 128, Float(8))] {
                    let d = 256, n = reUseMambaStateSize
                    // Half cases stay inside FP16 range: scale ≤ 2 and model-like step sizes
                    // (softplus(−3 ± 2)), so the recurrence cannot overflow on either path.
                    let scale = dtype == .float32 ? fullScale : min(fullScale, 2)
                    let shift: Float = dtype == .float32 ? 0 : -3
                    let deltaRaw = (MLXRandom.normal([batch, length, d], key: keys[0]) * scale + shift).asType(dtype)
                    let aLog = MLXRandom.uniform(low: -2, high: 2, [d, n], key: keys[1]).asType(dtype)
                    let b = (MLXRandom.normal([batch, length, n], key: keys[2]) * scale).asType(dtype)
                    let c = MLXRandom.normal([batch, length, n], key: keys[3]).asType(dtype)
                    let u = (MLXRandom.normal([batch, length, d], key: keys[4]) * scale).asType(dtype)
                    let dParam = MLXRandom.normal([d], key: keys[5]).asType(dtype)
                    let z = (MLXRandom.normal([batch, length, d], key: keys[6]) * scale).asType(dtype)
                    let metal = ReUseMetalSelectiveScan.run(deltaRaw: deltaRaw, A_log: aLog, B: b, C: c, u: u, D: dParam, z: z)
                    let stock = StockSelectiveScan.run(deltaRaw: deltaRaw, A_log: aLog, B: b, C: c, u: u, D: dParam, z: z).asType(dtype)
                    let err = relativeMaxError(metal, stock)
                    if !err.isFinite { return Result(passed: false, scanMaxAbsError: .infinity, convMaxAbsError: convErr, reason: "non-finite \(dtype) scan output") }
                    scanErr = max(scanErr, err)
                    if err > bound, failure.isEmpty { failure = "\(dtype) scan above \(bound)" }

                    let x = (MLXRandom.normal([batch, length, d], key: keys[7]) * scale).asType(dtype)
                    let weight = (MLXRandom.normal([d, 4, 1], key: keys[8]) * 0.5).asType(dtype)
                    let bias = (MLXRandom.normal([d], key: keys[5]) * 0.1).asType(dtype)
                    do {
                        let fast = try ReUseMetalCausalConv1d.run(x: x, weight: weight, bias: bias)
                        let ref = ReUseMamba.stockCausalConv(x, weight: weight, bias: bias)
                        let cerr = relativeMaxError(fast, ref)
                        guard cerr.isFinite else { return Result(passed: false, scanMaxAbsError: scanErr, convMaxAbsError: .infinity, reason: "non-finite \(dtype) conv output") }
                        convErr = max(convErr, cerr)
                        if cerr > bound, failure.isEmpty { failure = "\(dtype) conv above \(bound)" }
                    } catch {
                        return Result(passed: false, scanMaxAbsError: scanErr, convMaxAbsError: .infinity, reason: "conv kernel: \(error)")
                    }
                }
            }
            return Result(passed: failure.isEmpty, scanMaxAbsError: scanErr, convMaxAbsError: convErr, reason: failure)
        }
    }

    /// max |a − b| / max(max |b|, 1), evaluated in FP32 on the GPU (a numerical diagnostic, not model execution).
    static func relativeMaxError(_ a: MLXArray, _ b: MLXArray) -> Float {
        let a32 = a.asType(.float32), b32 = b.asType(.float32)
        let denom = maximum(abs(b32).max(), MLXArray(Float(1)))
        return (abs(a32 - b32).max() / denom).item(Float.self)
    }
}
