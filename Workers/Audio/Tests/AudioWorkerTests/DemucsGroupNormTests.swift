// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import Testing
@testable import AudioWorker

/// 22 Oct: every HTDemucs-ft separation at FP16 failed with HTTP 422 "Audio output is invalid or exceeds the output bound."
/// The time branch's first DConv normalizes one group of 6 × 85,995 values (the 7.8 s segment every input is padded to);
/// the pure-MLX fallback summed them in FP16, its mean was infinite and the whole separation became NaN.
@Suite(.serialized) struct DemucsGroupNormTests {
    init() throws {
        guard let path = (ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] ?? ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"]) else {
            throw audioFailure("The family GPU test needs the pinned no-fast-math metallib.")
        }
        setTestMetallib(URL(fileURLWithPath: path))
    }

    /// FP16 activations of the first time-branch DConv's shape: mean 0.75, so the FP16 sum passes 65,504.
    static func activations(channels: Int, length: Int) -> MLXArray {
        let values = (0..<(channels * length)).map { index in Float(0.75) + Float(0.5) * sin(Float(index % 9973) * 0.37) }
        return MLXArray(values, [1, channels, length]).asType(.float16)
    }

    /// Float32 GroupNorm (PyTorch semantics, one group) of the same FP16 values: the op-level reference.
    static func reference(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
        let x32 = x.asType(.float32)
        let mean = x32.mean(), variance = (x32 - mean).square().mean()
        return (x32 - mean) * rsqrt(variance + 1e-5) * weight.reshaped([1, -1, 1]) + bias.reshaped([1, -1, 1])
    }

    @Test(.heavy) func largeGroupGELUStaysFiniteAndHalf() throws {
        let x = Self.activations(channels: 6, length: 85_995)
        #expect(x.size == 515_970 && x.size > 32_768) // the pure-MLX path, past the fused kernel's threshold
        #expect(x.sum().item(Float.self).isInfinite) // the overflow that broke the old fallback
        let weight = MLXArray([Float(1), 0.5, 2, 1, 1.5, 0.25]).asType(.float16), bias = MLXArray([Float(0), 0.1, -0.1, 0.2, 0, -0.2]).asType(.float16)
        let out = fusedGroupNormGELU(x, weight: weight, bias: bias, numGroups: 1)
        #expect(out.dtype == .float16 && out.shape == [1, 6, 85_995])
        #expect(MLX.all(MLX.isFinite(out)).item(Bool.self))
        let expected = gelu(Self.reference(x, weight: weight.asType(.float32), bias: bias.asType(.float32)))
        let error = MLX.abs(out.asType(.float32) - expected).max().item(Float.self)
        #expect(error < 0.01)
    }

    @Test(.heavy) func largeGroupGLUStaysFiniteAndHalf() throws {
        let x = Self.activations(channels: 12, length: 85_995)
        let weight = MLXArray.ones([12]).asType(.float16), bias = MLXArray.zeros([12]).asType(.float16)
        let out = fusedGroupNormGLU(x, weight: weight, bias: bias, numGroups: 1)
        #expect(out.dtype == .float16 && out.shape == [1, 6, 85_995])
        let normed = Self.reference(x, weight: weight.asType(.float32), bias: bias.asType(.float32))
        let parts = split(normed, parts: 2, axis: 1), expected = parts[0] * sigmoid(parts[1])
        #expect(MLX.all(MLX.isFinite(out)).item(Bool.self))
        #expect(MLX.abs(out.asType(.float32) - expected).max().item(Float.self) < 0.01)
    }

    /// The fallback and the fused Metal kernel compute the same normalization on a group small enough for both.
    @Test(.heavy) func fallbackMatchesTheFusedKernel() throws {
        let x = Self.activations(channels: 6, length: 4_000)
        let weight = MLXArray([Float(1), 0.5, 2, 1, 1.5, 0.25]).asType(.float16), bias = MLXArray.zeros([6]).asType(.float16)
        let fused = fusedGroupNormGELU(x, weight: weight, bias: bias, numGroups: 1).asType(.float32)
        let statistics = demucsGroupStatistics(x.reshaped([1, 1, 6, 4_000]), eps: 1e-5).reshaped([1, 6, 4_000])
        let fallback = gelu(statistics * weight.asType(.float32).reshaped([1, 6, 1])).asType(.float16).asType(.float32)
        #expect(MLX.abs(fused - fallback).max().item(Float.self) < 0.01)
    }

    /// The DConv and encoder group norms share the float32 statistics.
    @Test(.heavy) func groupNormModulesStayFiniteOnLargeGroups() throws {
        let x = Self.activations(channels: 8, length: 85_995)
        let norm = GroupNormNCL(groupCount: 1, channels: 8)
        let out = norm(x)
        #expect(out.dtype == .float16 && MLX.all(MLX.isFinite(out)).item(Bool.self))
        let expected = Self.reference(x, weight: MLXArray.ones([8]), bias: MLXArray.zeros([8]))
        #expect(MLX.abs(out.asType(.float32) - expected).max().item(Float.self) < 0.01)
    }
}
