// SPDX-License-Identifier: Apache-2.0
import Foundation
// Vendored from faraday/SpeechLens Sources/Inference (Apache-2.0)
// @ d20e11bcafbc32435ef18a17ed56caf22fca6833; Vireo changes listed in CHANGES.md.

/// Numerical execution mode for the fixed-16 selective scan.
public enum ReUseScanMode: String, Sendable, Equatable, Codable {
    /// The existing production arithmetic and Metal kernel.
    case standard
    /// The strict fast-bitcast-scalar selective-scan arithmetic.
    case strict
}

/// Which implementation runs the ReUseMamba selective scan and causal conv (Vireo).
///
/// `.metal` uses SpeechLens's custom kernels (fast; family rule: only after the
/// on-device self-test in `ReUseKernelGate` agrees with `.stock`). `.stock`
/// uses plain MLX ops only (sequential scan over time; slow, but it is the
/// reference and the fallback when the gate fails or `VIREO_KERNELS=off`).
public enum ReUseKernels: String, Sendable, Equatable, Codable {
    case metal
    case stock

    /// `.stock` when `VIREO_FORCE_STOCK=1`, `VIREO_KERNELS=off` or `VIREO_REUSE_METAL=0` (Standard and Exact cells: the
    /// kernels are inexact), otherwise `preferred`.
    public static func resolved(
        preferred: ReUseKernels = .metal,
        configuration: FastPath = .current
    ) -> ReUseKernels {
        configuration.enabled("VIREO_REUSE_METAL") ? preferred : .stock
    }
}
