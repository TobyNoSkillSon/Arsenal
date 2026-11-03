// SPDX-License-Identifier: MIT
import Foundation

/// diffusers FlowMatchEulerDiscreteScheduler as QwenImage21Pipeline drives it, in float32 like the reference.
/// Base: linspace(1, 1/N, N) → exponential time shift by mu(target tokens) → stretch so the last sigma is 0.02.
/// Turbo: the checkpoint's saved `sample_sigmas` (8 steps, no shift); a step count cannot override it.
public struct FlowSchedule: Equatable, Sendable {
    /// One per denoising step plus the terminal 0.
    public let sigmas: [Float]
    public var steps: Int { sigmas.count - 1 }
    /// The model's timestep input for step i: sigma × 1000.
    public func timestep(_ step: Int) -> Float { sigmas[step] * 1000 }

    /// calculate_shift with the pipeline's config fallbacks.
    public static func mu(imageTokens: Int, _ c: SchedulerConfig) -> Double {
        let m = (c.maxShift - c.baseShift) / Double(c.maxImageSeqLen - c.baseImageSeqLen)
        let b = c.baseShift - m * Double(c.baseImageSeqLen)
        return Double(imageTokens) * m + b
    }

    public init(_ c: SchedulerConfig, steps requested: Int, imageTokens: Int, sigmas custom: [Double]? = nil) throws {
        var input: [Float]
        if let given = custom ?? c.sampleSigmas {
            input = given.map(Float.init)
        } else {
            guard requested >= 1 else { throw QwenImageError("At least one denoising step is needed") }
            // np.linspace(1.0, 1/N, N) in float64, then the scheduler's float32 array.
            let n = requested
            input = (0..<n).map { i in n == 1 ? 1 : Float(1.0 + (1.0 / Double(n) - 1.0) * Double(i) / Double(n - 1)) }
        }
        if c.dynamicShifting {
            let e = Float(exp(Self.mu(imageTokens: imageTokens, c)))
            input = input.map { e / (e + (1 / $0 - 1)) }
        } else if c.shift != 1 {
            let s = Float(c.shift)
            input = input.map { s * $0 / (1 + (s - 1) * $0) }
        }
        if let terminal = c.shiftTerminal, let last = input.last {
            let scale = (1 - last) / Float(1 - terminal)
            input = input.map { 1 - (1 - $0) / scale }
        }
        sigmas = input + [0]
    }

    /// One Euler step of the flow: x + (sigma_next − sigma) · v.
    public func delta(_ step: Int) -> Float { sigmas[step + 1] - sigmas[step] }
}
