// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXRandom
import ArsenalKernels

/// One resident speech model per worker; replacement sites are scoped to that model's load/unload.
public enum VireoFolderKernels {
    static let ropeSite = "qwen.rotate-half-rope"
    nonisolated(unsafe) static var current: FolderKernels?

    public static func configure(_ value: Any?, model: String, dtype: String) throws {
        let kernels = try FolderKernels(config: value)
        current = kernels
        guard model.hasPrefix("qwen3-tts") else { return }
        guard let candidate = kernels.candidate(ropeSite) else { return }
        // Synthetic q/k/cos/sin only: the check does not read the model's weights.
        let result = ExactGate.qualify(kernels: [candidate], model: model, dtype: dtype, weights: "-",
                                       checkRevision: "qwen-rope-v2-bf16-fp16-fp32-broadcast") { on in
            var outputs: [MLXArray] = []
            for type in [DType.bfloat16, .float16, .float32] {
                for batch in [1, 2] {
                    for length in [1, 3] {
                        let q = MLXRandom.normal([batch, length, 4, 128], key: MLXRandom.key(1)).asType(type)
                        let k = MLXRandom.normal([batch, length, 2, 128], key: MLXRandom.key(2)).asType(type)
                        let angles = MLXRandom.uniform(low: -3, high: 3, [1, length, 128], key: MLXRandom.key(3))
                        let c = MLX.cos(angles).asType(type), s = MLX.sin(angles).asType(type)
                        if on {
                            let (rq, rk) = VireoFusedRoPE.apply(q: q, k: k, cos: c, sin: s, candidate: candidate)
                            outputs += [rq, rk]
                        } else {
                            outputs += [VireoFusedRoPE.stock(q, cos: c, sin: s), VireoFusedRoPE.stock(k, cos: c, sin: s)]
                        }
                    }
                }
            }
            return outputs
        }
        if result.pass, candidate.failure == nil { kernels.activate(ropeSite, reason: result.detail) }
        else { kernels.deactivate(ropeSite, reason: candidate.failure ?? result.detail) }
    }

    public static func finalize(optimized: Bool) {
        if !optimized { current?.deactivate(ropeSite, reason: "speech model self-test selected the stock path") }
    }
    public static var wire: [[String: Any]] { current?.wire ?? [] }
    public static func reset() { current = nil }
}
