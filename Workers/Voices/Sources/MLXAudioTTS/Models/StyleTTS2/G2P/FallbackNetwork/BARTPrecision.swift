import CryptoKit
import Foundation
import MLX
import MLXNN

/// The phonemizer is a separate pinned asset. Round each tensor before constructing a learned
/// module; pack eligible projections one at a time and never retain its original FP32 dictionary.
public struct BARTPrecision: Sendable {
    let half: String
    let bits: Int
    let keeping: Set<String>
    let truncation: Bool
    public init(half: String = "fp16", bits: Int = 16, keeping: Set<String> = [], truncation: Bool = false) {
        self.half = half; self.bits = bits; self.keeping = keeping; self.truncation = truncation
    }
    var dtype: DType { half == "bf16" ? .bfloat16 : .float16 }
}

final class BARTWeightView {
    static let file = "asset-kokoro-g2p/us_bart.safetensors"
    let raw: [String: MLXArray]
    let precision: BARTPrecision
    var quantized: [String] = []
    var kept: [String: String] = [:]
    var matrices: [String: Int] = [:]
    var hashes: [String: String] = [:]
    var audits: [String: [String: Double]] = [:]
    var fallback: [String] = []
    init(_ raw: [String: MLXArray], precision: BARTPrecision) {
        self.raw = raw; self.precision = precision
        for (key, value) in raw {
            kept[Self.file + ":" + key] = value.ndim == 2 && key.hasSuffix(".weight") ? "fewer than 65,536 parameters or incompatible affine group 64" : "buffer, bias or norm stays selected half"
        }
    }
    subscript(_ key: String) -> MLXArray? {
        guard let raw = raw[key] else { return nil }
        if precision.truncation, raw.dtype == .float32 {
            return MLXArray(raw.asArray(Float.self).map { Float(bitPattern: $0.bitPattern & 0xffff0000) }, raw.shape).asType(.bfloat16)
        }
        return raw.dtype.isFloatingPoint ? raw.asType(precision.dtype) : raw
    }
    func linear(_ prefix: String) -> Linear {
        let key = prefix + ".weight", weight = self[key]!, bias = self[prefix + ".bias"]
        let selector = "matrix@" + Self.file + ":" + key
        let eligible = weight.ndim == 2 && weight.size >= 65_536 && weight.dim(1) % 64 == 0
        guard eligible else { return Linear(weight: weight, bias: bias) }
        matrices[selector] = weight.size
        if precision.bits == 16 || precision.keeping.contains(selector) {
            kept[Self.file + ":" + key] = precision.bits == 16 ? "selected half tier" : "selection gate fallback: kokoro-g2p; matrix retained " + precision.half
            if precision.bits < 16 { fallback.append(selector) }
            return Linear(weight: weight, bias: bias)
        }
        let (packed, scales, biases) = MLX.quantized(weight, groupSize: 64, bits: precision.bits, mode: .affine)
        eval(packed, scales, biases)
        hashes[selector] = SHA256.hash(data: scales.asData(access: .copy).data).map { String(format: "%02x", $0) }.joined()
        if precision.bits == 8, let biases, let audit = try? StoredVoiceWeights.affinePackingAudit(dense: weight, packed: packed, scales: scales, biases: biases) {
            audits[Self.file + ":" + key] = audit
        }
        quantized.append(Self.file + ":" + prefix); kept.removeValue(forKey: Self.file + ":" + key)
        return QuantizedLinear(weight: packed, bias: bias, scales: scales, biases: biases, groupSize: 64, bits: precision.bits)
    }
    var receipt: [String: Any] {
        let retained = fallback.reduce(0) { $0 + matrices[$1]! }
        let packed = precision.bits == 16 ? 0 : matrices.values.reduce(0, +) - retained
        return [
            "parameters": raw.values.reduce(0) { $0 + $1.size }, "quantized_parameters": packed,
            "fallback_parameters": retained, "quantized_layers": quantized.count, "linear_layers": matrices.count,
            "quantized": quantized.sorted(), "kept": kept, "fallback_matrices": fallback.sorted(),
            "component_eligible_parameters": ["kokoro-g2p": matrices.values.reduce(0, +)],
            "matrix_parameters": matrices, "matrix_components": matrices.mapValues { _ in "kokoro-g2p" }, "matrix_scales_sha256": hashes, "packing_audit": audits,
            "source_half": precision.half, "load_derivation": "Pinned phonemizer asset; tensor-wise half conversion and affine-g64 projection packing through ordinary Load"
        ]
    }
}
