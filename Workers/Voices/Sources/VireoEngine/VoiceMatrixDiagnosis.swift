// SPDX-License-Identifier: MIT
import CryptoKit
import Foundation
import MLX
import MLXNN
import MLXAudioTTS
import VireoCore

/// Capture-only ablation. This worker is discarded before any serving figure is collected.
final class VoiceMatrixDiagnosis {
    struct Pair {
        let root: Module
        let path: String
        let packed: Module
        let half: Module
        let selectors: Set<String>
    }
    let donor: MLXSpeech
    let pairs: [Pair]
    private var keeping: Set<String> = []
    init(candidate: MLXSpeech, donor: MLXSpeech, hashes: [String: String]) throws {
        self.donor = donor
        guard let a = candidate.model, let b = donor.model else { throw EngineError("Missing matrix diagnosis models") }
        self.pairs = try Self.pairs(roots: VoiceIntakeCapture.moduleRoots(a), denseRoots: Dictionary(uniqueKeysWithValues: VoiceIntakeCapture.moduleRoots(b)), hashes: hashes)
    }
    static func pairs(roots: [(String, Module)], denseRoots: [String: Module], hashes: [String: String]) throws -> [Pair] {
        guard Set(hashes.values).count == hashes.count else { throw EngineError("Ambiguous source scale fingerprints; matrix mapping cannot be diagnosed") }
        var pairs: [Pair] = []
        var seen: Set<ObjectIdentifier> = []
        for (rootName, root) in roots {
            guard let denseRoot = denseRoots[rootName] else { throw EngineError("Missing dense root") }
            let dense = Dictionary(uniqueKeysWithValues: denseRoot.leafModules().flattened() + [("", denseRoot)])
            for (path, layer) in root.leafModules().flattened() + [("", root)] {
                guard seen.insert(ObjectIdentifier(layer)).inserted else { continue }
                let scales: MLXArray
                if let q = layer as? QuantizedLinear { scales = q.scales }
                else if let q = layer as? QuantizedEmbedding { scales = q.scales }
                else if let q = layer as? StoredVoicePointwise { scales = q.scales }
                else { continue }
                let hash = SHA256.hash(data: scales.asData(access: .copy).data).map { String(format: "%02x", $0) }.joined()
                var selectors = Set(hashes.filter { $0.value == hash }.keys)
                // Codec Q/K/V source matrices become one stored runtime projection. Match all
                // three row blocks and retain/re-quantize that indivisible runtime group together.
                if selectors.isEmpty, path.hasSuffix("in_proj"), scales.dim(0) % 3 == 0 {
                    let rows = scales.dim(0) / 3
                    var group: Set<String> = []
                    for index in 0..<3 {
                        let block = scales[(index * rows)..<((index + 1) * rows)]
                        let fingerprint = SHA256.hash(data: block.asData(access: .copy).data).map { String(format: "%02x", $0) }.joined()
                        let matches = hashes.filter { $0.value == fingerprint }.keys
                        guard matches.count == 1 else { group = []; break }
                        group.formUnion(matches)
                    }
                    if group.count == 3 { selectors = group }
                }
                // Passing owners stay packed and need no source-to-runtime ablation mapping (e.g. fused codec Q/K/V).
                if selectors.isEmpty { continue }
                guard let half = dense[path], !path.isEmpty else { throw EngineError("Unmapped or root-level packed matrix: \(rootName)/\(path)") }
                pairs.append(Pair(root: root, path: path, packed: layer, half: half, selectors: selectors))
            }
        }
        let missing = Set(hashes.keys).subtracting(pairs.flatMap { $0.selectors })
        guard missing.isEmpty else { throw EngineError("Matrix diagnosis coverage differs from converted inventory: " + missing.sorted().joined(separator: ",")) }
        return pairs
    }
    func patch(_ selected: Set<String>) throws -> [[String]] {
        guard selected.isSubset(of: Set(pairs.flatMap { $0.selectors })) else { throw EngineError("Unknown matrix selector") }
        for pair in pairs {
            let before = !keeping.isDisjoint(with: pair.selectors), after = !selected.isDisjoint(with: pair.selectors)
            if before != after { try StoredVoiceWeights.replace(pair.root, path: pair.path, with: after ? pair.half : pair.packed) }
        }
        keeping = selected
        return Set(pairs.map { $0.selectors.sorted().joined(separator: ",") }).sorted().map { $0.split(separator: ",").map(String.init) }
    }
}

extension MLXSpeech {
    public func patchVoiceMatrices(halfDirectory: URL, coverage: URL, components: Set<String>, keeping: Set<String>) async throws -> [[String]] {
        if matrixDiagnosis == nil {
            let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: coverage)) as? [String: Any]
            guard let allHashes = raw?["matrix_scales_sha256"] as? [String: String], let owners = raw?["matrix_components"] as? [String: String] else { throw EngineError("Missing converted matrix inventory") }
            let hashes = allHashes.filter { components.contains(owners[$0.key] ?? "") }
            guard !hashes.isEmpty else { throw EngineError("No eligible matrices in failing owners") }
            let source = ModelSource(id: id, engine: "mlx-audio", directory: halfDirectory, precision: "bf16", assets: intakeAssets, overrides: configuration.switches)
            guard let donor = try await MLXSpeech.load(source) as? MLXSpeech else { throw EngineError("Missing half donor") }
            matrixDiagnosis = try VoiceMatrixDiagnosis(candidate: self, donor: donor, hashes: hashes)
        }
        return try matrixDiagnosis!.patch(keeping)
    }
}
