// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire
import CryptoKit
import DecisionsProtocol
import VerdictEngine
import VerdictCore

/// Production Get conversions: Von uniform FP16; derived decisions views quantize tensor by tensor at Load.
/// Von's versioned recipe packs every eligible Linear and the row-streamed embedding, with explicit int8-only half fallbacks.
enum Convert {
    static let manifest = "arsenal-decisions-view.json"
    static let legacyManifest = "mobius-decisions-view.json" // legacy-compat: read in snapshots written before the rename
    static let fp16Recipe = "cast:" + VonStore.converter
    static let quantizeRecipe = "mlx-affine-g64:verdict-at-load"

    static func isVonRecipe(_ recipe: String) -> Bool { recipe.hasPrefix("mlx-affine-g64:von-all-v2") }

    static func run(_ request: OrderedJSON) throws -> [String: Any] {
        guard let src = request["src_dir"]?.text, let dst = request["dst_dir"]?.text, let dtype = request["dtype"]?.text else {
            throw DecisionsError("bad_request", "convert needs src_dir, dst_dir and dtype.")
        }
        let group = request["group_size"].flatMap { Int($0.render()) } ?? 64
        guard group == 64 else { throw DecisionsError("unsupported_recipe", "Verdict's models quantize with group size 64 only.") }
        // Recipes are the model folders' [derive] strings (CONTRACTS §13.4); anything else is refused.
        let recipe = request["recipe"]?.text ?? ""
        let expected = dtype == "fp16" ? fp16Recipe : dtype == "bf16" ? "cast:von-bf16-2" : quantizeRecipe
        guard recipe == expected || (isVonRecipe(recipe) && ["int8", "int4"].contains(dtype)) else {
            throw DecisionsError("unsupported_recipe", "The decisions worker derives \(dtype) with recipe '\(expected)', not '\(recipe)'.")
        }
        let source = URL(fileURLWithPath: src, isDirectory: true), target = URL(fileURLWithPath: dst, isDirectory: true)
        let fm = FileManager.default
        let isVon =
            fm.fileExists(atPath: source.appendingPathComponent("option_marker.pt").path)
            || fm.fileExists(atPath: source.appendingPathComponent(VonStore.weightsFile).path)
        let model = request["config"]?["id"]?.text ?? request["model"]?.text ?? (isVon ? "von" : "laya")
        switch dtype {
        case "fp16", "bf16":
            guard isVon, fm.fileExists(atPath: source.appendingPathComponent("option_marker.pt").path) else {
                throw DecisionsError("unsupported_dtype", "fp16 is derived only for Von, from its fp32 option_marker.pt.")
            }
            let manifest = try VonStore.convert(
                snapshot: source, to: target, model: model, repository: "wfzyx/von", revision: request["revision"]?.text, format: dtype == "bf16" ? "BF16" : "F16")
            return ["files": try listing(target), "convert_s": manifest.convert_seconds]
        case "int8", "int4":
            if isVon && !fm.fileExists(atPath: source.appendingPathComponent(VonStore.weightsFile).path) {
                throw DecisionsError("unsupported_source", "Von's int8/int4 are derived from its fp16 view (convert dtype fp16 first).")
            }
            let start = Date()
            try? fm.removeItem(at: target)
            let work = target.deletingLastPathComponent().appendingPathComponent(target.lastPathComponent + ".partial", isDirectory: true)
            try? fm.removeItem(at: work)
            try fm.createDirectory(at: work, withIntermediateDirectories: true)
            for file in try files(source) where file != manifest && file != legacyManifest {
                let to = work.appendingPathComponent(file)
                try fm.createDirectory(at: to.deletingLastPathComponent(), withIntermediateDirectories: true)
                // APFS clone (copyItem clones on the same volume): no second copy of the 16-bit weights.
                try fm.copyItem(at: source.appendingPathComponent(file).resolvingSymlinksInPath(), to: to)
            }
            let bits = dtype == "int8" ? 8 : 4
            let fallback = recipe.components(separatedBy: ";half=").dropFirst().first?.components(separatedBy: ",") ?? []
            var coverage: [String: Any] = [:]
            if isVon {
                let tensors = try Safetensors.tensors(at: source.appendingPathComponent(VonStore.weightsFile))
                let eligible = tensors.filter { t in t.name.hasSuffix(".weight") && t.shape.count == 2 && t.shape[1] % 64 == 0 && t.count >= 65_536 }
                let selected = eligible.filter { t in !fallback.contains { t.name == $0 || t.name.hasPrefix($0 + ".") } }
                let total = tensors.reduce(0) { $0 + $1.count }, packed = selected.reduce(0) { $0 + $1.count }
                coverage = [
                    "total_parameters": total, "eligible_parameters": eligible.reduce(0) { $0 + $1.count }, "packed_parameters": packed,
                    "packed_tensors": selected.map(\.name), "fallback_parameters": eligible.reduce(0) { $0 + $1.count } - packed,
                    "kept": Dictionary(
                        uniqueKeysWithValues: tensors.filter { t in !selected.contains { $0.name == t.name } }.map { t in
                            (t.name, fallback.contains { t.name == $0 || t.name.hasPrefix($0 + ".") } ? "int8 saved-input quality fallback" : "norm, bias or small output matrix")
                        })
                ]
            }
            let info: [String: Any] = [
                "dtype": dtype, "bits": bits, "group_size": 64, "method": "mlx-affine-g64, quantized at load",
                "modules": isVon ? "every eligible Linear and row-streamed token embedding" : "Linear layers with in-dim % 64 == 0 and the token embedding",
                "half_modules": fallback, "coverage": coverage,
                "source": src, "rule": "Verdict f8177ed models.json convert_to"
            ]
            try ResponseJSON.data(info as NSDictionary).write(to: work.appendingPathComponent(manifest))
            try fm.moveItem(at: work, to: target)
            return ["files": try listing(target), "convert_s": (Date().timeIntervalSince(start) * 10).rounded() / 10]
        default:
            throw DecisionsError("unsupported_dtype", "convert makes fp16 (Von), int8 or int4, not \(dtype).")
        }
    }

    /// The dtype a derived view declares, or nil for a plain 16-bit snapshot.
    static func derivedDtype(_ dir: URL) -> String? {
        guard let data = try? Data(contentsOf: StoredNames.current(manifest, legacy: legacyManifest, in: dir)),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return object["dtype"] as? String
    }

    /// Relative paths of the regular files under `dir` (symlinks followed), sorted.
    static func files(_ dir: URL) throws -> [String] {
        let fm = FileManager.default
        guard let walker = fm.enumerator(at: dir, includingPropertiesForKeys: nil) else { return [] }
        var out: [String] = []
        let base = dir.resolvingSymlinksInPath().path + "/"
        for case let url as URL in walker {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: url.resolvingSymlinksInPath().path, isDirectory: &isDir), !isDir.boolValue else { continue }
            let full = url.path.hasPrefix(dir.path + "/") ? String(url.path.dropFirst(dir.path.count + 1)) : String(url.resolvingSymlinksInPath().path.dropFirst(base.count))
            out.append(full)
        }
        return out.sorted()
    }

    static func listing(_ dir: URL) throws -> [[String: Any]] {
        try files(dir).map { name in
            let url = dir.appendingPathComponent(name).resolvingSymlinksInPath()
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            var digest = SHA256(), bytes = 0
            while let chunk = try handle.read(upToCount: 16 << 20), !chunk.isEmpty { digest.update(data: chunk); bytes += chunk.count }
            return ["path": name, "bytes": bytes, "sha256": digest.finalize().map { String(format: "%02x", $0) }.joined()]
        }
    }
}
