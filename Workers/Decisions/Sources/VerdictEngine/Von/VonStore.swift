import Foundation
import CryptoKit
import VerdictCore

/// Converts the FP32-only pinned release to uniform half before execution. Original source and output hashes are separate.
/// BF16 nearest/truncation is restricted to source-format calibration; the catalog Get recipe writes FP16.
public enum VonStore {
    public static let weightsFile = "von-fp16.safetensors"
    public static let manifestFile = "manifest.json"
    public static let sidecarFiles = ["config.json", "tokenizer.json", "tokenizer_config.json", "marker_calibration.json"]
    public static let converter = "von-half-2"

    /// Kept for compatibility with stored-manifest readers; no learned tensor stays FP32.
    static func keepsF32(_ name: String) -> Bool { false }

    /// True when `dir` holds a complete converted model (weights, sidecars, manifest).
    public static func isComplete(_ dir: URL) -> Bool {
        let fm = FileManager.default
        return ([weightsFile, manifestFile] + sidecarFiles).allSatisfy { fm.fileExists(atPath: dir.appendingPathComponent($0).path) }
    }

    public static func manifest(_ dir: URL) -> StoredModelManifest? {
        (try? Data(contentsOf: dir.appendingPathComponent(manifestFile))).flatMap { try? JSONDecoder().decode(StoredModelManifest.self, from: $0) }
    }

    /// Bytes on disk of a stored model (weights + sidecars + manifest).
    public static func storedBytes(_ dir: URL) -> Int64 {
        ([weightsFile, manifestFile] + sidecarFiles).reduce(Int64(0)) { sum, name in
            sum + ((try? FileManager.default.attributesOfItem(atPath: dir.appendingPathComponent(name).path)[.size] as? NSNumber)?.int64Value ?? 0)
        }
    }

    /// Converts a Von snapshot (option_marker.pt + sidecars) into `dir`, replacing an incomplete one. Built in
    /// `dir`.partial and renamed on success, so a crash never leaves a half model that looks complete.
    @discardableResult
    public static func convert(snapshot: URL, to dir: URL, model: String, repository: String, revision: String?, format: String = "F16", truncate: Bool = false) throws -> StoredModelManifest {
        let start = Date()
        let fm = FileManager.default
        let config = try LayaEncoderConfiguration(data: Data(contentsOf: snapshot.appendingPathComponent("config.json")))
        let source = snapshot.appendingPathComponent("option_marker.pt")
        let zip = try VonZip(source)
        try VonWeights.validate(zip, config: config)
        let names = VonWeights.encoderNames(config) + VonWeights.head
        let tensors = names.map { Safetensors.Tensor(name: $0.0, shape: $0.1, dtype: format) }
        let work = dir.deletingLastPathComponent().appendingPathComponent(dir.lastPathComponent + ".partial", isDirectory: true)
        try? fm.removeItem(at: work)
        try fm.createDirectory(at: work, withIntermediateDirectories: true)
        for name in sidecarFiles {
            try fm.copyItem(at: snapshot.appendingPathComponent(name).resolvingSymlinksInPath(), to: work.appendingPathComponent(name))
        }
        let stored = try Safetensors.write(to: work.appendingPathComponent(weightsFile), tensors: tensors,
                                           metadata: ["format": "pt", "converter": converter, "model": model, "rounding": truncate ? "truncate" : "nearest"]) { i in
            try zip.read("data/\(i)")
        }
        // The Hugging Face cache names a blob by its content SHA-256; otherwise hash the source.
        let resolved = source.resolvingSymlinksInPath()
        let blobName = resolved.lastPathComponent
        let sha = blobName.count == 64 && blobName.allSatisfy(\.isHexDigit) ? blobName : try sha256(resolved)
        let sourceBytes = (try? fm.attributesOfItem(atPath: resolved.path)[.size] as? NSNumber)?.int64Value ?? 0
        var manifest = StoredModelManifest(model: model, repository: repository, revision: revision, source_file: "option_marker.pt",
                                           source_bytes: sourceBytes, source_sha256: sha, source_dtype: "fp32", dtype: format == "BF16" ? "bf16" : "fp16",
                                           f32_tensors: tensors.filter { $0.dtype == "F32" }.map(\.name), weights_file: weightsFile,
                                           stored_bytes: 0, convert_seconds: 0, converter: converter,
                                           created: ISO8601DateFormatter().string(from: Date()))
        manifest.convert_seconds = (Date().timeIntervalSince(start) * 10).rounded() / 10
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: work.appendingPathComponent(manifestFile))
        manifest.stored_bytes = storedBytes(work)
        _ = stored
        try encoder.encode(manifest).write(to: work.appendingPathComponent(manifestFile))
        try? fm.removeItem(at: dir)
        try fm.moveItem(at: work, to: dir)
        return manifest
    }

    static func sha256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var digest = SHA256()
        while let chunk = try handle.read(upToCount: 16 << 20), !chunk.isEmpty { digest.update(data: chunk) }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
