// SPDX-License-Identifier: MIT
// Get-time cast of the pinned mlx-community/VoxCPM2-bf16 release to FP16, file by file: BF16 language models and the
// FP32 AudioVAE become FP16 (AGENTS.md precision rule); the integer sample-rate buckets are kept as stored. Every
// cast tensor is checked for values outside FP16's range first, so an overflow fails the Get instead of the audio.
import Foundation
import MLX

public enum VoxCPM2Cast {
    public static let recipe = "cast:voxcpm2"
    public static let fp16Max: Float = 65504

    public struct Report: Sendable {
        public var files: [String] = []
        public var castTensors = 0
        public var keptTensors = 0
        public var largestMagnitude: Float = 0
    }

    /// Tensors that stay in their stored dtype: integer buffers (the rotary tables the release ships are cast too; the
    /// loader recomputes them in float32).
    static func keeps(_ key: String, _ dtype: DType) -> Bool { !dtype.isFloatingPoint }

    /// Casts one tensor dictionary; throws on any value an FP16 tensor cannot hold.
    static func cast(_ tensors: [String: MLXArray], file: String, report: inout Report) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in tensors {
            if keeps(key, value.dtype) { out[key] = value; report.keptTensors += 1; continue }
            if value.dtype != .float16 {
                let magnitude = MLX.abs(value.asType(.float32, stream: .cpu), stream: .cpu).max(stream: .cpu).item(Float.self)
                guard magnitude.isFinite, magnitude <= fp16Max else {
                    throw VoxCPM2Error("\(file): \(key) holds \(magnitude), outside FP16's range; this release cannot be cast to FP16")
                }
                report.largestMagnitude = max(report.largestMagnitude, magnitude)
            }
            out[key] = value.asType(.float16, stream: .cpu)
            report.castTensors += 1
        }
        return out
    }

    /// The pinned files of the model folder (`[weights] files` in model.toml): only these are cast or copied.
    public static func manifestFiles(modelDirectory: URL) throws -> [String] {
        let text = try String(contentsOf: modelDirectory.appendingPathComponent("model.toml"), encoding: .utf8)
        var inWeights = false
        for line in text.split(separator: "\n", omittingEmptySubsequences: false).map({ $0.trimmingCharacters(in: .whitespaces) }) {
            if line.hasPrefix("[") { inWeights = line == "[weights]"; continue }
            guard inWeights, line.hasPrefix("files"), let open = line.firstIndex(of: "["), let close = line.lastIndex(of: "]") else { continue }
            let files = line[line.index(after: open)..<close].split(separator: ",").map { $0.trimmingCharacters(in: CharacterSet(charactersIn: " \"")) }
                .filter { !$0.isEmpty }
            guard !files.isEmpty, files.allSatisfy({ !$0.hasPrefix("/") && !$0.contains("..") }) else { break }
            return files
        }
        throw VoxCPM2Error("model.toml declares no [weights] files")
    }

    /// Writes the FP16 snapshot of `source` into the empty `destination`; non-weight files are copied unchanged. On any
    /// failure (or `isCancelled`) everything it wrote is removed again.
    public static func convert(source: URL, files: [String], into destination: URL, isCancelled: () -> Bool = { false }) throws -> Report {
        do { return try write(source: source, files: files, into: destination, isCancelled: isCancelled) } catch {
            for path in files { try? FileManager.default.removeItem(at: destination.appendingPathComponent(path)) }
            throw error
        }
    }

    static func write(source: URL, files: [String], into destination: URL, isCancelled: () -> Bool) throws -> Report {
        var report = Report()
        try Device.withDefaultDevice(.cpu) {
            for path in files.sorted() {
                if isCancelled() { throw CancellationError() }
                let input = source.appendingPathComponent(path).resolvingSymlinksInPath()
                let output = destination.appendingPathComponent(path)
                try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
                if path.hasSuffix(".safetensors") {
                    let tensors = try MLX.loadArrays(url: input, stream: .cpu)
                    try MLX.save(arrays: try cast(tensors, file: path, report: &report), metadata: ["format": "mlx"], url: output)
                } else {
                    try FileManager.default.copyItem(at: input, to: output)
                }
                report.files.append(path)
            }
        }
        return report
    }
}
