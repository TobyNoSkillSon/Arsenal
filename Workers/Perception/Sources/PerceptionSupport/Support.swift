// SPDX-License-Identifier: MIT
import Foundation
import CoreGraphics
import MLX
import MLXNN
import IntegrationKit
import ArsenalWire
import ModelPrecision

public enum PerceptionIO {
    public static func error(_ text: String) -> WorkerFailure { WorkerFailure(code: "invalid_request", message: text) }
    public static func json(_ url: URL) throws -> JSONObject { try JSONDecoder().decode(JSONObject.self, from: Data(contentsOf: url)) }
    public static func path(_ input: JSONObject, _ key: String) throws -> URL {
        guard let path = input[key]?.object?["path"]?.string, !path.isEmpty else { throw error("Missing \(key) file.") }
        return URL(fileURLWithPath: path)
    }
    /// Directory of a declared `[[asset]]` (verified by Get), or of a lab fixture when random weights name one.
    public static func asset(_ context: NativeLoadContext, _ id: String) throws -> URL {
        guard let directory = context.assets[id] else { throw error("Missing asset \(id).") }
        return directory
    }
    public static func archives(_ folder: URL) throws -> [URL] {
        let files = try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil).filter { $0.pathExtension == "safetensors" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        guard !files.isEmpty else { throw error("Checkpoint has no safetensors files.") }; return files
    }
    public static func eligible(_ name: String, _ shape: [Int]) -> Bool { name.hasSuffix(".weight") && shape.count == 2 && shape[1] % 64 == 0 }
    public static func prepare(_ module: Module, context: NativeLoadContext, map: (String, MLXArray) throws -> (String, MLXArray)? = { ($0, $1) }, cancellation: NativeCancellation)
        throws
    {
        try cancellation.check()
        if context.randomWeights {
            try NativeParameters.initializeRandom(module)
            if context.precision == "int8" || context.precision == "int4" {
                quantize(model: module) { name, layer in
                    guard let weight = (layer as? Linear)?.weight ?? (layer as? Embedding)?.weight, eligible(name + ".weight", weight.shape) else { return nil }
                    return (groupSize: 64, bits: context.precision == "int8" ? 8 : 4, mode: .affine)
                }
            }
        } else {
            _ = try NativePrecisionTier(context.precision).load(
                module, archives: archives(context.weights),
                map: { key, value in
                    try cancellation.check(); return try map(key, value)
                }, eligible: eligible)
        }
        module.train(false); try NativeParameters.requireHalf(Dictionary(uniqueKeysWithValues: module.parameters().flattened()))
    }
    /// The ordinary file-by-file half conversion, except that `dropping` names checkpoint tensors the model never reads.
    /// They are left out of the snapshot, so their values can neither fail the FP16 range check nor occupy memory.
    public static func convert(
        _ context: NativeConversionContext, cancellation: NativeCancellation, dropping: Set<String>, eligible: @escaping (String, [Int]) -> Bool
    ) throws -> JSONObject {
        for path in try NativeConversionContext.files(in: context.source) {
            try cancellation.check()
            if StoredNames.reserved.contains(URL(fileURLWithPath: path).lastPathComponent) { continue }
            let source = context.source.appendingPathComponent(path), target = context.destination.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            guard !FileManager.default.fileExists(atPath: target.path) else { throw PrecisionError("Duplicate conversion output: " + path) }
            guard source.pathExtension == "safetensors" else {
                try FileManager.default.copyItem(at: source, to: target)
                continue
            }
            let archive = try TensorArchive(source)
            let keys = archive.entries.keys.filter { !dropping.contains($0) }.sorted()
            _ = try StreamedPrecision.convert(
                keys: keys,
                read: { key in
                    try cancellation.check(); return try archive.array(key)
                }, to: target, half: context.half, bits: context.bits, eligible: eligible)
        }
        return try context.receipt(cancellation: cancellation)
    }
    public static func bytes(_ module: Module?) -> Int { module?.parameters().flattened().reduce(0) { $0 + $1.1.nbytes } ?? 0 }
    public static func pixels(_ image: CGImage, size: Int) throws -> MLXArray {
        var bytes = [UInt8](repeating: 0, count: size * size * 4)
        guard
            let context = CGContext(
                data: &bytes, width: size, height: size, bitsPerComponent: 8, bytesPerRow: size * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw error("Cannot resize image.") }
        context.interpolationQuality = .high; context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
        var rgb = [Float](); rgb.reserveCapacity(size * size * 3)
        for i in 0..<(size * size) { for channel in 0..<3 { rgb.append(Float(bytes[i * 4 + channel]) / Float(255)) } }
        return MLXArray(rgb, [1, size, size, 3]).asType(.float16)
    }
}

private final class AsyncResult<T>: @unchecked Sendable {
    var value: Result<T, Error>?
    let semaphore = DispatchSemaphore(value: 0)
}
public extension PerceptionIO {
    static func awaitResult<T>(_ action: @escaping () async throws -> T) throws -> T {
        let box = AsyncResult<T>()
        Task.detached {
            do { box.value = .success(try await action()) } catch { box.value = .failure(error) }; box.semaphore.signal()
        }
        box.semaphore.wait()
        return try box.value!.get()
    }
}
