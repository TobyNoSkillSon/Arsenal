// SPDX-License-Identifier: MIT
// Arsenal new-runner kit: helpers for Model.swift. Do not edit; call these from your model.
import AVFoundation
import CoreGraphics
import CryptoKit
import Foundation
import ImageIO
import MLX
import MLXNN
import Tokenizers
import UniformTypeIdentifiers

enum RunnerError: Error {
    /// The request's input is wrong (bad text, missing file): the caller can fix it.
    case invalid(String)
    /// The model failed: a bug or an unsupported checkpoint.
    case failed(String)
    /// Arsenal cancelled the request (Params.checkCancelled()).
    case cancelled
}

/// One request's inputs, as the op's input schema declares them.
struct Params {
    let raw: [String: Any]
    private let cancelled: () -> Bool
    init(_ raw: [String: Any], cancelled: @escaping () -> Bool = { false }) { self.raw = raw; self.cancelled = cancelled }

    /// Call between steps of long work (each layer, chunk or frame): throws when Arsenal cancelled the request.
    func checkCancelled() throws { if cancelled() { throw RunnerError.cancelled } }

    func string(_ key: String) throws -> String {
        guard let value = raw[key] as? String, !value.isEmpty else { throw RunnerError.invalid("\(key) must be a nonempty string.") }
        return value
    }
    func string(_ key: String, default value: String) -> String { raw[key] as? String ?? value }
    func strings(_ key: String) throws -> [String] {
        if let one = raw[key] as? String { return [one] }
        guard let list = raw[key] as? [String], !list.isEmpty else { throw RunnerError.invalid("\(key) must be a string or a list of strings.") }
        return list
    }
    func double(_ key: String, default value: Double) -> Double { (raw[key] as? NSNumber)?.doubleValue ?? value }
    func int(_ key: String, default value: Int) -> Int { (raw[key] as? NSNumber)?.intValue ?? value }
    func bool(_ key: String, default value: Bool) -> Bool { raw[key] as? Bool ?? value }
    func numbers(_ key: String) throws -> [Double] {
        guard let list = raw[key] as? [Any], !list.isEmpty else { throw RunnerError.invalid("\(key) must be a nonempty list of numbers.") }
        return try list.map {
            guard let number = $0 as? NSNumber else { throw RunnerError.invalid("\(key) must hold only numbers.") }
            return number.doubleValue
        }
    }
    /// A file input: Arsenal passes {"path": "/absolute/path"} for a schema property with x-arsenal-file.
    func file(_ key: String) throws -> URL {
        let path = (raw[key] as? [String: Any])?["path"] as? String ?? raw[key] as? String
        guard let path, FileManager.default.fileExists(atPath: path) else { throw RunnerError.invalid("\(key) must be an existing file.") }
        return URL(fileURLWithPath: path)
    }
}

enum Files {
    static func size(_ url: URL) throws -> Int { (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.intValue ?? 0 }
    static func sha256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
    /// config.json (or another JSON file) from the weights folder.
    static func json(_ dir: URL, _ name: String = "config.json") throws -> [String: Any] {
        let url = dir.appendingPathComponent(name)
        guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw RunnerError.failed("\(name) is not a JSON object.")
        }
        return object
    }
}

enum Weights {
    /// Every tensor of every .safetensors file in the folder, floating tensors as `dtype` (float16 by default).
    static func load(_ dir: URL, dtype: DType = .float16) throws -> [String: MLXArray] {
        var all: [String: MLXArray] = [:]
        let files = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path }
        guard !files.isEmpty else { throw RunnerError.failed("No .safetensors file in \(dir.path).") }
        for file in files {
            for (key, value) in try loadArrays(url: file) {
                guard value.dtype.isFloatingPoint else { all[key] = value; continue }
                all[key] = dtype == .float16 && !fitsHalf(value, name: key) ? value.asType(.float32) : value.asType(dtype)
            }
        }
        return all
    }

    /// Float16 holds at most 65,504: a larger weight would become infinity, so such a tensor stays float32.
    static func fitsHalf(_ value: MLXArray, name: String) -> Bool {
        guard value.dtype != .float16 else { return true }
        let largest = abs(value.asType(.float32)).max().item(Float.self)
        if largest <= 65_504 { return true }
        FileHandle.standardError.write("runner: \(name) holds \(largest), beyond float16; kept as float32\n".data(using: .utf8)!)
        return false
    }

    /// Puts checkpoint tensors into a Module. `rename` maps a checkpoint key to the module's key
    /// (return nil to skip a tensor the model does not use). Throws with the exact keys that are missing or unused.
    static func apply(_ weights: [String: MLXArray], to module: Module, rename: (String) -> String? = { $0 }) throws {
        var mapped: [String: MLXArray] = [:]
        for (key, value) in weights { if let name = rename(key) { mapped[name] = value } }
        let shapes = Dictionary(module.parameters().flattened().map { ($0.0, $0.1.shape) }, uniquingKeysWith: { a, _ in a })
        let expected = Set(shapes.keys)
        let missing = expected.subtracting(mapped.keys).sorted(), unused = Set(mapped.keys).subtracting(expected).sorted()
        // PyTorch BatchNorm's step counter is not a weight.
        guard missing.isEmpty, unused.allSatisfy({ $0.hasSuffix("num_batches_tracked") }) else {
            throw RunnerError.failed(
                "Weights do not match the module. Missing: \(missing.prefix(12).joined(separator: ", ")) (\(missing.count)). "
                    + "Unused: \(unused.prefix(12).joined(separator: ", ")) (\(unused.count)).")
        }
        mapped = mapped.filter { expected.contains($0.key) }
        for (key, value) in mapped {
            if let shape = shapes[key], shape != value.shape {
                throw RunnerError.failed("\(key): checkpoint shape \(value.shape), module shape \(shape). PyTorch Conv weights need .transposed(0, 2, 3, 1) (2-D) or .transposed(0, 2, 1) (1-D).")
            }
        }
        module.update(parameters: ModuleParameters.unflattened(mapped))
        module.train(false)  // inference: BatchNorm uses its running statistics, Dropout is off
        eval(module)
    }
}

enum Images {
    /// An image as float32 [height, width, 3], values 0...1, resized to `width`×`height` when given.
    static func load(_ url: URL, width: Int? = nil, height: Int? = nil) throws -> (pixels: MLXArray, width: Int, height: Int) {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw RunnerError.invalid("\(url.lastPathComponent) is not a readable image.")
        }
        let w = width ?? image.width, h = height ?? image.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        guard let context = CGContext(
            data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw RunnerError.failed("Could not draw the image.") }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        let rgba = MLXArray(bytes, [h, w, 4]).asType(.float32) / 255
        return (rgba[0..., 0..., ..<3], image.width, image.height)
    }

    /// Normalizes [H, W, 3] pixels with per-channel mean and std (e.g. ImageNet: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).
    static func normalize(_ pixels: MLXArray, mean: [Float], std: [Float]) -> MLXArray {
        (pixels - MLXArray(mean)) / MLXArray(std)
    }

    /// Writes [H, W, 3] or [H, W] values 0...1 as a PNG.
    static func savePNG(_ pixels: MLXArray, to url: URL) throws {
        let array = pixels.ndim == 2 ? stacked([pixels, pixels, pixels], axis: -1) : pixels
        let h = array.dim(0), w = array.dim(1)
        let alpha = MLXArray.ones([h, w, 1])
        var bytes = (clip(concatenated([array.asType(.float32), alpha], axis: -1), min: 0, max: 1) * 255).round().asType(.uint8).asArray(UInt8.self)
        guard let context = CGContext(
            data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
            let image = context.makeImage(),
            let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw RunnerError.failed("Could not write \(url.lastPathComponent).") }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw RunnerError.failed("Could not write \(url.lastPathComponent).") }
    }
}

enum Audio {
    /// Any audio file as mono float32 samples at `sampleRate`, values -1...1.
    static func load(_ url: URL, sampleRate: Double) throws -> MLXArray {
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url) } catch { throw RunnerError.invalid("\(url.lastPathComponent) is not a readable audio file.") }
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false)!
        guard let converter = AVAudioConverter(from: file.processingFormat, to: format),
            let input = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))
        else { throw RunnerError.failed("Could not convert \(url.lastPathComponent).") }
        try file.read(into: input)
        let capacity = AVAudioFrameCount(Double(input.frameLength) * sampleRate / file.processingFormat.sampleRate) + 1024
        guard let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { throw RunnerError.failed("Out of memory.") }
        var fed = false
        var failure: NSError?
        converter.convert(to: output, error: &failure) { _, status in
            if fed { status.pointee = .endOfStream; return nil }
            fed = true; status.pointee = .haveData; return input
        }
        if let failure { throw RunnerError.failed(failure.localizedDescription) }
        let samples = Array(UnsafeBufferPointer(start: output.floatChannelData![0], count: Int(output.frameLength)))
        return MLXArray(samples)
    }
}

enum Text {
    /// The Hugging Face tokenizer in the weights folder (tokenizer.json and tokenizer_config.json).
    static func tokenizer(_ dir: URL) throws -> Tokenizer {
        let semaphore = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var result: Result<Tokenizer, Error> = .failure(RunnerError.failed("Tokenizer did not load."))
        Task {
            do { result = .success(try await AutoTokenizer.from(modelFolder: dir)) } catch { result = .failure(error) }
            semaphore.signal()
        }
        semaphore.wait()
        return try result.get()
    }
}

/// A float array as a plain list for a JSON reply.
func list(_ array: MLXArray) -> [Double] { array.asType(.float32).asArray(Float.self).map(Double.init) }

