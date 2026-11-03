// SPDX-License-Identifier: MIT
import Foundation
import ImageIO
import IntegrationKit
import MLX
import MLXIdol
import ModelPrecision
import ArsenalWire

/// IDOL: one image → a 3D Gaussian splat (.ply). IDOL's own modules FP16 as IDOL runs them (the FP32 originals cast
/// once at Get; the latent positions stay FP32); DINOv3 ViT-H+ BF16 as IDOL runs it; the FLUX.2 VAE encoder FP16 (in
/// MLX it is closer to FP32 in FP16, 0.4 % relative error, than in BF16, 3.1 %; IDOL's own BF16 run is 0.7 %).
public final class IdolAdapter: NativeModel {
    public let architecture = "idol"
    public let operations: Set<String> = ["image_to_3d"]
    public var peakBytes: Int { Memory.peakMemory }
    private var pipeline: IdolPipeline?
    public var parameters: [String: MLXArray] { pipeline?.parameters ?? [:] }
    public init() {}

    /// `[[asset]]` id → its `into` folder in the converted snapshot.
    public static let components = ["idol-dinov3": "components/idol-dinov3", "idol-flux2-vae": "components/idol-flux2-vae"]

    /// Get: IDOL's FP32 modules → FP16 (latent positions kept FP32), DINOv3 FP32 → BF16, the BF16 VAE → FP16.
    public static var conversion: NativeConversion {
        NativeConversion(recipe: "cast:geometry-idol-v1") { context, cancellation in
            guard context.precision == "fp16" else { throw PrecisionError("IDOL ships its 16-bit format only") }
            try convert(context.source, into: context.destination, cancellation: cancellation) { key in
                IdolFiles.float32Tensors.contains(key) ? .float32 : .float16
            }
            for (id, directory) in components.sorted(by: { $0.key < $1.key }) {
                let target = context.destination.appendingPathComponent(directory)
                guard !FileManager.default.fileExists(atPath: target.path) else { continue }
                guard let asset = context.assets[id] else { throw PrecisionError("Missing component asset " + id) }
                try convert(asset, into: target, cancellation: cancellation) { _ in id == "idol-dinov3" ? .bfloat16 : .float16 }
            }
            return try context.receipt(cancellation: cancellation)
        }
    }

    /// Streams every safetensors file of `source` into `target` with each floating tensor in `dtype(key)`; other files
    /// are copied. A cast that leaves a non-finite value is refused.
    static func convert(_ source: URL, into target: URL, cancellation: NativeCancellation, dtype: (String) -> DType) throws {
        for path in try NativeConversionContext.files(in: source) {
            try cancellation.check()
            if StoredNames.reserved.contains(URL(fileURLWithPath: path).lastPathComponent) { continue }
            let input = source.appendingPathComponent(path), output = target.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
            guard input.pathExtension == "safetensors" else { try FileManager.default.copyItem(at: input, to: output); continue }
            let archive = try TensorArchive(input), writer = try TensorWriter(output)
            for key in archive.entries.keys.sorted() {
                try autoreleasepool {
                    try cancellation.check()
                    let tensor = try archive.array(key)
                    var value = tensor
                    if [.float16, .bfloat16, .float32, .float64].contains(tensor.dtype) {
                        value = tensor.asType(dtype(key), stream: .cpu)
                        guard !MLX.any(.!MLX.isFinite(value, stream: .cpu), stream: .cpu).item(Bool.self) else { throw PrecisionError("Non-finite tensor after the cast: " + key) }
                    }
                    eval(value); try writer.append(key, value)
                }
                Memory.clearCache()
            }
            try writer.finish()
        }
    }

    public func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        try cancellation.check()
        guard context.precision == "fp16" else { throw WorkerFailure(code: "invalid_request", message: "IDOL ships its 16-bit format only.") }
        if context.randomWeights {
            pipeline = try IdolPipeline.lab()
        } else {
            do {
                pipeline = try IdolPipeline.load(context.weights, cancellation: { try cancellation.check() })
            } catch let error as IdolError {
                throw WorkerFailure(code: "missing_asset", message: error.description)
            }
        }
        try NativeParameters.requireHalf(parameters)
        try cancellation.check()
    }

    public func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operation == "image_to_3d", let pipeline else { throw WorkerFailure(code: "not_loaded", message: "Load IDOL before making a splat.") }
        guard let path = input["image"]?.object?["path"]?.string, let output = input["out_dir"]?.string else {
            throw WorkerFailure(code: "invalid_request", message: "image_to_3d requires an image and an output directory.")
        }
        let request = try Self.request(input, lab: pipeline.imageSize < 1024)
        try cancellation.check()
        let image = try NativeMedia.image(URL(fileURLWithPath: path))
        let result: IdolResult
        do {
            result = try pipeline.generate(
                image, request: request, cancellation: { try cancellation.check() },
                progress: { label, fraction in
                    try? emit(.json(["event": "progress", "message": .string(label), "label": .string(label), "step": .number((fraction * 100).rounded()), "total": 100]))
                })
        } catch let error as IdolError {
            throw WorkerFailure(code: "invalid_request", message: error.description)
        }
        try cancellation.check()
        let root = URL(fileURLWithPath: output), ply = root.appendingPathComponent("splat.ply"), preview = root.appendingPathComponent("preview.png")
        do { try result.gaussians.writePLY(to: ply) } catch let error as IdolError {
            throw WorkerFailure(code: "invalid_model_output", message: error.description)
        }
        try cancellation.check()
        // the splat seen from the direction IDOL estimated for the picture, framed to fill the image
        let camera = result.camera, fov = camera.radius < 100 ? camera.fov : 40
        let side = pipeline.imageSize < 1024 ? 64 : 768
        let pixels = IdolPreview.render(
            result.gaussians, size: side, yaw: camera.yaw, pitch: camera.pitch, radius: IdolPreview.fittingRadius(result.gaussians, fov: fov), fov: fov)
        try IdolPreview.writePNG(pixels, width: side, height: side, channels: 4, to: preview)
        return [
            "files": [
                .object(try NativeMedia.file(ply, root: root, mime: "application/x-ply", id: "splat")),
                .object(try NativeMedia.file(preview, root: root, mime: "image/png", id: "preview"))
            ],
            "gaussians": .number(Double(result.gaussians.count)),
            "camera": ["yaw": .number(camera.yaw), "pitch": .number(camera.pitch), "radius": .number(camera.radius), "fov": .number(camera.fov)]
        ]
    }

    /// Request options with their reference defaults and limits (sample.py).
    static func request(_ input: JSONObject, lab: Bool) throws -> IdolRequest {
        func number(_ key: String, _ fallback: Double, _ range: ClosedRange<Double>, integer: Bool = false) throws -> Double {
            guard let value = input[key] else { return fallback }
            guard let n = value.number, n.isFinite, range.contains(n), !integer || n.rounded(.towardZero) == n else {
                throw WorkerFailure(code: "invalid_request", message: "IDOL \(key) must be \(integer ? "an integer" : "a number") in \(range.lowerBound)...\(range.upperBound).")
            }
            return n
        }
        var r = IdolRequest()
        let mode = input["mode"]?.string ?? "synthetic"
        guard let margin = IdolImage.cropMargin[mode] else { throw WorkerFailure(code: "invalid_request", message: "IDOL mode must be synthetic or real.") }
        r.margin = try number("crop_margin", margin, 1...4)
        r.seed = UInt64(try number("seed", 42, 0...4_294_967_295, integer: true))
        r.steps = Int(try number("steps", 50, 1...200, integer: true))
        r.guidance = Float(try number("cfg_strength", 3, 0...20))
        r.rescaleT = Float(try number("rescale_t", 3, 0.1...20))
        let count = Int(try number("num_gaussians", 262_144, 32_768...1_048_576, integer: true))
        r.gaussians = lab ? 2048 : count / 32 * 32
        return r
    }

    public func unload() { pipeline = nil; Memory.clearCache() }
}
