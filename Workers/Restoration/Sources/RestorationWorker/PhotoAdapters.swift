// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import ArsenalWire
import DDColorNative
import LaMaNative

/// Loads one flat safetensors snapshot in the selected half format (Get already stored it there).
private func halfWeights(_ context: NativeLoadContext) throws -> [String: MLXArray] {
    let half: DType = context.precision == "bf16" ? .bfloat16 : .float16
    guard ["fp16", "bf16"].contains(context.precision) else { throw MediaBridge.error("This model runs at its 16-bit format only.") }
    let arrays = try MLX.loadArrays(url: context.weights.appendingPathComponent("model.safetensors"))
    for (key, value) in arrays where value.dtype != half {
        throw WorkerFailure(code: "precision", message: "Prepare the \(context.precision) snapshot before Load: \(key) is \(value.dtype)")
    }
    return arrays
}

private func outputFile(_ root: URL, _ name: String, label: String) throws -> JSONValue {
    var file = try NativeMedia.file(root.appendingPathComponent(name), root: root, mime: "image/png", id: "image")
    file["label"] = .string(label)
    return .object(file)
}

/// DDColor (ConvNeXt-L, ModelScope weights): colourizes a photo. The provider's pipeline keeps the photo's own
/// luminance at full size and predicts colour at `input_size` (512 by default).
final class DDColorAdapter: NativeModel {
    let architecture = "ddcolor"
    let operations: Set<String> = ["colorize"]
    static let inputSizes: Set<Int> = [256, 512, 768, 1024]
    private var model: DDColorModel?
    private var weights: [String: MLXArray] = [:]
    var peakBytes: Int { Memory.peakMemory }
    var parameters: [String: MLXArray] { weights }
    static func eligible(_ name: String, _ shape: [Int]) -> Bool { false }

    func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        try cancellation.check()
        let half: DType = context.precision == "bf16" ? .bfloat16 : .float16
        let w = context.randomWeights ? DDColorWeights.random(dtype: half) : try halfWeights(context)
        eval(Array(w.values))
        model = try DDColorModel(weights: w); weights = w
    }

    func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operation == "colorize", let model else { throw MediaBridge.error("DDColor is not loaded or the operation is unsupported.") }
        let root = try MediaBridge.output(input)
        let size = Int(exactly: input["input_size"]?.number ?? 512) ?? -1
        guard Self.inputSizes.contains(size) else { throw MediaBridge.error("input_size must be 256, 512, 768 or 1024.") }
        let photo = try PhotoIO.rgb(try MediaBridge.path(input["image"]))
        try cancellation.check()
        try MediaBridge.stage(1, of: 2, "Colorizing", emit: emit)
        let out = DDColorPipeline.colorize(model, rgb: photo.bytes, width: photo.width, height: photo.height, inputSize: size)
        try cancellation.check()
        try PhotoIO.png(out, width: photo.width, height: photo.height, space: photo.colorSpace, at: root.appendingPathComponent("colorized.png"))
        try MediaBridge.stage(2, of: 2, "Colorizing", emit: emit)
        return ["files": [try outputFile(root, "colorized.png", label: "Colorized")]]
    }
    func unload() { model = nil; weights = [:] }
}

/// Big-LaMa: fills the masked pixels of a photo. advimman/lama predict: mask pixels above zero are the hole, the
/// image is padded symmetrically to a multiple of eight, only the hole changes, and 8-bit output truncates as
/// `np.clip(x * 255, 0, 255).astype("uint8")`.
final class LaMaAdapter: NativeModel {
    let architecture = "lama"
    let operations: Set<String> = ["inpaint"]
    private var model: LaMaModel?
    private var weights: [String: MLXArray] = [:]
    var peakBytes: Int { Memory.peakMemory }
    var parameters: [String: MLXArray] { weights }
    static func eligible(_ name: String, _ shape: [Int]) -> Bool { false }

    /// Get-time preparation: fold the float32 source's BatchNorms, then store the selected half.
    static let conversion = NativeConversion(recipe: "cast:restoration-lama-v1") { context, cancellation in
        try cancellation.check()
        guard context.precision == "fp16" || context.precision == "bf16" else { throw MediaBridge.error("LaMa ships at its 16-bit format only.") }
        let source = try MLX.loadArrays(url: context.source.appendingPathComponent("model.safetensors"), stream: .cpu)
        for (key, value) in source where value.dtype != .float32 && !key.hasSuffix("num_batches_tracked") {
            throw WorkerFailure(code: "precision", message: "LaMa folding needs the float32 source; \(key) is \(value.dtype)")
        }
        let folded = try LaMaFold.fold(source).mapValues { $0.asType(context.half, stream: .cpu) }
        for (key, value) in folded where MLX.any(.!MLX.isFinite(value, stream: .cpu), stream: .cpu).item(Bool.self) {
            throw WorkerFailure(code: "precision", message: "Folded LaMa tensor overflows \(context.precision): \(key)")
        }
        try cancellation.check()
        try MLX.save(arrays: folded, url: context.destination.appendingPathComponent("model.safetensors"))
        return try context.receipt(cancellation: cancellation)
    }

    func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        try cancellation.check()
        let half: DType = context.precision == "bf16" ? .bfloat16 : .float16
        let w = context.randomWeights ? LaMaFold.random(dtype: half) : try halfWeights(context)
        eval(Array(w.values))
        model = try LaMaModel(weights: w); weights = w
    }

    func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operation == "inpaint", let model else { throw MediaBridge.error("LaMa is not loaded or the operation is unsupported.") }
        let root = try MediaBridge.output(input)
        let photo = try PhotoIO.rgb(try MediaBridge.path(input["image"]))
        try Self.checkSize(width: photo.width, height: photo.height)
        let mask = try PhotoIO.mask(try MediaBridge.path(input["mask"]), photo: photo)
        guard mask.width == photo.width, mask.height == photo.height else {
            throw MediaBridge.error("The mask must be the image's size (\(photo.width)×\(photo.height)); it is \(mask.width)×\(mask.height).")
        }
        guard mask.bytes.contains(where: { $0 > 0 }) else { throw MediaBridge.error("The mask marks no pixels to fill.") }
        try cancellation.check()
        try MediaBridge.stage(1, of: 2, "Filling", emit: emit)
        let out = LaMaPipeline.inpaint(model, rgb: photo.bytes, mask: mask.bytes, width: photo.width, height: photo.height)
        try cancellation.check()
        try PhotoIO.png(out, width: photo.width, height: photo.height, space: photo.colorSpace, at: root.appendingPathComponent("inpainted.png"))
        try MediaBridge.stage(2, of: 2, "Filling", emit: emit)
        return ["files": [try outputFile(root, "inpainted.png", label: "Filled")]]
    }
    func unload() { model = nil; weights = [:] }

    /// One untiled, uncancellable generator pass: checked against the makers' graph up to 3840×2160; past that it is unverified,
    /// and 4K already drifts in tint, so larger photos are refused rather than run.
    static let maxPixels = 3840 * 2160
    static func checkSize(width: Int, height: Int) throws {
        guard width >= 16, height >= 16 else { throw MediaBridge.error("The image must be at least 16 pixels on each side.") }
        guard width * height <= maxPixels else {
            let mp = (Double(width * height) / 1e5).rounded() / 10
            throw MediaBridge.error("Big LaMa fills photos up to 8.3 megapixels (3840×2160); this one is \(width)×\(height) (\(mp) MP). Downscale it first.")
        }
    }
}
