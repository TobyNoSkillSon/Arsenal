// SPDX-License-Identifier: MIT
// Synthetic graph used only by the integration package's service tests; not bundled with Arsenal.
import Foundation
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire
import ModelPrecision
import CoreGraphics
import ImageIO

final class Graph: Module {
    @ModuleInfo var projection: Linear
    override init() { _projection.wrappedValue = Linear(64, 4); super.init() }
}
final class Adapter: NativeModel {
    let architecture = "integration_fixture"
    let operations: Set<String> = ["project", "upscale_image", "project_image", "remove_background"]
    var model = Graph()
    var half: DType = .float16
    var peakBytes: Int { Memory.peakMemory }
    var parameters: [String: MLXArray] { Dictionary(uniqueKeysWithValues: model.parameters().flattened()) }
    func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        half = context.config["serving_dtype"]?.string == "bf16" ? .bfloat16 : .float16
        if context.architecture != nil {
            let component = try TensorArchive(context.weights.appendingPathComponent("codec/nested/component.safetensors"))
            let value = try component.array("spatial.weight")
            guard value.dtype == half, value.item(Float.self) == 2,
                FileManager.default.fileExists(atPath: context.weights.appendingPathComponent("codec/tokenizer/config.json").path)
            else { throw PrecisionError("Converted component missing") }
        }
        _ = try NativePrecisionTier(context.precision, half: half).load(
            model, archives: [context.weights.appendingPathComponent("model.safetensors")], map: { ($0, $1) }, eligible: { _, _ in true })
    }
    func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        if operation == "remove_background" {
            let original = input["image"]?["path"]?.string ?? ""
            let root = URL(fileURLWithPath: input["out_dir"]!.string!), file = root.appendingPathComponent("cutout.png")
            var pixels: [UInt8] = [255, 0, 0, 255, 0, 0, 0, original.contains("opaque") ? 255 : 0]
            let image = pixels.withUnsafeMutableBytes { buffer in
                CGContext(
                    data: buffer.baseAddress, width: 2, height: 1, bitsPerComponent: 8, bytesPerRow: 8, space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!.makeImage()!
            }
            let destination = CGImageDestinationCreateWithURL(file as CFURL, "public.png" as CFString, 1, nil)!
            CGImageDestinationAddImage(destination, image, nil)
            guard CGImageDestinationFinalize(destination) else { throw PrecisionError("PNG fixture failed") }
            return ["files": [.object(try NativeMedia.file(file, root: root, mime: "image/png", id: original.contains("wrong-id") ? "wrong" : "foreground"))]]
        }
        if operation == "project_image" {
            let url = URL(fileURLWithPath: input["image"]!["path"]!.string!), image = try NativeMedia.image(url)
            guard image.width == 2, [.first, .last, .premultipliedFirst, .premultipliedLast].contains(image.alphaInfo) else {
                throw PrecisionError("Target did not receive prepared RGBA")
            }
            let value = model.projection(MLXArray.ones([1, 64], dtype: half)); eval(value)
            return ["values": .array(value.asArray(Float.self).map { .number(Double($0)) }), "prepared_path": .string(url.path)]
        }
        if operation == "upscale_image" {
            let root = URL(fileURLWithPath: input["out_dir"]!.string!), file = root.appendingPathComponent("output.png")
            try Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aJ1sAAAAASUVORK5CYII=")!.write(to: file)
            var metadata = try NativeMedia.file(file, root: root, mime: "image/png", id: "output")
            if input["omit_id"]?.bool == true { metadata["id"] = nil }
            return ["files": [.object(metadata)]]
        }
        let value = model.projection(MLXArray.ones([1, 64], dtype: half)); eval(value)
        return ["values": .array(value.asArray(Float.self).map { .number(Double($0)) })]
    }
    func unload() { model = Graph() }
}
extension Adapter: NativeOutputParsing {
    /// The fixture's "generated text" for `project` is a JSON number array, parsed into the values `run` returns.
    func parseOutput(_ operation: String, input: JSONObject, raw: String) throws -> JSONObject {
        guard operation == "project", let values = (try? JSONSerialization.jsonObject(with: Data(raw.utf8))) as? [Double], !values.isEmpty else {
            throw WorkerFailure(code: NativeOutputParsingFailure.code, message: "The fixture output is not a nonempty number array.")
        }
        return ["values": .array(values.map { .number($0) })]
    }
}
let components: (NativeConversionContext, NativeCancellation) throws -> JSONObject = { context, cancellation in
    try context.convertTree(from: context.source, cancellation: cancellation, eligible: { _, _ in true })
    if !FileManager.default.fileExists(atPath: context.destination.appendingPathComponent("codec").path) {
        guard let asset = context.assets["codec"] else { throw PrecisionError("Missing codec component") }
        try context.convertTree(from: asset, into: "codec", cancellation: cancellation, eligible: { _, _ in false })
    }
    return try context.receipt(cancellation: cancellation)
}
NativeFamilyWorker.run(
    family: "integration-fixture", kinds: ["images"], operations: ["project", "upscale_image", "project_image", "remove_background"],
    registrations: [
        NativeRegistration(["integration_fixture"], conversion: NativeConversion(recipe: "fixture-components", convert: components)) { Adapter() },
        NativeRegistration([], conversion: NativeConversion(recipe: "cast:fixture-components", convert: components)) { Adapter() },
        NativeRegistration([], conversion: NativeConversion(recipe: "fixture-components-bf16", half: .bfloat16, convert: components)) { Adapter() },
        NativeRegistration([], conversion: NativeConversion(recipe: "cast:fixture-components-bf16", half: .bfloat16, convert: components)) { Adapter() }
    ])
