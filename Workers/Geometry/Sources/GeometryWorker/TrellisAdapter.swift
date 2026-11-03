// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXTrellis
import ArsenalWire
import ImageIO
import UniformTypeIdentifiers

public final class TrellisAdapter: NativeModel {
    public let architecture = "trellis2"
    public let operations: Set<String> = ["image_to_3d"]
    public var peakBytes: Int { Memory.peakMemory }
    private var banks: [String: TrellisWeights] = [:]
    private var engine: TrellisEngine?
    private var lab = false
    public var parameters: [String: MLXArray] {
        var all: [String: MLXArray] = [:]
        for (component, bank) in banks { for (name, value) in bank.checkpointParameters { all[component + "." + name] = value } }
        return all
    }
    public init() {}
    public static let files = [
        "struct_flow": "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
        "shape_flow_512": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        "shape_flow_1024": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
        "tex_flow_512": "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
        "tex_flow_1024": "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
        "shape_dec": "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        "tex_dec": "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"
    ]
    /// Both auxiliary repositories are `[[asset]] into` components: Get converts them once into `components/<id>` of the
    /// FP16 base; int8/int4 receive that base (components included) and pack it. Load reads only `context.weights`.
    public static let components = ["trellis-structure-decoder": "components/trellis-structure-decoder", "trellis-dinov3": "components/trellis-dinov3"]
    public static var baseConversion: NativeConversion {
        NativeConversion(recipe: "cast:geometry-trellis-v1", eligible: TrellisWeights.eligible, components: components)
    }
    public func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        try cancellation.check()
        lab = context.randomWeights && context.config["full_inventory"]?.bool != true
        let spec = TrellisShapeSpec(lab: lab)
        for (index, component) in spec.components.keys.sorted().enumerated() {
            try cancellation.check()
            let bank = TrellisWeights(shapes: spec.components[component]!)
            banks[component] = bank
            if context.randomWeights {
                if lab { try bank.initializeLab(precision: context.precision, seed: UInt64(101 + index)) } else { try NativeParameters.initializeRandom(bank) }
            } else {
                let archive: URL
                if let file = Self.files[component] {
                    archive = context.weights.appendingPathComponent(file)
                } else if component == "struct_dec" {
                    archive = context.weights.appendingPathComponent("components/trellis-structure-decoder/ckpts/ss_dec_conv3d_16l8_fp16.safetensors")
                } else if component == "dino" {
                    archive = context.weights.appendingPathComponent("components/trellis-dinov3/model.safetensors")
                } else {
                    throw WorkerFailure(code: "missing_asset", message: "Missing converted TRELLIS component: " + component)
                }
                try bank.load(archives: [archive], precision: context.precision, cancellation: cancellation)
            }
        }
        if context.randomWeights && !lab { return }
        let candidates = [context.weights.appendingPathComponent("pipeline.json"), context.folder.appendingPathComponent("pipeline.json")]
        guard let pipelineFile = candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }),
            let pipeline = try JSONSerialization.jsonObject(with: Data(contentsOf: pipelineFile)) as? [String: Any]
        else {
            throw WorkerFailure(code: "missing_asset", message: "TRELLIS requires its pinned pipeline.json normalization config.")
        }
        engine = try TrellisEngine(banks: banks, spec: spec, pipeline: pipeline)
        try NativeParameters.requireHalf(parameters); try cancellation.check()
    }
    public func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operation == "image_to_3d", let engine else { throw WorkerFailure(code: "not_loaded", message: "Load TRELLIS.2 before generating a mesh.") }
        guard let path = input["image"]?.object?["path"]?.string, let output = input["out_dir"]?.string else {
            throw WorkerFailure(code: "invalid_request", message: "image_to_3d requires an RGBA image and output directory.")
        }
        func integer(_ key: String, _ fallback: Int, _ range: ClosedRange<Int>) throws -> Int {
            guard let value = input[key] else { return fallback }
            guard let number = value.number, number.isFinite, number.rounded(.towardZero) == number,
                number >= Double(range.lowerBound), number <= Double(range.upperBound)
            else {
                throw WorkerFailure(code: "invalid_request", message: "TRELLIS \(key) must be an integer in \(range).")
            }
            return Int(number)
        }
        let resolution = try integer("resolution", 1024, 512...1536)
        let textureSize = try integer("texture_size", 1024, 256...4096)
        let steps = try integer("steps", 12, 1...100)
        let seed = UInt64(try integer("seed", 0, 0...4294967295))
        let targetFaces = try integer("target_faces", 120000, 32...500000)
        guard [512, 1024, 1536].contains(resolution), [256, 512, 1024, 2048, 4096].contains(textureSize) else {
            throw WorkerFailure(code: "invalid_request", message: "Unsupported TRELLIS resolution or texture size.")
        }
        try cancellation.check()
        let image = try NativeMedia.image(URL(fileURLWithPath: path))
        // Both user-provided RGBA and the host's future BiRefNet result enter this exact path.
        let pixels = try TrellisImage.pixels(image, size: lab ? 32 : 512)
        let highPixels = resolution > 512 ? try TrellisImage.pixels(image, size: lab ? 32 : 1024) : nil
        // The engine names each of its stages once: six, plus "Refining shape" above 512.
        let stages = resolution > 512 ? 7 : 6
        var stage = 0
        let mesh = try engine.generate(
            pixels: pixels, highResolutionPixels: highPixels, resolution: resolution, textureSize: textureSize,
            targetFaces: targetFaces, steps: steps, seed: seed, cancellation: cancellation,
            log: { message in
                stage = Swift.min(stage + 1, stages)
                try? emit(
                    .json([
                        "event": "progress", "message": .string(message), "label": .string(String(message.prefix(128))), "step": .number(Double(stage)),
                        "total": .number(Double(stages))
                    ]))
            }
        ).yUp()
        try cancellation.check()
        let root = URL(fileURLWithPath: output), glb = root.appendingPathComponent("mesh.glb"), preview = root.appendingPathComponent("preview.png")
        try GLTFExport.writeGLB(
            to: glb, positions: mesh.vertices, indices: mesh.faces, normals: mesh.normals, uvs: mesh.uvs,
            baseColorRGBA: (mesh.texRGBA, mesh.atlasSize, mesh.atlasSize),
            metallicRoughnessRGBA: mesh.mrRGBA.map { ($0, mesh.atlasSize, mesh.atlasSize) })
        try cancellation.check()
        try TrellisPreview.write(mesh, to: preview, cancellation: cancellation)
        try cancellation.check()
        return [
            "files": [
                .object(try NativeMedia.file(glb, root: root, mime: "model/gltf-binary", id: "mesh")),
                .object(try NativeMedia.file(preview, root: root, mime: "image/png", id: "preview"))
            ]
        ]
    }
    public func unload() { engine = nil; banks.removeAll(); Memory.clearCache() }
}
