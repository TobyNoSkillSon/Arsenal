// SPDX-License-Identifier: MIT
import CoreGraphics
import Foundation
import GeometryWorker
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire
import Testing
@testable import MLXTrellis

@Suite(.serialized)
struct TrellisTests {
    @Test func vendorInventoriesIncludeAllNineComponents() {
        let spec = TrellisShapeSpec(lab: false).components
        #expect(spec.count == 9)
        #expect(spec.values.reduce(0) { $0 + $1.count } == 4265)
        #expect(spec["dino"]?["embeddings.mask_token"] == [1, 1, 1024])
        #expect(spec["shape_dec"]?["blocks.0.4.to_subdiv.weight"] == [8, 1024])
        #expect(spec["tex_dec"]?["blocks.0.4.to_subdiv.weight"] == nil)
    }

    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func halfNormalizationAvoidsSquaredOverflow() {
        NativeRuntime.prepare()
        let input = MLXArray([Float(1000), 1000], [1, 1, 1, 2]).asType(.float16)
        let normalized = trellisQKNorm(input, gamma: MLXArray([Float(2), 3], [1, 2]).asType(.float16), numHeads: 1, headDim: 2)
        #expect(normalized.dtype == .float16)
        #expect(normalized.asType(.float32).asArray(Float.self) == [2, 3])
    }

    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func streamedComponentsLoadStoredPacksAndKeepBiasOrder() throws {
        NativeRuntime.prepare()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("trellis-component-test-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source"), assets = ["trellis-structure-decoder", "trellis-dinov3"]
        try FileManager.default.createDirectory(at: source.appendingPathComponent("ckpts"), withIntermediateDirectories: true)
        let values = [Float](repeating: 0.125, count: 64) + [Float](repeating: 0.25, count: 64)
        let arrays = ["projection.weight": MLXArray(values, [2, 64]), "projection.bias": MLXArray([Float(3), 7])]
        try MLX.save(arrays: arrays, url: source.appendingPathComponent("ckpts/model.safetensors"))
        try Data("{\"normalization\":true}".utf8).write(to: source.appendingPathComponent("pipeline.json"))
        var assetPaths: JSONObject = [:]
        for id in assets {
            let path = root.appendingPathComponent(id)
            try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true)
            try MLX.save(arrays: arrays, url: path.appendingPathComponent("model.safetensors"))
            assetPaths[id] = .string(path.path)
        }
        func convert(_ precision: String, from: URL, to: URL, withAssets: Bool) throws {
            _ = try TrellisAdapter.baseConversion.run(
                [
                    "src_dir": .string(from.path), "dst_dir": .string(to.path), "dtype": .string(precision),
                    "group_size": 64, "recipe": "cast:geometry-trellis-v1", "assets": .object(withAssets ? assetPaths : [:])
                ],
                cancellation: NativeCancellation())
        }
        let half = root.appendingPathComponent("fp16")
        try convert("fp16", from: source, to: half, withAssets: true)
        for precision in ["fp16", "int8", "int4"] {
            let directory = root.appendingPathComponent(precision)
            if precision != "fp16" { try convert(precision, from: half, to: directory, withAssets: false) }
            for file in ["ckpts/model.safetensors"] + assets.map({ "components/" + $0 + "/model.safetensors" }) {
                let bank = TrellisWeights(shapes: ["projection.weight": [2, 64], "projection.bias": [2]])
                try bank.load(archives: [directory.appendingPathComponent(file)], precision: precision, cancellation: NativeCancellation())
                let result = bank.matrix("projection.weight")(MLXArray.ones([1, 64], dtype: .float16)) + bank["projection.bias"]!
                #expect(result.asType(.float32).asArray(Float.self) == [11, 23])
                #expect(bank.checkpointParameters["projection.weight"]?.dtype == (precision == "fp16" ? .float16 : .uint32))
            }
            #expect(FileManager.default.fileExists(atPath: directory.appendingPathComponent("pipeline.json").path))
        }
    }

    /// Regression (22 Oct, "TRELLIS predicted no occupied voxels" on every image): the trained DINOv3-L residual stream
    /// reaches about 1.5e5 from layer 0, past FP16's 65,504. An FP16 stream became inf, the final LayerNorm made the
    /// condition NaN, and the sparse-structure decoder had no positive logit. Tiny graph on the CPU, same overflow.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func dinoResidualBeyondHalfRangeStaysFinite() throws {
        NativeRuntime.prepare()
        try Device.withDefaultDevice(.cpu) {
            let bank = TrellisWeights(shapes: TrellisShapeSpec(lab: true).components["dino"]!)
            try bank.initializeLab(precision: "fp16", seed: 7)
            // MLP output 4e4 (FP16-representable) times LayerScale 4 puts the residual at 1.6e5, as in the trained layer 0.
            let c = 64
            let updates = [
                bank.internalName("layer.0.mlp.down_proj.bias")!: MLXArray.ones([c], dtype: .float16) * 40000,
                bank.internalName("layer.0.layer_scale2.lambda1")!: MLXArray.ones([c], dtype: .float16) * 4
            ]
            try bank.update(parameters: ModuleParameters.unflattened(updates), verify: [.shapeMismatch])
            let dino = DINOv3(weights: bank, hidden: c, numHeads: 4, numLayers: 1)
            let pixels = MLXArray((0..<3072).map { Float($0 % 17) / 8 - 1 }, [1, 3, 32, 32]).asType(.float16)
            let half = dino.block(dino.embed(pixels), layer: 0, cos: dino.ropeCosSin(2, 2).0, sin: dino.ropeCosSin(2, 2).1)
            #expect(!MLX.all(MLX.isFinite(half)).item(Bool.self)) // the old FP16 stream overflows here
            let condition = try dino(pixels)
            #expect(condition.dtype == .float16)
            #expect(MLX.all(MLX.isFinite(condition)).item(Bool.self))
            #expect(MLX.abs(condition.asType(.float32)).max().item(Float.self) < 100)
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func rgbaValidationAndLanczosReference() throws {
        NativeRuntime.prepare()
        func image(_ alpha: UInt8) -> CGImage {
            let bytes: [UInt8] = [20, 40, 60, alpha, 80, 100, 120, alpha, 20, 40, 60, alpha, 80, 100, 120, alpha]
            return CGImage(
                width: 2, height: 2, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 8,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: CGDataProvider(data: Data(bytes) as CFData)!, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        }
        #expect(throws: WorkerFailure.self) { try TrellisImage.pixels(image(255), size: 32) }
        #expect(throws: WorkerFailure.self) { try TrellisImage.pixels(image(0), size: 32) }
        let gray = CGImage(
            width: 2, height: 2, bitsPerComponent: 8, bitsPerPixel: 16, bytesPerRow: 4,
            space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
            provider: CGDataProvider(data: Data([80, 0, 80, 255, 80, 255, 80, 255]) as CFData)!,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        #expect(throws: WorkerFailure.self) { try TrellisImage.pixels(gray, size: 32) }
        // Equal box coverage averages two symmetric Lanczos contributions; constant rows remain constant.
        #expect(
            TrellisResample.resize(
                [0, 255, 0, 255], width: 2, height: 2, channels: 1,
                targetWidth: 1, targetHeight: 1, filter: .lanczos) == [128])
    }
}
