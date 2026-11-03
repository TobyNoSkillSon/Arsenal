// SPDX-License-Identifier: MIT
import CoreGraphics
import Foundation
import ImageIO
import IntegrationKit
import MLX
import ArsenalWire
import Testing
import UniformTypeIdentifiers
@testable import GeometryWorker
@testable import MLXIdol

/// IDOL without weights: host arithmetic, request limits, Get's precision policy and the whole pipeline at miniature
/// size with random weights.
@Suite(.serialized)
struct IdolTests {
    // MARK: host arithmetic (no GPU)

    @Test func systematicSplitKeepsTheCountAndFollowsTheProbabilities() {
        let probs: [Float] = [0.5, 0, 0.25, 0.25, 0, 0, 0, 0]
        let split = IdolOctree.systematic(probs[...], count: 8, u0: 0.01)
        #expect(split.reduce(0, +) == 8)
        #expect(split == [4, 0, 2, 2, 0, 0, 0, 0])
        // a zero row falls back to uniform; zero items give zero counts
        #expect(IdolOctree.systematic([Float](repeating: 0, count: 8)[...], count: 8, u0: 0.05) == [Int](repeating: 1, count: 8))
        #expect(IdolOctree.systematic(probs[...], count: 0, u0: 0) == [Int](repeating: 0, count: 8))
        // one item lands where the offset falls in the cumulative distribution
        #expect(IdolOctree.systematic(probs[...], count: 1, u0: 0.6) == [0, 0, 1, 0, 0, 0, 0, 0])
    }

    @Test func timeGridIsTheReferenceShiftedSchedule() {
        let grid = IdolPipeline.timeGrid(steps: 4, rescale: 3)
        #expect(grid.count == 5)
        #expect(grid.first == 1 && grid.last == 0)
        // 3t / (1 + 2t) at t = 0.75, 0.5, 0.25
        for (value, t) in zip(grid[1...3], [Float(0.75), 0.5, 0.25]) { #expect(abs(value - 3 * t / (1 + 2 * t)) < 1e-6) }
    }

    @Test func cameraDecodesYawPitchRadiusAndFieldOfView() {
        let c = IdolCamera(latent: [0, 1, 0, 0.5, 0.2])
        #expect(abs(c.yaw) < 1e-9 && abs(c.pitch) < 1e-9)
        #expect(abs(c.radius - 2) < 1e-6)
        #expect(abs(c.fov - 2 * atan(0.1) * 180 / .pi) < 1e-5)
        let ortho = IdolCamera(latent: [0, 0, 0, 0, 1])
        #expect(ortho.radius == 1000 && abs(ortho.pitch - .pi / 2) < 1e-9)
    }

    @Test func latentPositionTableMatchesTheReferenceLayout() {
        // channels 18: 3 coordinates × (3 sines + 3 cosines), frequencies 2^0, 2^1, 2^2 (max resolution 16 > 3)
        let table = PositionEmbedding.latentTable([0.25, 0, 0.5], channels: 18)
        #expect(table.count == 18)
        #expect(abs(table[0] - 1) < 1e-6) // sin(2π · 0.25)
        #expect(abs(table[3] - 0) < 1e-6) // cos(2π · 0.25)
        #expect(abs(table[6] - 0) < 1e-6 && abs(table[9] - 1) < 1e-6) // y = 0
        #expect(abs(table[12] - 0) < 1e-6 && abs(table[15] + 1) < 1e-6) // sin(π), cos(π)
    }

    @Test func requestDefaultsAndLimitsFollowTheReference() throws {
        let r = try IdolAdapter.request([:], lab: false)
        #expect(r.steps == 50 && r.seed == 42 && r.guidance == 3 && r.rescaleT == 3 && r.gaussians == 262_144 && r.margin == 1.2)
        #expect(try IdolAdapter.request(["mode": "real"], lab: false).margin == 1.7)
        #expect(try IdolAdapter.request(["num_gaussians": 1_000_001], lab: false).gaussians == 1_000_000)
        for bad: JSONObject in [["mode": "photo"], ["steps": 0], ["steps": 2.5], ["num_gaussians": 1024], ["cfg_strength": -1], ["seed": -1]] {
            #expect(throws: WorkerFailure.self) { try IdolAdapter.request(bad, lab: false) }
        }
    }

    @Test func registryOffersIDOLWithItsOwnConverter() {
        let registration = GeometryWorker.registrations.first { $0.modelTypes.contains("idol") }
        #expect(registration?.conversion?.recipe == "cast:geometry-idol-v1")
        #expect(registration?.make().operations == ["image_to_3d"])
    }

    // MARK: GPU, no weights

    /// Get: IDOL's FP32 tensors become FP16 except the latent positions; DINOv3 becomes BF16, the VAE FP16.
    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func getKeepsEachPartInIDOLsOwnPrecision() throws {
        NativeRuntime.prepare()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("idol-get-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source"), dino = root.appendingPathComponent("dino"), vae = root.appendingPathComponent("vae")
        for dir in [source, dino, vae.appendingPathComponent("vae")] { try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true) }
        try MLX.save(
            arrays: ["pos_pe": MLXArray([Float(0.123456789), 0.5, 0.75], [1, 1, 3]), "input_layer.weight": MLXArray([Float(1.0001), 2], [1, 2])],
            url: source.appendingPathComponent("denoiser.safetensors"))
        try Data("{}".utf8).write(to: source.appendingPathComponent("config.json"))
        try MLX.save(arrays: ["layer.0.norm1.weight": MLXArray([Float(3), 4])], url: dino.appendingPathComponent("model.safetensors"))
        try MLX.save(arrays: ["encoder.conv_in.bias": MLXArray([Float(5), 6]).asType(.bfloat16)], url: vae.appendingPathComponent("vae/diffusion_pytorch_model.safetensors"))
        let out = root.appendingPathComponent("out")
        let receipt = try IdolAdapter.conversion.run(
            [
                "src_dir": .string(source.path), "dst_dir": .string(out.path), "dtype": "fp16", "group_size": 64, "recipe": "cast:geometry-idol-v1",
                "assets": ["idol-dinov3": .string(dino.path), "idol-flux2-vae": .string(vae.path)]
            ],
            cancellation: NativeCancellation())
        #expect(receipt["files"]?.array?.count == 4)
        let denoiser = try MLX.loadArrays(url: out.appendingPathComponent("denoiser.safetensors"))
        #expect(denoiser["pos_pe"]?.dtype == .float32 && denoiser["pos_pe"]?.asArray(Float.self).first == Float(0.123456789))
        #expect(denoiser["input_layer.weight"]?.dtype == .float16)
        #expect(try MLX.loadArrays(url: out.appendingPathComponent(IdolFiles.dinov3))["layer.0.norm1.weight"]?.dtype == .bfloat16)
        #expect(try MLX.loadArrays(url: out.appendingPathComponent(IdolFiles.vae + "/diffusion_pytorch_model.safetensors"))["encoder.conv_in.bias"]?.dtype == .float16)
        // int8/int4 are not offered
        #expect(throws: (any Error).self) {
            _ = try IdolAdapter.conversion.run(
                [
                    "src_dir": .string(source.path), "dst_dir": .string(root.appendingPathComponent("int8").path), "dtype": "int8", "group_size": 64,
                    "recipe": "cast:geometry-idol-v1", "assets": ["idol-dinov3": .string(dino.path), "idol-flux2-vae": .string(vae.path)]
                ],
                cancellation: NativeCancellation())
        }
    }

    /// The whole request through the adapter with miniature random weights: a cutout in, a valid splat PLY and a
    /// rendered preview out; an opaque image is refused with the remove-background hint; cancellation stops it.
    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func miniaturePipelineWritesASplatAndPreview() throws {
        NativeRuntime.prepare()
        let previousLab = getenv("ARSENAL_NATIVE_LAB").map { String(cString: $0) }
        setenv("ARSENAL_NATIVE_LAB", "1", 1)
        defer { if let previousLab { setenv("ARSENAL_NATIVE_LAB", previousLab, 1) } else { unsetenv("ARSENAL_NATIVE_LAB") } }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("idol-scenario-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root.appendingPathComponent("out"), withIntermediateDirectories: true)
        let adapter = IdolAdapter()
        try adapter.load(
            NativeLoadContext(["model_dir": .string(root.path), "weights_dir": .string(root.path), "dtype": "fp16", "config": ["random_weights": true]]),
            cancellation: NativeCancellation())
        try NativeParameters.requireHalf(adapter.parameters)
        #expect(adapter.parameters.keys.contains { $0.hasPrefix("dinov3.") } && adapter.parameters.keys.contains { $0.hasPrefix("vae.") })

        let cutout = root.appendingPathComponent("cutout.png"), opaque = root.appendingPathComponent("opaque.png")
        try Self.writeDisc(to: cutout, transparentBackground: true)
        try Self.writeDisc(to: opaque, transparentBackground: false)
        var events = 0
        let result = try adapter.run(
            "image_to_3d", input: ["image": ["path": .string(cutout.path)], "out_dir": .string(root.appendingPathComponent("out").path), "steps": 2, "seed": 3],
            cancellation: NativeCancellation(), emit: { _ in events += 1 })
        let files = result["files"]?.array?.compactMap(\.object) ?? []
        #expect(files.count == 2)
        #expect(files.map { $0["mime"]?.string } == ["application/x-ply", "image/png"])
        #expect(result["gaussians"]?.number == 2048)
        #expect(events >= 4)

        let ply = try Data(contentsOf: root.appendingPathComponent("out/splat.ply"))
        let headerEnd = ply.range(of: Data("end_header\n".utf8))!
        let header = String(decoding: ply[..<headerEnd.upperBound], as: UTF8.self)
        #expect(header.hasPrefix("ply\nformat binary_little_endian 1.0\nelement vertex 2048\n"))
        #expect(header.components(separatedBy: "property float ").count - 1 == 17)
        #expect(ply.count - headerEnd.upperBound == 2048 * 17 * 4)
        let values = ply[headerEnd.upperBound...].withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        #expect(values.allSatisfy { $0.isFinite })
        // random weights put Gaussians anywhere their unbounded offset scale allows; rotations are unit quaternions
        for i in 0..<2048 {
            let row = Array(values[(i * 17)..<(i * 17 + 17)])
            let q = row[13..<17].reduce(0) { $0 + $1 * $1 }
            #expect(abs(q - 1) < 1e-3)
        }
        let preview = CGImageSourceCreateImageAtIndex(CGImageSourceCreateWithURL(root.appendingPathComponent("out/preview.png") as CFURL, nil)!, 0, nil)
        #expect(preview?.width == 64)

        #expect(throws: WorkerFailure.self) {
            _ = try adapter.run(
                "image_to_3d", input: ["image": ["path": .string(opaque.path)], "out_dir": .string(root.appendingPathComponent("out").path)], cancellation: NativeCancellation(),
                emit: { _ in })
        }
        let cancelled = NativeCancellation()
        cancelled.cancel()
        #expect(throws: WorkerFailure.self) {
            _ = try adapter.run(
                "image_to_3d", input: ["image": ["path": .string(cutout.path)], "out_dir": .string(root.appendingPathComponent("out").path)], cancellation: cancelled,
                emit: { _ in })
        }
        adapter.unload()
        #expect(adapter.parameters.isEmpty)
    }

    /// A 96 × 80 image: a coloured disc, on transparency or on opaque grey.
    static func writeDisc(to url: URL, transparentBackground: Bool) throws {
        let (w, h) = (96, 80)
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<w {
                let inside = (x - 50) * (x - 50) + (y - 40) * (y - 40) < 25 * 25, i = (y * w + x) * 4
                if inside {
                    bytes[i] = 220; bytes[i + 1] = UInt8(x * 2); bytes[i + 2] = 40; bytes[i + 3] = 255
                } else if !transparentBackground {
                    bytes[i] = 128; bytes[i + 1] = 128; bytes[i + 2] = 128; bytes[i + 3] = 255
                }
            }
        }
        try IdolPreview.writePNG(bytes, width: w, height: h, channels: 4, to: url)
    }
}
