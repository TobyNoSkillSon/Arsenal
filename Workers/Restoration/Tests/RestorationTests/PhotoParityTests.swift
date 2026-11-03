// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import ArsenalWire
import Testing
import DDColorNative
import LaMaNative
@testable import RestorationWorker

/// Fixed-tensor parity of the DDColor and LaMa ports against the provider code's captures (spandrel on MPS, captured
/// once by lab/scripts/reference.py into lab/evidence/<model>/), plus one end-to-end image each through the adapters.
/// Heavy and opt-in: ARSENAL_DDCOLOR_SOURCE / ARSENAL_LAMA_SOURCE name the pinned mlx-community safetensors.
/// Each test writes its Swift captures beside the reference for lab/scripts/compare.py.
@Suite(.serialized) struct PhotoParityTests {
    static let env = ProcessInfo.processInfo.environment
    static let repo = (0..<5).reduce(URL(fileURLWithPath: #filePath)) { url, _ in url.deletingLastPathComponent() }
    static let evidence = repo.appendingPathComponent("lab/evidence")
    static var ddcolor: URL? { env["ARSENAL_DDCOLOR_SOURCE"].map(URL.init(fileURLWithPath:)) }
    static var lama: URL? { env["ARSENAL_LAMA_SOURCE"].map(URL.init(fileURLWithPath:)) }

    /// Max |difference| and PSNR over `peak` (the reference's value range when nil).
    static func psnr(_ a: MLXArray, _ b: MLXArray, peak: Float? = nil) -> (maxAbs: Float, psnr: Float) {
        let x = a.asType(.float32), y = b.asType(.float32)
        let range = peak ?? (x.max() - x.min()).item(Float.self), mse = MLX.mean(MLX.square(x - y)).item(Float.self)
        return (MLX.max(MLX.abs(x - y)).item(Float.self), 10 * log10f(range * range / mse))
    }

    @Test(.heavy, .enabled(if: env["ARSENAL_DDCOLOR_SOURCE"] != nil)) func ddcolorComponents() throws {
        NativeRuntime.prepare()
        let source = try MLX.loadArrays(url: try #require(Self.ddcolor))
        let input = try MLX.loadArrays(url: Self.evidence.appendingPathComponent("ddcolor/input.safetensors"))["input"]!
        let reference = try MLX.loadArrays(url: Self.evidence.appendingPathComponent("ddcolor/reference-fp32.safetensors"))
        for (name, dtype) in [("fp16", DType.float16)] {
            let model = try DDColorModel(weights: source.mapValues { $0.asType(dtype) })
            model.capture = true
            _ = model(input)
            eval(Array(model.captured.values))
            #expect(Set(model.captured.keys) == Set(reference.keys))
            try MLX.save(arrays: model.captured.mapValues { $0.asType(.float32) }, url: Self.evidence.appendingPathComponent("ddcolor/swift-\(name).safetensors"))
            let ab = Self.psnr(reference["ab"]!, model.captured["ab"]!)
            print("ddcolor \(name) ab max|d| \(ab.maxAbs) PSNR \(ab.psnr)")
            // 22 Oct capture: 53.4 dB (max |d| 4.2 on ab values up to 47); every component 51 dB or more.
            #expect(ab.psnr > 45)
        }
    }

    @Test(.heavy, .enabled(if: env["ARSENAL_LAMA_SOURCE"] != nil)) func lamaFoldedComponents() throws {
        NativeRuntime.prepare()
        let root = Self.evidence.appendingPathComponent("lama/convert-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let src = root.appendingPathComponent("src"), dst = root.appendingPathComponent("dst")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: dst, withIntermediateDirectories: true)
        let source = try #require(Self.lama)
        if clonefile(source.path, src.appendingPathComponent("model.safetensors").path, 0) != 0 {
            try FileManager.default.copyItem(at: source, to: src.appendingPathComponent("model.safetensors"))
        }
        _ = try LaMaAdapter.conversion.run(
            ["src_dir": .string(src.path), "dst_dir": .string(dst.path), "dtype": "fp16", "group_size": 64, "recipe": "cast:restoration-lama-v1"],
            cancellation: NativeCancellation())
        let folded = try MLX.loadArrays(url: dst.appendingPathComponent("model.safetensors"))
        #expect(folded.count == LaMaFold.foldedShapes().count)
        #expect(folded.values.allSatisfy { $0.dtype == .float16 })
        let model = try LaMaModel(weights: folded)
        model.capture = true
        let input = try MLX.loadArrays(url: Self.evidence.appendingPathComponent("lama/input.safetensors"))["input"]!
        let reference = try MLX.loadArrays(url: Self.evidence.appendingPathComponent("lama/reference-fp32.safetensors"))
        _ = model(input)
        eval(Array(model.captured.values))
        #expect(Set(model.captured.keys) == Set(reference.keys))
        try MLX.save(arrays: model.captured.mapValues { $0.asType(.float32) }, url: Self.evidence.appendingPathComponent("lama/swift-fp16.safetensors"))
        let p = Self.psnr(reference["predicted"]!, model.captured["predicted"]!, peak: 1)
        print("lama fp16 predicted max|d| \(p.maxAbs) PSNR \(p.psnr)")
        // 22 Oct capture: 38.0 dB over the [0, 1] pixel range (max |d| 0.036); the unfolded FP16 graph read 0.6 dB.
        #expect(p.psnr > 34)
    }

    /// One image each through the adapters' serving path (decode, provider pipeline, PNG), written to lab/evidence.
    @Test(.heavy, .enabled(if: env["ARSENAL_DDCOLOR_SOURCE"] != nil && env["ARSENAL_LAMA_SOURCE"] != nil && env["ARSENAL_PHOTO_E2E"] == "1"))
    func endToEnd() throws {
        NativeRuntime.prepare()
        let fixtures = Self.repo.appendingPathComponent("lab/fixtures")
        let dtype = Self.env["ARSENAL_DDCOLOR_DTYPE"] ?? "fp16"
        for (adapter, source, op, input, dtypeName) in [
            (
                DDColorAdapter() as NativeModel, Self.ddcolor!, "colorize",
                ["image": JSONValue.object(["path": .string(fixtures.appendingPathComponent("ddcolor-input.png").path)])], dtype
            ),
            (
                LaMaAdapter() as NativeModel, Self.lama!, "inpaint",
                [
                    "image": JSONValue.object(["path": .string(fixtures.appendingPathComponent("lama-input.png").path)]),
                    "mask": .object(["path": .string(fixtures.appendingPathComponent("lama-mask.png").path)])
                ], "fp16"
            )
        ] {
            let root = Self.evidence.appendingPathComponent("e2e-\(op)")
            try? FileManager.default.removeItem(at: root)
            let weights = root.appendingPathComponent("weights"), out = root.appendingPathComponent("out")
            try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
            let half: DType = dtypeName == "bf16" ? .bfloat16 : .float16
            if op == "inpaint" {
                let src = root.appendingPathComponent("src")
                try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
                _ = clonefile(source.path, src.appendingPathComponent("model.safetensors").path, 0)
                _ = try LaMaAdapter.conversion.run(
                    ["src_dir": .string(src.path), "dst_dir": .string(weights.path), "dtype": "fp16", "group_size": 64, "recipe": "cast:restoration-lama-v1"],
                    cancellation: NativeCancellation())
                try FileManager.default.removeItem(at: src)
            } else {
                try MLX.save(arrays: try MLX.loadArrays(url: source).mapValues { $0.asType(half) }, url: weights.appendingPathComponent("model.safetensors"))
            }
            let context = try NativeLoadContext(["weights_dir": .string(weights.path), "model_dir": .string(root.path), "dtype": .string(dtypeName)])
            Memory.peakMemory = 0
            let started = Date()
            try adapter.load(context, cancellation: NativeCancellation())
            var request = input; request["out_dir"] = .string(out.path)
            let result = try adapter.run(op, input: request, cancellation: NativeCancellation(), emit: { _ in })
            let seconds = Date().timeIntervalSince(started)
            let files = try #require(result["files"]?.array)
            #expect(files.count == 1)
            print("\(op) \(dtypeName): \(String(format: "%.2f", seconds)) s load+run, MLX peak \(Memory.peakMemory / 1_048_576) MiB, \(files[0].object?["path"]?.string ?? "")")
            adapter.unload()
            try FileManager.default.removeItem(at: weights)
        }
    }

    /// A 3840x2160 inpaint (8.3 megapixels): FP16 stays finite where the spectral blocks see their largest values,
    /// and the result is written for comparison with the provider's FP32 output (lab/evidence/lama/output-large-fp32.png).
    @Test(.heavy, .enabled(if: env["ARSENAL_LAMA_SOURCE"] != nil && env["ARSENAL_PHOTO_LARGE"] == "1")) func largeInpaint() throws {
        NativeRuntime.prepare()
        let fixtures = Self.repo.appendingPathComponent("lab/fixtures"), root = Self.evidence.appendingPathComponent("e2e-inpaint-large")
        try? FileManager.default.removeItem(at: root)
        let src = root.appendingPathComponent("src"), weights = root.appendingPathComponent("weights")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
        _ = clonefile(try #require(Self.lama).path, src.appendingPathComponent("model.safetensors").path, 0)
        _ = try LaMaAdapter.conversion.run(
            ["src_dir": .string(src.path), "dst_dir": .string(weights.path), "dtype": "fp16", "group_size": 64, "recipe": "cast:restoration-lama-v1"],
            cancellation: NativeCancellation())
        let adapter = LaMaAdapter()
        try adapter.load(try NativeLoadContext(["weights_dir": .string(weights.path), "model_dir": .string(root.path), "dtype": "fp16"]), cancellation: NativeCancellation())
        Memory.peakMemory = 0
        let started = Date()
        let result = try adapter.run(
            "inpaint",
            input: [
                "image": .object(["path": .string(fixtures.appendingPathComponent("lama-large-input.png").path)]),
                "mask": .object(["path": .string(fixtures.appendingPathComponent("lama-large-mask.png").path)]), "out_dir": .string(root.appendingPathComponent("out").path)
            ], cancellation: NativeCancellation(), emit: { _ in })
        print("inpaint 3840x2160 fp16: \(String(format: "%.2f", Date().timeIntervalSince(started))) s run, MLX peak \(Memory.peakMemory / 1_048_576) MiB")
        #expect(result["files"]?.array?.count == 1)
        adapter.unload()
        try FileManager.default.removeItem(at: src); try FileManager.default.removeItem(at: weights)
    }
}
