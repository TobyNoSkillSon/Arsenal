// SPDX-License-Identifier: MIT
import CoreGraphics
import Foundation
import ImageIO
import IntegrationKit
import MLX
import MLXRandom
import Testing
@testable import MLXIdol

/// The trained model end to end on real cutouts: IDOL_E2E_ROOT is a weights folder in the converted layout
/// (`IdolFiles`; unconverted sources are cast as Get does), IDOL_E2E_IMAGES a comma-separated list of RGBA cutouts,
/// IDOL_E2E_OUT where the PLY files and renders (the predicted camera and three turns around the object) go.
@Suite(.serialized)
struct IdolEndToEndTests {
    static let env = ProcessInfo.processInfo.environment

    @Test(.heavy, .enabled(if: env["IDOL_E2E_ROOT"] != nil && env["IDOL_E2E_IMAGES"] != nil && env["IDOL_E2E_OUT"] != nil))
    func trainedModelMakesSplats() throws {
        NativeRuntime.prepare()
        let started = Date()
        let pipeline = try IdolPipeline.load(URL(fileURLWithPath: Self.env["IDOL_E2E_ROOT"]!), cast: true)
        print("E2E load \(String(format: "%.1f", Date().timeIntervalSince(started))) s")
        let out = URL(fileURLWithPath: Self.env["IDOL_E2E_OUT"]!)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        var request = IdolRequest()
        if let steps = Self.env["IDOL_E2E_STEPS"].flatMap(Int.init) { request.steps = steps }
        if let seed = Self.env["IDOL_E2E_SEED"].flatMap(UInt64.init) { request.seed = seed }
        for path in Self.env["IDOL_E2E_IMAGES"]!.split(separator: ",").map(String.init) {
            let name = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
            let source = try #require(CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil))
            let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
            Memory.peakMemory = 0
            let t0 = Date()
            let result = try pipeline.generate(image, request: request)
            let seconds = Date().timeIntervalSince(t0)
            let g = result.gaussians
            #expect(g.count == request.gaussians)
            try g.writePLY(to: out.appendingPathComponent(name + ".ply"))
            try IdolPreview.writePNG(result.prepared, width: pipeline.imageSize, height: pipeline.imageSize, channels: 3, to: out.appendingPathComponent(name + "_input.png"))
            let c = result.camera, fov = c.radius < 100 ? c.fov : 40, radius = IdolPreview.fittingRadius(g, fov: fov)
            for (index, turn) in [0.0, Double.pi / 2, Double.pi, 3 * Double.pi / 2].enumerated() {
                let pixels = IdolPreview.render(
                    g, size: 768, yaw: c.yaw + turn, pitch: c.pitch, radius: radius, fov: fov)
                try IdolPreview.writePNG(pixels, width: 768, height: 768, channels: 4, to: out.appendingPathComponent("\(name)_view\(index).png"))
            }
            let opacity = g.opacity.mean().item(Float.self), extent = MLX.abs(g.xyz).max().item(Float.self)
            print(
                "E2E \(name): \(String(format: "%.1f", seconds)) s, peak \(Memory.peakMemory / 1_048_576) MB, \(g.count) Gaussians, mean opacity \(opacity), "
                    + "max |xyz| \(extent), camera yaw \(c.yaw) pitch \(c.pitch) radius \(c.radius) fov \(c.fov)")
            #expect(extent <= 0.75)
        }
    }

    /// Stage by stage against idol-sdk's own run of the same picture (IDOL_E2E_REF, from `idol_reference_e2e.py`): the
    /// image tokens from its prepared pixels, the splat decoded from its final latent, and the prepared pixels.
    @Test(.heavy, .enabled(if: env["IDOL_E2E_ROOT"] != nil && env["IDOL_E2E_REF"] != nil))
    func stagesMatchIDOLsOwnRun() throws {
        NativeRuntime.prepare()
        let pipeline = try IdolPipeline.load(URL(fileURLWithPath: Self.env["IDOL_E2E_ROOT"]!), cast: true)
        let ref = try MLX.loadArrays(url: URL(fileURLWithPath: Self.env["IDOL_E2E_REF"]!))
        let pixels = ref["pixels"]! // [1, 3, 1024, 1024] in [0, 1]
        let rgb = (pixels[0].transposed(1, 2, 0) * 255).round().asType(.uint8).reshaped([-1]).asArray(UInt8.self)
        let (f1, f2) = try pipeline.condition(rgb: rgb, size: 1024, key: MLXRandom.key(0))
        IdolParityTests.expectClose("e2e feature1 (DINOv3)", f1, ref["feature1"]!, relative: 1.0 / 8, cosine: 0.99)
        // feature2 is a sample of a wide VAE posterior: two samples agree only at cosine ~0.71, so it is not compared here
        // (the posterior itself is checked in IdolParityTests.vaePosteriorMatches)
        #expect(f2.shape == ref["feature2"]!.shape)
        // the sampler from idol-sdk's own noise and image tokens (IDOL_E2E_STEPS steps; idol-sdk ran 50)
        if let noise = ref["noise_latent"], Self.env["IDOL_E2E_SAMPLE"] == "1" {
            let s = try pipeline.sample(
                latent: noise, camera: ref["noise_camera"]!, feature1: ref["feature1"]!, feature2: ref["feature2"]!, steps: 50, guidance: 3, rescale: 3)
            IdolParityTests.expectClose("e2e sampled latent from idol-sdk's noise", s.latent, ref["latent"]!, relative: 1.0 / 8, cosine: 0.99)
            IdolParityTests.expectClose("e2e sampled camera from idol-sdk's noise", s.camera, ref["camera"]!, relative: 1.0 / 8, cosine: 0.99)
        }
        if Self.env["IDOL_E2E_SAMPLE"] == "1", let noise = ref["noise_latent"] {
            func extents(_ latent: MLXArray) throws -> [Float] {
                let g = try pipeline.decode(latent, count: 262_144, key: MLXRandom.key(1))
                let mask = (g.opacity .> 0.5)[0..., 0].asArray(Bool.self)
                return (0..<3).map { axis in
                    let kept = zip(g.xyz[0..., axis].asArray(Float.self), mask).filter(\.1).map(\.0).sorted()
                    return kept[kept.count * 99 / 100] - kept[kept.count / 100]
                }
            }
            for seed in [UInt64(42), 7] {
                let mine = MLXRandom.normal(noise.shape, key: MLXRandom.key(seed)), camera = MLXRandom.normal([1, 1, 5], key: MLXRandom.key(seed + 1))
                let a = try pipeline.sample(latent: mine, camera: camera, feature1: ref["feature1"]!, feature2: ref["feature2"]!, steps: 50, guidance: 3, rescale: 3)
                print("E2E MLX noise \(seed) + idol-sdk tokens: extents \(try extents(a.latent))")
                let b = try pipeline.sample(latent: noise, camera: ref["noise_camera"]!, feature1: f1, feature2: f2, steps: 50, guidance: 3, rescale: 3)
                if seed == 42 { print("E2E idol-sdk noise + MLX tokens: extents \(try extents(b.latent))") }
            }
        }
        let g = try pipeline.decode(ref["latent"]!, count: 262_144, key: MLXRandom.key(1))
        let xyz = g.xyz, opaque = g.opacity .> 0.5
        // idol-sdk's own splat of this latent (armchair, seed 42): 1st-99th percentile extents of the opaque Gaussians
        let reference = (Self.env["IDOL_E2E_REF_EXTENTS"] ?? "0.552,0.456,0.511").split(separator: ",").compactMap { Float(String($0)) }
        for axis in 0..<3 {
            let values = xyz[0..., axis].asArray(Float.self), mask = opaque[0..., 0].asArray(Bool.self)
            let kept = zip(values, mask).filter(\.1).map(\.0).sorted()
            let lo = kept[kept.count / 100], hi = kept[kept.count * 99 / 100]
            print("E2E decoded from idol-sdk's latent, IDOL axis \(axis) extent \(hi - lo)")
            #expect(abs((hi - lo) - reference[axis]) < 0.02, "axis \(axis): \(hi - lo) against idol-sdk's \(reference[axis])")
        }
        if let images = Self.env["IDOL_E2E_IMAGES"], let first = images.split(separator: ",").first,
            let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: String(first)) as CFURL, nil), let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        {
            let mine = MLXArray(try IdolImage.prepare(image, margin: 1.2).map { Float($0) / 255 }, [1024, 1024, 3])
            IdolParityTests.expectClose("e2e prepared pixels", mine, pixels[0].transposed(1, 2, 0), relative: 1.0 / 16, cosine: 0.99)
        }
    }
}
