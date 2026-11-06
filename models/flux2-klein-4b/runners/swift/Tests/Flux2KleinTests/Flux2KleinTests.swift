// SPDX-License-Identifier: MIT
// Pure contract checks and tiny half precision GPU layout probes. No model runs on the CPU.
import Foundation
import MLX
import MLXNN
import Testing
@testable import Flux2Klein
@testable import FluxWorker

/// The no-fast-math metallib scripts/test.sh lays out beside the native helpers, as every other native suite uses.
func useTestMetallib() {
    if let path = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) }
}

@Suite struct Schedule {
    init() { useTestMetallib() }
    /// diffusers 0.41.0 FlowMatchEulerDiscreteScheduler with the snapshot's scheduler_config and Klein's
    /// linspace(1, 1/N, N) sigmas plus compute_empirical_mu (values printed by the reference environment).
    @Test(arguments: [
        (1024, 4, [1.0, 0.9580853581428528, 0.8839818835258484, 0.7174965739250183, 0.0]),
        (4096, 4, [1.0, 0.9673840403556824, 0.908143937587738, 0.7671999335289001, 0.0]),
        (2304, 4, [1.0, 0.9622337222099304, 0.8946577906608582, 0.7389686703681946, 0.0]),
        (256, 1, [1.0, 0.0]),
        (4096, 8, [1.0, 0.9854254722595215, 0.9666411280632019, 0.9415147304534912, 0.906182587146759, 0.8528416752815247, 0.7630144953727722, 0.5798067450523376, 0.0]),
        (9216, 4, [1.0, 0.9575182795524597, 0.8825350403785706, 0.7146438956260681, 0.0])
    ])
    func matchesVendor(seq: Int, steps: Int, expected: [Double]) {
        let sigmas = KleinSchedule.sigmas(seqLen: seq, steps: steps)
        #expect(sigmas.count == expected.count)
        for (a, b) in zip(sigmas, expected) { #expect(abs(Double(a) - b) <= 1e-7) }
    }
}

@Suite struct Layout {
    init() { useTestMetallib() }
    @Test(.heavy) func bf16Defaults() {
        let precision = KleinPrecision()
        #expect(precision.weights == .bfloat16 && precision.encoder == .bfloat16 && precision.activations == .bfloat16 && precision.vae == .bfloat16)
        Device.withDefaultDevice(.gpu) {
            let noise = KleinPipeline.noise(seed: 7, width: 64, height: 64)
            #expect(noise.dtype == .bfloat16 && noise.shape == [1, 16, 128] && all(isFinite(noise)).item(Bool.self))
            #expect(KleinPipeline.ropeTables(textLength: 3, gridH: 2, gridW: 2).0.dtype == .bfloat16)
        }
    }

    @Test(.heavy) func rmsNormalizationKeepsLargeHalfActivations() throws {
        try Device.withDefaultDevice(.gpu) {
            let norm = QwenHalfRMSNorm(dimensions: 4, eps: 1e-6)
            try norm.update(parameters: ModuleParameters.unflattened(["weight": MLXArray.ones([4], dtype: .float16)]), verify: .all)
            let x = MLXArray([Float16(700), -700, 300, -300], [1, 4])
            let y = norm(x)
            #expect(y.dtype == .float16 && all(isFinite(y)).item(Bool.self))
            #expect(abs(y[0, 0].item(Float.self) - 1.299867) < 0.002)
            #expect(abs(mean(square(y)).item(Float.self) - 1) < 0.002)
        }
    }
    @Test(.heavy) func unpatchifyMatchesDiffusers() {
        // channel c·4 + dy·2 + dx of token (h, w) lands at pixel (2h + dy, 2w + dx), channel c.
        Device.withDefaultDevice(.gpu) {
            let vae = Flux2VAEDecoder()
            let packed = MLXArray((0..<(2 * 3 * 128)).map(Float16.init), [1, 6, 128])
            let z = vae.latents(packed, gridH: 2, gridW: 3) // bn mean 0, var 1 → scale sqrt(1 + 1e-4)
            #expect(z.shape == [1, 4, 6, 32])
            let scale = Float(1.0001).squareRoot()
            for (h, w, c, dy, dx) in [(0, 0, 0, 0, 0), (1, 2, 31, 1, 1), (0, 1, 5, 1, 0), (1, 0, 7, 0, 1)] {
                let expected = Float((h * 3 + w) * 128 + c * 4 + dy * 2 + dx) * scale
                let got = z[0, 2 * h + dy, 2 * w + dx, c].item(Float.self)
                #expect(abs(got - expected) <= 1e-3 * max(1, expected))
            }
        }
    }

    @Test(.heavy) func ropeTablesFollowTextThenImage() {
        Device.withDefaultDevice(.gpu) {
            let (cos, sin) = KleinPipeline.ropeTables(textLength: 3, gridH: 2, gridW: 2, dtype: .float16)
            #expect(cos.shape == [7, 64] && sin.shape == [7, 64])
            // Text token 2: axis 3 carries position 2, axes 0–2 are 0 → cos 1 / sin 0 on their columns.
            #expect(cos[2, 0..<48].min().item(Float.self) == 1)
            #expect(abs(sin[2, 48].item(Float.self) - Foundation.sin(2)) < 1e-3)
            // Image token (1, 1) = row 3 + 3: axes 1 and 2 carry 1.
            #expect(abs(cos[6, 16].item(Float.self) - Foundation.cos(1)) < 1e-3)
            #expect(abs(cos[6, 32].item(Float.self) - Foundation.cos(1)) < 1e-3)
        }
    }

    @Test func chatTemplateIsTheVendorString() {
        #expect(KleinTextEncoder.chatPrompt("hi") == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }
}

@Suite struct Requests {
    init() { useTestMetallib() }
    let support = URL(fileURLWithPath: "/tmp/arsenal-flux-tests-support")
    func parse(_ extra: JSONObject) throws -> ImageRequest {
        var r: JSONObject = ["prompt": "A red square", "out_dir": .string(support.path + "/jobs/a")]
        r.merge(extra) { _, new in new }
        return try ImageRequest.parse(r, support: support, randomSeed: { 7 })
    }

    @Test func defaults() throws {
        let r = try parse([:])
        #expect(r.width == 1024 && r.height == 1024 && r.steps == 4 && r.n == 1 && r.seed == 7)
    }
    @Test func fixedDistilledGuidanceAcceptsOnlyOne() throws {
        _ = try parse(["guidance": 1])
    }
    @Test func resolutionAndExplicitDimensions() throws {
        #expect(try parse(["resolution": "512x512"]).width == 512)
        let r = try parse(["resolution": "512x512", "width": 768])
        #expect(r.width == 768 && r.height == 512)
    }
    @Test(
        arguments: [
            ["resolution": "640x640"], ["width": 100], ["width": 4096], ["steps": 0], ["n": 9], ["seed": -1], ["seed": 4_294_967_296],
            ["guidance": 21], ["guidance": 2], ["guidance": 0], ["prompt": "  "], ["out_dir": "/tmp/elsewhere"], ["out_dir": "relative/dir"], ["steps": 2.5]
        ] as [JSONObject])
    func rejects(_ extra: JSONObject) {
        #expect(throws: WorkerFailure.self) { try parse(extra) }
    }
    @Test func jobsDirectoryItselfIsRejected() {
        #expect(throws: WorkerFailure.self) { try parse(["out_dir": .string(support.path + "/jobs")]) }
        #expect(throws: WorkerFailure.self) { try parse(["out_dir": .string(support.path + "/jobs/../outside")]) }
    }
}

/// Fake engine: a gradient image, one callback per step.
final class FakeEngine: ImageEngine {
    var calls = 0
    func image(prompt: String, width: Int, height: Int, steps: Int, seed: UInt64, step: (Int, Int) throws -> Void) throws -> (pixels: [UInt8], width: Int, height: Int) {
        calls += 1
        for i in 1...steps { try step(i, steps) }
        return ((0..<(width * height * 3)).map { UInt8(truncatingIfNeeded: $0 &+ Int(seed)) }, width, height)
    }
}

@Suite(.serialized) struct Wire {
    init() { useTestMetallib() }
    func frames(_ pipe: Pipe) throws -> [JSONObject] {
        try pipe.fileHandleForWriting.close()
        return try Frames.decode(pipe.fileHandleForReading.readDataToEndOfFile())
    }

    @Test(arguments: ["bf16", "fp16", "int8", "int4"])
    func precisionIsPassedToTheEngine(_ dtype: String) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("flux-intake-wire-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        for file in KleinWeights.requiredFiles {
            let url = root.appendingPathComponent(file)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: url)
        }
        let pipe = Pipe(), jobs = Jobs(); #expect(jobs.admit(1))
        var observed: String?
        let runner = Runner(
            wire: WorkerChannel(output: pipe.fileHandleForWriting), jobs: jobs, support: root,
            precisionLoader: { _, requested in
                observed = requested; return FakeEngine()
            })
        _ = try runner.run(["id": 1, "op": "load", "dtype": .string(dtype), "weights_dir": .string(root.path)])
        #expect(observed == dtype && runner.dtype == dtype && runner.engine != nil)
        _ = try runner.run(["id": 2, "op": "unload"])
    }

    @Test(arguments: ["fp16", "int8", "int4"])
    func rejectsUnqualifiedPrecision(_ dtype: String) {
        let pipe = Pipe()
        let runner = Runner(wire: WorkerChannel(output: pipe.fileHandleForWriting), jobs: Jobs(), loader: { _ in FakeEngine() })
        #expect(throws: WorkerFailure.self) { try runner.run(["id": 1, "op": "load", "dtype": .string(dtype), "weights_dir": "/tmp/unused"]) }
        #expect(runner.engine == nil && runner.dtype == nil)
    }

    @Test(arguments: [false, true]) func cancelledLoadNeverPublishesAnEngine(cancelDuringLoad: Bool) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("flux-cancel-load-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        for file in KleinWeights.requiredFiles {
            let url = root.appendingPathComponent(file)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: url)
        }
        let pipe = Pipe(), jobs = Jobs()
        _ = jobs.admit(7)
        if !cancelDuringLoad { _ = jobs.cancel(7) }
        var loads = 0
        let runner = Runner(
            wire: WorkerChannel(output: pipe.fileHandleForWriting), jobs: jobs,
            loader: { _ in
                loads += 1
                _ = jobs.cancel(7)
                return FakeEngine()
            })
        #expect(throws: KleinRequestCancelled.self) {
            try runner.run(["id": 7, "op": "load", "weights_dir": .string(root.path)])
        }
        #expect(loads == (cancelDuringLoad ? 1 : 0))
        #expect(runner.engine == nil && runner.dtype == nil)
        #expect(try frames(pipe).allSatisfy { $0["ok"]?.bool != true && $0["event"] != "status" })
    }

    @Test func loadGenerateUnload() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("flux-wire-\(UUID().uuidString)")
        let weights = root.appendingPathComponent("weights"), support = root.appendingPathComponent("support")
        for file in KleinWeights.requiredFiles {
            let url = weights.appendingPathComponent(file)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: url)
        }
        setenv("ARSENAL_SUPPORT_DIR", support.path, 1)
        defer { try? FileManager.default.removeItem(at: root) }
        let pipe = Pipe(), jobs = Jobs(), engine = FakeEngine()
        let runner = Runner(wire: WorkerChannel(output: pipe.fileHandleForWriting), jobs: jobs, loader: { _ in engine })
        for id in 1...4 { _ = jobs.admit(Double(id)) }
        #expect(try runner.run(["id": 1, "op": "load", "weights_dir": .string(weights.path)]) == false)
        #expect(runner.dtype == "bf16")
        let out = support.appendingPathComponent("jobs/j1")
        #expect(
            try runner.run([
                "id": 2, "op": "generate", "prompt": "p", "width": 64, "height": 96, "steps": 3, "n": 2, "seed": 4_294_967_295,
                "out_dir": .string(out.path)
            ]) == false)
        #expect(throws: WorkerFailure.self) { try runner.run(["id": 3, "op": "load", "weights_dir": .string(weights.path)]) }
        #expect(try runner.run(["id": 4, "op": "unload"]) == true)
        let sent = try frames(pipe)
        let reply = try #require(sent.first { $0["id"]?.number == 2 && $0["ok"]?.bool == true })
        let images = try #require(reply["images"]?.array)
        #expect(images.count == 2)
        #expect(reply["seeds"]?.array == [4_294_967_295, 0]) // seed + index wraps at 2³²
        for image in images {
            let path = try #require(image["path"]?.string)
            #expect(path.hasPrefix(out.resolvingSymlinksInPath().path) || path.hasPrefix(out.path))
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            #expect(data.prefix(4) == Data([137, 80, 78, 71]) && image["bytes"]?.number == Double(data.count))
            #expect(image["width"]?.number == 64 && image["height"]?.number == 96)
        }
        let steps = sent.filter { $0["id"]?.number == 2 && $0["phase"]?.string == "step" && $0["step"] != nil }
        #expect(steps.map { $0["step"]?.number ?? 0 } == [1, 2, 3, 4, 5, 6] && steps.allSatisfy { $0["total"]?.number == 6 })
    }

    @Test func cancelBetweenStepsRemovesWrittenImages() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("flux-cancel-\(UUID().uuidString)")
        let support = root.appendingPathComponent("support")
        setenv("ARSENAL_SUPPORT_DIR", support.path, 1)
        defer { try? FileManager.default.removeItem(at: root) }
        let pipe = Pipe(), jobs = Jobs()
        final class Cancelling: ImageEngine {
            let jobs: Jobs
            init(jobs: Jobs) { self.jobs = jobs }
            var calls = 0
            func image(prompt: String, width: Int, height: Int, steps: Int, seed: UInt64, step: (Int, Int) throws -> Void) throws -> (pixels: [UInt8], width: Int, height: Int) {
                calls += 1
                if calls == 2 { _ = jobs.cancel(7) }
                for i in 1...steps { try step(i, steps) }
                return ([UInt8](repeating: 9, count: width * height * 3), width, height)
            }
        }
        let runner = Runner(wire: WorkerChannel(output: pipe.fileHandleForWriting), jobs: jobs, loader: { _ in Cancelling(jobs: jobs) })
        runner.engine = Cancelling(jobs: jobs)
        _ = jobs.admit(7)
        let out = support.appendingPathComponent("jobs/j2")
        #expect(throws: KleinRequestCancelled.self) {
            try runner.run(["id": 7, "op": "generate", "prompt": "p", "width": 64, "height": 64, "steps": 2, "n": 3, "out_dir": .string(out.path)])
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: out.path).isEmpty)
    }
}
