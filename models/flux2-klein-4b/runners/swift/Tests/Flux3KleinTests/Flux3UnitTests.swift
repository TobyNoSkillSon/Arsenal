// SPDX-License-Identifier: MIT
// Default-suite checks: host math, shapes, parsing and request contracts. No MLX array is evaluated here.
import Foundation
import Testing
@testable import Flux3Klein
@testable import FluxWorker
import Flux2Klein

@Suite struct Flux3Schedules {
    /// mflux Flux3FlowMapScheduler(N).timesteps, captured on the GPU in float32.
    @Test(arguments: [1, 2, 4, 8, 16, 50])
    func matchesMflux(steps: Int) throws {
        let expected = try need((try TinyFixture.caseJSON["timesteps"] as? [String: [Double]])?[String(steps)], "no schedule \(steps)")
        let got = Flux3Schedule.timesteps(steps: steps)
        #expect(got.count == steps + 1 && expected.count == steps + 1)
        #expect(got.first == 1 && got.last == 0)
        for (a, b) in zip(got, expected) { #expect(abs(Double(a) - b) <= Double(Float(b).ulp), "\(a) vs \(b)") }
    }

    @Test func defaultIsSixteenDistilledSteps() {
        #expect(Flux3Schedule.defaultSteps == 16)
        let t = Flux3Schedule.timesteps(steps: 16)
        #expect(zip(t, t.dropFirst()).allSatisfy { $0 > $1 })
    }
}

@Suite struct Flux3PositionTables {
    @Test func idsMatchMflux() throws {
        let ids = try need(try TinyFixture.caseJSON["ids"] as? [String: [[Double]]], "no ids")
        func same(_ a: [[Float]], _ b: [[Double]]?) -> Bool { a.map { $0.map(Double.init) } == b }
        #expect(same(Flux3Prompt.textIDs(80), ids["text"]))
        #expect(same(Flux3Positions.imageIDs(gridH: 4, gridW: 4), ids["image"]))
        #expect(same(Flux3Positions.imageIDs(gridH: 2, gridW: 4, layer: -1), ids["ref1"]))
        #expect(same(Flux3Positions.imageIDs(gridH: 4, gridW: 2, layer: -2), ids["ref2"]))
    }

    @Test func tinyRopeTablesMatchMflux() throws {
        let ids =
            Flux3Positions.imageIDs(gridH: 4, gridW: 4) + Flux3Positions.imageIDs(gridH: 2, gridW: 4, layer: -1)
            + Flux3Positions.imageIDs(gridH: 4, gridW: 2, layer: -2) + Flux3Prompt.textIDs(80)
        let (cos, sin) = Flux3Rope.tables(ids, axes: [8, 8, 8, 8], theta: 10000)
        let rc = try TinyFixture.hostTensor("rope_cos"), rs = try TinyFixture.hostTensor("rope_sin")
        #expect(rc.shape == [112, 16] && cos.count == 112 * 16)
        #expect(zip(cos, rc.values).map { abs($0 - $1) }.max()! <= 1.2e-7)
        #expect(zip(sin, rs.values).map { abs($0 - $1) }.max()! <= 1.2e-7)
    }

    /// The release's head (4 × 32) at real positions: 1024² image, a third reference layer, 4096 text tokens.
    @Test func realRopeRowsMatchMflux() throws {
        let rows = try need(try TinyFixture.caseJSON["rope_real_rows"] as? [Int], "no rows")
        let ids = Flux3Positions.imageIDs(gridH: 64, gridW: 64) + Flux3Positions.imageIDs(gridH: 3, gridW: 5, layer: -3) + Flux3Prompt.textIDs(4096)
        let (cos, sin) = Flux3Rope.tables(rows.map { ids[$0] }, axes: [32, 32, 32, 32], theta: 10000)
        let rc = try TinyFixture.hostTensor("rope_real_cos"), rs = try TinyFixture.hostTensor("rope_real_sin")
        #expect(rc.shape == [rows.count, 64])
        #expect(zip(cos, rc.values).map { abs($0 - $1) }.max()! <= 1.2e-7)
        #expect(zip(sin, rs.values).map { abs($0 - $1) }.max()! <= 1.2e-7)
    }
}

@Suite struct Flux3PromptRules {
    @Test func padsToMultiplesOfEightyCappedAt4096() {
        #expect(Flux3Prompt.paddedLength(1) == 80 && Flux3Prompt.paddedLength(80) == 80 && Flux3Prompt.paddedLength(81) == 160)
        #expect(Flux3Prompt.paddedLength(4090) == 4096 && Flux3Prompt.paddedLength(5000) == 4096)
        let (ids, valid) = Flux3Prompt.pad(Array(0..<11))
        #expect(ids.count == 80 && valid == 11 && ids[11] == Flux3Prompt.padToken && ids[79] == Flux3Prompt.padToken)
        let long = Flux3Prompt.pad(Array(repeating: 5, count: 5000))
        #expect(long.ids.count == 4096 && long.valid == 4096)
    }

    @Test func positionsMatchMflux() throws {
        let expected = try need(try TinyFixture.caseJSON["positions"] as? [[Int]], "no positions")
        #expect(Flux3Prompt.positions(length: 80, valid: 37).map(Int.init) == expected[0])
    }

    @Test func chatTemplateIsQwen3VLInstruct() {
        #expect(Flux3Prompt.chatPrompt("A red fox") == "<|im_start|>user\nA red fox<|im_end|>\n<|im_start|>assistant\n")
        #expect(Flux3Prompt.taps == [4, 8, 12, 16, 20, 24, 28, 32])
    }
}

@Suite struct Flux3Configs {
    /// The text_config of Qwen/Qwen3-VL-4B-Instruct (byte-identical to the release's text_encoder/config.json).
    @Test func parsesQwen3VLTextConfig() throws {
        let json =
            #"{"architectures":["Qwen3VLForConditionalGeneration"],"model_type":"qwen3_vl","text_config":{"attention_bias":false,"head_dim":128,"hidden_size":2560,"intermediate_size":9728,"num_attention_heads":32,"num_hidden_layers":36,"num_key_value_heads":8,"rms_norm_eps":1e-06,"rope_scaling":{"mrope_interleaved":true,"mrope_section":[24,20,20],"rope_type":"default"},"rope_theta":5000000,"vocab_size":151936},"vision_config":{"depth":24}}"#
        let c = try Flux3TextConfig(json: Data(json.utf8))
        #expect(c == Flux3TextConfig())
        #expect(c.hidden * Flux3Prompt.taps.count == Flux3TransformerConfig().jointDim)
        #expect(throws: Flux3Error.self) { try Flux3TextConfig(json: Data(#"{"text_config":{"hidden_size":8,"num_hidden_layers":12}}"#.utf8)) }
    }

    @Test func infersReleaseTransformerFromShapes() throws {
        var shapes: [String: [Int]] = [
            "x_embedders.image.weight": [3072, 128], "x_embedders.image_cond.weight": [3072, 128], "context_embedder.weight": [3072, 20480],
            "timestep_embedder.linear_1.weight": [3072, 256], "target_timestep_embedder.linear_1.weight": [3072, 256]
        ]
        for i in 0..<5 {
            for m in ["mode_blocks.image", "mode_blocks.image_cond", "txt_mode_blocks"] { shapes["\(m).\(i).attn.to_q.weight"] = [3072, 3072] }
        }
        for i in 0..<28 {
            shapes["single_transformer_blocks.\(i).attn.norm_q.weight"] = [128]
            shapes["single_transformer_blocks.\(i).ff.linear_out.weight"] = [3072, 9216]
        }
        let c = try Flux3TransformerConfig.inferred(from: shapes)
        #expect(c == Flux3TransformerConfig())
        #expect(c.axes == [32, 32, 32, 32] && c.inner == 3072)
        shapes.removeValue(forKey: "target_timestep_embedder.linear_1.weight")
        #expect(throws: Flux3Error.self) { try Flux3TransformerConfig.inferred(from: shapes) }
    }

    @Test func weightNamesFollowTheCheckpoint() {
        #expect(Flux3Weights.transformerName("single_transformer_blocks.3.attn.to_out.0.weight") == "single_transformer_blocks.3.attn.to_out.weight")
        #expect(Flux3Weights.textEncoderName("model.language_model.layers.31.mlp.up_proj.weight") == "layers.31.mlp.up_proj.weight")
        #expect(Flux3Weights.textEncoderName("model.language_model.layers.32.mlp.up_proj.weight") == nil)
        #expect(Flux3Weights.textEncoderName("model.language_model.norm.weight") == nil)
        #expect(Flux3Weights.textEncoderName("model.visual.blocks.0.attn.qkv.weight") == nil)
        #expect(Flux3Weights.textEncoderName("model.language_model.embed_tokens.weight") == "embed_tokens.weight")
        #expect(Flux3Weights.vaeName("bn.running_var")! == (false, "bn_var"))
        #expect(Flux3Weights.vaeName("encoder.mid_block.attentions.0.to_out.0.bias")! == (true, "encoder.mid_block.attentions.0.to_out.bias"))
        #expect(Flux3Weights.vaeName("quant_conv.weight")! == (true, "quant_conv.weight"))
        #expect(Flux3Weights.vaeName("bn.num_batches_tracked") == nil)
    }
}

@Suite struct Flux3ReferenceGeometry {
    /// mflux Flux3ReferenceImages.reference_canvas on the same sizes (values printed by the reference environment).
    static let canvases: [([Int], [Int])] = [
        ([512, 512], [512, 512, 512, 512]), ([1000, 750], [1000, 750, 1008, 752]), ([4000, 3000], [2341, 1756, 2352, 1760]),
        ([17, 900], [17, 900, 64, 912]), ([3000, 10], [3000, 10, 3008, 64]), ([1536, 1024], [1536, 1024, 1536, 1024]),
        ([2049, 2049], [2048, 2048, 2048, 2048]), ([1023, 1025], [1023, 1025, 1024, 1040]), ([5000, 100], [4096, 82, 4096, 96])
    ]
    @Test(arguments: canvases)
    func singleCanvas(size: [Int], expected: [Int]) {
        let f = Flux3Canvas.single(width: size[0], height: size[1])
        #expect([f.fit.width, f.fit.height, f.canvas.width, f.canvas.height] == expected)
    }

    static let multis: [([Int], [Int])] = [
        ([1500, 1000], [1254, 836, 3, 2, 1248, 832]), ([640, 480], [640, 480, 0, 0, 640, 480]), ([1023, 1025], [1023, 1025, 7, 0, 1008, 1024]),
        ([20, 4000], [20, 4000, 2, 0, 16, 4000]), ([3000, 2000], [1254, 836, 3, 2, 1248, 832])
    ]
    @Test(arguments: multis)
    func multiCrop(size: [Int], expected: [Int]) throws {
        let m = try Flux3Canvas.multi(width: size[0], height: size[1])
        #expect([m.scaled.width, m.scaled.height, m.crop.x, m.crop.y, m.crop.width, m.crop.height] == expected)
    }

    @Test func tooSmallReferenceIsRefused() {
        #expect(throws: Flux3Error.self) { try Flux3Canvas.multi(width: 15, height: 400) }
    }

    /// np.pad(np.arange(n), (before, after), mode) from the reference environment.
    static let pads: [(Int, Int, Int, Bool, [Int])] = [
        (5, 3, 4, false, [3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0]), (3, 7, 6, false, [1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0]),
        (4, 5, 9, true, [3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3]), (2, 3, 3, false, [1, 0, 1, 0, 1, 0, 1, 0]),
        (1, 2, 2, true, [0, 0, 0, 0, 0])
    ]
    @Test(arguments: pads)
    func padMatchesNumpy(n: Int, before: Int, after: Int, symmetric: Bool, expected: [Int]) {
        #expect((0..<(n + before + after)).map { Flux3Canvas.padIndex($0 - before, size: n, symmetric: symmetric) } == expected)
    }

    @Test func resolveCanvasFollowsMflux() {
        let box = (x: 4, y: 0, width: 1000, height: 752)
        let none = Flux3Canvas.resolve(canvas: (1008, 752), box: box, width: nil, height: nil)
        #expect(none.width == 1008 && none.height == 752 && none.crop?.width == 1000)
        let same = Flux3Canvas.resolve(canvas: (1008, 752), box: box, width: 1000, height: 752)
        #expect(same.width == 1008 && same.crop != nil)
        let other = Flux3Canvas.resolve(canvas: (1008, 752), box: box, width: 512, height: nil)
        #expect(other.width == 512 && other.height == 752 && other.crop == nil)
    }

    /// Pillow 13 `Image.resize(..., LANCZOS)` on random RGB images, bit for bit.
    @Test func lanczosMatchesPillow() throws {
        let cases = try need(try TinyFixture.json("lanczos.json")["cases"] as? [[String: Any]], "no cases")
        #expect(cases.count == 4)
        for c in cases {
            let size = try ["width", "height", "out_width", "out_height"].map { try need(c[$0] as? Int, "no \($0)") }
            let source = try need(c["pixels"] as? [Int], "no pixels"), expected = try need(c["out"] as? [Int], "no out")
            let image = RGBImage(width: size[0], height: size[1], pixels: source.map(UInt8.init))
            let out = image.resizedLanczos(width: size[2], height: size[3])
            #expect(out.pixels == expected.map(UInt8.init), "\(image.width)x\(image.height) → \(out.width)x\(out.height)")
        }
    }

    @Test func paddingAndCropRoundTrip() {
        let image = RGBImage(width: 3, height: 2, pixels: (0..<18).map(UInt8.init))
        let padded = image.padded(width: 7, height: 4, left: 2, top: 1)
        #expect(padded.width == 7 && padded.height == 4)
        #expect(padded.cropped(x: 2, y: 1, width: 3, height: 2) == image)
    }
}

@Suite struct Flux3Requests {
    let support = URL(fileURLWithPath: "/tmp/flux3-support")
    var job: JSONValue { .string("/tmp/flux3-support/jobs/1") }

    @Test func generateDefaults() throws {
        let r = try Flux3ImageRequest.parse(["prompt": "a fox", "out_dir": job], operation: "generate", support: support, randomSeed: { 9 })
        #expect(r.width == 1024 && r.height == 1024 && r.steps == 16 && r.n == 1 && r.seed == 9 && r.images.isEmpty)
    }

    @Test func refusesWhatTheModelLacks() {
        let base: JSONObject = ["prompt": "a fox", "out_dir": job]
        for extra: JSONObject in [
            ["guidance": 3.5], ["negative_prompt": "blur"], ["resolution": "512x512"], ["width": 1000], ["height": 8],
            ["width": 4096, "height": 2048], ["steps": 0], ["n": 5], ["seed": -1], ["images": [["path": "/etc/hosts"]]]
        ] {
            #expect(throws: WorkerFailure.self) { try Flux3ImageRequest.parse(base.merging(extra) { $1 }, operation: "generate", support: support) }
        }
        #expect(throws: WorkerFailure.self) { try Flux3ImageRequest.parse(["prompt": " ", "out_dir": job], operation: "generate", support: support) }
        #expect(throws: WorkerFailure.self) { try Flux3ImageRequest.parse(["prompt": "x", "out_dir": "/tmp/elsewhere"], operation: "generate", support: support) }
    }

    @Test func editTakesOneToTenReadableReferences() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("flux3-ref-\(UUID().uuidString).png")
        try Data([0]).write(to: file); defer { try? FileManager.default.removeItem(at: file) }
        let one: JSONValue = ["path": .string(file.path)]
        let r = try Flux3ImageRequest.parse(["prompt": "x", "images": [one], "out_dir": job], operation: "edit", support: support)
        #expect(r.images == [file] && r.width == nil && r.height == nil)
        #expect(throws: WorkerFailure.self) { try Flux3ImageRequest.parse(["prompt": "x", "out_dir": job], operation: "edit", support: support) }
        #expect(throws: WorkerFailure.self) {
            try Flux3ImageRequest.parse(["prompt": "x", "images": .array(Array(repeating: one, count: 11)), "out_dir": job], operation: "edit", support: support)
        }
        #expect(throws: WorkerFailure.self) {
            try Flux3ImageRequest.parse(["prompt": "x", "images": [["path": "/nonexistent/ref.png"]], "out_dir": job], operation: "edit", support: support)
        }
    }
}

/// Fake FLUX 3 engine: a gradient at the requested (or the first reference's) size, one callback per step.
final class FakeFlux3: Flux3Engine {
    var requests: [Flux3Request] = []
    func image(_ request: Flux3Request, step: (Int, Int) throws -> Void) throws -> (pixels: [UInt8], width: Int, height: Int) {
        requests.append(request)
        for i in 1...request.steps { try step(i, request.steps) }
        let w = request.crop?[2] ?? request.width ?? 64, h = request.crop?[3] ?? request.height ?? 64
        return ((0..<(w * h * 3)).map { UInt8(truncatingIfNeeded: $0) }, w, h)
    }
}

@Suite(.serialized) struct Flux3Wire {
    init() { useTestMetallib() }
    func frames(_ pipe: Pipe) throws -> [JSONObject] {
        try pipe.fileHandleForWriting.close()
        return try Frames.decode(pipe.fileHandleForReading.readDataToEndOfFile())
    }

    func snapshot(_ root: URL) throws -> URL {
        let weights = root.appendingPathComponent("weights")
        for file in Flux3Weights.requiredFiles {
            let url = weights.appendingPathComponent(file)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: url)
        }
        return weights
    }

    @Test func architectureSelectsFlux3AndServesGenerateAndEdit() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("flux3-wire-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let weights = try snapshot(root), support = root.appendingPathComponent("support")
        let reference = root.appendingPathComponent("ref.png")
        try PNGFixture.write(width: 48, height: 32, to: reference)
        let pipe = Pipe(), jobs = Jobs(), engine = FakeFlux3()
        let runner = Runner(wire: WorkerChannel(output: pipe.fileHandleForWriting), jobs: jobs, support: support, loader: { _ in FakeEngine() })
        runner.flux3Loader = { _, _ in engine }
        for id in 1...6 { _ = jobs.admit(Double(id)) }
        let config: JSONValue = ["architecture": "flux3-image-klein"]
        #expect(throws: WorkerFailure.self) {
            try runner.run(["id": 1, "op": "load", "dtype": "fp16", "weights_dir": .string(weights.path), "config": config])
        }
        #expect(try runner.run(["id": 2, "op": "load", "dtype": "bf16", "weights_dir": .string(weights.path), "config": config]) == false)
        #expect(runner.flux3 != nil && runner.engine == nil && runner.status()["model"] == "flux3-image-klein")
        let out = support.appendingPathComponent("jobs/j1")
        _ = try runner.run(["id": 3, "op": "generate", "prompt": "p", "width": 64, "height": 96, "steps": 2, "n": 2, "seed": 7, "out_dir": .string(out.path)])
        _ = try runner.run(["id": 4, "op": "edit", "prompt": "q", "images": [["path": .string(reference.path)]], "steps": 1, "out_dir": .string(out.path)])
        // A reference too small to condition on is the caller's error, refused before the engine runs.
        let tiny = root.appendingPathComponent("tiny.png")
        try PNGFixture.write(width: 10, height: 40, to: tiny)
        let pair: JSONValue = [["path": .string(reference.path)], ["path": .string(tiny.path)]]
        #expect(throws: WorkerFailure(code: "invalid_request", message: "A reference image is too small to condition on: it needs at least 16 pixels per side.")) {
            try runner.run(["id": 6, "op": "edit", "prompt": "q", "images": pair, "out_dir": .string(out.path)])
        }
        #expect(try runner.run(["id": 5, "op": "unload"]) == true)
        #expect(engine.requests.count == 3)
        #expect(engine.requests[0].width == 64 && engine.requests[0].height == 96 && engine.requests[0].seed == 7 && engine.requests[1].seed == 8)
        // The 48 x 32 reference is planned before the engine runs: padded to the 64 x 64 minimum canvas, cropped back.
        #expect(engine.requests[2].references.map { [$0.width, $0.height] } == [[64, 64]] && engine.requests[2].width == 64)
        #expect(engine.requests[2].crop == [8, 16, 48, 32] && engine.requests[2].planned)
        let sent = try frames(pipe)
        let generated = try #require(sent.first { $0["id"]?.number == 3 && $0["ok"]?.bool == true })
        let files = try #require(generated["files"]?.array)
        #expect(files.count == 2 && files.allSatisfy { $0["mime"] == "image/png" && $0["id"]?.string?.hasPrefix("image-") == true })
        let edited = try #require(sent.first { $0["id"]?.number == 4 && $0["ok"]?.bool == true })
        #expect(edited["files"]?.array?.first?["width"]?.number == 48)
    }

    @Test func editIsRefusedForFlux2() throws {
        let pipe = Pipe(), jobs = Jobs()
        _ = jobs.admit(1)
        let runner = Runner(wire: WorkerChannel(output: pipe.fileHandleForWriting), jobs: jobs, loader: { _ in FakeEngine() })
        #expect(throws: WorkerFailure.self) { try runner.run(["id": 1, "op": "edit", "prompt": "x", "out_dir": "/tmp/x"]) }
    }
}

/// A tiny RGB PNG for reference-image tests.
enum PNGFixture {
    static func write(width: Int, height: Int, to url: URL) throws {
        try Flux2Klein.PNG.write((0..<(width * height * 3)).map { UInt8(truncatingIfNeeded: $0 * 7) }, width: width, height: height, to: url)
    }
}

final class FakeEngine: ImageEngine {
    func image(prompt: String, width: Int, height: Int, steps: Int, seed: UInt64, step: (Int, Int) throws -> Void) throws -> (pixels: [UInt8], width: Int, height: Int) {
        ([UInt8](repeating: 0, count: width * height * 3), width, height)
    }
}

@Suite struct Flux3ImageDecoding {
    @Test func readsPNGPixelsAsStored() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("flux3-read-\(UUID().uuidString).png")
        defer { try? FileManager.default.removeItem(at: url) }
        try PNGFixture.write(width: 5, height: 3, to: url)
        let image = try RGBImage.read(url)
        #expect(image.width == 5 && image.height == 3)
        #expect(image.pixels == (0..<45).map { UInt8(truncatingIfNeeded: $0 * 7) })
    }
}

@Suite struct Flux3Planning {
    /// The 4 MP cap holds for the resolved size, not only when both sides are given.
    @Test func resolvedSizeIsCapped() throws {
        var request = Flux3Request(prompt: "x")
        request.references = [RGBImage(width: 2048, height: 2048, pixels: [UInt8](repeating: 9, count: 2048 * 2048 * 3))]
        request.width = 3200
        #expect(throws: Flux3Error.self) { try Flux3Pipeline.plan(request) }
        request.width = nil
        let planned = try Flux3Pipeline.plan(request)
        #expect(planned.width == 2048 && planned.height == 2048 && planned.crop == nil && planned.planned)
        #expect(try Flux3Pipeline.plan(planned) == planned)
        var generate = Flux3Request(prompt: "x")
        generate.width = 4096
        #expect(try Flux3Pipeline.plan(generate).height == 1024) // 4096 x 1024 is exactly 4 MP
        generate.height = 1040
        #expect(throws: Flux3Error.self) { try Flux3Pipeline.plan(generate) }
    }
}

/// Pillow 13 `ImageOps.exif_transpose(Image.open(p)).convert("RGB")`: stored colours, alpha dropped, grey repeated.
@Suite struct Flux3ReadsLikePillow {
    @Test func transparentAndGreyPNGsKeepTheirStoredColour() throws {
        let expected = try TinyFixture.json("read.json")
        let rgba = try RGBImage.read(TinyFixture.root.appendingPathComponent("read-rgba.png"))
        #expect(rgba.width == 7 && rgba.height == 5 && rgba.pixels.map(Int.init) == expected["rgba"] as? [Int])
        let la = try RGBImage.read(TinyFixture.root.appendingPathComponent("read-la.png"))
        #expect(la.width == 6 && la.height == 4 && la.pixels.map(Int.init) == expected["la"] as? [Int])
    }

    @Test(arguments: 2...8) func orientationMatchesExifTranspose(_ orientation: Int) throws {
        let cases = try need(try TinyFixture.json("read.json")["orient"] as? [String: [String: [Int]]], "no orientation cases")
        let want = try need(cases[String(orientation)], "no case \(orientation)")
        let out = RGBImage(width: 3, height: 2, pixels: (0..<18).map(UInt8.init)).oriented(orientation)
        #expect([out.width, out.height] == want["size"] && out.pixels.map(Int.init) == want["pixels"])
    }
}
