// SPDX-License-Identifier: MIT
// The worker boundary: both folders' operation schemas, request parsing, and the adapter's random-weight path at every
// declared tier (load, generate, edit, PNG receipts) — the same path the lab smoke drives through the family worker.
import Foundation
import IntegrationKit
import MLX
import ArsenalCore
import ArsenalWire
import Testing
@testable import QwenImageNative
@testable import QwenImageWorker

@Suite(.serialized) struct WorkerTests {
    static let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
    static let folders = ["qwen-image-2.1", "qwen-image-2.1-turbo"].map { repo.appendingPathComponent("retired-models/\($0)") }
    init() {
        if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { GPU.metallib = URL(fileURLWithPath: path) }
        setenv("ARSENAL_NATIVE_LAB", "1", 1)
    }

    func schema(_ folder: URL, _ name: String) throws -> JSONValue { .object(try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("ops/\(name)")))) }
    func valid(_ value: JSONObject, _ schema: JSONValue) -> Bool { (try? OperationSchemaValidator.validate(.object(value), schema: schema)) != nil }

    @Test func operationSchemasAcceptTheDocumentedRequestsAndRefuseTheRest() throws {
        for folder in Self.folders {
            let turbo = folder.lastPathComponent.hasSuffix("turbo")
            let generate = try schema(folder, "generate.input.json"), edit = try schema(folder, "edit.input.json")
            #expect(valid(["prompt": "a lighthouse", "width": 2752, "height": 1536, "seed": 1, "n": 2], generate))
            #expect(!valid(["prompt": ""], generate) && !valid(["prompt": "x", "width": 1000], generate) && !valid(["prompt": "x", "width": 3072], generate))
            #expect(!valid(["prompt": "x", "images": [["path": "/a.png"]]], generate))
            #expect(valid(["prompt": "swap", "images": [["path": "/a.png"], ["path": "/b.png"]]], edit))
            #expect(!valid(["prompt": "swap"], edit) && !valid(["prompt": "swap", "images": []], edit))
            #expect(!valid(["prompt": "swap", "images": .array(Array(repeating: ["path": "/a.png"], count: 11))], edit))
            // Turbo runs its saved 8-step schedule without guidance; the base model takes both.
            #expect(valid(["prompt": "x", "steps": 40, "guidance": 4, "negative_prompt": "blur"], generate) == !turbo)
            #expect(OperationSchemaValidator.declares(path: "images", in: edit.object!))
            // Transparency is a generate switch an agent can find in the schema; edit takes the sentences in its prompt.
            #expect(valid(["prompt": "a sticker", "transparent": true], generate) && !valid(["prompt": "x", "transparent": "yes"], generate))
            #expect(!valid(["prompt": "swap", "images": [["path": "/a.png"]], "transparent": true], edit))
            let output = try schema(folder, "image.output.json")
            let result: JSONObject = [
                "files": [
                    [
                        "id": "image-1", "path": "/j/a.png", "mime": "image/png", "width": 1024, "height": 1024, "bytes": 10,
                        "sha256": .string(String(repeating: "a", count: 64)), "alpha": true
                    ]
                ],
                "seed": 3, "seconds": 1.5
            ]
            #expect(valid(result, output))
        }
    }

    @Test func foldersValidateWithTheirDeclaredOperationsAndTiers() throws {
        for folder in Self.folders {
            let model = try ModelManifest.load(folder, requirePins: false)
            #expect(model.problems(folder: folder, requirePins: false).isEmpty, "\(model.problems(folder: folder, requirePins: false))")
            #expect(model.operations.map(\.name).sorted() == ["edit", "generate"])
            #expect(model.precisions == ["bf16"] && model.derive == nil)
            #expect(model.runners.map(\.worker) == ["qwenimage"] && model.architecture == "qwen-image-2.1")
        }
    }

    @Test func requestParsingFollowsTheContract() throws {
        let out = FileManager.default.temporaryDirectory.path
        var r = try ImageRequest(operation: "generate", input: ["prompt": "x", "out_dir": .string(out)], turbo: false, randomSeed: { 9 })
        #expect(r.width == 1024 && r.height == 1024 && r.steps == 40 && r.seed == 9 && r.count == 1 && r.guidance == 1)
        r = try ImageRequest(operation: "edit", input: ["prompt": "x", "images": [["path": "/a.png"]], "out_dir": .string(out)], turbo: true)
        #expect(r.width == nil && r.steps == 8 && r.images.count == 1)
        #expect(throws: WorkerFailure.self) { try ImageRequest(operation: "generate", input: ["prompt": "x", "steps": 4, "out_dir": .string(out)], turbo: true) }
        #expect(throws: WorkerFailure.self) { try ImageRequest(operation: "generate", input: ["prompt": "x", "width": 1000, "out_dir": .string(out)], turbo: false) }
        #expect(throws: WorkerFailure.self) { try ImageRequest(operation: "edit", input: ["prompt": "x", "out_dir": .string(out)], turbo: false) }
        #expect(throws: WorkerFailure.self) { try ImageRequest(operation: "generate", input: ["prompt": "x", "out_dir": "/nonexistent-job"], turbo: false) }
        // transparent wraps the description in the maker's RGBA format once, whether or not the prompt already has it.
        let opening = "This is an RGBA image with transparency.", closing = "The image has alpha channel and the background is transparent."
        let wrapped = opening + " A cute cartoon dragon sticker. " + closing
        for prompt in ["A cute cartoon dragon sticker", " A cute cartoon dragon sticker. ", wrapped, opening + " A cute cartoon dragon sticker."] {
            r = try ImageRequest(operation: "generate", input: ["prompt": .string(prompt), "transparent": true, "out_dir": .string(out)], turbo: true)
            #expect(r.prompt == wrapped, "\(prompt)")
        }
        r = try ImageRequest(operation: "generate", input: ["prompt": "a dragon", "transparent": false, "out_dir": .string(out)], turbo: false)
        #expect(r.prompt == "a dragon")
        #expect(throws: WorkerFailure.self) {
            try ImageRequest(operation: "edit", input: ["prompt": "x", "images": [["path": "/a.png"]], "transparent": true, "out_dir": .string(out)], turbo: false)
        }
    }

    /// Native sizes are 2048 × 2048 (4.19 MP) and 2752 × 1536; anything larger is refused before any work starts.
    @Test func sizeCapAdmitsNativeShapesAndRefusesLarger() throws {
        let out = FileManager.default.temporaryDirectory.path
        func generate(_ width: Int, _ height: Int) throws -> ImageRequest {
            let input: JSONObject = ["prompt": "x", "width": .number(Double(width)), "height": .number(Double(height)), "out_dir": .string(out)]
            return try ImageRequest(operation: "generate", input: input, turbo: false)
        }
        for (w, h) in [(2048, 2048), (2752, 1536), (1024, 1024), (512, 2048)] { #expect(try generate(w, h).width == w) }
        for (w, h) in [(2752, 2752), (2048, 2080), (1536, 2752)] {
            #expect(throws: WorkerFailure.self) { try generate(w, h) }
        }
    }

    @Test(.heavy) func registryAndEligibility() {
        #expect(QwenImageWorker.registrations.count == 1 && QwenImageWorker.registrations[0].modelTypes == ["qwen-image-2.1"])
        // BF16 only: Get normalizes to the served half and derives no quantized tier.
        #expect(QwenImageWorker.registrations[0].conversion == nil)
        #expect(QwenImageWeights.eligible("transformer_blocks.0.attn.to_q.weight", [4096, 4096]))
        #expect(QwenImageWeights.eligible("model.language_model.embed_tokens.weight", [151936, 4096]))
        #expect(!QwenImageWeights.eligible("model.visual.pos_embed.weight", [2304, 1152]))
        #expect(!QwenImageWeights.eligible("model.visual.patch_embed.proj.weight", [1152, 3, 2, 16, 16]))
        #expect(!QwenImageWeights.eligible("decoder.conv_in.weight", [1152, 64, 3, 3]) && !QwenImageWeights.eligible("txt_in.text_norm.weight", [4096]))
        #expect(QwenImageWeights.encoderName("lm_head.weight") == nil && QwenImageWeights.encoderName("model.language_model.norm.weight") == nil)
        #expect(QwenImageWeights.encoderName("model.visual.blocks.3.attn.qkv.weight") == "visual.blocks.3.attn.qkv.weight")
        #expect(QwenImageWeights.transformerName("modulation.1.weight") == "modulation.linear.weight")
        #expect(QwenImageWeights.vaeName("decoder.up_blocks.0.upsampler.time_conv.weight") == nil)
        #expect(QwenImageWeights.vaeName("encoder.down_blocks.1.downsampler.resample.1.bias") == "encoder.down_blocks.1.downsampler.conv.bias")
        #expect(QwenImageWeights.vaeTensor("decoder.norm_out.gamma", MLXArray.ones([8, 1, 1, 1])).shape == [8])
        #expect(QwenImageWeights.vaeTensor("decoder.conv_in.weight", MLXArray.ones([8, 4, 3, 3])).shape == [8, 3, 3, 4])
    }

    /// The declared tier loads the random graph and serves both operations end to end, base and Turbo.
    @Test(.heavy, arguments: [("bf16", false), ("bf16", true)])
    func randomWeightsServeEveryTier(precision: String, turbo: Bool) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("qwen-image-tests-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let job = root.appendingPathComponent("job"), weights = root.appendingPathComponent("weights")
        try FileManager.default.createDirectory(at: job, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
        let reference = root.appendingPathComponent("reference.png")
        try ImageOps.writePNG(RGBAImage(width: 48, height: 32, bytes: (0..<(48 * 32 * 4)).map { UInt8(($0 * 13) % 256) }), to: reference)
        var config: JSONObject = ["random_weights": true, "architecture": "qwen-image-2.1"]
        if turbo { config["turbo"] = true }
        let adapter = QwenImageAdapter()
        try adapter.load(
            NativeLoadContext(["weights_dir": .string(weights.path), "model_dir": .string(root.path), "dtype": .string(precision), "config": .object(config)]),
            cancellation: NativeCancellation())
        defer { adapter.unload() }
        let packed = adapter.parameters.values.filter { $0.dtype == .uint32 }.count
        #expect(packed == 0)
        #expect(adapter.parameters.values.allSatisfy { ![.float32, .float64].contains($0.dtype) })
        var events: [JSONObject] = []
        let generated = try adapter.run(
            "generate", input: ["prompt": "a red apple", "width": 256, "height": 256, "seed": 5, "n": 2, "out_dir": .string(job.path)].merging(turbo ? [:] : ["steps": 2]) { $1 },
            cancellation: NativeCancellation(), emit: { if case .json(let e) = $0 { events.append(e) } })
        let images = generated["files"]?.array ?? []
        #expect(images.count == 2 && generated["seeds"]?.array?.compactMap(\.number) == [5, 6])
        #expect(images.allSatisfy { $0["width"]?.number == 256 && $0["height"]?.number == 256 && FileManager.default.fileExists(atPath: $0["path"]?.string ?? "") })
        let total = turbo ? 6 : 4 // the lab Turbo schedule has 3 sigmas
        #expect(events.last?["step"]?.number == Double(total) && events.last?["total"]?.number == Double(total))
        let edited = try adapter.run(
            "edit", input: ["prompt": "make it night", "images": [["path": .string(reference.path)]], "out_dir": .string(job.path)].merging(turbo ? [:] : ["steps": 1]) { $1 },
            cancellation: NativeCancellation(), emit: { _ in })
        // No size given: the reference's 3:2 aspect at about one megapixel.
        #expect(edited["files"]?.array?.first?["width"]?.number == 1248 && edited["files"]?.array?.first?["height"]?.number == 832)
    }

    @Test(.heavy) func cancellationStopsBetweenSteps() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("qwen-image-cancel-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let adapter = QwenImageAdapter()
        try adapter.load(
            NativeLoadContext([
                "weights_dir": .string(root.path), "model_dir": .string(root.path), "dtype": "bf16", "config": ["random_weights": true, "architecture": "qwen-image-2.1"]
            ]), cancellation: NativeCancellation())
        let token = NativeCancellation()
        #expect(throws: WorkerFailure.self) {
            _ = try adapter.run(
                "generate", input: ["prompt": "x", "width": 256, "height": 256, "steps": 50, "out_dir": .string(root.path)], cancellation: token,
                emit: { _ in token.cancel() })
        }
        let pngs = try FileManager.default.contentsOfDirectory(atPath: root.path).filter { $0.hasSuffix(".png") }
        #expect(pngs.isEmpty)
    }
}
