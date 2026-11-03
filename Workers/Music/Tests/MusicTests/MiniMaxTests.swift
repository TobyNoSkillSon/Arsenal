// SPDX-License-Identifier: MIT
import Foundation
import Testing
import IntegrationKit
import ArsenalWire
import MLX
import MLXNN
@testable import MusicWorker

/// MiniMax Music 3 contract checks without weights or GPU work: the maker's prompt assembly and scheduler (diffusers
/// values in `models/minimax-music3/gate/golden/contract-fixture.json`), window/crop arithmetic, request bounds and
/// the Get layout transforms.
@Suite struct MiniMaxTests {
    private var repo: URL { (0..<5).reduce(URL(fileURLWithPath: #filePath)) { url, _ in url.deletingLastPathComponent() } }
    private func fixture() throws -> [String: Any] {
        let url = repo.appendingPathComponent("models/minimax-music3/gate/golden/contract-fixture.json")
        return try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }

    static func useMetallib() throws {
        let metallib = try #require(
            ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] ?? ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"],
            "the GPU check needs the pinned no-fast-math metallib")
        GPU.metallib = URL(fileURLWithPath: metallib)
    }

    @Test func promptAssemblyMatchesTheMakersEncoder() throws {
        let cases = try #require(try fixture()["cases"] as? [[String: Any]])
        #expect(cases.count == 5)
        for row in cases {
            let caption = try #require(row["caption"] as? String), lyrics = try #require(row["lyrics"] as? String)
            #expect(MiniMaxPrompt.cleanCaption(caption) == row["clean_caption"] as? String, "caption \(caption.debugDescription)")
            #expect(MiniMaxPrompt.normalizeLyrics(lyrics) == row["normalized_lyrics"] as? String, "lyrics \(lyrics.debugDescription)")
            #expect(MiniMaxPrompt.text(caption: caption, lyrics: lyrics) == row["text"] as? String)
            let ids = try #require(row["ids"] as? [Int])
            #expect(MiniMaxPrompt.unconditional(ids) == row["unconditional_ids"] as? [Int])
        }
    }

    @Test func eulerScheduleMatchesTheMakersScheduler() throws {
        let scheduler = try #require(try fixture()["scheduler"] as? [String: [String: [Double]]])
        #expect(scheduler.count == 4)
        for (steps, values) in scheduler {
            let sigmas = MiniMaxPipeline.sigmas(Int(steps)!)
            let expected = try #require(values["sigmas"])
            #expect(sigmas.count == expected.count)
            for (a, b) in zip(sigmas, expected) { #expect(abs(Double(a) - b) < 1e-6) }
            #expect(Array(sigmas.dropLast()).map(Double.init).elementsEqual(try #require(values["timesteps"])) { abs($0 - $1) < 1e-6 })
        }
    }

    @Test func windowsLatentsAndCropsFollowThePipeline() {
        let c = MiniMaxConfiguration()
        #expect(c.frameRate == 25)
        #expect(c.latentLength(200) == 689 && c.latentLength(150) == 516 && c.latentLength(1) == 3)
        #expect(MiniMaxPipeline.chunkStarts(1) == [0] && MiniMaxPipeline.chunkStarts(200) == [0])
        #expect(MiniMaxPipeline.chunkStarts(201) == [0, 100] && MiniMaxPipeline.chunkStarts(250) == [0, 100])
        #expect(MiniMaxPipeline.chunkStarts(400) == [0, 100, 200])
        // Windows hop 100 frames = 344.5 latents; the kept spans tile: 0 ..< 689-258, then 86 ..< 689-258 of each next window.
        #expect(MiniMaxPipeline.cropLeftLatents + MiniMaxPipeline.cropRightLatents == 344)
        #expect(MiniMaxPipeline.overlapLatents * 2 == 344)
        #expect(c.upsampling.reduce(1, *) == c.outputHop)
    }

    @Test func requestBoundsAndDefaults() throws {
        let base: JSONObject = ["prompt": "warm acoustic pop", "lyrics": "[verse]\nhello"]
        let options = try MiniMaxAdapter.options(base)
        #expect(options.request.duration == 60 && options.request.steps == 30 && abs(options.request.guidance - 1.7) < 1e-6)
        #expect(options.batch == 1 && options.format == .wav && options.sampleRate == nil && !options.metadata)
        #expect(options.request.lyrics == "[verse]\nhello")
        // No lyrics, blank lyrics or the switch: the maker's [instrumental] tag (the schema leaves lyrics optional).
        var instrumental = base; instrumental["instrumental"] = true
        #expect(try MiniMaxAdapter.options(instrumental).request.lyrics == "[instrumental]")
        #expect(try MiniMaxAdapter.options(["prompt": "x"]).request.lyrics == "[instrumental]")
        #expect(try MiniMaxAdapter.options(["prompt": "x", "lyrics": " \n"]).request.lyrics == "[instrumental]")
        var rate = base; rate["sample_rate"] = 32000
        #expect(try MiniMaxAdapter.options(rate).sampleRate == 32000)
        var seeded = base; seeded["seed"] = 7
        #expect(try MiniMaxAdapter.options(seeded).request.seed == 7)
        #expect(try MiniMaxAdapter.options(base.merging(["duration": 0.04]) { $1 }).request.duration == 0.04)
        #expect(try MiniMaxAdapter.options(base.merging(["duration": 60, "batch_size": 12]) { $1 }).batch == 12)
        let rejected: [JSONObject] = [
            ["prompt": " ", "lyrics": "x"], ["lyrics": "x"], base.merging(["duration": 361]) { $1 }, base.merging(["duration": 0.03]) { $1 },
            base.merging(["steps": 0]) { $1 }, base.merging(["steps": 2.5]) { $1 }, base.merging(["guidance": 21]) { $1 },
            base.merging(["batch_size": 17]) { $1 }, base.merging(["duration": 60, "batch_size": 13]) { $1 }, base.merging(["seed": -1]) { $1 },
            base.merging(["sample_rate": 48000]) { $1 }, base.merging(["format": "mp3"]) { $1 },
            // Wrong JSON types are refused, not replaced by defaults.
            ["prompt": 5], base.merging(["lyrics": 5]) { $1 }, base.merging(["steps": "30"]) { $1 }, base.merging(["seed": "7"]) { $1 },
            base.merging(["instrumental": "yes"]) { $1 }, base.merging(["duration": "10"]) { $1 }, base.merging(["format": 1]) { $1 },
            base.merging(["include_metadata": 1]) { $1 }, base.merging(["sample_rate": "32000"]) { $1 }, base.merging(["guidance": .null]) { $1 }
        ]
        for input in rejected { #expect(throws: WorkerFailure.self, "\(input)") { try MiniMaxAdapter.options(input) } }
    }

    /// The fusion encoder's nearest resampling uses torch's float32 index formula, which differs from exact arithmetic
    /// at some window lengths; the fixture lists, from torch on CPU and MPS, every such position for 1–400 frames.
    @Test func nearestResamplingMatchesTorch() throws {
        let nearest = try #require(try fixture()["nearest"] as? [String: Any])
        let differs = try #require(nearest["differs_from_exact"] as? [String: [[Int]]])
        #expect(differs.count == 16 && nearest["cpu_equals_mps"] as? Bool == true)
        let c = MiniMaxConfiguration()
        for frames in 1...400 {
            let target = c.latentLength(frames)
            let indices = MiniMaxConditionEncoder.nearestIndices(frames: frames, target: target)
            #expect(indices.count == target)
            let changed = (0..<target).filter { indices[$0] != Swift.min(frames - 1, $0 * frames / target) }.map { [$0, indices[$0]] }
            #expect(changed == (differs[String(frames)] ?? []), "frames \(frames)")
        }
    }

    @Test func getLayoutsMatchMLXConvolutions() {
        typealias L = MusicConversion.MiniMaxLayout
        let cases: [(String, String, L)] = [
            ("condition_encoder", "proj.weight", .conv), ("condition_encoder", "proj.bias", .keep), ("condition_encoder", "layer_scale", .keep),
            ("transformer", "preprocess_conv.weight", .conv), ("transformer", "postprocess_conv.weight", .conv), ("transformer", "proj_in.weight", .keep),
            ("transformer", "time_proj.weight", .keep), ("vocoder", "blocks.0.conv_t1.weight_v", .convTranspose), ("vocoder", "conv_in.weight_v", .conv),
            ("vocoder", "blocks.2.res_unit3.conv2.weight_v", .conv), ("vocoder", "dec_in_proj.weight", .conv), ("vocoder", "conv_out.bias", .keep),
            ("vocoder", "snake_out.alpha", .vector), ("language_model", "model.layers.0.mlp.up_proj.weight", .keep),
            ("rvq_depth_decoder", "audio_heads.3.weight", .keep)
        ]
        for (component, key, layout) in cases { #expect(MusicConversion.miniMaxLayout(component: component, key: key) == layout, "\(component) \(key)") }
    }

    @Test func registryResolvesTheFolderArchitecture() throws {
        let registration = try NativeRegistration.resolve(checkpoint: [:], architecture: "minimax_music3", registrations: MusicWorker.registrations)
        #expect(registration.conversion?.recipe == "cast:minimax-music3-v1")
        #expect(registration.make().architecture == "minimax_music3")
    }

    @Test func componentConfigsMatchTheBuiltInArchitecture() throws {
        let configs = repo.appendingPathComponent("models/minimax-music3/gate/configs")
        #expect(try MiniMaxConfiguration.load(configs) == MiniMaxConfiguration())
    }

    /// Every tensor of the five pinned checkpoints (names, shapes and dtypes read from the safetensors headers) maps to
    /// exactly one port parameter of the same shape after Get's layout recipe; nothing is left over on either side.
    /// The full-size graph is built lazily and never evaluated.
    @Test(.heavy) func everyCheckpointTensorMapsToThePort() throws {
        try Self.useMetallib()
        let url = repo.appendingPathComponent("models/minimax-music3/gate/golden/checkpoint-keys.json")
        let doc = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let components = try #require(doc["components"] as? [String: [String: [Any]]])
        let c = MiniMaxConfiguration()
        let modules: [String: Module] = [
            "language_model": MiniMaxLanguageModel(c), "rvq_depth_decoder": MiniMaxDepthDecoder(c), "condition_encoder": MiniMaxConditionEncoder(c),
            "transformer": MiniMaxFlowTransformer(c), "vocoder": MiniMaxVocoder(c)
        ]
        #expect(Set(components.keys) == Set(MiniMaxComponents.all) && components.values.map(\.count).reduce(0, +) == 1012)
        var total = 0
        for (component, tensors) in components {
            let module = try #require(modules[component])
            let port = Dictionary(uniqueKeysWithValues: module.parameters().flattened().map { ($0.0, $0.1.shape) })
            var mapped: [String: [Int]] = [:]
            for (key, value) in tensors where !key.hasSuffix(".weight_g") {
                let shape = value.dropFirst().compactMap { ($0 as? NSNumber)?.intValue }
                let name = key.hasSuffix(".weight_v") ? String(key.dropLast(9)) + ".weight" : key
                let layout: [Int]
                switch MusicConversion.miniMaxLayout(component: component, key: key) {
                case .keep: layout = shape
                case .conv: layout = [shape[0], shape[2], shape[1]]
                case .convTranspose: layout = [shape[1], shape[2], shape[0]]
                case .vector: layout = [shape.reduce(1, *)]
                }
                mapped[name] = layout
                if component != "vocoder" || !key.hasSuffix(".alpha") {
                    #expect(["BF16", "F32"].contains(value.first as? String ?? ""), "\(component) \(key) dtype")
                }
            }
            #expect(Set(mapped.keys) == Set(port.keys), "\(component): \(Set(mapped.keys).symmetricDifference(Set(port.keys)).sorted().prefix(6))")
            for (name, shape) in mapped { #expect(port[name] == shape, "\(component) \(name) \(shape) vs \(port[name] ?? [])") }
            total += port.count
        }
        #expect(total == 1012 - 30) // 30 vocoder weight-norm gains fold into their kernels
    }

    /// Random tiny weights through the whole serving graph: frames, windows, overlap carry and stitching.
    @Test(.heavy) func tinyRandomGraphMakesStereoAudio() throws {
        try Self.useMetallib()
        let c = MiniMaxConfiguration.tiny
        let lm = MiniMaxLanguageModel(c), depth = MiniMaxDepthDecoder(c), condition = MiniMaxConditionEncoder(c)
        let transformer = MiniMaxFlowTransformer(c), vocoder = MiniMaxVocoder(c)
        for module in [lm, depth, condition, transformer, vocoder] as [Module] { try MusicSupport.initialize(module, precision: "fp16") }
        try MiniMaxAdapter.computeInBF16(lm)
        #expect(lm.parameters().flattened().allSatisfy { $0.1.dtype == .bfloat16 })
        let pipeline = MiniMaxPipeline(
            configuration: c, lm: lm, depth: depth, condition: condition, transformer: transformer, vocoder: vocoder,
            tokenizer: try MusicSupport.fixtureTokenizer())
        let tokens = MLXArray([Int32](repeating: 5, count: 12)).reshaped(2, 6)
        // 260 forced frames: two windows, one overlap.
        var forced: [[Int32]] = []
        for i in 0..<260 {
            var row: [Int32] = [Int32(i % 64)]
            for j in 1..<8 { row.append(Int32((j * 7 + i) % 32)) }
            forced.append(row)
        }
        let ar = try pipeline.frames(tokens: tokens, maxFrames: 260, seed: 1, forced: forced)
        #expect(ar.hidden.shape == [1, 260, 8 * c.hidden] && ar.codes.count == 260)
        // The BF16 global model hands FP16 states to the local model and the fusion encoder.
        #expect(ar.hidden.dtype == .float16)
        let chunks = try pipeline.latents(frameHidden: ar.hidden, steps: 2, guidance: 1.7, key: MLXRandom.key(2), cancelled: { false }, progress: { _ in })
        #expect(chunks.map { $0.shape } == [[1, c.latentLength(200), c.latent], [1, c.latentLength(160), c.latent]])
        let audio = try pipeline.decode(chunks, cancelled: { false })
        let expected = (c.latentLength(200) - MiniMaxPipeline.cropRightLatents + c.latentLength(160) - MiniMaxPipeline.cropLeftLatents) * c.outputHop
        #expect(audio.left.count == expected && audio.right.count == expected)
        let finite = audio.left.allSatisfy { $0.isFinite } && audio.right.allSatisfy { $0.isFinite }
        #expect(finite)
        // Only the restricted head stays resident; guidance 1 runs the conditional pass alone.
        #expect(lm.lmHead.weight.shape == [MiniMaxPrompt.semanticVocabulary + 1, c.hidden])
        let unguided = try pipeline.latents(
            frameHidden: ar.hidden[0..., 0..<40], steps: 2, guidance: 1, key: MLXRandom.key(2), cancelled: { false }, progress: { _ in })
        #expect(unguided.map { $0.shape } == [[1, c.latentLength(40), c.latent]])
        // 260 forced frames feed back 259 times: the last frame is never fed back.
        #expect(pipeline.lastFeedbackPasses == 259)
        // Sampled frames stop at the frame budget.
        let sampled = try pipeline.frames(tokens: tokens, maxFrames: 5, seed: 3)
        #expect(sampled.hidden.dim(1) <= 5 && sampled.hidden.dim(2) == 8 * c.hidden)
        if !sampled.ended { #expect(pipeline.lastFeedbackPasses == 5) } // frames 0…4 fed back; frame 5 ends the budget
    }

    /// FP16 `alpha + 1e-9` loses the epsilon; the snake runs in float32 so zero or tiny scales stay finite.
    @Test(.heavy) func snakeStaysFiniteForZeroAndTinyScales() throws {
        try Self.useMetallib()
        let snake = MiniMaxSnake(4)
        try snake.update(parameters: ModuleParameters.unflattened(["alpha": MLXArray([Float(0), 1e-8, 1e-3, 1]).asType(.float16)]), verify: .none)
        let y = snake(MLXArray([Float](repeating: 0.5, count: 8)).reshaped(1, 2, 4).asType(.float16))
        #expect(y.dtype == .float16)
        #expect(MLX.all(MLX.isFinite(y)).item(Bool.self))
    }
}
