import Foundation
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire
import Testing
@testable import AudioWorker
@testable import VellaSTT

/// The Vella transcription folders: Parakeet v3 Ultra and both Qwen3-ASR sizes on VellaSTT.
private let vellaRepository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent()

private func vellaMetallib() throws {
    guard let path = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] ?? ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] else {
        throw audioFailure("The family GPU test needs the pinned no-fast-math metallib.")
    }
    setTestMetallib(URL(fileURLWithPath: path))
    setenv("ARSENAL_NATIVE_LAB", "1", 1)
}

/// A shrunk checkpoint directory for one folder's no-weight fixture: config.json plus Qwen's gate tokenizer.
private struct VellaFixture {
    let id: String
    let folder: URL
    let config: JSONObject
    let root: URL
    let source: URL
    let architecture: String
    init(_ id: String) throws {
        self.id = id
        folder = vellaRepository.appendingPathComponent("models/" + id)
        config = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("gate/no-weights.json")))["config"]!.object!
        let manifest = try String(contentsOf: folder.appendingPathComponent("model.toml"), encoding: .utf8)
        architecture = manifest.split(separator: "\n").first { $0.hasPrefix("architecture = ") }!.split(separator: "\"")[1].description
        root = FileManager.default.temporaryDirectory.appendingPathComponent("vella-asr-" + UUID().uuidString)
        source = root.appendingPathComponent("source")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try WireJSON.encode(config).write(to: source.appendingPathComponent("config.json"))
    }
    var registration: NativeRegistration { AudioWorker.registrations.first { $0.modelTypes.contains(architecture) }! }
    func load(_ precision: String = "fp16", weights: URL? = nil) throws -> AudioAdapter {
        guard let adapter = registration.make() as? AudioAdapter else { throw audioFailure("Expected an audio adapter.") }
        var settings = config
        settings["architecture"] = .string(architecture)
        if weights == nil { settings["random_weights"] = true }
        let context = try NativeLoadContext([
            "model_dir": .string(folder.path), "weights_dir": .string((weights ?? source).path), "dtype": .string(precision), "config": .object(settings)
        ])
        try adapter.load(context, cancellation: NativeCancellation())
        return adapter
    }
    func cleanup() { try? FileManager.default.removeItem(at: root); Memory.clearCache() }
}

/// Five seconds of a deterministic voiced-like signal: harmonics with a slow amplitude envelope and seeded noise.
private func vellaSignal(seconds: Double = 5) -> [Float] {
    var state: UInt64 = 0x9E37_79B9_7F4A_7C15
    return (0..<Int(seconds * 16000)).map { index in
        let t = Double(index) / 16000
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        let noise = Double(Int64(bitPattern: state >> 11) % 2000) / 2000 - 0.5
        let envelope = 0.5 + 0.5 * sin(2 * .pi * 1.3 * t)
        return Float(envelope * (0.3 * sin(2 * .pi * 180 * t) + 0.15 * sin(2 * .pi * 360 * t) + 0.08 * sin(2 * .pi * 1100 * t)) + 0.02 * noise)
    }
}

private func writeWave(_ samples: [Float], to url: URL) throws {
    var data = Data()
    func append<T: FixedWidthInteger>(_ value: T) { var little = value.littleEndian; data.append(Data(bytes: &little, count: MemoryLayout<T>.size)) }
    data.append(contentsOf: Array("RIFF".utf8)); append(UInt32(36 + samples.count * 2)); data.append(contentsOf: Array("WAVEfmt ".utf8))
    append(UInt32(16)); append(UInt16(1)); append(UInt16(1)); append(UInt32(16000)); append(UInt32(32000)); append(UInt16(2)); append(UInt16(16))
    data.append(contentsOf: Array("data".utf8)); append(UInt32(samples.count * 2))
    for sample in samples { append(Int16(max(-1, min(1, sample)) * 32767)) }
    try data.write(to: url)
}

extension PrecisionTests {
    /// Get's path for each folder: a BF16 checkpoint under the published tensor names (header-map renames inverted) loads
    /// into the adapter's graph, every tier the folder declares installs, and the reloaded model transcribes exactly
    /// like the graph it was saved from. Qwen3-ASR's tiers pack the text model only; the audio tower stays half.
    @Test(.heavy) func vellaCheckpointsInstallThroughTheGetPath() throws {
        try vellaMetallib()
        for (id, tiers) in [("parakeet-v3-ultra", ["fp16"]), ("qwen3-asr-0.6b", ["fp16"]), ("qwen3-asr-1.7b", ["fp16"])] {
            try autoreleasepool {
                let fixture = try VellaFixture(id)
                defer { fixture.cleanup() }
                let audio = fixture.root.appendingPathComponent("signal.wav")
                try writeWave(vellaSignal(), to: audio)
                let request: JSONObject = ["audio": ["path": .string(audio.path)]]
                let dense = try fixture.load()
                // Random graphs never emit an end token: both sides generate the same bounded number of tokens.
                (dense as? VellaQwenASRAdapter)?.tokenLimit = 6
                let expected = try dense.run("transcribe", input: request, cancellation: NativeCancellation(), emit: { _ in })
                let layout = try WireJSON.decode(Data(contentsOf: fixture.folder.appendingPathComponent("gate/header-map.json")))
                var published: [String: String] = [:]
                for (name, rule) in layout["tensors"]?.object ?? [:] { published[rule["name"]!.string!] = name }
                var tensors: [String: MLXArray] = [:]
                for (name, tensor) in dense.graph!.parameters().flattened() { tensors[published[name] ?? name] = tensor.asType(.bfloat16) }
                let eligible = dense.parameters.filter { vellaQwenEligible($0.key, $0.value.shape) }.count
                dense.unload()
                try save(arrays: tensors, url: fixture.source.appendingPathComponent("model.safetensors"))
                for file in ["tokenizer.json", "tokenizer_config.json"] where fixture.architecture == "qwen3_asr_vella" {
                    try FileManager.default.copyItem(
                        at: fixture.folder.appendingPathComponent("gate/tokenizer/" + file), to: fixture.source.appendingPathComponent(file))
                }
                for precision in tiers {
                    var weights = fixture.source
                    if precision != "fp16" {
                        weights = fixture.root.appendingPathComponent(precision)
                        let converter = fixture.registration.conversion!
                        _ = try converter.run(
                            [
                                "src_dir": .string(fixture.source.path), "dst_dir": .string(weights.path), "dtype": .string(precision), "group_size": 64,
                                "recipe": .string(converter.recipe)
                            ], cancellation: NativeCancellation())
                    }
                    let model = try fixture.load(precision, weights: weights)
                    try NativeParameters.requireHalf(model.parameters)
                    let packed = model.parameters.filter { $0.value.dtype == .uint32 }
                    #expect(packed.count == (precision == "fp16" ? 0 : eligible), "\(id) \(precision)")
                    #expect(eligible > 0 || id != "qwen3-asr-1.7b")
                    #expect(!packed.keys.contains { $0.hasPrefix("audio_tower.") }, "\(id) \(precision) keeps the audio tower half")
                    if precision == "fp16" {
                        (model as? VellaQwenASRAdapter)?.tokenLimit = 6
                        let reloaded = try model.run("transcribe", input: request, cancellation: NativeCancellation(), emit: { _ in })
                        #expect(reloaded == expected, "\(id): the checkpoint layout reloads into the same graph")
                    }
                    model.unload()
                }
            }
        }
    }

    /// Vella's optimized Parakeet components (fused Conformer encoder, compiled TDT decoder, tail block sizing) are exact:
    /// at FP16 they give stock's token IDs, and the fused encoder stays within half rounding of the stock modules.
    @Test(.heavy) func vellaParakeetOptimizedPathMatchesStock() throws {
        try vellaMetallib()
        let fixture = try VellaFixture("parakeet-v3-ultra")
        defer { fixture.cleanup() }
        let adapter = try fixture.load()
        defer { adapter.unload() }
        let model = try #require(adapter.graph as? ParakeetModel)
        #expect(model.keptLevers.contains("VELLA_PARAKEET_TAILBLOCK"), "Ultra keeps Vella's tail block sizing")
        #expect(model.computeDType == .float16)
        let audio = MLXArray(vellaSignal())
        _ = model.configureFastPath(enabled: false)
        let stock = model.qualificationTokens(audio: audio)
        let mel = ParakeetAudio.logMelSpectrogram(audio, config: model.preprocessConfig).asType(.float16)
        let stockEncoder = model.encodeBatchFeatures(mel).0.asType(.float32)
        guard VellaFastPath.available() else { return }
        #expect(model.configureFastPath(enabled: true))
        #expect(model.fastPathComponents["encoder"] == true && model.fastPathComponents["decoder"] == true)
        #expect(model.fastPathComponents["nax_gemm"] != true, "the inexact GEMM stays off")
        let fast = model.qualificationTokens(audio: audio)
        #expect(model.fastPathFinite)
        #expect(!stock.isEmpty)
        #expect(fast == stock)
        let fastEncoder = model.encodeBatchFeatures(mel).0.asType(.float32)
        let delta = fastEncoder - stockEncoder
        let relative = MLX.sqrt((delta * delta).sum() / (stockEncoder * stockEncoder).sum()).item(Float.self)
        #expect(relative < 0.02, "fused encoder deviation \(relative)")
        model.keptLevers = KeptLevers(enabled: [])
        #expect(model.qualificationTokens(audio: audio) == stock, "fixed 32-step blocks give the same tokens")
    }

    /// Vella's optimized Qwen decoder (pipelined greedy decode, prefill without the discarded vocabulary projection)
    /// generates stock's tokens.
    @Test(.heavy) func vellaQwenOptimizedDecoderMatchesStock() throws {
        try vellaMetallib()
        let fixture = try VellaFixture("qwen3-asr-0.6b")
        defer { fixture.cleanup() }
        let adapter = try fixture.load()
        defer { adapter.unload() }
        let model = try #require(adapter.graph as? VellaSTT.Qwen3ASRModel)
        let audio = MLXArray(vellaSignal(seconds: 4))
        for language in [nil, "Polish"] as [String?] {
            _ = model.configureFastPath(enabled: false, component: "decoder")
            _ = model.generate(audio: audio, maxTokens: 24, context: "Arsenal", language: language, chunkDuration: 30)
            let stock = model.lastTokens
            _ = model.configureFastPath(enabled: true, component: "decoder")
            _ = model.generate(audio: audio, maxTokens: 24, context: "Arsenal", language: language, chunkDuration: 30)
            #expect(model.fastPathComponents["decoder"] == true && model.fastPathComponents["encoder"] == false)
            #expect(model.fastPathFinite)
            #expect(stock.count > 0)
            #expect(model.lastTokens == stock, "language \(language ?? "auto")")
        }
        try NativeParameters.requireHalf(adapter.parameters)
    }

    /// The optimized decoder's finiteness probe reads finite FP16 logits as finite even when their sum passes FP16's
    /// 65,504 (real Qwen3-ASR logits sum to about ±1.5M, which failed the self-test on every FP16 Load), and still
    /// catches a single NaN or infinite logit.
    @Test(.heavy) func vellaQwenFinitenessProbeSurvivesHalfOverflow() throws {
        try vellaMetallib()
        let logits = MLXArray.ones([151_936], dtype: .float16) * 30
        let index = MLXArray(0..<151_936)
        #expect(VellaSTT.Qwen3ASRModel.decoderFiniteProbe(logits).item(Float.self).isFinite)
        #expect(VellaSTT.Qwen3ASRModel.decoderFiniteProbe(-logits).item(Float.self).isFinite)
        for bad in [Float.nan, .infinity, -.infinity] {
            let poisoned = MLX.where(index .== 7, MLXArray(bad).asType(.float16), logits)
            #expect(poisoned.dtype == .float16)
            #expect(!VellaSTT.Qwen3ASRModel.decoderFiniteProbe(poisoned).item(Float.self).isFinite, "\(bad)")
        }
    }

    /// A degenerate decoder (all-zero tied vocabulary: every step's argmax is token 0, never EOS) loops. Both decode
    /// loops stop it with the stock repeat backstop (last 24 tokens, at most 3 distinct) at the same token, well
    /// before the per-segment budget, so a looping segment cannot run to Vella's 1024-token cap on either path.
    @Test(.heavy) func vellaQwenRepeatBackstopStopsBothDecodeLoopsAlike() throws {
        try vellaMetallib()
        let fixture = try VellaFixture("qwen3-asr-0.6b")
        defer { fixture.cleanup() }
        let adapter = try fixture.load()
        defer { adapter.unload() }
        let model = try #require(adapter.graph as? VellaSTT.Qwen3ASRModel)
        let (key, table) = try #require(model.parameters().flattened().first { $0.0.hasSuffix("embed_tokens.weight") })
        try model.update(parameters: ModuleParameters.unflattened([key: MLXArray.zeros(like: table)]), verify: .none)
        let audio = MLXArray(vellaSignal(seconds: 4))
        var runs: [[Int]] = []
        for optimized in [false, true] {
            _ = model.configureFastPath(enabled: optimized, component: "decoder")
            _ = model.generate(audio: audio, maxTokens: 200, context: "Arsenal", language: nil, chunkDuration: 30)
            #expect(model.fastPathComponents["decoder"] == optimized)
            runs.append(model.lastTokens)
        }
        #expect(runs[0] == Array(repeating: 0, count: 24), "stock stops after 24 repeats")
        #expect(runs[1] == runs[0], "the optimized loop stops at the same token")
    }

    /// The Qwen tokenizer that `loadTokenizer` synthesizes from vocab.json/merges.txt lives in a private scratch
    /// directory: the installed weights stay unchanged.
    @Test func vellaQwenTokenizerLeavesTheWeightsUntouched() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("vella-tokenizer-test-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let vocabulary = ["a": 0, "b": 1, "ab": 2, "Ġ": 3, "<|im_start|>": 4, "<|im_end|>": 5]
        try JSONSerialization.data(withJSONObject: vocabulary).write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\na b\n".write(to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)
        let added: [String: Any] = ["4": ["content": "<|im_start|>", "special": true], "5": ["content": "<|im_end|>", "special": true]]
        try JSONSerialization.data(withJSONObject: ["tokenizer_class": "Qwen2Tokenizer", "added_tokens_decoder": added])
            .write(to: directory.appendingPathComponent("tokenizer_config.json"))
        let before = try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
        let tokenizer = try blockingTokenizer { try await VellaSTT.Qwen3ASRModel.loadTokenizer(directory) }
        #expect(tokenizer.encode(text: "<|im_start|>ab") == [4, 2])
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted() == before)
    }

    @Test(.heavy) func vellaFallbackFollowsVellasRuntimeRule() throws {
        final class Fake: FastPathCapable {
            var enabled = false, finite = true, configurations = 0
            func configureFastPath(enabled: Bool, component: String) -> Bool { configurations += 1; self.enabled = enabled; return true }
            var fastPathFinite: Bool { finite }
            func qualificationTokens(audio: MLXArray) -> [Int] { [] }
        }
        // Non-finite optimized output: the request reruns stock and the model stays stock.
        let model = Fake(), fallback = VellaFallback(component: "both")
        fallback.enable(model, allowed: true)
        #expect(fallback.active && fallback.stockReason == nil && model.enabled)
        model.finite = false
        var runs: [Bool] = []
        let first = try fallback.run(
            model, text: { $0 },
            body: {
                runs.append(model.enabled); return "stock text"
            })
        #expect(first == "stock text")
        #expect(runs == [true, false] && !fallback.active && !model.enabled && fallback.stockReason != nil)
        runs = []
        _ = try fallback.run(
            model, text: { $0 },
            body: {
                runs.append(model.enabled); return "x"
            })
        #expect(runs == [false], "stock stays selected until reload")
        // Empty optimized text that stock confirms keeps the optimized path; text stock finds switches to stock.
        let silent = Fake()
        fallback.enable(silent, allowed: true)
        #expect(try fallback.run(silent, text: { $0 }, body: { "" }) == "")
        #expect(fallback.active && silent.enabled)
        let speech = Fake()
        fallback.enable(speech, allowed: true)
        #expect(try fallback.run(speech, text: { $0 }, body: { speech.enabled ? " " : "words" }) == "words")
        #expect(!fallback.active && !speech.enabled)
        // A throwing optimized run reruns stock; cancellation is not a fallback.
        let failing = Fake()
        fallback.enable(failing, allowed: true)
        let recovered = try fallback.run(
            failing, text: { $0 },
            body: {
                if failing.enabled { throw audioFailure("kernel") }
                return "ok"
            })
        #expect(recovered == "ok")
        let cancelled = Fake()
        fallback.enable(cancelled, allowed: true)
        #expect(throws: WorkerFailure.self) {
            try fallback.run(cancelled, text: { $0 }, body: { () -> String in throw WorkerFailure(code: "cancelled", message: "Request cancelled.") })
        }
        #expect(fallback.active)
        // Macs outside the optimized GPU family run stock from the start.
        fallback.enable(Fake(), allowed: false)
        #expect(!fallback.active && fallback.stockReason != nil)
        #expect(!VellaFastPath.available(environment: ["ARSENAL_ASR_STOCK": "1"]))
    }

    /// Vella's self-test, ported (both reviews, finding 4): the first Load per model files, GPU and macOS build compares
    /// stock and optimized token IDs on the bundled clip, stays stock on any mismatch, and saves the verdict; a saved
    /// verdict is reused without running the models again, and a new GPU or macOS build tests again.
    @Test(.heavy) func vellaSelfTestKeepsStockOnMismatchAndSavesTheVerdict() throws {
        final class Fake: FastPathCapable {
            var enabled = false, finite = true, runs = 0
            var fastTokens: [Int] = [5, 6, 7], stockTokens: [Int] = [5, 6, 7]
            func configureFastPath(enabled: Bool, component: String) -> Bool { self.enabled = enabled; return true }
            var fastPathFinite: Bool { finite }
            func qualificationTokens(audio: MLXArray) -> [Int] { runs += 1; return enabled ? fastTokens : stockTokens }
        }
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent("vella-gate-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: storage) }
        let host = VellaFastPathGate.Host(gpuFamily: "apple9", gpuName: "Test GPU", osBuild: "26A1")
        let clip = VellaFastPathGate.bundledClip()
        let gate = VellaFastPathGate(storage: storage, host: host, clip: { clip })
        let weights = storage.appendingPathComponent("weights")
        try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
        try Data([1, 2, 3]).write(to: weights.appendingPathComponent("model.safetensors"))
        func key(_ gate: VellaFastPathGate, model: String = "parakeet-v3-ultra") -> String {
            gate.key(model: model, precision: "fp16", weights: weights, component: "both", configuration: "disabled=nax_gemm")
        }

        // Identical tokens: optimized, saved as fast; the next Load reuses it without transcribing.
        let good = Fake(), fallback = VellaFallback(component: "both")
        fallback.enable(good, allowed: true, qualification: (gate, key(gate), "parakeet-v3-ultra"))
        #expect(fallback.active && good.enabled && good.runs == 2)
        let record = try JSONSerialization.jsonObject(with: Data(contentsOf: storage.appendingPathComponent(key(gate) + ".json"))) as? [String: String]
        #expect(record?["status"] == "fast"); #expect(record?["os_build"] == "26A1"); #expect(record?["revision"] == VellaFastPathGate.revision)
        let again = Fake()
        fallback.enable(again, allowed: true, qualification: (VellaFastPathGate(storage: storage, host: host, clip: { clip }), key(gate), "parakeet-v3-ultra"))
        #expect(fallback.active && again.runs == 0)

        // A finite but different transcript (a kernel regression on another chip or macOS): stock, saved, sticky.
        let other = VellaFastPathGate.Host(gpuFamily: "apple9", gpuName: "Test GPU", osBuild: "27A1")
        let otherGate = VellaFastPathGate(storage: storage, host: other, clip: { clip })
        #expect(key(otherGate) != key(gate), "a new macOS build tests again")
        let wrong = Fake(); wrong.fastTokens = [5, 6, 8]
        fallback.enable(wrong, allowed: true, qualification: (otherGate, key(otherGate), "parakeet-v3-ultra"))
        #expect(!fallback.active && !wrong.enabled && wrong.runs == 2)
        #expect(fallback.stockReason == "The optimized path failed its self-test against stock MLX on this Mac.")
        let sticky = Fake()
        fallback.enable(sticky, allowed: true, qualification: (VellaFastPathGate(storage: storage, host: other, clip: { clip }), key(otherGate), "parakeet-v3-ultra"))
        #expect(!fallback.active && !sticky.enabled && sticky.runs == 0)

        // Non-finite optimized values and an empty stock transcript are failures too; a missing clip is stock for this
        // Load only (not saved, so the next Load tests again).
        let nonFinite = Fake(); nonFinite.finite = false
        #expect(VellaFastPathGate.check(nonFinite, component: "both", audio: MLXArray(clip ?? [])) != .fast)
        let silent = Fake(); silent.stockTokens = []; silent.fastTokens = []
        #expect(VellaFastPathGate.check(silent, component: "both", audio: MLXArray(clip ?? [])) != .fast)
        let missing = VellaFastPathGate(storage: storage, host: host, clip: { nil })
        fallback.enable(Fake(), allowed: true, qualification: (missing, key(missing, model: "qwen3-asr-0.6b"), "qwen3-asr-0.6b"))
        #expect(!fallback.active)
        #expect(!FileManager.default.fileExists(atPath: storage.appendingPathComponent(key(missing, model: "qwen3-asr-0.6b") + ".json").path))
        // Another GPU family keeps stock without testing.
        let fake = Fake()
        fallback.enable(fake, allowed: false, qualification: (gate, key(gate), "parakeet-v3-ultra"))
        #expect(!fallback.active && fake.runs == 0)
    }

    /// The bundled clip ships with the worker and decodes as 2.69 s of 16 kHz speech, found the way the shipped worker finds it.
    @Test func vellaSelfTestClipIsBundled() throws {
        let samples = try #require(VellaFastPathGate.bundledClip())
        #expect(samples.count == 43_040)
        #expect(samples.contains { abs($0) > 0.05 })
        #expect(VellaFastPathGate.pcm16(wav: Data("RIFF....WAVE".utf8)) == nil)
    }

    @Test func parakeetWordsJoinSubwordPieces() {
        let words = parakeetWords([(" Hel", 0.0, 0.2), ("lo", 0.2, 0.3), (" ", 0.3, 0.32), (" wor", 0.4, 0.5), ("ld", 0.5, 0.7), (".", 0.7, 0.72)])
        #expect(words.map(\.word) == ["Hello", "world."])
        #expect(words.map(\.start) == [0.0, 0.4] && words.map(\.end) == [0.3, 0.72])
        #expect(parakeetWords([("lo", 0.1, 0.2)]).map(\.word) == ["lo"], "a leading piece without a space still starts a word")
    }

    @Test func vellaFoldersDeclareWhatTheAdaptersRun() throws {
        #expect(manifestID("schema = 3\nid = \"parakeet-v3-ultra\"\n[licence]\nid = \"x\"\n") == "parakeet-v3-ultra")
        #expect(manifestID("[weights]\nid = \"x\"\n") == nil)
        for id in ["qwen3-asr-1.7b", "qwen3-asr-0.6b"] {
            let folder = vellaRepository.appendingPathComponent("models/" + id)
            let schema = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("ops/transcribe.input.json")))
            let codes = schema["properties"]!["language"]!["enum"]!.array!.compactMap(\.string)
            #expect(codes.count == 30)
            let supported = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("gate/no-weights.json")))["config"]!["support_languages"]!.array!
                .compactMap(\.string)
            // Every request code names a language the checkpoint's config lists, and round-trips to the same code.
            for code in codes {
                let name = try #require(qwenLanguageNames[code], "\(code)")
                #expect(supported.contains(name), "\(code) → \(name)")
                #expect(audioLanguageCode(name) == code)
            }
        }
        #expect(vellaQwenEligible("model.layers.0.mlp.up_proj.weight", [128, 64]))
        #expect(!vellaQwenEligible("audio_tower.layers.0.fc1.weight", [128, 64]))
        #expect(!vellaQwenEligible("thinker.audio_tower.proj1.weight", [128, 64]))
    }
}
