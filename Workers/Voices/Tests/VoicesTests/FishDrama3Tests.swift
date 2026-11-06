// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
import Tokenizers
@testable import FishDrama3

/// Drama 3 port: text rules and tokens against fixtures captured once from the reference (mlx-audio fish_drama3 dad2087f,
/// lab/drama3-port/scripts), the checkpoint key map from safetensors headers (no weight bytes), and tiny random graphs on
/// the CPU. Real-weight parity is the opt-in heavy check in lab/drama3-port.md.
@Suite(.serialized) struct FishDrama3Tests {
    init() { if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) } }

    static let snapshot = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
        ".cache/huggingface/hub/models--mlx-community--fish-audio-drama-3-4bit/snapshots/cc639e27aba7edb73429d3e3806f471dcd79d316")
    /// Only a real cache snapshot counts: another app once linked this path to its own conversion, other weights.
    static var hasSnapshot: Bool {
        let config = snapshot.appendingPathComponent("config.json").resolvingSymlinksInPath().path
        let hub = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub").resolvingSymlinksInPath().path
        return FileManager.default.fileExists(atPath: config) && config.hasPrefix(hub + "/")
    }

    func fixture(_ name: String) throws -> [String: Any] {
        let url = try #require(Bundle.module.url(forResource: name, withExtension: "json"))
        return try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }

    // MARK: - Text and prompt

    @Test func textRulesMatchTheReference() throws {
        let f = try fixture("drama3-text")
        let splits = try #require(f["split"] as? [[String: Any]])
        #expect(splits.count == 11)
        for case let (text?, chunks?) in splits.map({ ($0["text"] as? String, $0["chunks"] as? [String]) }) {
            #expect(Drama3Text.split(text) == chunks, "split differs for \(text.prefix(40))")
        }
        for entry in try #require(f["speaker_tag"] as? [[String: String]]) { #expect(Drama3Text.withSpeakerTag(entry["text"]!) == entry["tagged"]) }
        for entry in try #require(f["reference_transcript"] as? [[String: String]]) {
            #expect(Drama3Text.referenceTranscript(entry["text"]!) == entry["transcript"])
        }
        let invalid = try #require(f["invalid"] as? [String]), valid = try #require(f["valid"] as? [String])
        #expect(invalid.count == 8 && valid.count == 3)
        for text in invalid { #expect(throws: Drama3Error.self) { try Drama3Text.validate(text, location: "t") } }
        for text in valid { try Drama3Text.validate(text, location: "t") }
    }

    struct StubTokenizer: Drama3Tokenizing {
        /// One id per Unicode scalar, kept below the semantic band.
        func encode(_ text: String) -> [Int] { text.unicodeScalars.map { Int($0.value % 50) + 3 } }
        func id(of token: String) -> Int? { nil }
    }

    @Test func promptRowsFollowReferenceBracketV1() throws {
        let codebooks = 10, begin = 248_099
        let reference = Drama3Codes(rows: (0..<codebooks).map { book in (0..<3).map { Int32(book * 10 + $0) } })
        let history = Drama3Codes(rows: (0..<codebooks).map { _ in [7, 8] })
        let tokenizer = StubTokenizer()
        let prompt = try Drama3PromptBuilder.build(
            references: [("Hello world.", reference)], history: [("Earlier.", history)], target: "Now.", tokenizer: tokenizer, codebooks: codebooks,
            vocab: 4096, semanticBegin: begin)
        #expect(prompt.width == 11)
        let systemText = [
            "<|im_start|>system\n", "convert the provided text to speech reference to the following:",
            "\n\n[Reference 1]\nText: <|speaker:0|> Hello world.\nSpeech: ", "<|im_end|>\n"
        ]
        let systemRows = systemText.map { tokenizer.encode($0).count }.reduce(0, +) + 3
        #expect(prompt.systemRows == systemRows)
        let turns = [
            "<|im_start|>user\n", "<|speaker:0|> Earlier.", "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>", "<|im_end|>\n",
            "<|im_start|>user\n", "<|speaker:0|> Now.", "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"
        ]
        #expect(prompt.count == systemRows + turns.map { tokenizer.encode($0).count }.reduce(0, +) + 2)
        // The reference's audio rows sit right after its "Speech: " text and carry [c0 + begin, c0…c9].
        let audioStart = systemText.prefix(3).map { tokenizer.encode($0).count }.reduce(0, +)
        for frame in 0..<3 {
            #expect(Array(prompt.row(audioStart + frame)) == [Int32(begin + frame)] + (0..<codebooks).map { Int32($0 * 10 + frame) })
        }
        #expect(Array(prompt.row(0).dropFirst()) == Array(repeating: 0, count: codebooks))
        // Counts first: bad codes are refused, never silently clipped.
        for bad in [Drama3Codes(rows: [[1]]), Drama3Codes(rows: (0..<codebooks).map { _ in [] }), Drama3Codes(rows: (0..<codebooks).map { _ in [4096] })] {
            #expect(throws: Drama3Error.self) {
                try Drama3PromptBuilder.build(
                    references: [("x", bad)], history: [], target: "y", tokenizer: tokenizer, codebooks: codebooks, vocab: 4096, semanticBegin: begin)
            }
        }
    }

    @Test(.enabled(if: FishDrama3Tests.hasSnapshot)) func tokenizerMatchesTheReference() async throws {
        let f = try fixture("drama3-tokens")
        let tokenizer = Drama3TokenizerAdapter(tokenizer: try await AutoTokenizer.from(modelFolder: Self.snapshot))
        let segments = try #require(f["segments"] as? [[String: Any]])
        #expect(segments.count == 11)
        for segment in segments {
            let text = try #require(segment["text"] as? String)
            #expect(tokenizer.encode(text) == segment["ids"] as? [Int], "ids differ for \(text.debugDescription)")
        }
        for (token, id) in try #require(f["specials"] as? [String: Int]) { #expect(tokenizer.id(of: token) == id) }
        let config = try Drama3Config(data: Data(contentsOf: Self.snapshot.appendingPathComponent("config.json")))
        try Drama3Speech.checkTokenizer(tokenizer, config: config)
    }

    // MARK: - Checkpoint map (headers only, nothing evaluated)

    static func headers(_ directory: URL) throws -> [String: (dtype: String, shape: [Int])] {
        var result: [String: (String, [Int])] = [:]
        for file in try FileManager.default.contentsOfDirectory(atPath: directory.path) where file.hasPrefix("model") && file.hasSuffix(".safetensors") {
            let handle = try FileHandle(forReadingFrom: directory.appendingPathComponent(file)); defer { try? handle.close() }
            let size = try #require(try handle.read(upToCount: 8)).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
            let bytes = try #require(try handle.read(upToCount: Int(size)))
            let header = try #require(try JSONSerialization.jsonObject(with: bytes) as? [String: Any])
            for (key, value) in header {
                guard let entry = value as? [String: Any], let dtype = entry["dtype"] as? String, let shape = entry["shape"] as? [Int] else { continue }
                result[key] = (dtype, shape)
            }
        }
        return result
    }

    @Test(.enabled(if: FishDrama3Tests.hasSnapshot)) func moduleTreeMatchesTheCheckpointKeysAndShapes() throws {
        let config = try Drama3Config(data: Data(contentsOf: Self.snapshot.appendingPathComponent("config.json")))
        #expect(config.text.layers == 40 && config.text.experts == 256 && config.text.expertsPerToken == 8 && config.tower.codebooks == 10)
        #expect(config.semanticBegin == 248_099 && config.semanticEnd == 252_194 && config.eosID == 248_046)
        let quantization = try #require(config.quantization)
        #expect(quantization.spec("backbone.model.layers.0.mlp.switch_mlp.gate_proj").bits == 4)
        #expect(quantization.spec("backbone.model.layers.3.self_attn.q_proj").bits == 8)
        let stored = try Self.headers(Self.snapshot)
        #expect(stored.count == 1_841)
        try Device.withDefaultDevice(.cpu) {
            // Lazily built and quantized: shapes are known without allocating or evaluating anything.
            let network = Drama3Network(config)
            quantize(model: network) { path, _ in
                stored["\(path).scales"] == nil ? nil : (quantization.spec(path).groupSize, quantization.spec(path).bits, .affine)
            }
            let parameters = Dictionary(uniqueKeysWithValues: network.parameters().flattened().map { ($0.0, $0.1.shape) })
            #expect(Set(parameters.keys) == Set(stored.keys))
            for (key, shape) in parameters where stored[key]?.shape != shape { Issue.record("\(key): module \(shape), checkpoint \(stored[key]?.shape ?? [])") }
            #expect(stored.filter { $0.value.dtype == "F32" }.keys.allSatisfy { $0.hasSuffix("norm.weight") })
        }
    }

    // MARK: - Tiny random graphs (CPU)

    static let tiny = Drama3Config(
        text: Drama3TextConfig(
            hiddenSize: 32, layers: 4, attentionHeads: 2, kvHeads: 1, headDim: 16, linearKeyHeads: 2, linearValueHeads: 4, linearKeyHeadDim: 8,
            linearValueHeadDim: 8, convKernel: 4, experts: 4, expertsPerToken: 2, moeIntermediate: 16, sharedExpertIntermediate: 16, vocabSize: 80),
        tower: Drama3TowerConfig(dim: 32, heads: 2, kvHeads: 1, headDim: 16, intermediate: 32, vocabSize: 16, codebooks: 4, sharedLayers: 2, backboneCodebooks: 1),
        semanticBegin: 60, semanticEnd: 75, eosID: 2)

    @Test func samplerKeepsTheReferenceRules() throws {
        try Device.withDefaultDevice(.cpu) {
            let logits = MLXArray([Float(0.1), 3, 2.9, -1, 0.5, 2.8]).reshaped(1, 6)
            MLXRandom.seed(7)
            for _ in 0..<20 {
                #expect(drama3SampleTopKTopP(logits, temperature: 1, topK: 1, topP: 1).item(Int.self) == 1)
                #expect(drama3SampleTopKTopP(logits, temperature: 1, topK: 6, topP: 1e-6).item(Int.self) == 1)
                #expect(drama3SampleTopKTopP(logits, temperature: 0, topK: 6, topP: 1).item(Int.self) == 1)
            }
            var seen = Set<Int>()
            for _ in 0..<60 { seen.insert(drama3SampleTopKTopP(logits, temperature: 1, topK: 3, topP: 1).item(Int.self)) }
            #expect(seen.isSubset(of: [1, 2, 5]) && seen.count >= 2)
        }
    }

    @Test func tinyBackboneCachesMatchOneFullPass() throws {
        try Device.withDefaultDevice(.cpu) {
            MLXRandom.seed(3)
            let network = Drama3Network(Self.tiny)
            eval(network.parameters())
            let model = network.backbone.model
            let embeddings = MLXRandom.normal([1, 9, 32])
            let full = model(embeddings: embeddings, cache: model.makeCache())
            let cache = model.makeCache()
            let first = model(embeddings: embeddings[0..., ..<5], cache: cache)
            let second = model(embeddings: embeddings[0..., 5..<8], cache: cache)
            let last = model(embeddings: embeddings[0..., 8...], cache: cache)
            let pieces = concatenated([first, second, last], axis: 1)
            #expect(abs(full - pieces).max().item(Float.self) < 1e-4)
            // Copied caches are independent: advancing a copy leaves the original able to reproduce the same step.
            let copy = cache.map { $0.copy() }
            let next = MLXRandom.normal([1, 1, 32])
            let a = model(embeddings: next, cache: copy), b = model(embeddings: next, cache: cache)
            #expect(abs(a - b).max().item(Float.self) < 1e-6)
        }
    }

    @Test func maskedGatedDeltaStepKeepsTheState() throws {
        try Device.withDefaultDevice(.cpu) {
            MLXRandom.seed(4)
            let q = MLXRandom.normal([1, 2, 2, 8]), k = MLXRandom.normal([1, 2, 2, 8]), v = MLXRandom.normal([1, 2, 4, 8])
            let a = MLXRandom.normal([1, 2, 4]), b = MLXRandom.normal([1, 2, 4]), aLog = MLXRandom.normal([4]), dtBias = MLXRandom.normal([4])
            let start = MLXRandom.normal([1, 4, 8, 8])
            let before = start.asArray(Float.self)
            let (_, masked) = drama3GatedDeltaOps(q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: start, mask: MLXArray([false, false]).reshaped(1, 2))
            #expect(masked.asArray(Float.self) == before, "a fully masked step leaves the state as it was")
            let (_, moved) = drama3GatedDeltaOps(q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: start, mask: nil)
            #expect(moved.asArray(Float.self) != before)
            #expect(start.asArray(Float.self) == before, "the caller's state array is never updated in place")
        }
    }

    final class StubCodec: Drama3Codec {
        let sampleRate = 100, frameLength = 4
        var resets = 0, encoded = 0
        func encodeCodes(_ samples: [Float]) throws -> MLXArray {
            encoded += 1
            return MLXArray((0..<(4 * 3)).map { Int32($0 % 16) }, [4, 3])
        }
        func resetStreaming() { resets += 1 }
        func decodeStep(_ codes: MLXArray) -> MLXArray {
            repeated(codes.asType(.float32).sum(axis: 1)[0] / 100, count: frameLength, axis: 0)
        }
    }

    @Test func tinyGenerationIsSeededAndReusesTheVoicePrefix() async throws {
        try await Device.withDefaultDevice(.cpu) {
            MLXRandom.seed(11)
            let model = Drama3Model(config: Self.tiny)
            eval(model.network.parameters())
            let codec = StubCodec()
            model.tokenizer = StubTokenizer(); model.codec = codec
            func run(_ text: String) async throws -> [[Float]] {
                var pieces: [[Float]] = []
                try await model.generate(
                    text: text, references: [([0.1, 0.2], "Hello there.")], sampling: Drama3Sampling(temperature: 1, topK: 8, topP: 0.95, maxFrames: 12),
                    seed: 5, emit: { pieces.append($0) })
                return pieces
            }
            let a = try await run("One short sentence.")
            let b = try await run("One short sentence.")
            #expect(codec.encoded == 1, "the second request reuses the cached voice")
            #expect(codec.resets == 4, "reset before and after each request")
            #expect(a == b)
            let frames = a.map(\.count).reduce(0, +) / codec.frameLength
            #expect(frames > 0 && frames <= 12)
            if a.count > 1 { #expect(a[0].count == Drama3Defaults.firstChunkFrames * codec.frameLength) }
            let finite = a.allSatisfy { $0.allSatisfy(\.isFinite) }
            #expect(finite)
        }
    }
}
