// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import Testing
import Tokenizers
@testable import VoicesWorker
@testable import VoxCPM2
import ArsenalWire
import VireoCore

/// Weight-free VoxCPM2 checks (CPU MLX, tiny shapes): fixed port bugs, the maker's prompt layouts and request rules.
@Suite struct VoxCPM2Tests {
    init() { if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) } }

    /// A network small enough for the CPU with every module the real one has.
    static func tinyConfig() -> VoxCPM2Configuration {
        let lm = VoxCPM2LMConfig(
            hiddenSize: 32, numHiddenLayers: 2, numAttentionHeads: 2, numKeyValueHeads: 1, intermediateSize: 64, vocabSize: 128,
            ropeShortFactor: Array(repeating: 1, count: 8), maxPositionEmbeddings: 4096, useMup: false, kvChannels: 16)
        return VoxCPM2Configuration(
            lmConfig: lm, encoderConfig: VoxCPM2EncoderConfig(hiddenDim: 16, ffnDim: 32, numHeads: 2, numLayers: 1, kvChannels: 16),
            ditConfig: VoxCPM2DiTConfig(hiddenDim: 16, ffnDim: 32, numHeads: 2, numLayers: 1, kvChannels: 16),
            audioVaeConfig: VoxCPM2AudioVAEConfig(encoderDim: 4, encoderRates: [2, 5], latentDim: 8, decoderDim: 16, decoderRates: [2, 3], sampleRate: 16000, outSampleRate: 48000),
            patchSize: 2, featDim: 8, scalarQuantizationLatentDim: 8, residualLmNumLayers: 1, residualLmNoRope: true)
    }

    /// Port bug (fixed): the maker's causal conv pads 2·padding − output_padding. The stride-5 encoder conv (kernel 10,
    /// padding 3, output_padding 1) pads 5, so output i reads input 5i + 4 at its last tap; the port padded 6 (5i + 3).
    @Test func strideFiveCausalConvPadsLikeTheMaker() throws {
        try Device.withDefaultDevice(.cpu) {
            let conv = VoxCausalConv1d(inChannels: 1, outChannels: 1, kernelSize: 10, stride: 5, padding: 3, outputPadding: 1, bias: false)
            var w = [Float](repeating: 0, count: 10); w[9] = 1
            try conv.conv.update(parameters: ModuleParameters.unflattened(["weight": MLXArray(w).reshaped(1, 10, 1)]), verify: [.shapeMismatch])
            let x = MLXArray((0..<40).map(Float.init)).reshaped(1, 40, 1)
            let y = conv(x).reshaped(-1).asArray(Float.self)
            #expect(y == (0..<8).map { Float(5 * $0 + 4) })
            // The encoder block builds exactly this conv for stride 5.
            #expect(VoxCausalEncoderBlock(outputDim: 8, inputDim: 4, stride: 5).conv.padVal == 5)
            #expect(VoxCausalEncoderBlock(outputDim: 8, inputDim: 4, stride: 8).conv.padVal == 8)
        }
    }

    /// Port bug (fixed): torch.bucketize puts a value equal to a boundary in the lower bucket.
    @Test func sampleRateBucketsFollowTorchBucketize() throws {
        try Device.withDefaultDevice(.cpu) {
            let decoder = VoxCPM2Model(Self.tinyConfig()).audioVAE.decoder
            let buckets = [16000, 20000, 20001, 40000, 48000].map { decoder.getSRIdx(MLXArray([Int32($0)])).item(Int.self) }
            #expect(buckets == [0, 0, 1, 2, 3])
        }
    }

    /// The maker's four prompt layouts (`_generate_with_prompt_cache`): counts first, then tokens and masks.
    @Test func promptLayoutsMatchTheMaker() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = VoxCPM2Model(Self.tinyConfig())
            let ref = MLXArray.ones([3, 2, 8]), prompt = MLXArray.ones([4, 2, 8]) * 2
            let zero = model.prompt(text: [5, 6])
            #expect(zero.tokens == [5, 6, 101] && zero.textMask == [1, 1, 1] && zero.audioMask == [0, 0, 0] && zero.contextPatches == 0)
            #expect(zero.feats.shape == [3, 2, 8])
            let reference = model.prompt(text: [5, 6], refFeat: ref)
            #expect(reference.tokens == [103, 0, 0, 0, 104, 5, 6, 101])
            #expect(reference.textMask == [1, 0, 0, 0, 1, 1, 1, 1] && reference.audioMask == [0, 1, 1, 1, 0, 0, 0, 0])
            #expect(reference.contextPatches == 0)
            let continuation = model.prompt(text: [7, 5, 6], promptFeat: prompt)
            #expect(continuation.tokens == [7, 5, 6, 101, 0, 0, 0, 0])
            #expect(continuation.textMask == [1, 1, 1, 1, 0, 0, 0, 0] && continuation.audioMask == [0, 0, 0, 0, 1, 1, 1, 1])
            #expect(continuation.contextPatches == 3)
            let both = model.prompt(text: [7, 5, 6], promptFeat: prompt, refFeat: ref)
            #expect(both.tokens == [103, 0, 0, 0, 104, 7, 5, 6, 101, 0, 0, 0, 0])
            #expect(both.feats.shape == [13, 2, 8])
            #expect(both.feats[9...].sum().item(Float.self) == 4 * 2 * 8 * 2)
            #expect(both.feats[5..<9].sum().item(Float.self) == 0)
        }
    }

    /// Reference clips are padded to whole patches: right for the isolated reference, left for a continuation prompt.
    @Test func referencePaddingSideFollowsTheMode() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = VoxCPM2Model(Self.tinyConfig())
            let patchSamples = model.patchSize * model.audioVAE.chunkSize // 2 × 10
            #expect(patchSamples == 20)
            let right = model.encodeAudio([Float](repeating: 0.1, count: 25), padLeft: false)
            let left = model.encodeAudio([Float](repeating: 0.1, count: 25), padLeft: true)
            #expect(right.shape == [2, 2, 8] && left.shape == [2, 2, 8])
        }
    }

    /// One tiny end-to-end pass: the loop stops at the cap or the stop head, and decoding drops the context's samples.
    @Test func tinyGenerationDecodesWithoutTheContext() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = VoxCPM2Model(Self.tinyConfig())
            var generation = VoxCPM2Generation(); generation.inferenceTimesteps = 2
            let p = model.prompt(text: [7, 5, 6], promptFeat: MLXArray.zeros([4, 2, 8]))
            let (patches, _) = try model.generatePatches(p, maxLength: 5, generation: generation, isCancelled: { false })
            #expect((1...5).contains(patches.count))
            let perPatch = model.patchSize * model.audioVAE.decodeChunkSize
            let context = p.feats[(p.feats.dim(0) - p.contextPatches)...]
            #expect(model.decode(context: context, patches: patches).count == patches.count * perPatch)
            #expect(model.decode(context: nil, patches: patches).count == patches.count * perPatch)
        }
    }

    @Test func weightsMustCoverTheNetworkExactly() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = VoxCPM2Model(Self.tinyConfig())
            // The release's names (mlx-community layout): AudioVAE blocks under `layers`, snake-case conditioning, no inner conv.
            var weights: [String: MLXArray] = [:]
            for (key, value) in model.parameters().flattened() where key != "audio_vae.decoder.srBoundaries" {
                let name =
                    key.hasPrefix("audio_vae.")
                    ? key.replacingOccurrences(of: #"\.conv\.(weight|bias)$"#, with: ".$1", options: .regularExpression)
                        .replacingOccurrences(of: #"\.blocks\.(\d+)"#, with: ".blocks.layers.$1", options: .regularExpression)
                        .replacingOccurrences(of: "srCondLayers", with: "sr_cond_layers") : key
                weights[name] = value
            }
            weights["audio_vae.decoder._sr_boundaries"] = MLXArray([20000, 30000, 40000] as [Int32])
            weights["base_lm.rope.inv_freq"] = MLXArray.ones([8]) // shipped by the release, recomputed here
            try model.apply(weights: weights)
            weights["stop_head.extra"] = MLXArray.ones([2])
            #expect(throws: VoxCPM2Error.self) { try model.apply(weights: weights) }
            weights["stop_head.extra"] = nil
            weights["stop_head.weight"] = nil
            #expect(throws: VoxCPM2Error.self) { try model.apply(weights: weights) }
        }
    }

    @Test func cloneModeRules() throws {
        #expect(try VoxCPM2Speech.cloneMode(hasClip: false, transcript: nil, requested: nil) == nil)
        #expect(try VoxCPM2Speech.cloneMode(hasClip: true, transcript: nil, requested: nil) == .reference)
        #expect(try VoxCPM2Speech.cloneMode(hasClip: true, transcript: "Hello.", requested: nil) == .both)
        #expect(try VoxCPM2Speech.cloneMode(hasClip: true, transcript: "Hello.", requested: .reference) == .reference)
        #expect(try VoxCPM2Speech.cloneMode(hasClip: true, transcript: "Hello.", requested: .continuation) == .continuation)
        #expect(throws: VoxCPM2Error.self) { try VoxCPM2Speech.cloneMode(hasClip: true, transcript: " ", requested: .continuation) }
        #expect(throws: VoxCPM2Error.self) { try VoxCPM2Speech.cloneMode(hasClip: false, transcript: "Hello.", requested: nil) }
        #expect(throws: VoxCPM2Error.self) { try VoxCPM2Speech.cloneMode(hasClip: false, transcript: nil, requested: .reference) }
    }

    @Test func textCleanupAndVoiceDescription() throws {
        #expect(try VoxCPM2Speech.text("Hello\n  there.", instructions: nil) == "Hello there.")
        #expect(try VoxCPM2Speech.text("Hello.", instructions: " A calm older man ") == "(A calm older man)Hello.")
        #expect(throws: VoxCPM2Error.self) { try VoxCPM2Speech.text(" \n ", instructions: nil) }
        #expect(throws: VoxCPM2Error.self) { try VoxCPM2Speech.text("Hi.", instructions: "loud (very)") }
    }

    @Test func longTextIsChunkedAtSentenceEnds() {
        #expect(VoxCPM2Speech.chunks("Short text. Two sentences.") == ["Short text. Two sentences."])
        let sentence = String(repeating: "word ", count: 19) + "end." // 99 characters
        let text = Array(repeating: sentence, count: 10).joined(separator: " ")
        let pieces = VoxCPM2Speech.chunks(text)
        #expect(pieces.count == 3)
        #expect(pieces.allSatisfy { $0.count <= VoxCPM2Speech.chunkCharacters && $0.hasSuffix("end.") })
        #expect(pieces.joined(separator: " ") == text)
    }

    /// The Get cast: floats to FP16, integers kept, any value outside FP16's range refused.
    @Test func fp16CastChecksTheRange() throws {
        try Device.withDefaultDevice(.cpu) {
            var report = VoxCPM2Cast.Report()
            let out = try VoxCPM2Cast.cast(
                ["a": MLXArray([1.5, -2] as [Float]), "b": MLXArray([3] as [Float]).asType(.bfloat16), "audio_vae.decoder._sr_boundaries": MLXArray([20000] as [Int32])],
                file: "x.safetensors", report: &report)
            #expect(out["a"]?.dtype == .float16 && out["b"]?.dtype == .float16 && out["audio_vae.decoder._sr_boundaries"]?.dtype == .int32)
            #expect(report.castTensors == 2 && report.keptTensors == 1 && report.largestMagnitude == 3)
            #expect(throws: VoxCPM2Error.self) { try VoxCPM2Cast.cast(["big": MLXArray([70000] as [Float])], file: "x", report: &report) }
        }
    }

    @Test func runnerOptionsAreValidated() throws {
        let options = try Runner.voxOptions(["cfg_value": 1.5, "inference_timesteps": 20, "clone_mode": "continuation", "retry_badcase": false])
        #expect(options.cfgValue == 1.5 && options.inferenceTimesteps == 20 && options.cloneMode == .continuation && !options.retryBadcase)
        #expect(try Runner.voxOptions([:]) == VoxCPM2Options())
        for bad: JSONObject in [["cfg_value": 0], ["cfg_value": 11], ["inference_timesteps": 0], ["inference_timesteps": 2.5], ["clone_mode": "isolated"], ["retry_badcase": 1]] {
            #expect(throws: WorkerFailure.self) { try Runner.voxOptions(bad) }
        }
    }

    /// Fixed bug (real check, 2 Nov): an impossible VoxCPM2 request came back as HTTP 500; invalid_request maps to 422.
    @Test func impossibleVoxRequestsAreInvalidInput() throws {
        let clip: JSONObject = ["path": "/tmp/a.wav"]
        try Runner.checkVoxRequest(["text": "Cześć.", "reference": .object(clip)], options: VoxCPM2Options())
        var continuation = VoxCPM2Options(); continuation.cloneMode = .continuation
        for (request, options): (JSONObject, VoxCPM2Options) in [
            (["text": "Cześć.", "reference": .object(clip)], continuation),
            (["text": "Cześć.", "reference": ["text": "Hej."]], VoxCPM2Options()),
            (["text": "Cześć.", "instructions": "loud (very)"], VoxCPM2Options())
        ] {
            #expect { try Runner.checkVoxRequest(request, options: options) } throws: { ($0 as? WorkerFailure)?.code == "invalid_request" }
        }
    }

    /// Review fix: text opening with "(" and no ")" crashed the worker (force unwrap); a complete "(…)" is the prefix.
    @Test func descriptionPrefixNeverCrashes() throws {
        #expect(try VoxCPM2Speech.parts("(unclosed and spoken", instructions: nil) == ("", "(unclosed and spoken"))
        #expect(try VoxCPM2Speech.parts("(calm voice) Hello.", instructions: nil) == ("(calm voice)", "Hello."))
        #expect(try VoxCPM2Speech.parts("(x) Hello.", instructions: "warm") == ("(warm)", "(x) Hello."))
        #expect(throws: VoxCPM2Error.self) { try VoxCPM2Speech.parts("(only a description)", instructions: nil) }
        #expect(throws: VoxCPM2Error.self) { try VoxCPM2Speech.parts(String(repeating: "a", count: VoxCPM2Speech.maxTextCharacters + 1), instructions: nil) }
    }

    /// Review fix: Hindi, Burmese, Arabic, Urdu and Khmer sentence marks split; a sentence without them is cut at spaces,
    /// and a run without spaces at the limit; nothing is lost.
    @Test func chunkerHandlesEveryScript() {
        for mark in ["।", "။", "؟", "۔", "។"] {
            let text = Array(repeating: String(repeating: "x", count: 150) + mark, count: 6).joined()
            let pieces = VoxCPM2Speech.chunks(text)
            #expect(pieces.count == 3 && pieces.allSatisfy { $0.hasSuffix(mark) && $0.count <= VoxCPM2Speech.chunkCharacters }, "\(mark)")
        }
        let spaced = Array(repeating: "słowo", count: 300).joined(separator: " ")
        let spacedPieces = VoxCPM2Speech.chunks(spaced)
        #expect(spacedPieces.allSatisfy { $0.count <= VoxCPM2Speech.chunkCharacters && !$0.hasPrefix(" ") })
        #expect(spacedPieces.joined(separator: " ") == spaced)
        let han = String(repeating: "你", count: 1000)
        let hanPieces = VoxCPM2Speech.chunks(han)
        #expect(hanPieces.count == 3 && hanPieces.allSatisfy { $0.count <= VoxCPM2Speech.chunkCharacters } && hanPieces.joined() == han)
    }

    /// Review fix: chunk k's retries (seed + 0…2) never reuse another chunk's seed; one chunk keeps the request seed.
    @Test func chunkSeedsAreIndependent() {
        #expect(VoxCPM2Speech.chunkSeed(7, index: 0) == 7)
        let seeds = (0..<50).flatMap { chunk in (0..<3).map { VoxCPM2Speech.chunkSeed(7, index: chunk) &+ UInt64($0) } }
        #expect(Set(seeds).count == seeds.count)
    }

    /// Review fix: requests VoxCPM2 cannot serve are refused as invalid_request (HTTP 422), not engine errors.
    @Test func unservableRequestsAreRefusedAsInvalidRequest() throws {
        let invalid: (Error) -> Bool = { ($0 as? WorkerFailure)?.code == "invalid_request" }
        for request: JSONObject in [
            ["text": "Hi.", "voice": "af_heart"], ["text": " \n\t "], ["text": .string(String(repeating: "a", count: VoxCPM2Speech.maxTextCharacters + 1))]
        ] {
            #expect { try Runner.checkVoxRequest(request, options: VoxCPM2Options()) } throws: { invalid($0) }
        }
        try Runner.checkVoxRequest(["text": "Hi.", "voice": "voice-1", "reference": ["path": "/tmp/a.wav"]], options: VoxCPM2Options())
        #expect { try Runner.checkVoxClip(AudioClip(samples: Array(repeating: 0, count: 4000), sampleRate: 16000)) } throws: { invalid($0) }
        try Runner.checkVoxClip(AudioClip(samples: Array(repeating: 0, count: 8000), sampleRate: 16000))
    }

    /// Review fix: Get casts only the folder's pinned files, and leaves nothing behind when it fails or is cancelled.
    @Test func castCopiesOnlyPinnedFilesAndCleansUp() throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let pinned = try VoxCPM2Cast.manifestFiles(modelDirectory: repo.appendingPathComponent("models/voxcpm2"))
        #expect(pinned.count == 7 && pinned.contains("model-00001.safetensors") && pinned.contains("tokenizer.json") && !pinned.contains("README.md"))
        try Device.withDefaultDevice(.cpu) {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent("voxcpm2-cast-" + UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let source = root.appendingPathComponent("src"), out = root.appendingPathComponent("dst")
            try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
            try MLX.save(arrays: ["w": MLXArray([1, 2] as [Float])], url: source.appendingPathComponent("a.safetensors"))
            try Data("{}".utf8).write(to: source.appendingPathComponent("config.json"))
            try Data("extra".utf8).write(to: source.appendingPathComponent("README.md"))
            let report = try VoxCPM2Cast.convert(source: source, files: ["a.safetensors", "config.json"], into: out)
            #expect(report.files == ["a.safetensors", "config.json"])
            #expect(try FileManager.default.contentsOfDirectory(atPath: out.path).sorted() == ["a.safetensors", "config.json"])
            for path in ["a.safetensors", "config.json"] { try FileManager.default.removeItem(at: out.appendingPathComponent(path)) }
            #expect(throws: (any Error).self) { try VoxCPM2Cast.convert(source: source, files: ["a.safetensors", "missing.json"], into: out) }
            #expect(try FileManager.default.contentsOfDirectory(atPath: out.path).isEmpty)
            #expect(throws: CancellationError.self) { try VoxCPM2Cast.convert(source: source, files: ["a.safetensors"], into: out, isCancelled: { true }) }
            #expect(try FileManager.default.contentsOfDirectory(atPath: out.path).isEmpty)
        }
    }

    /// Review fix (L2): a sentence too long for one chunk kept no space before it, so "Hi. " and a 450-character
    /// sentence became "Hi.abcdefgh…" in one group.
    @Test func chunkerKeepsTheSpaceBeforeASplitSentence() {
        let text = "Hi. " + Array(repeating: "abcdefgh", count: 50).joined(separator: " ") + "."
        let pieces = VoxCPM2Speech.chunks(text)
        #expect(pieces.count == 2 && pieces[0].hasPrefix("Hi. abcdefgh"))
        #expect(pieces.allSatisfy { $0.count <= VoxCPM2Speech.chunkCharacters })
        #expect(pieces.joined(separator: " ") == text)
    }

    /// One id per character, as the stub tokenizer below; never a special id.
    static func characterIDs(_ text: String) -> [Int] { text.unicodeScalars.map { Int($0.value % 90) + 5 } }

    /// Review fix (M2): without a clip every chunk was its own generation with its own seed, so a long text changed
    /// speaker every ~400 characters. Chunks after the first now continue chunk 0 (its text, then theirs).
    @Test func chunksAfterTheFirstContinueTheFirstVoice() {
        let generation = VoxCPM2Generation()
        let sentence = String(repeating: "word ", count: 19) + "end."
        let body = Array(repeating: sentence, count: 10).joined(separator: " ")
        let pieces = VoxCPM2Speech.chunks(body)
        let (alone, continuing) = VoxCPM2Speech.plans(
            body: body, pieces: pieces, prefix: "(calm)", promptText: nil, anchored: true, generation: generation, tokenize: Self.characterIDs)
        #expect(alone.count == 3 && continuing.count == 3 && continuing[0] == nil)
        for index in 1..<3 {
            let plan = continuing[index]!
            #expect(plan.continuesFirst && plan.ids == Self.characterIDs("(calm)" + pieces[0] + " " + pieces[index]))
            #expect(plan.target == ("(calm)" + pieces[index]).count && plan.maxLength == plan.target * 6 + 10)
        }
        #expect(alone[1].ids == Self.characterIDs("(calm)" + pieces[1]) && !alone[1].continuesFirst)
        // A chunk continues chunk 0 only once chunk 0 left audio, and only where the continuation fits the context;
        // otherwise it speaks on its own (review: Hindi or Thai chunks are ~3 tokens per letter and must not be refused).
        #expect(VoxCPM2Speech.plan(1, alone: alone, continuing: continuing, firstPatches: nil, context: 8192) == alone[1])
        #expect(VoxCPM2Speech.plan(1, alone: alone, continuing: continuing, firstPatches: 300, context: 8192) == continuing[1])
        #expect(VoxCPM2Speech.plan(0, alone: alone, continuing: continuing, firstPatches: 300, context: 8192) == alone[0])
        let need = VoxCPM2Speech.contextNeed(continuing[1]!, referencePatches: nil, promptPatches: 300)
        #expect(need == continuing[1]!.ids.count + 1 + 300 + continuing[1]!.maxLength)
        #expect(VoxCPM2Speech.plan(1, alone: alone, continuing: continuing, firstPatches: 300, context: need - 1) == alone[1])
        // Scripts without spaces join chunk 0 and the next with nothing between them.
        let han = String(repeating: "你好。", count: 200)
        let hanPieces = VoxCPM2Speech.chunks(han)
        let hanPlans = VoxCPM2Speech.plans(body: han, pieces: hanPieces, prefix: "", promptText: nil, anchored: true, generation: generation, tokenize: Self.characterIDs)
        #expect(hanPlans.continuing[1]?.ids == Self.characterIDs(hanPieces[0] + hanPieces[1]))
        // With a clip (its timbre or transcript is the voice) and for one chunk, nothing continues chunk 0.
        let clip = VoxCPM2Speech.plans(body: body, pieces: pieces, prefix: "", promptText: "Hej.", anchored: false, generation: generation, tokenize: Self.characterIDs)
        #expect(clip.continuing.allSatisfy { $0 == nil } && clip.alone[2].ids == Self.characterIDs("Hej." + pieces[2]))
        let one = VoxCPM2Speech.plans(body: "Hi.", pieces: ["Hi."], prefix: "", promptText: nil, anchored: true, generation: generation, tokenize: Self.characterIDs)
        #expect(one.continuing == [nil])
    }

    /// A tiny random network whose stop head always fires (after the maker's two-patch minimum).
    static func stoppingModel() throws -> VoxCPM2Model {
        let model = VoxCPM2Model(tinyConfig())
        let h = model.config.lmConfig.hiddenSize
        try model.update(
            parameters: ModuleParameters.unflattened([
                "stop_proj.weight": MLXArray.zeros([h, h]), "stop_proj.bias": MLXArray.ones([h]),
                "stop_head.weight": MLX.concatenated([MLXArray.zeros([1, h]), MLXArray.ones([1, h])], axis: 0)
            ]), verify: [.shapeMismatch])
        return model
    }

    /// M2 end to end on a tiny network: chunk 0's generated patches become the continuation prompt of the later chunks,
    /// whose audio decodes after chunk 0's last patches.
    @Test func longTextWithoutAClipSpeaksThroughTheContinuation() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let model = try Self.stoppingModel()
            let anchor = VoxCPM2Speech.continuationPrompt((0..<4).map { _ in MLXArray.ones([1, 2, 8]) })
            #expect(anchor.shape == [4, 2, 8])
            let speech = VoxCPM2Speech(id: "voxcpm2", model: model, tokenizer: CharacterTokenizer())
            let sentence = String(repeating: "word ", count: 19) + "end."
            var request = SpeechRequest(text: Array(repeating: sentence, count: 10).joined(separator: " "))
            request.seed = 3
            var options = VoxCPM2Options(); options.inferenceTimesteps = 2
            var chunks: [Int] = [], continued: [Bool] = []
            // Release review E (L-E9): sample counts alone match a standalone run (a continued chunk trims its warm-up),
            // so the plans each chunk ran are checked too.
            speech.onPlan = { _, plan in continued.append(plan.continuesFirst) }
            try await speech.speak(request, options: options, control: GenerationControl()) { chunks.append($0.count) }
            let perPatch = model.patchSize * model.audioVAE.decodeChunkSize
            #expect(chunks == [4, 4, 4].map { $0 * perPatch })
            #expect(continued == [false, true, true])
        }
    }

    /// Review fix (L4): a clip at the wrong rate is the caller's fault inside the worker (it resamples every clip), an
    /// engine error; review fix (L1): a transcript that leaves no room for speech is refused when the voice is saved.
    @Test func voxFaultsAndSavedVoiceLimits() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let speech = VoxCPM2Speech(id: "voxcpm2", model: try Self.stoppingModel(), tokenizer: CharacterTokenizer())
            var request = SpeechRequest(text: "Hi.")
            request.reference = AudioClip(samples: Array(repeating: 0, count: 48000), sampleRate: 48000)
            await #expect(throws: EngineError.self) { try await speech.speak(request, options: VoxCPM2Options(), control: GenerationControl()) { _ in } }
            // The tiny network's patches are 20 samples, so one second is 800 patches.
            try speech.checkVoice(samples: 16000, transcript: "Hej, to ja.")
            try speech.checkVoice(samples: 16000, transcript: nil)
            #expect(throws: VoxCPM2Error.self) { try speech.checkVoice(samples: 16000, transcript: String(repeating: "a", count: 6000)) }
        }
        // The real network: a 30 s clip (188 patches) with a 100,000-character transcript would refuse every request.
        let full = VoxCPM2Speech.fullChunkTokens
        #expect(full == 1200 + 1 + 4096)
        #expect(VoxCPM2Speech.voicePromptTokens(patches: 188, transcriptTokens: 2000) + full <= 8192)
        #expect(VoxCPM2Speech.voicePromptTokens(patches: 188, transcriptTokens: 100_000) + full > 8192)
        #expect(VoxCPM2Speech.voicePromptTokens(patches: 188, transcriptTokens: nil) == 190)
    }

    /// Review fix (L3): load and conversion errors reached the caller as "VoxCPM2Error error 1".
    @Test func errorsCarryTheirMessage() {
        #expect(VoxCPM2Error("config.json is x, not voxcpm2").localizedDescription == "config.json is x, not voxcpm2")
    }

    /// Review fix (L4): a saved voice whose stored recording is gone is Arsenal's fault (engine error, 500); the
    /// caller's own unreadable clip stays invalid_request (422).
    @Test func savedVoiceRecordingFaultsAreEngineErrors() throws {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent("voxcpm2-missing-\(UUID().uuidString).wav").path
        #expect(throws: EngineError.self) { try Runner.readReference(missing, rate: 16000, fromSavedVoice: true) }
        #expect { try Runner.readReference(missing, rate: 16000, fromSavedVoice: false) } throws: { ($0 as? WorkerFailure)?.code == "invalid_request" }
        #expect(try Runner.readReference(nil, rate: 16000, fromSavedVoice: true) == nil)
    }

    /// Review fix (L5): an unexpected AudioVAE tensor name or a zero patch size crashed the worker; both now throw.
    @Test func brokenSnapshotsThrowInsteadOfCrashing() throws {
        try Device.withDefaultDevice(.cpu) {
            let vae = VoxCPM2Model(Self.tinyConfig()).audioVAE
            for key in ["encoder.block.x.weight", "decoder.model.weight", "decoder.sr_cond_model.0.weight", "encoder"] {
                #expect(throws: VoxCPM2Error.self, "\(key)") { try vae.sanitize(weights: ["decoder.model.0.weight": MLXArray.ones([2]), key: MLXArray.ones([2])]) }
            }
            #expect(throws: VoxCPM2Error.self) {
                try vae.sanitize(weights: ["encoder.block.0.weight_g": MLXArray.ones([3, 1, 1]), "encoder.block.0.weight_v": MLXArray.ones([2, 1, 1])])
            }
        }
        var config = Self.tinyConfig()
        try config.check()
        config.patchSize = 0
        #expect(throws: VoxCPM2Error.self) { try config.check() }
        config = Self.tinyConfig(); config.audioVaeConfig.encoderRates = [2, 0]
        #expect(throws: VoxCPM2Error.self) { try config.check() }
    }

    @Test func voiceOperationsAcceptVoxCPM2() throws {
        try SpeechOperation.check(["op": "clone_voice", "audio": ["path": "/tmp/a.wav"], "name": "Me"], modelID: "voxcpm2")
        try SpeechOperation.check(["op": "speak", "text": "Hi", "max_tokens": 4096, "reference": ["path": "/tmp/a.wav"]], modelID: "voxcpm2")
        #expect(throws: WorkerFailure.self) { try SpeechOperation.check(["op": "speak", "text": "Hi", "max_tokens": 4097], modelID: "voxcpm2") }
        #expect(throws: WorkerFailure.self) { try SpeechOperation.check(["op": "speak", "text": "Hi", "temperature": 0.5], modelID: "voxcpm2") }
    }
}

/// One token per character (ids 5–94, never a special id), for tiny-network tests that need a tokenizer.
struct CharacterTokenizer: Tokenizer {
    func tokenize(text: String) -> [String] { text.unicodeScalars.map { String($0) } }
    func encode(text: String) -> [Int] { VoxCPM2Tests.characterIDs(text) }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { VoxCPM2Tests.characterIDs(text) }
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { token.unicodeScalars.count == 1 ? VoxCPM2Tests.characterIDs(token)[0] : nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { nil }
    var eosTokenId: Int? { nil }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }
    func applyChatTemplate(messages: [Message]) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] { [] }
    func applyChatTemplate(
        messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [ToolSpec]?
    ) throws -> [Int] { [] }
    func applyChatTemplate(
        messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
