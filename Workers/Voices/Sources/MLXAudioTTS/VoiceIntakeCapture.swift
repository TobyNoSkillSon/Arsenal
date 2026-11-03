// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXLMCommon
import MLXAudioCore

/// Saved-input replay on the already-loaded production model. Candidates never feed their output
/// into the next component. Only the reference may create the fixture; archives become read-only.
public final class VoiceIntakeCapture {
    public static var active: VoiceIntakeCapture?
    var prefix = ""
    public var caseIndex = 0
    public let profiles: [[String: Any]]
    var profile: [String: Any] { profiles[caseIndex] }
    var suffix: String { profile.isEmpty ? "" : "-" + String(caseIndex) }
    var text: String { profile["text"] as? String ?? "Hello. This is a short voice test." }
    var lengths: [Int] { profile["prefix_lengths"] as? [Int] ?? [1, 2, 3] }
    var frames: Int { profile["code_frames"] as? Int ?? 3 }
    var offset: Int { profile["offset"] as? Int ?? 0 }
    var modulus: Int { offset == 0 ? 17 : 101 }
    var preset: String { profile["preset"] as? String ?? "alba" }
    var inputs: [String: MLXArray]
    var outputs: [String: MLXArray] = [:]
    var traces: [String: Any] = [:]
    var auxiliaryCoverage: [String: Any] = [:]
    let create: Bool
    let only: Set<String>?
    let referenceOutputs: [String: MLXArray]
    public init(fixture: URL, create: Bool, only: Set<String>? = nil, reference: URL? = nil) throws {
        let plan = fixture.appendingPathComponent("fixture-plan.json")
        if FileManager.default.fileExists(atPath: plan.path) {
            let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: plan)) as? [String: Any]
            guard let cases = raw?["cases"] as? [[String: Any]], !cases.isEmpty else { throw NSError(domain: "VoiceIntake", code: 1) }
            profiles = cases
        } else { profiles = [[:]] }
        self.only = only
        if let only, let reference {
            referenceOutputs = try MLX.loadArrays(url: reference.appendingPathComponent("outputs.safetensors"))
            let original = try JSONSerialization.jsonObject(with: Data(contentsOf: reference.appendingPathComponent("activations.json"))) as! [String: Any]
            traces = original.filter { key, _ in !only.contains(key.components(separatedBy: "/")[0]) }
        } else { referenceOutputs = [:] }
        self.create = create
        inputs = create ? [:] : try MLX.loadArrays(url: fixture.appendingPathComponent("inputs.safetensors"))
    }
    public static func moduleRoots(_ model: SpeechGenerationModel) throws -> [(String, Module)] {
        if let q = model as? Qwen3TTSModel {
            var roots: [(String, Module)] = [("talker", q.talker)]
            if let s = q.speakerEncoder { roots.append(("speaker", s)) }
            if let c = q.speechTokenizer { roots.append(("codec", c)) }
            return roots
        }
        if let b = model as? BreezeTTSModel {
            var roots: [(String, Module)] = [("breeze", b)]
            if let c = b.audioTokenizer { roots.append(("codec", c)) }
            return roots
        }
        if let p = model as? PocketTTSModel { return [("pocket", p)] }
        if let k = model as? KokoroModel, let p = k.textProcessor as? KokoroMultilingualProcessor {
            return [("kokoro", k), ("kokoro-g2p", try p.intakeModel())]
        }
        throw NSError(domain: "VoiceIntake", code: 1)
    }
    func input(_ name: String, _ make: () throws -> MLXArray) throws -> MLXArray {
        let key = name + suffix
        if create {
            let raw = try make(); eval(raw)
            // Cache updates mutate MLXArray objects. Own a materialized copy before the next step.
            let value = MLXArray(raw.asData(access: .copy).data, raw.shape, dtype: raw.dtype)
            inputs[key] = value; return value
        }
        guard let value = inputs[key] else { throw failure("Missing immutable input: " + key) }
        return value
    }
    @discardableResult
    func component(_ name: String, _ run: () throws -> MLXArray) rethrows -> MLXArray {
        let key = name + suffix
        if let only, !only.contains(key), let saved = referenceOutputs[key] { outputs[key] = saved; return saved }
        let old = prefix; prefix = key; defer { prefix = old }
        let value = try run(); eval(value)
        outputs[key] = value
        observe("output", value)
        return value
    }
    public static func observeLastToken(_ key: String, _ value: MLXArray) {
        guard let capture = active else { return }
        capture.observe(key, value.ndim == 3 ? value[0..., -1, 0...] : value)
    }
    public static func observe(_ key: String, _ value: MLXArray, clamps: @autoclosure () -> Int = 0) {
        guard let active else { return }
        active.observe(key, value, clamps: clamps())
    }
    func observe(_ key: String, _ value: MLXArray, clamps: Int = 0) {
        guard !prefix.isEmpty else { return }
        eval(value)
        let finite = MLX.all(MLX.isFinite(value)).item(Bool.self)
        traces[prefix + "/" + key] = ["shape": value.shape, "count": value.size, "dtype": String(describing: value.dtype),
            "finite": finite, "max_abs": finite ? Double(MLX.max(MLX.abs(value)).item(Float.self)) : 1e300, "clamp_count": clamps]
    }
    public func finish(fixture: URL, output: URL) throws -> Int {
        let fm = FileManager.default
        if create {
            try fm.createDirectory(at: fixture, withIntermediateDirectories: true)
            let file = fixture.appendingPathComponent("inputs.safetensors")
            guard !fm.fileExists(atPath: file.path) else { throw failure("Fixture already exists") }
            try MLX.save(arrays: inputs, url: file)
            try fm.setAttributes([.posixPermissions: 0o444], ofItemAtPath: file.path)
        }
        try fm.createDirectory(at: output, withIntermediateDirectories: true)
        try MLX.save(arrays: outputs, url: output.appendingPathComponent("outputs.safetensors"))
        try JSONSerialization.data(withJSONObject: traces, options: [.sortedKeys]).write(to: output.appendingPathComponent("activations.json"))
        if !auxiliaryCoverage.isEmpty {
            try JSONSerialization.data(withJSONObject: auxiliaryCoverage, options: [.sortedKeys]).write(to: output.appendingPathComponent("auxiliary-coverage.json"))
        }
        return outputs.count
    }
    func failure(_ text: String) -> NSError { NSError(domain: "VoiceIntake", code: 1, userInfo: [NSLocalizedDescriptionKey: text]) }
    func audio(_ rate: Int) throws -> MLXArray {
        try input("reference/audio") { MLXArray((0..<rate).map { Float(sin(Double($0) * 2 * .pi * Double(profile["audio_hz"] as? Int ?? 180 + offset * 7) / Double(rate))) * 0.05 }).reshaped(1, 1, rate) }
    }
}

extension BreezeTTSModel {
    public func intakeCapture(_ c: VoiceIntakeCapture) throws {
        guard let tokenizer, let codec = audioTokenizer else { throw c.failure("Breeze tokenizers missing") }
        let ids = try c.input("text/ids") { MLXArray(tokenizer.encode(text: "[S0]" + c.text).map(Int32.init)).reshaped(1, -1) }
        let encoder = c.component("text-encoder") { textEncoder(ids) }
        let encInput = try c.input("text-projection/input") { encoder }
        let projection = c.component("text-projection") { textEncoderProjection(encInput.asType(StoredVoiceWeights.dtype(textEncoderProjection))) }
        _ = c.component("text-embedding") { embedTextTokens(ids) }
        let codes = try c.input("audio/codes") { MLXArray((0..<(c.frames * config.numCodebooks)).map { Int32(1 + ($0 + c.offset) % c.modulus) }).reshaped(1, c.frames, config.numCodebooks) }
        let audioEmbed = c.component("audio-embedding") { backboneModel.embedTokens(codes) }
        for index in 0..<3 {
            let saved = try c.input("backbone-\(index)/input") { index == 0 ? projection : audioEmbed[0..., 0..<c.lengths[index], 0...] }
            let hidden = c.component("backbone-\(index)") { backboneModel(inputEmbeddings: saved.asType(StoredVoiceWeights.halfType(backboneModel))) }
            let h = try c.input("head-\(index)/input") { hidden[0..., -1, 0...] }
            _ = c.component("head-\(index)") { lmHead(h.asType(StoredVoiceWeights.dtype(lmHead))) }
            for head in [0, (config.numCodebooks - 2) / 2, config.numCodebooks - 2] {
                let depthIDs = try c.input("depth-\(index)-\(head)/ids") { MLXArray((0..<(head + 2)).map { Int32(($0 + c.offset) % c.modulus) }).reshaped(1, head + 2) }
                _ = c.component("depth-\(index)-\(head)") {
                    let state = h.asType(StoredVoiceWeights.halfType(depthDecoder))
                    guard fastOptions.depthCache else { return depthDecoder.nextLogits(tokenIDs: depthIDs, backboneHiddenState: state) }
                    depthDecoder.model.fused = fastOptions.fusedDepth
                    depthDecoder.model.smallMFFN = fastOptions.smallM
                    let cache = depthDecoder.model.makeCache()
                    var hidden = depthDecoder.model.prime(firstToken: depthIDs[0..., 1], backboneHiddenState: state, cache: cache)
                    for position in 2..<depthIDs.dim(1) { hidden = depthDecoder.model.step(token: depthIDs[0..., position], position: position, cache: cache) }
                    return depthDecoder.logits(hidden, headIndex: head)
                }
            }
        }
        try codec.intakeCapture(c, codes: codes, audio: c.audio(sampleRate))
    }
}

extension Qwen3TTSModel {
    public func intakeCapture(_ c: VoiceIntakeCapture) throws {
        guard let tokenizer, let codec = speechTokenizer else { throw c.failure("Qwen tokenizers missing") }
        let ids = try c.input("text/ids") { MLXArray(tokenizer.encode(text: c.text).map(Int32.init)).reshaped(1, -1) }
        let text = c.component("text-embedding") { talker.getTextEmbeddings()(ids) }
        let textIn = try c.input("text-projection/input") { text }
        let projected = c.component("text-projection") { talker.textProjection(textIn.asType(StoredVoiceWeights.halfType(talker))) }
        let codes = try c.input("audio/codes") { MLXArray((0..<(c.frames * talker.config.numCodeGroups)).map { Int32(1 + ($0 + c.offset) % c.modulus) }).reshaped(1, c.frames, talker.config.numCodeGroups) }
        _ = c.component("codec-embedding") { talker.getInputEmbeddings()(codes[0..., 0..., 0]) }
        for index in 0..<3 {
            let input = try c.input("talker-\(index)/input") { projected[0..., 0..<min(projected.dim(1), c.lengths[index]), 0...] }
            var hidden: MLXArray!
            _ = c.component("talker-\(index)") { let result = talker(input.asType(StoredVoiceWeights.halfType(talker))); hidden = result.1; return result.0 }
            let state = try c.input("predictor-\(index)/input") { hidden![0..., (-1)..., 0...] }
            let predictorEmbeddings = (0..<(talker.config.numCodeGroups - 1)).map { head in
                c.component("predictor-embedding-\(index)-\(head)") {
                    talker.codePredictor.codecEmbedding[head](codes[0..., index..<(index + 1), head + 1])
                }
            }
            let referenceCache = talker.codePredictor.makeCache()
            for head in 0..<(talker.config.numCodeGroups - 1) {
                let saved = try c.input("predictor-\(index)-\(head)/input") {
                    head == 0 ? concatenated([MLX.repeated(state, count: 1 + c.offset % 3, axis: 1), talker.getInputEmbeddings()(codes[0..., index..<(index + 1), 0])], axis: 1) : predictorEmbeddings[head - 1]
                }
                var cache = c.create ? referenceCache : talker.codePredictor.makeCache()
                if head > 0 {
                    for layer in cache.indices {
                        let prior = referenceCache[layer].state
                        let keys = try c.input("predictor-\(index)-\(head)/cache-\(layer)/keys") { prior[0] }
                        let values = try c.input("predictor-\(index)-\(head)/cache-\(layer)/values") { prior[1] }
                        if !c.create { cache[layer].state = [keys.asType(StoredVoiceWeights.halfType(talker)), values.asType(StoredVoiceWeights.halfType(talker))] }
                    }
                }
                _ = c.component("predictor-\(index)-\(head)") {
                    talker.codePredictor(saved.asType(StoredVoiceWeights.halfType(talker)), cache: cache, generationStep: head).0
                }
            }
        }
        let audio = try c.audio(sampleRate)
        if let speakerEncoder {
            let mel = try c.input("speaker/mel") { computeMelSpectrogram(audio: audio.flattened(), sampleRate: speakerEncoder.config.sampleRate, nFft: 1024, hopLength: 256, nMels: 128).expandedDimensions(axis: 0) }
            _ = c.component("speaker-encoder") { speakerEncoder(mel.asType(StoredVoiceWeights.halfType(speakerEncoder))) }
        }
        try codec.intakeCapture(c, codes: codes, audio: audio)
    }
}

extension Qwen3TTSSpeechTokenizer {
    func intakeCapture(_ c: VoiceIntakeCapture, codes: MLXArray, audio: MLXArray) throws {
        if let encoder = encoderModel {
            encoder.encoder.resetState()
            for cache in encoder.encoderCache { cache.trim(cache.offset) }
            encoder.downsample.resetState()
            let hidden = c.component("codec-encoder") { encoder.encoder(audio.asType(StoredVoiceWeights.halfType(encoder))) }
            let saved = try c.input("codec-encoder-transformer/input") { hidden }
            let transformed = c.component("codec-encoder-transformer") { encoder.encoderTransformer(saved.asType(StoredVoiceWeights.halfType(encoder)), cache: encoder.encoderCache)[0] }
            let downInput = try c.input("codec-downsample/input") { transformed }
            _ = c.component("codec-downsample") { encoder.downsample(downInput.asType(StoredVoiceWeights.halfType(encoder))) }
        }
        decoder.resetStreamingState()
        _ = c.component("codec-decoder") { decoder(codes.transposed(0, 2, 1)) }
    }
}

extension PocketTTSModel {
    public func intakeCapture(_ c: VoiceIntakeCapture) throws {
        let dtype = StoredVoiceWeights.dtype(flow_lm.input_linear)
        let tokens = try c.input("text/ids") { flow_lm.conditioner.prepare(c.text).tokens }
        let text = c.component("text-conditioner") { flow_lm.conditioner(TokenizedText(tokens: tokens)) }
        let preset = try c.input("preset/value") {
            guard let value = try MLX.loadArrays(url: modelFolder.appendingPathComponent("embeddings/" + c.preset + ".safetensors"))["audio_prompt"] else { throw c.failure("Missing Alba preset") }; return value
        }
        _ = preset // Saved baseline conditioning remains immutable for independent downstream inputs.
        _ = try c.component("preset-conditioning") {
            guard let current = try MLX.loadArrays(url: modelFolder.appendingPathComponent("embeddings/" + c.preset + ".safetensors"))["audio_prompt"] else {
                throw c.failure("Missing current Alba preset")
            }
            return current.asType(dtype)
        }
        let audio = try c.audio(sampleRate)
        mimi.resetState()
        let latent = c.component("mimi-encoder") { mimi.encodeToLatent(audio) }
        let speakerInput = try c.input("speaker/input") { latent.transposed(0, 2, 1) }
        _ = c.component("speaker-projection") { speaker_proj(speakerInput.asType(dtype)) }
        let textIn = try c.input("backbone/text") { text }
        for index in 0..<3 {
            let sequence = try c.input("backbone-\(index)/sequence") { MLXArray((0..<(c.lengths[index] * flow_lm.ldim)).map { Float(($0 + c.offset) % (c.offset == 0 ? 23 : 31) - (c.offset == 0 ? 11 : 15)) / 32 }).reshaped(1, c.lengths[index], flow_lm.ldim).asType(dtype) }
            let embedded = c.component("latent-projection-\(index)") { flow_lm.input_linear(sequence.asType(dtype)) }
            let backboneIn = try c.input("backbone-\(index)/input") { concatenated([textIn, embedded], axis: 1) }
            let state = c.component("backbone-\(index)") { flow_lm.out_norm(flow_lm.transformer(backboneIn.asType(dtype), cache: flow_lm.makeCache()))[0..., -1, 0...] }
            let hidden = try c.input("flow-\(index)/condition") { state }
            _ = c.component("eos-\(index)") { flow_lm.out_eos(hidden.asType(dtype)) }
            let noise = try c.input("flow-\(index)/noise") { MLXArray((0..<flow_lm.ldim).map { Float(($0 + c.offset) % (c.offset == 0 ? 17 : 29) - (c.offset == 0 ? 8 : 14)) / 16 }).reshaped(1, flow_lm.ldim).asType(dtype) }
            let s = try c.input("flow-\(index)/s") { MLXArray([Float(index) / 3]).reshaped(1, 1).asType(dtype) }
            let t = try c.input("flow-\(index)/t") { MLXArray([Float(index + 1) / 3]).reshaped(1, 1).asType(dtype) }
            let direction = c.component("flow-\(index)") { flow_lm.flow_net(hidden.asType(dtype), s.asType(dtype), t.asType(dtype), noise.asType(dtype)) }
            let decoderIn = try c.input("mimi-transformer-upsampling-\(index)/input") {
                let frames = c.lengths[index]
                let trajectory = MLX.repeated((noise + direction).expandedDimensions(axis: 2), count: frames, axis: 2)
                let variation = MLXArray((0..<frames).map { Float($0) / 100 }).reshaped(1, 1, frames).asType(dtype)
                return (trajectory + variation) * flow_lm.emb_std.reshaped(1, -1, 1) + flow_lm.emb_mean.reshaped(1, -1, 1)
            }
            mimi.resetState()
            let nextInput = c.component("mimi-transformer-upsampling-\(index)") {
                mimi.decoderInput(mimi.quantizer(decoderIn.asType(dtype)))
            }
            // Independent, well-conditioned saved input. The standard shared bounds are frozen
            // before any candidate: rel L2 <= .05, cosine >= .999, finite and no new clamps.
            let waveformInput = try c.input("mimi-waveform-\(index)/input") {
                let rms = MLX.sqrt(MLX.mean(nextInput * nextInput))
                return nextInput / MLX.maximum(rms, MLXArray(Float(0.01)).asType(dtype))
            }
            mimi.decoder.resetState()
            _ = c.component("mimi-waveform-\(index)") { mimi.decoder(waveformInput.asType(dtype)) }
        }
        mimi.resetState()
    }
}


extension KokoroModel {
    public func intakeCapture(_ c: VoiceIntakeCapture) throws {
        let dtype = storedHalf ?? StoredVoiceWeights.dtype(bertEncoder)
        guard let processor = textProcessor as? KokoroMultilingualProcessor else { throw c.failure("Missing Kokoro phonemizer") }
        let g2p = try processor.intakeModel()
        c.auxiliaryCoverage = g2p.precisionReceipt
        let graphemes = try c.input("kokoro/g2p-graphemes") { MLXArray((0..<(5 + c.offset % 13)).map { Int32(4 + ($0 + c.offset) % 20) }).reshaped(1, -1) }
        let phonemePrefix = try c.input("kokoro/g2p-prefix") { MLXArray((0..<(2 + c.offset % 7)).map { Int32(4 + ($0 + c.offset) % 50) }).reshaped(1, -1) }
        let g2pEncoded = c.component("kokoro-g2p-0") { g2p.encode(graphemes).asType(dtype) }
        let g2pMemory = try c.input("kokoro/g2p-memory") { g2pEncoded }
        _ = c.component("kokoro-g2p-1") { g2p.decode(phonemePrefix, encoderOutput: g2pMemory.asType(dtype)).asType(dtype) }
        let phonemes = try textProcessor?.process(text: c.text, language: "en-us") ?? c.text
        let ids = try c.input("kokoro/ids") { MLXArray([Int32(0)] + tokenize(phonemes).map(Int32.init) + [0]).reshaped(1, -1) }
        let length = ids.dim(1), lengths = MLXArray([Int32(ids.dim(1))])
        let mask = MLXArray.zeros([1, length], type: Bool.self)
        let voice = ["af_heart", "bf_emma", "am_adam"][c.offset % 3]
        let style = try c.input("kokoro/style") { try loadVoice(named: voice)[min(length, 509)..<min(length, 509) + 1].asType(dtype) }
        _ = try c.component("kokoro-style") { try loadVoice(named: voice)[min(length, 509)..<min(length, 509) + 1].asType(dtype) }
        let encoded = c.component("kokoro-bert") { bert(ids, attentionMask: MLXArray.ones([1, length], type: Int32.self)).0.asType(dtype) }
        let projectionIn = try c.input("kokoro/projection") { encoded }
        let projected = c.component("kokoro-projection") { bertEncoder(projectionIn.asType(dtype)).transposed(0, 2, 1) }
        let prosodyIn = try c.input("kokoro/prosody") { projected }
        let d = c.component("kokoro-prosody-0") { predictor.textEncoder(prosodyIn.asType(dtype), style: style[0..., 128...].asType(dtype), textLengths: lengths, mask: mask).asType(dtype) }
        let durationIn = try c.input("kokoro/duration") { d }
        _ = c.component("kokoro-prosody-1") { predictor.durationProj(predictor.lstm(durationIn.asType(dtype)).0).asType(dtype) }
        // Fixed alignment, independent of rounded candidate durations.
        let aligned = try c.input("kokoro/aligned-prosody") { MLX.repeated(d.transposed(0, 2, 1), count: 2, axis: 2).asType(dtype) }
        let f0 = c.component("kokoro-prosody-2") { predictor.predict(aligned.asType(dtype), style[0..., 128...].asType(dtype)).0.asType(dtype) }
        let noise = c.component("kokoro-prosody-3") { predictor.predict(aligned.asType(dtype), style[0..., 128...].asType(dtype)).1.asType(dtype) }
        let text = c.component("kokoro-text") { textEncoder(ids, inputLengths: lengths, mask: mask).asType(dtype) }
        let asr = try c.input("kokoro/asr") { MLX.repeated(text, count: 2, axis: 2).asType(dtype) }
        let savedF0 = try c.input("kokoro/f0") { f0 }
        let savedNoise = try c.input("kokoro/noise") { noise }
        _ = c.component("kokoro-decoder") {
            MLXRandom.seed(UInt64(24301 + c.offset))
            return decoder(asr.asType(dtype), savedF0.asType(dtype), savedNoise.asType(dtype), style[0..., ..<128].asType(dtype)).asType(dtype)
        }
    }
}
