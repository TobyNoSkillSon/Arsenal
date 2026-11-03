// SPDX-License-Identifier: MIT
// Drama 3 behind the voices family's engine boundary (VireoCore `SpeechModel`): loading a pinned snapshot and serving
// speak requests. Every request clones the voice of a reference clip with its exact transcript, or draws an unprompted
// voice when none is given; there are no preset voices and no speed control, and delivery cues go inline ([whispers]).
import Foundation
import MLX
import MLXNN
import Tokenizers
import VireoCore

struct Drama3TokenizerAdapter: Drama3Tokenizing {
    let tokenizer: any Tokenizer
    func encode(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }
    func id(of token: String) -> Int? { tokenizer.convertTokenToId(token) }
}

public final class Drama3Speech: SpeechModel {
    public static let modelID = "fish-drama-3"
    public let id: String
    public let sampleRate: Int
    private(set) var model: Drama3Model?

    init(id: String, model: Drama3Model, sampleRate: Int) {
        self.id = id; self.model = model; self.sampleRate = sampleRate
    }

    public var engineLabel: String { "mlx" }
    public var engineReason: String { "Native Swift port of mlx-audio fish_drama3 (dad2087f); stock MLX kernels" }

    /// Loads a snapshot directory as stored (vendor mixed 4/8-bit affine backbone and tower, float32 norms) with the codec
    /// cast to `codecDType` in memory. Nothing is written to the snapshot.
    public static func load(directory: URL, id: String = modelID, codecDType: DType = .float16) async throws -> Drama3Speech {
        let config = try Drama3Config(data: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let model = Drama3Model(config: config)
        try model.load(weights: try Drama3Model.shardWeights(directory))
        eval(model.network.parameters())
        let tokenizer = Drama3TokenizerAdapter(tokenizer: try await AutoTokenizer.from(modelFolder: directory))
        try checkTokenizer(tokenizer, config: config)
        model.tokenizer = tokenizer
        let codec = try Drama3CodecLoader.load(directory: directory, dtype: codecDType)
        model.codec = codec
        eval(model.candidateIDs)
        return Drama3Speech(id: id, model: model, sampleRate: codec.sampleRate)
    }

    /// The tokenizer and config.json must agree on the semantic band and EOS.
    static func checkTokenizer(_ tokenizer: Drama3Tokenizing, config: Drama3Config) throws {
        let last = config.tower.vocabSize - 1
        for (token, expected) in [("<|semantic:0|>", config.semanticBegin), ("<|semantic:\(last)|>", config.semanticEnd), ("<|im_end|>", config.eosID)] {
            guard tokenizer.id(of: token) == expected else { throw Drama3Error("tokenizer and config disagree on \(token)") }
        }
    }

    public func unload() async {
        model?.clearVoices()
        model = nil
        Memory.clearCache()
    }

    public func speak(_ request: SpeechRequest, control: GenerationControl, emit: @escaping ([Float]) async throws -> Void) async throws {
        guard let model else { throw EngineError("Drama 3 is not loaded") }
        if let preset = request.preset, !preset.isEmpty { throw EngineError("Drama 3 has no preset voices; pass a reference clip with its transcript or a saved voice") }
        if let instructions = request.description ?? request.style, !instructions.isEmpty {
            throw EngineError("Drama 3 takes delivery cues inline, e.g. [whispers], not instructions")
        }
        var references: [([Float], String)] = []
        if let clip = request.reference {
            guard clip.sampleRate == sampleRate, clip.channels == 1 else { throw EngineError("Reference audio must be mono at \(sampleRate) Hz") }
            guard let text = request.referenceText, !Drama3Text.strip(text).isEmpty else {
                throw EngineError("Drama 3 cloning needs the exact transcript of the reference clip (reference.text)")
            }
            references.append((clip.samples, text))
        } else if request.referenceText?.isEmpty == false {
            throw EngineError("reference.text needs its reference.path clip")
        }
        let sampling = Drama3Sampling(temperature: request.temperature, topK: request.topK, topP: request.topP, maxFrames: request.maxTokens)
        do {
            try await model.generate(
                text: request.text, references: references, sampling: sampling, seed: request.seed, isCancelled: { control.isCancelled },
                emit: { samples in
                    guard !control.isCancelled else { return }
                    guard samples.allSatisfy(\.isFinite) else { throw EngineError("Drama 3 produced non-finite audio") }
                    try await emit(samples)
                })
        } catch let error as Drama3Error { throw EngineError(error.message) }
    }
}
