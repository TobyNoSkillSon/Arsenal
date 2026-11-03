// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import MLXLLM
import ModelPrecision
import ArsenalWire
import Tokenizers

final class ACEAdapter: NativeModel {
    let architecture = "acestep"
    let operations: Set<String> = ["generate_music"]
    private var pipeline: ACEStepPipeline?
    var peakBytes: Int { Memory.peakMemory }
    var parameters: [String: MLXArray] {
        guard let pipeline else { return [:] }
        var tensors = MusicSupport.parameters([
            ("dit.", pipeline.network), ("vae.", pipeline.vae), ("text_encoder.", pipeline.textEncoder.model), ("planner.", pipeline.planner.model)
        ])
        tensors["silence.silence"] = pipeline.silence
        return tensors
    }
    func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        try cancellation.check()
        let small = context.randomWeights && context.config["small"]?.bool == true
        let config: ACEStepConfiguration
        let vaeConfig: OobleckConfiguration
        let qwenText: Qwen3Configuration, qwenPlanner: Qwen3Configuration
        if small {
            config = try JSONDecoder().decode(
                ACEStepConfiguration.self,
                from: JSONSerialization.data(withJSONObject: [
                    "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
                    "text_hidden_dim": 128, "num_lyric_encoder_hidden_layers": 1, "num_attention_pooler_hidden_layers": 1, "num_timbre_encoder_hidden_layers": 1,
                    "timbre_fix_frame": 16, "is_turbo": true
                ]))
            vaeConfig = OobleckConfiguration(
                audioChannels: 2, channelMultiples: [1, 2, 4, 4, 4], decoderChannels: 2, decoderInputChannels: 64, downsamplingRatios: [2, 4, 4, 6, 10], encoderHiddenSize: 128,
                samplingRate: 48000)
            qwenText = try MusicSupport.qwenConfiguration(Data(), small: true); qwenPlanner = qwenText
        } else {
            let source = context.randomWeights ? context.folder.appendingPathComponent("gate/configs") : context.weights
            config = try JSONDecoder().decode(ACEStepConfiguration.self, from: Data(contentsOf: source.appendingPathComponent("acestep-v15-turbo/config.json")))
            vaeConfig = try JSONDecoder().decode(OobleckConfiguration.self, from: Data(contentsOf: source.appendingPathComponent("vae/config.json")))
            qwenText = try MusicSupport.qwenConfiguration(Data(contentsOf: source.appendingPathComponent("Qwen3-Embedding-0.6B/config.json")), small: false)
            qwenPlanner = try MusicSupport.qwenConfiguration(Data(contentsOf: source.appendingPathComponent("acestep-5Hz-lm-1.7B/config.json")), small: false)
        }
        let network = ACEStepNetwork(config), vae = OobleckVAE(vaeConfig), encoder = Qwen3Model(qwenText), planner = Qwen3Model(qwenPlanner)
        let textTokenizer: any Tokenizers.Tokenizer, plannerTokenizer: any Tokenizers.Tokenizer, codeIDs: [String: Int], silence: MLXArray
        if context.randomWeights {
            for module in [network as Module, vae as Module, encoder as Module, planner as Module] {
                try MusicSupport.initialize(module, precision: context.precision); try cancellation.check()
            }
            textTokenizer = try MusicSupport.fixtureTokenizer(); plannerTokenizer = textTokenizer
            codeIDs = Dictionary(uniqueKeysWithValues: (0..<64).map { ("<|audio_code_\($0)|>", 256 + $0) })
            silence = MLXArray.zeros([small ? 32 : 15000, 64], dtype: .float16)
        } else {
            for (module, path) in [
                (network as Module, "acestep-v15-turbo/model.safetensors"), (vae as Module, "vae/diffusion_pytorch_model.safetensors"),
                (encoder as Module, "Qwen3-Embedding-0.6B/model.safetensors"), (planner as Module, "acestep-5Hz-lm-1.7B/model.safetensors")
            ] {
                try cancellation.check(); try MusicSupport.load(module, file: context.weights.appendingPathComponent(path), precision: context.precision)
            }
            textTokenizer = try MusicSupport.wait { try await AutoTokenizer.from(modelFolder: context.weights.appendingPathComponent("Qwen3-Embedding-0.6B")) }
            plannerTokenizer = try PlannerTokenizer.load(context.weights.appendingPathComponent("acestep-5Hz-lm-1.7B"))
            codeIDs = try PlannerTokenizer.audioCodeIDs(context.weights.appendingPathComponent("acestep-5Hz-lm-1.7B"))
            silence = try TensorArchive(context.weights.appendingPathComponent("acestep-v15-turbo/silence.safetensors")).array("silence").asType(.float16)
        }
        pipeline = ACEStepPipeline(
            configuration: config, network: network, vae: vae, textEncoder: TextEncoder(model: encoder, tokenizer: textTokenizer),
            planner: Planner(model: planner, tokenizer: plannerTokenizer, codeIDs: codeIDs), silence: silence)
    }
    func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operation == "generate_music", let pipeline else { throw MusicSupport.failure("ACE-Step is not loaded or operation is unsupported.") }
        try cancellation.check(); let request = try MusicRequests.decode(input, maxSeconds: 600)
        pipeline.cancellation = cancellation; pipeline.planner.cancellation = cancellation
        defer { pipeline.cancellation = nil; pipeline.planner.cancellation = nil }
        do {
            return try withoutActuallyEscaping(emit) { forward in
                let songs = try pipeline.generate(
                    request,
                    progress: { progress in
                        try? forward(MusicSupport.progress(progress, batch: request.batchSize))
                    }, cancelled: { (try? cancellation.check()) == nil })
                return try MusicRequests.files(songs, input: input, cancellation: cancellation)
            }
        } catch {
            try cancellation.check()
            if let failure = error as? ACEStepError {
                switch failure {
                case .invalidRequest(let message), .unreadableAudio(let message): throw MusicSupport.failure(message)
                default: break
                }
            }
            if let failure = error as? YuE2Error, case .invalidRequest(let message) = failure { throw MusicSupport.failure(message) }
            throw error
        }
    }
    func unload() { pipeline = nil }
}
