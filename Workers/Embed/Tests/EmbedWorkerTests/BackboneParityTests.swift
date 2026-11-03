import Foundation
import IntegrationKit
import MLX
import MLXNN
import MLXEmbedders
import MLXRandom
import MLXLMCommon
import MLXVLM
import Testing
@testable import EmbedWorker

@Suite(.serialized) struct RetrievalGPUTests {
    @Test(.heavy) func qwenBackboneMatchesPinnedEmbedder() throws {
        NativeRuntime.prepare()
        let data = Data(
            """
            {"hidden_size":128,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,"intermediate_size":256,"rms_norm_eps":0.000001,"vocab_size":256}
            """.utf8)
        MLXRandom.seed(7)
        let reference = MLXEmbedders.Qwen3Model(try JSONDecoder().decode(MLXEmbedders.Qwen3Configuration.self, from: data))
        try NativeParameters.initializeRandom(reference)
        let candidate = RetrievalGraph(try JSONDecoder().decode(EmbedQwenConfiguration.self, from: data), ranking: false, jina: false)
        try candidate.update(parameters: reference.parameters(), verify: .all)
        let ids = MLXArray([1, 23, 51, 9]).reshaped(1, -1)
        let expected = try #require(reference(ids).hiddenStates)
        let actual = try candidate.backbone(ids, cancellation: NativeCancellation())
        #expect(actual.shape == expected.shape)
        #expect(MLX.max(abs(actual - expected)).item(Float.self) == 0)
    }

    @Test(.heavy) func storedSourceAndPacksUseManifestIdentity() throws {
        NativeRuntime.prepare()
        let base = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: base) }
        let source = base.appendingPathComponent("source"), folder = base.appendingPathComponent("anonymous-folder")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data("id = \"qwen3-reranker-0.6b\"\n".utf8).write(to: folder.appendingPathComponent("model.toml"))
        let data = Data(
            """
            {"model_type":"qwen3","hidden_size":128,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,"intermediate_size":256,"rms_norm_eps":0.000001,"vocab_size":256,"tie_word_embeddings":true,"max_position_embeddings":40960}
            """.utf8)
        try data.write(to: source.appendingPathComponent("config.json"))
        let graph = RetrievalGraph(try JSONDecoder().decode(EmbedQwenConfiguration.self, from: data), ranking: true, jina: false)
        try NativeParameters.initializeRandom(graph, half: .bfloat16)
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: graph.parameters().flattened()), url: source.appendingPathComponent("model.safetensors"))
        try Data(#"{"tokenizer_class":"PreTrainedTokenizer","unk_token":"[UNK]"}"#.utf8).write(to: source.appendingPathComponent("tokenizer_config.json"))
        try Data(
            #"{"version":"1.0","added_tokens":[{"id":1,"content":"yes","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},{"id":2,"content":"no","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],"pre_tokenizer":{"type":"Whitespace"},"model":{"type":"BPE","unk_token":"[UNK]","vocab":{"[UNK]":0,"yes":1,"no":2},"merges":[]}}"#
                .utf8
        ).write(to: source.appendingPathComponent("tokenizer.json"))
        let converter = try #require(EmbedWorker.registrations.compactMap(\.conversion).first { $0.recipe == "cast:embed-qwen3-v2" })
        for tier in ["fp16", "int8", "int4"] {
            let destination = base.appendingPathComponent(tier)
            _ = try converter.run(
                [
                    "src_dir": .string(tier == "fp16" ? source.path : base.appendingPathComponent("fp16").path), "dst_dir": .string(destination.path), "dtype": .string(tier),
                    "group_size": 64, "recipe": "cast:embed-qwen3-v2"
                ],
                cancellation: NativeCancellation())
            let adapter = TextRetrievalAdapter()
            try adapter.load(
                NativeLoadContext(["model_dir": .string(folder.path), "weights_dir": .string(destination.path), "dtype": .string(tier)]), cancellation: NativeCancellation())
            #expect(adapter.operations == ["rerank"])
            // The shipping 8B (and its retired 0.6B/4B siblings) declare 40,960 positions; the card's max_length is 8,192.
            #expect(adapter.limit == 8192)
            #expect(throws: (any Error).self) {
                try adapter.run("rerank", input: ["query": "query", "documents": ["first"], "max_tokens_per_doc": 8193], cancellation: NativeCancellation(), emit: { _ in })
            }
            let result = try adapter.run("rerank", input: ["query": "query", "documents": ["first", "second"]], cancellation: NativeCancellation(), emit: { _ in })
            #expect(result["results"]?.array?.count == 2)
            #expect(adapter.parameters.values.allSatisfy { $0.dtype == .float16 || $0.dtype == .uint32 })
            if tier != "fp16" { #expect(adapter.parameters.values.contains { $0.dtype == .uint32 }) }
            adapter.unload()
            #expect(adapter.parameters.isEmpty)
        }
    }

    @Test(.heavy) func ovisNormMappingUsesTheServingHalfBeforeOffset() {
        NativeRuntime.prepare()
        let source = MLXArray([Float(0.00390625)]).asType(.bfloat16)
        let direct = OvisAdapter.nativeNorm(source)
        let derived = OvisAdapter.nativeNorm(source.asType(.float16))
        #expect(direct.dtype == .float16)
        #expect(direct.item(Float.self) == 1.00390625)
        #expect(direct.item(Float.self) == derived.item(Float.self))
    }
    @Test(.heavy) func packedVisionPositionsKeepFractionalInterpolation() throws {
        NativeRuntime.prepare()
        let data = Data(
            """
            {"model_type":"qwen3_5_vision","depth":1,"hidden_size":64,"intermediate_size":128,"out_hidden_size":64,"num_heads":4,"patch_size":16,"spatial_merge_size":2,"temporal_patch_size":2,"num_position_embeddings":16}
            """.utf8)
        let model = OvisVision.VisionModel(try JSONDecoder().decode(MLXVLM.Qwen3VLConfiguration.VisionConfiguration.self, from: data))
        try NativeParameters.initializeRandom(model)
        let values = (sin(MLXArray(0..<1024).asType(.float32)) * 0.1).reshaped(16, 64).asType(.float16)
        try model.update(parameters: ModuleParameters.unflattened(["pos_embed.weight": values]), verify: .none)
        let frames = [THW(1, 6, 6)]
        let reference = model.positionalEmbeddings(frames)
        eval(reference)
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
        defer { try? FileManager.default.removeItem(at: file) }
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()), url: file)
        _ = try NativePrecisionTier("int8").load(model, archives: [file], map: { ($0, $1) }, eligible: { $0 == "pos_embed.weight" && $1.count == 2 })
        let actual = model.positionalEmbeddings(frames)
        #expect(actual.dtype == .float16)
        #expect(actual.shape == reference.shape)
        #expect(MLX.max(abs(actual - reference)).item(Float.self) < 0.002)
    }
}
