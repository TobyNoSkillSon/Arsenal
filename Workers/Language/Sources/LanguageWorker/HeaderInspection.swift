// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM
import ArsenalWire
import IntegrationKit

extension LanguageWorker {
    public static func inspectTokenizer(directory: URL, specification: URL, output: URL) throws {
        let tokenizer = try localAwait { try await LocalTokenizer(directory: directory) }
        let spec = try WireJSON.decode(Data(contentsOf: specification))
        let text = tokenizer.encode(text: "Hello", addSpecialTokens: false)
        let decoded = tokenizer.decode(tokenIds: text, skipSpecialTokens: true)
        let extra: [String: any Sendable] = ["enable_thinking": false, "template": "{\"name\":\"string\"}"]
        let plain = try tokenizer.applyChatTemplate(messages: [["role": "user", "content": "Hello"]], tools: nil, additionalContext: extra)
        let hasImages = spec["operations"]?.array?.contains(where: { ["describe", "read_document", "extract", "ground_screen"].contains($0.string ?? "") }) == true
        let image =
            hasImages
            ? try tokenizer.applyChatTemplate(messages: [["role": "user", "content": [["type": "image"], ["type": "text", "text": "Hello"]]]], tools: nil, additionalContext: extra)
            : []
        guard !text.isEmpty, !plain.isEmpty, decoded == "Hello", !hasImages || !image.isEmpty else {
            throw WorkerFailure(code: "tokenizer", message: "Offline tokenizer/template checks failed.")
        }
        try WireJSON.encode(["roundtrip": .string(decoded), "plain_tokens": .number(Double(plain.count)), "image_tokens": .number(Double(image.count))]).write(to: output)
    }
    /// Lab-only shape inspection uses lazy arrays and never evaluates a real-size model.
    public static func inspectHeader(config: URL, header: URL, output: URL) throws {
        guard ProcessInfo.processInfo.environment["ARSENAL_NATIVE_LAB"] == "1" else {
            throw WorkerFailure(code: "invalid_request", message: "Header inspection requires ARSENAL_NATIVE_LAB.")
        }
        NativeRuntime.prepare()
        let data = try Data(contentsOf: config), raw = try WireJSON.decode(data)
        let type = raw["model_type"]?.string ?? ""
        let module: Module
        var language: (any LanguageModel)?
        var paddle: PaddleOCRVLModel?
        if type == "hunyuan_v1_dense" {
            language = HunyuanModel(try JSONDecoder().decode(HunyuanConfiguration.self, from: data)); module = language!
        } else if type == "paddleocr_vl" {
            paddle = PaddleOCRVLModel(config: try PaddleOCRVLConfig.load(from: config.deletingLastPathComponent())); module = paddle!
        } else {
            language = try LanguageGraph.make(data, type: type); module = language!
        }
        try LanguageGraph.removeUnusedSharedKV(module, config: data)
        let expected = Dictionary(uniqueKeysWithValues: module.parameters().flattened().map { ($0.0, $0.1.shape) })
        let source = try WireJSON.decode(Data(contentsOf: header))
        let tensors = source.compactMapValues { value -> MLXArray? in
            guard let shape = value.object?["shape"]?.array?.compactMap({ $0.number.map(Int.init) }) else { return nil }
            return MLX.zeros(shape, dtype: .bfloat16)
        }
        let nativeLayout = tensors.keys.contains { $0.hasPrefix("language_model.") }
        let inactive: Set<String> =
            type.hasPrefix("lfm") && raw["projector_use_layernorm"]?.bool == false ? ["multi_modal_projector.layer_norm.bias", "multi_modal_projector.layer_norm.weight"] : []
        let activeTensors = tensors.filter { !inactive.contains($0.key) }
        let mapped: [String: MLXArray]
        if let paddle {
            mapped = Dictionary(uniqueKeysWithValues: activeTensors.compactMap { paddle.mapCheckpoint($0.key, $0.value) })
        } else if let language {
            mapped = nativeLayout ? activeTensors : language.sanitize(weights: activeTensors)
        } else {
            mapped = [:]
        }
        let missing = Set(expected.keys).subtracting(mapped.keys).sorted()
        let unexpected = Set(mapped.keys).subtracting(expected.keys).sorted()
        let wrong = expected.keys.sorted().compactMap { name -> ArsenalWire.JSONValue? in
            guard let tensor = mapped[name], tensor.shape != expected[name] else { return nil }
            return ["name": .string(name), "checkpoint": .array(tensor.shape.map { .number(Double($0)) }), "graph": .array(expected[name]!.map { .number(Double($0)) })]
        }
        var ignore: [ArsenalWire.JSONValue] = []; var rules: JSONObject = [:]; var casts: JSONObject = [:]
        for (name, tensor) in tensors {
            let values: [String: MLXArray]
            if inactive.contains(name) {
                values = [:]
            } else if let paddle {
                values = paddle.mapCheckpoint(name, tensor).map { [$0.0: $0.1] } ?? [:]
            } else if let language {
                values = nativeLayout ? [name: tensor] : language.sanitize(weights: [name: tensor])
            } else {
                values = [:]
            }
            if values.isEmpty { ignore.append(.string(name)); continue }
            if values.count == 1, let (mappedName, mappedTensor) = values.first {
                var rule: JSONObject = ["name": .string(mappedName)]
                if tensor.shape != mappedTensor.shape {
                    let candidates = tensor.ndim == 3 ? [[0, 2, 1], [2, 0, 1]] : tensor.ndim == 4 ? [[0, 2, 3, 1]] : tensor.ndim == 5 ? [[0, 2, 3, 4, 1]] : []
                    if let axes = candidates.first(where: { $0.map { tensor.shape[$0] } == mappedTensor.shape }) { rule["permutation"] = .array(axes.map { .number(Double($0)) }) }
                }
                rules[name] = .object(rule)
                if let dtype = source[name]?.object?["dtype"]?.string, ["F32", "BF16"].contains(dtype) {
                    casts[name] = ["from": .string(dtype), "to": "F16", "at": "Get"]
                }
            }
        }
        let inventory = expected.mapValues { shape -> ArsenalWire.JSONValue in ["shape": .array(shape.map { .number(Double($0)) }), "dtype": "float16"] }
        let receipt: JSONObject = [
            "architecture": .string(type), "source_tensors": .number(Double(source.count)), "graph_tensors": .number(Double(expected.count)),
            "mapped_tensors": .number(Double(mapped.count)), "missing": .array(missing.map { .string($0) }), "unexpected": .array(unexpected.map { .string($0) }),
            "shape_mismatches": .array(wrong), "shape_match": .bool(missing.isEmpty && unexpected.isEmpty && wrong.isEmpty),
            "mapping": ["ignore": .array(ignore), "tensors": .object(rules), "casts": .object(casts)], "parameters": .object(inventory),
            "dtype_note": "Exact source names/shapes; explicit BF16/F32 to FP16 storage conversion at Get. Trained numerical quality remains unqualified."
        ]
        try WireJSON.encode(receipt).write(to: output)
    }
}
