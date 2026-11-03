// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import IntegrationKit
import PerceptionSupport
import ArsenalWire
import Tokenizers

public final class NativeLocateAnything: NativeModel {
    public let architecture = "locateanything"
    public let operations: Set<String> = ["detect", "ground_screen"]
    private var model: LocateAnythingModel?
    private var session: LocateAnythingSession?
    private var scratch: URL?
    public init() {}
    public var parameters: [String: MLXArray] { Dictionary(uniqueKeysWithValues: model?.parameters().flattened() ?? []) }
    public var peakBytes: Int { PerceptionIO.bytes(model) }
    public func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        var config = context.randomWeights ? context.config : try PerceptionIO.json(context.weights.appendingPathComponent("config.json"))
        if context.randomWeights && context.config["full_inventory"]?.bool != true {
            config["vision_config"] = [
                "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4, "intermediate_size": 256, "init_pos_emb_height": 8, "init_pos_emb_width": 8
            ]
            // Keep the real tokenizer's vocabulary and spatial token IDs; shrink only hidden layers.
            config["text_config"] = ["hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 256]
            config["eos_token_id"] = 151645
        }
        let parsed = try JSONDecoder().decode(LocateAnythingConfiguration.self, from: JSONEncoder().encode(config))
        let loaded = LocateAnythingModel(parsed)
        try PerceptionIO.prepare(loaded, context: context, map: Self.map, cancellation: cancellation)
        // Legacy tokenizer synthesis must not mutate the verified snapshot. Use a private temporary copy of small assets.
        let temp = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-locate-tokenizer-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        scratch = temp
        let source = context.randomWeights ? try PerceptionIO.asset(context, "tokenizer") : context.weights
        for name in ["vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"] {
            let from = source.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: from.path) { try FileManager.default.copyItem(at: from, to: temp.appendingPathComponent(name)) }
        }
        try LocateAnythingTokenizerSetup.ensureTokenizerJSON(in: temp)
        let tokenizer = try PerceptionIO.awaitResult { try await AutoTokenizer.from(modelFolder: temp) }
        model = loaded; session = LocateAnythingSession(model: loaded, config: parsed, tokenizer: tokenizer)
    }
    public static func map(_ key: String, _ value: MLXArray) throws -> (String, MLXArray)? {
        if key.contains("position_ids") || key.contains("rotary_emb.inv_freq") || key == "language_model.lm_head.weight" { return nil }
        return (key, value)
    }
    /// Get's packing rule, on checkpoint names: exactly the matrices Load packs. Load keeps a tensor only when `map` does
    /// and packs it only when it is an eligible matrix of the graph owned by a Linear or Embedding; packing by name alone
    /// would pack a matrix Load drops (a tied `language_model.lm_head.weight`), leaving scales/biases that Load refuses as
    /// "Orphan affine companions". The Language family follows the same rule (`LanguageNative.storesPacked`).
    static func packingRule(_ graph: Module) -> (String, [Int]) -> Bool {
        let shapes = Dictionary(uniqueKeysWithValues: graph.parameters().flattened().map { ($0.0, $0.1.shape) })
        let owners = Set(graph.leafModules().flattened().compactMap { name, module in module is Linear || module is Embedding ? name : nil })
        // `map` reads names only; a scalar placeholder stands in for the tensor.
        let placeholder = MLXArray(Float16(0))
        return { key, shape in
            guard let (name, _) = try? map(key, placeholder), shapes[name] == shape, PerceptionIO.eligible(name, shape) else { return false }
            return owners.contains(String(name.dropLast(".weight".count)))
        }
    }
    /// Get-time derive with Load's packing rule, on the graph config.json describes (built lazily: no tensor is allocated).
    public static func conversion(recipe: String) -> NativeConversion {
        NativeConversion(recipe: recipe) { context, cancellation in
            let json = try PerceptionIO.json(context.source.appendingPathComponent("config.json"))
            let graph = LocateAnythingModel(try JSONDecoder().decode(LocateAnythingConfiguration.self, from: JSONEncoder().encode(json)))
            try context.convertTree(from: context.source, cancellation: cancellation, eligible: packingRule(graph))
            return try context.receipt(cancellation: cancellation)
        }
    }
    public func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operations.contains(operation), let session else { throw PerceptionIO.error("Unsupported operation or unloaded model.") }
        let image = try NativeMedia.image(PerceptionIO.path(input, "image"))
        let question = try Self.question(operation: operation, input: input)
        // The maker's predict() default is 2,048 tokens; its card suggests up to 8,192 against truncated lists.
        let maxTokens = Int(input["max_tokens"]?.number ?? 2048)
        guard (1...8192).contains(maxTokens) else { throw PerceptionIO.error("Token bound must be 1...8192.") }
        session.cancellation = cancellation
        let result: LocateAnythingResult
        do {
            result = try session.predict(image: image, question: question, maxNewTokens: maxTokens, mode: .hybrid)
        } catch let error as LocateAnythingOutputError {
            // Truncated or malformed generations are the parser's rejection of the model's output, not a runner fault.
            throw WorkerFailure(code: NativeOutputParsingFailure.code, message: error.localizedDescription)
        }
        try cancellation.check()
        return Self.output(result.boxes, operation: operation, width: image.width, height: image.height)
    }
    /// The operation's reply for parsed detections: normalized for `detect`, original pixels for `ground_screen`.
    static func output(_ boxes: [DetectedBox], operation: String, width: Int, height: Int) -> JSONObject {
        let pixels = operation == "ground_screen"
        let detections: [JSONValue] = boxes.prefix(1024).map { box in
            let values = pixels ? box.boxPixels : box.box
            func n(_ i: Int) -> JSONValue { .number(Double(max(0, values[i]))) }
            var entry: JSONObject = ["label": .string(box.label ?? "object")]
            if box.isPoint { entry["point"] = ["x": n(0), "y": n(1)] } else { entry["box"] = ["xmin": n(0), "ymin": n(1), "xmax": n(2), "ymax": n(3)] }
            return .object(entry)
        }
        var output: JSONObject = ["detections": .array(detections), "coordinate_system": .string(pixels ? "pixels" : "normalized")]
        if pixels { output["image_size"] = ["width": .number(Double(width)), "height": .number(Double(height))] }
        return output
    }
    static func question(operation: String, input: JSONObject) throws -> String {
        let phrase = input["phrase"]?.string ?? input["instruction"]?.string ?? input["labels"]?.array?.compactMap(\.string).joined(separator: ",")
        guard let phrase, !phrase.isEmpty, phrase.utf8.count <= 32768 else { throw PerceptionIO.error("Provide labels, a referring expression, or an instruction.") }
        let task: LocateAnythingTask
        if input["output_mode"]?.string == "points" {
            task = .pointing
        } else if operation == "ground_screen" {
            task = .gui
        } else if input["phrase"] != nil {
            task = .grounding
        } else if input["instruction"] != nil {
            task = .custom
        } else {
            task = .detection
        }
        return task.instruction(phrase)
    }
    public func unload() { model = nil; session = nil; if let scratch { try? FileManager.default.removeItem(at: scratch) }; scratch = nil }
    deinit { unload() }
}

extension NativeLocateAnything: NativeOutputParsing {
    /// The parser `run` applies after generation, for maker-documented `gate/parser-fixtures.json` cases. `raw` is the
    /// decoded generation with its special tokens kept (`<|im_end|>` included), as the maker's `generate` returns it; it
    /// is tokenized with the loaded tokenizer and goes through the same termination and grammar checks as served output.
    public func parseOutput(_ operation: String, input: JSONObject, raw: String) throws -> JSONObject {
        guard operations.contains(operation), let session else { throw PerceptionIO.error("Unsupported operation or unloaded model.") }
        let image = try NativeMedia.image(PerceptionIO.path(input, "image"))
        do {
            let boxes = try session.boxes(generated: session.tokens(decoded: raw), imageWidth: image.width, imageHeight: image.height)
            return Self.output(boxes, operation: operation, width: image.width, height: image.height)
        } catch let error as LocateAnythingOutputError {
            throw WorkerFailure(code: NativeOutputParsingFailure.code, message: error.localizedDescription)
        }
    }
}
