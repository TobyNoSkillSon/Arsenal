// SPDX-License-Identifier: MIT
import Foundation
import CoreImage
import PDFKit
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM
import IntegrationKit
import ArsenalWire

final class LanguageNative: NativeModel {
    private(set) var model: (any LanguageModel)?
    private var paddle: PaddleOCRVLModel?
    private var tokenizer: (any MLXLMCommon.Tokenizer)?
    private var controlTokens: [String] = []
    private var processor: (any UserInputProcessor)?
    private var eos = Set<Int>()
    private var id = ""
    private var maximumContext = 32768
    private var maxOutput = 4096
    private var spec: JSONObject = [:]
    /// The operation being served; it sizes the repetition guard.
    private var operation = ""
    /// Serving half: FP16 unless the folder declares a `[precision] dtype = "bf16"` exception (host `serving_dtype`).
    private var half: DType = .float16
    /// Graph parameter names, computed once per load for the per-tensor checkpoint map.
    private var parameterNames = Set<String>()
    /// Module paths of the graph's Linear and Embedding layers, the only owners the tier installer packs.
    private var packableOwners = Set<String>()
    private(set) var architecture = ""
    private(set) var operations = Set<String>()
    var peakBytes: Int { Memory.peakMemory }
    var parameters: [String: MLXArray] { Dictionary(uniqueKeysWithValues: graph?.parameters().flattened() ?? []) }
    private var graph: Module? { if let model { return model }; return paddle }
    static func eligible(_ key: String, _ shape: [Int]) -> Bool {
        key.hasSuffix(".weight") && shape.count == 2 && shape[1] % 64 == 0
            && !key.contains("position_embedding") && !key.contains("pos_embed")
    }
    /// Load's packing rule, on graph names: an eligible matrix owned by a Linear or Embedding.
    func packs(_ name: String, _ shape: [Int]) -> Bool {
        Self.eligible(name, shape) && packableOwners.contains(String(name.dropLast(7)))
    }
    /// Get's packing rule, on checkpoint names: exactly the matrices Load packs. Load's map drops what the graph does not
    /// use (Qwen3.5's `mtp.*` head, a tied `lm_head`, position ids); packing those left scales/biases that Load refused
    /// as "Orphan affine companions", so a dropped matrix stays half and Load discards it whole.
    func storesPacked(_ key: String, _ shape: [Int]) -> Bool {
        // A lazy placeholder: the map reads names and shapes only, so no tensor is materialized.
        guard shape.count == 2, let (name, mapped) = try? map(key, MLXArray.zeros(shape, dtype: half)) else { return false }
        return mapped.shape == shape && packs(name, shape)
    }
    /// Get-time derive that packs with the Load rule (`storesPacked`), from the same graph Load builds from config.json.
    static func conversion(recipe: String) -> NativeConversion {
        NativeConversion(recipe: recipe) { context, cancellation in
            guard let folder = context.folder else { throw WorkerFailure(code: "invalid_request", message: "Language conversion requires model_dir.") }
            let native = LanguageNative()
            native.half = context.half
            try native.makeGraph(weights: context.source, folder: folder, architecture: nil)
            defer { native.unload() }
            try context.convertTree(from: context.source, cancellation: cancellation, eligible: native.storesPacked)
            return try context.receipt(cancellation: cancellation)
        }
    }
    /// The weightless graph config.json describes, plus the folder spec and name sets the checkpoint map reads.
    @discardableResult
    private func makeGraph(weights: URL, folder: URL, architecture declared: String?) throws -> (data: Data, config: JSONObject) {
        let data = try Data(contentsOf: weights.appendingPathComponent("config.json"))
        let config = try WireJSON.decode(data)
        architecture = declared ?? config["model_type"]?.string ?? ""
        spec = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("runners/native/language.json")))
        if architecture == "hunyuan_v1_dense" {
            model = HunyuanModel(try JSONDecoder().decode(HunyuanConfiguration.self, from: data))
        } else if architecture == "paddleocr_vl" {
            paddle = PaddleOCRVLModel(config: try PaddleOCRVLConfig.load(from: weights))
        } else {
            model = try LanguageGraph.make(data, type: architecture)
        }
        guard let graph else { throw WorkerFailure(code: "unsupported_model", message: "Missing language graph.") }
        try LanguageGraph.removeUnusedSharedKV(graph, config: data)
        try VisionActivation.apply(graph, config: data, architecture: architecture)
        parameterNames = Set(graph.parameters().flattened().map(\.0))
        packableOwners = Set(graph.leafModules().flattened().compactMap { $0.1 is Linear || $0.1 is Embedding ? $0.0 : nil })
        return (data, config)
    }
    func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        try cancellation.check()
        // Worker start-up already caps MLX's buffer cache (NativeRuntime); lab tools that load directly get it here.
        Memory.cacheLimit = min(Memory.cacheLimit, NativeRuntime.cacheLimitBytes)
        id = context.folder.lastPathComponent
        half = context.precision == "bf16" || context.config["serving_dtype"]?.string == "bf16" ? .bfloat16 : .float16
        let (data, config) = try makeGraph(weights: context.weights, folder: context.folder, architecture: context.architecture)
        guard let graph else { throw WorkerFailure(code: "unsupported_model", message: "Missing language graph.") }
        operations = Set(spec["operations"]?.array?.compactMap(\.string) ?? [])
        maximumContext = Int(config["text_config"]?.object?["max_position_embeddings"]?.number ?? config["max_position_embeddings"]?.number ?? 32768)
        // A maker serving preset (`max_context`, e.g. Index-Translate's `--max-model-len 32768`) bounds prompt plus output
        // below the checkpoint's position limit, so one request cannot prefill the whole trained context.
        if let serving = spec["max_context"]?.number { maximumContext = min(maximumContext, Int(serving)) }
        maxOutput = Int(context.config["max_output_tokens"]?.number ?? Double(min(maximumContext, 32768)))
        if context.randomWeights && context.config["inventory_only"]?.bool == true {
            try NativeParameters.initializeRandom(graph, half: half)
            return
        }
        if context.randomWeights {
            tokenizer = RandomTokenizer(
                imageID: Int(config["image_token_id"]?.number ?? 260), startID: Int(config["vision_start_token_id"]?.number ?? 261),
                endID: Int(config["vision_end_token_id"]?.number ?? 262), imageToken: architecture.hasPrefix("lfm") ? "<image>" : "<|image_pad|>")
        } else {
            let local = try localAwait { try await LocalTokenizer(directory: context.weights) }
            tokenizer = local; controlTokens = local.controlTokens
        }
        if architecture != "hunyuan_v1_dense" && architecture != "paddleocr_vl", let tokenizer {
            let processorName = spec["processor"]?.string ?? ""
            var processorData: Data? = context.randomWeights ? try spec["processor_config"].map { try JSONEncoder().encode($0) } : nil
            for name in ["preprocessor_config.json", "processor_config.json"] {
                if let d = try? Data(contentsOf: context.weights.appendingPathComponent(name)) { processorData = d; break }
            }
            guard let processorData else { throw WorkerFailure(code: "missing_asset", message: "Missing local processor configuration.") }
            let preparedConfig = try processorConfiguration(processorData, model: config, random: context.randomWeights)
            switch architecture {
            case "qwen2_5_vl":
                processor = TeleOCRVLProcessor(try JSONDecoder().decode(TeleOCRVLProcessorConfiguration.self, from: preparedConfig), tokenizer: tokenizer)
            case "qwen3_5", "qwen3_vl":
                processor = QwenVisionProcessor(try JSONDecoder().decode(QwenVisionConfiguration.self, from: preparedConfig), tokenizer: tokenizer)
            case "lfm2_vl", "lfm2-vl":
                let imageToken = config["image_token_id"]?.number.map { Int($0) } ?? tokenizer.convertTokenToId("<image>")
                guard let imageToken else { throw WorkerFailure(code: "missing_asset", message: "LFM2-VL config names no image token.") }
                processor = LFMVisionProcessor(
                    try JSONDecoder().decode(LFMVisionConfiguration.self, from: preparedConfig), tokenizer: tokenizer, imageTokenID: imageToken)
            case "gemma4":
                processor = GemmaVisionProcessor(
                    try JSONDecoder().decode(GemmaVisionConfiguration.self, from: preparedConfig), tokenizer: tokenizer,
                    imageTokenID: Int(config["image_token_id"]?.number ?? 258_880), boiTokenID: Int(config["boi_token_id"]?.number ?? 255_999),
                    eoiTokenID: Int(config["eoi_token_id"]?.number ?? 258_882))
            default: throw WorkerFailure(code: "unsupported_model", message: "No processor for \(processorName).")
            }
        }
        let generation = (try? WireJSON.decode(Data(contentsOf: context.weights.appendingPathComponent("generation_config.json")))) ?? config
        if let value = generation["eos_token_id"]?.number { eos.insert(Int(value)) }
        for value in generation["eos_token_id"]?.array ?? [] { if let n = value.number { eos.insert(Int(n)) } }
        if let token = tokenizer?.eosToken, let value = tokenizer?.convertTokenToId(token) { eos.insert(value) }
        if context.randomWeights {
            try NativeParameters.initializeRandom(graph, half: half)
            var clipping: [String: MLXArray] = [:]
            for (key, value) in graph.parameters().flattened()
            where key.hasSuffix("input_min") || key.hasSuffix("output_min") || key.hasSuffix("input_max") || key.hasSuffix("output_max") {
                clipping[key] = MLX.full(value.shape, values: MLXArray(key.hasSuffix("min") ? Float(-65504) : Float(65504)), dtype: half)
            }
            if !clipping.isEmpty { try graph.update(parameters: ModuleParameters.unflattened(clipping), verify: .none) }
            if context.precision != "fp16" {
                // Serialize the small random graph once, then exercise the production streamed tier installer.
                let temp = context.weights.appendingPathComponent("random-half.safetensors")
                try MLX.save(arrays: parameters, url: temp)
                defer { try? FileManager.default.removeItem(at: temp) }
                _ = try NativePrecisionTier(context.precision, half: half).load(graph, archives: [temp], map: { ($0, $1) }, eligible: packs)
            }
        } else {
            let archives = try FileManager.default.contentsOfDirectory(at: context.weights, includingPropertiesForKeys: nil).filter { $0.pathExtension == "safetensors" }
            _ = try NativePrecisionTier(context.precision, half: half).load(
                graph, archives: archives,
                map: { name, tensor in
                    try cancellation.check(); return try self.map(name, tensor)
                }, eligible: packs)
        }
        parameterNames.removeAll(); packableOwners.removeAll()
        graph.train(false)
        try cancellation.check()
    }
    private func processorConfiguration(_ data: Data, model: JSONObject, random: Bool) throws -> Data {
        var value = try WireJSON.decode(data)
        if let image = value["image_processor"]?.object {
            value.merge(image) { _, imageValue in imageValue }
        }
        // Transformers 5 Qwen-VL processors spell min/max_pixels as size.shortest_edge/longest_edge; the pinned
        // Swift configurations read only the flat keys and would otherwise fall back to Qwen2-VL defaults.
        if ["qwen3_5", "qwen3_vl", "qwen2_5_vl"].contains(architecture), let size = value["size"]?.object {
            if value["min_pixels"] == nil, let shortest = size["shortest_edge"] { value["min_pixels"] = shortest }
            if value["max_pixels"] == nil, let longest = size["longest_edge"] { value["max_pixels"] = longest }
        }
        try Self.applyPixelLimits(&value, spec: spec, architecture: architecture)
        if random {
            for key in ["image_token_id", "boi_token_id", "eoi_token_id", "tile_size", "encoder_patch_size", "max_tiles", "downsample_factor"] {
                if let entry = model[key] { value[key] = entry }
            }
            value["size"] = ["shortest_edge": 1024, "longest_edge": 4096, "width": 64, "height": 64]
            value["min_pixels"] = 1024; value["max_pixels"] = 4096
            value["image_seq_length"] = 4
            // Gemma 4 counts soft tokens from its own patch and pooling geometry, which must be the synthetic vision
            // tower's (the checkpoint's processor_config agrees with its config.json the same way).
            if architecture == "gemma4", let vision = model["vision_config"]?.object {
                for key in ["patch_size", "pooling_kernel_size"] { if let entry = vision[key] { value[key] = entry } }
                value["max_soft_tokens"] = 70
            }
        }
        return try WireJSON.encode(value)
    }
    /// The folder's `min_pixels`/`max_pixels` narrow the checkpoint processor's own bounds, as a maker's pre-resize runs
    /// before the Transformers processor (Infinity's 2,048 floor stays under the processor's 65,536; Ovis's 448² raises
    /// it). LFM2-VL and Gemma 4 budget tokens, so the limits become token counts at their pixels-per-token.
    static func applyPixelLimits(_ value: inout JSONObject, spec: JSONObject, architecture: String) throws {
        let low = spec["min_pixels"]?.number, high = spec["max_pixels"]?.number
        switch architecture {
        case "qwen3_5", "qwen3_vl", "qwen2_5_vl":
            if let low { value["min_pixels"] = .number(max(low, value["min_pixels"]?.number ?? 0)) }
            if let high { value["max_pixels"] = .number(min(high, value["max_pixels"]?.number ?? high)) }
            if let fit = spec["scale_to_fit"] { value["arsenal_scale_to_fit"] = fit }
        case "lfm2_vl", "lfm2-vl":
            let patch = value["encoder_patch_size"]?.number ?? 16, factor = value["downsample_factor"]?.number ?? 2
            let perToken = patch * patch * factor * factor
            var most = value["max_image_tokens"]?.number ?? 256, least = value["min_image_tokens"]?.number ?? 64
            if let high { most = min(most, (high / perToken).rounded(.down)) }
            if let low { least = max(least, (low / perToken).rounded(.up)) }
            guard most >= 1 else { throw WorkerFailure(code: "unsupported_model", message: "max_pixels leaves no LFM2-VL image token.") }
            value["max_image_tokens"] = .number(most); value["min_image_tokens"] = .number(min(least, most))
        case "gemma4":
            guard let high else { return }
            let patch = value["patch_size"]?.number ?? 16, pool = value["pooling_kernel_size"]?.number ?? 3
            let current = value["max_soft_tokens"]?.number ?? 280
            // image_processing_gemma4.py accepts only these soft-token budgets.
            guard let budget = [70.0, 140, 280, 560, 1120].last(where: { $0 <= current && $0 * patch * patch * pool * pool <= high }) else {
                throw WorkerFailure(code: "unsupported_model", message: "max_pixels is below Gemma 4's smallest image budget.")
            }
            value["max_soft_tokens"] = .number(budget)
        default: break
        }
    }
    /// Gemma 4's clippable linears store activation bounds as scalars (`…input_min`, `…input_max`, `…output_min`,
    /// `…output_max`); an infinite or out-of-range bound only means "no clip", so it is clamped to ±65,504 instead of
    /// failing the non-finite half check. The pinned E4B checkpoint's 928 bounds were all finite (|x| ≤ 212) when read.
    static func boundedClipping(_ name: String, _ tensor: MLXArray) -> MLXArray {
        guard [".input_min", ".input_max", ".output_min", ".output_max"].contains(where: { name.hasSuffix($0) }), tensor.dtype.isFloatingPoint else {
            return tensor
        }
        return MLX.minimum(MLX.maximum(tensor.asType(.float32), MLXArray(Float(-65504))), MLXArray(Float(65504)))
    }
    private func map(_ name: String, _ tensor: MLXArray) throws -> (String, MLXArray)? {
        var tensor = Self.boundedClipping(name, tensor)
        // Cast before model-specific norm offsets so direct Load and Get-derived tiers share one half basis.
        tensor = tensor.dtype.isFloatingPoint ? tensor.asType(half) : tensor
        if let paddle { return paddle.mapCheckpoint(name, tensor) }
        if architecture.hasPrefix("lfm"), spec["inactive_parameters"]?.array?.contains(.string(name)) == true { return nil }
        guard let model else { return nil }
        if parameterNames.contains(name) { return (name, tensor) }
        let values = model.sanitize(weights: [name: tensor])
        guard values.count <= 1 else { throw WorkerFailure(code: "checkpoint_mapping", message: "Tensor needs a multi-output layout conversion: \(name).") }
        return values.first.map { ($0.key, $0.value) }
    }
    func unload() {
        model = nil; paddle = nil; parameterNames.removeAll(); packableOwners.removeAll()
        tokenizer = nil; controlTokens = []; processor = nil; eos.removeAll(); operations.removeAll(); Memory.clearCache()
    }

    func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        let result = try perform(operation, input: input, cancellation: cancellation, emit: emit)
        var envelope = result; envelope["id"] = .number(Double(Int32.max)); envelope["ok"] = true
        do { _ = try WireFrame.json(envelope).encoded() } catch { throw WorkerFailure(code: "output_limit", message: "Language output exceeds the worker JSON frame limit.") }
        return result
    }
    private func perform(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        try cancellation.check()
        guard operations.contains(operation), tokenizer != nil else { throw failure("Unsupported operation or unloaded model.") }
        self.operation = operation
        switch operation {
        case "count_tokens":
            guard let text = input["text"]?.string else { throw failure("count_tokens requires text.") }
            return ["tokens": .number(Double(tokenizer!.encode(text: text, addSpecialTokens: false).count))]
        case "translate":
            let texts = input["text"]?.array ?? []
            let imagePath = input["image"]?.object?["path"]?.string
            guard !texts.isEmpty || imagePath != nil, let target = input["target_lang"]?.string, !target.isEmpty else {
                throw failure("Translate requires text or image and target_lang.")
            }
            if let imagePath {
                guard id.hasPrefix("index-translate-") else { throw failure("This translator does not accept images.") }
                let prompt = try LanguagePrompts.imageTranslation(text: texts.compactMap(\.string).joined(separator: "\n"), target: target, input: input)
                let output = try generate(
                    messages: LanguagePrompts.textThenImage(prompt: prompt), images: [.ciImage(try image(imagePath))], extra: ["enable_thinking": false], input: input,
                    cancellation: cancellation, emit: emit)
                return [
                    "translations": [
                        ["text": .string(LanguagePrompts.indexAnswer(output.text)), "detected_source_language": "und", "finish_reason": .string(output.reason)]
                    ]
                ]
            }
            var result: [ArsenalWire.JSONValue] = []
            for item in texts {
                try cancellation.check()
                guard let text = item.string, !text.isEmpty else { throw failure("Translation text must be nonempty.") }
                let prompt = try LanguagePrompts.translation(id: id, text: text, target: target, input: input)
                let output = try generate(
                    messages: [["role": "user", "content": prompt]], images: [], extra: ["enable_thinking": false], input: input, cancellation: cancellation, emit: emit)
                result.append([
                    "text": .string(id.hasPrefix("index-translate-") ? LanguagePrompts.indexAnswer(output.text) : output.text),
                    "detected_source_language": input["source_lang"]?.string.flatMap { $0.lowercased() == "auto" ? nil : ArsenalWire.JSONValue.string($0) } ?? "und",
                    "finish_reason": .string(output.reason)
                ])
            }
            return ["translations": .array(result)]
        case "chat", "describe", "tag":
            var images: [UserInput.Image] = []
            let messages: [Message]
            if operation == "chat" {
                guard let raw = input["messages"]?.array, !raw.isEmpty else { throw failure("Chat requires messages.") }
                messages = try raw.map { item in
                    guard let object = item.object, let role = object["role"]?.string else { throw failure("Message requires role and content.") }
                    if let text = object["content"]?.string { return ["role": role, "content": text] }
                    guard let parts = object["content"]?.array else { throw failure("Message content is invalid.") }
                    var content: [[String: any Sendable]] = []
                    for part in parts {
                        if let text = part.object?["text"]?.string {
                            content.append(["type": "text", "text": text])
                        } else if let path = part.object?["image"]?.object?["path"]?.string {
                            images.append(.ciImage(try image(path))); content.append(["type": "image"])
                        } else if let url = part.object?["image_url"]?.object?["url"]?.string {
                            images.append(.ciImage(try inlineImage(url))); content.append(["type": "image"])
                        } else {
                            throw failure("Only text and image content are supported.")
                        }
                    }
                    return ["role": role, "content": content]
                }
            } else {
                guard let path = input["image"]?.object?["path"]?.string else { throw failure("An image is required.") }
                images = [.ciImage(try image(path))]
                messages = [
                    [
                        "role": "user",
                        "content": [
                            ["type": "image"],
                            [
                                "type": "text",
                                "text": input["prompt"]?.string ?? (operation == "tag" ? "List the visible objects and scene as comma-separated tags." : "Describe this image.")
                            ]
                        ]
                    ]
                ]
            }
            // Describe and tag stay deterministic unless the request samples, with at least a 1.05 repetition penalty
            // against the greedy loops small VLMs fall into.
            var decodingInput = input
            if operation != "chat", decodingInput["temperature"] == nil { decodingInput["temperature"] = 0 }
            let output = try generate(
                messages: messages, images: images, extra: ["enable_thinking": input["thinking"]?.bool ?? false], input: decodingInput,
                maxPixels: spec["operation_max_pixels"]?.object?[operation]?.number.map { Int($0) }, repetitionFloor: operation == "chat" ? nil : 1.05,
                splitThinking: true, cancellation: cancellation, emit: emit)
            if operation == "tag" || operation == "describe" { return try finished(output) { try parseOutput(operation, input: input, raw: output.text) } }
            var chat: JSONObject = [
                "text": .string(output.text), "finish_reason": .string(output.reason),
                "usage": ["prompt_tokens": .number(Double(output.prompt)), "completion_tokens": .number(Double(output.count))], "ttft_s": .number(output.first),
                "tokens_per_s": .number(output.rate)
            ]
            if let reasoning = output.reasoning { chat["reasoning"] = .string(reasoning) }
            return chat
        case "read_document":
            guard let path = input["document"]?.object?["path"]?.string else { throw failure("A document path is required.") }
            var output: [ArsenalWire.JSONValue] = [], files: [ArsenalWire.JSONValue] = []
            try eachDocumentPage(path: path, selected: input["pages"]?.array?.compactMap { $0.number.map(Int.init) }) { index, image in
                try cancellation.check()
                if orchestrated(input) { return output.append(try teleOCRPage(index: index, image: image, input: input, cancellation: cancellation, emit: emit)) }
                // TeleOCR's direct layout modes send the client's layout prompt with the 1036×1036 layout image.
                let mode = input["mode"]?.string ?? "ocr"
                let layout = id == "teleocr" && ["layout", "distorted_layout"].contains(mode)
                let prompt = try layout ? TeleDocument.layoutPrompt(distorted: mode == "distorted_layout") : LanguagePrompts.document(id: id, input: input, spec: spec)
                let result = try generate(
                    messages: LanguagePrompts.messages(prompt: prompt, system: spec["system"]?.string),
                    images: [.ciImage(layout ? TeleDocument.layoutImage(image) : modelInput(image, document: path))],
                    extra: ["enable_thinking": false, "mode": id == "nuextract3" ? "markdown" : "content"],
                    input: input, kind: TeleDocument.kind(mode: input["mode"]?.string ?? "ocr"), cancellation: cancellation, emit: emit)
                output.append(try documentPage(result.text, index: index, image: image, input: input, files: &files, reason: result.reason, cancellation: cancellation))
            }
            return try documentResult(output, files: files, input: input)
        case "extract":
            let mode = input["mode"]?.string ?? "structured"
            let template = input["template"]?.object
            guard mode != "structured" || template != nil else { throw failure("NuExtract structured mode requires a JSON template.") }
            // The host forwards the client's template text under ordered_json; its key order is the output's key order.
            let templateString = try input["ordered_json"]?.object?["template"]?.string ?? template.map { String(data: try WireJSON.encode($0), encoding: .utf8)! } ?? ""
            let examples = try NuExamples.prepare(input["examples"]?.array ?? [], template: templateString, loadImage: image)
            let text = input["text"]?.string ?? ""
            var images: [UserInput.Image] = []
            if let path = input["image"]?.object?["path"]?.string { images = [.ciImage(try image(path))] }
            if let path = input["document"]?.object?["path"]?.string {
                images = try extractionPages(
                    path: path, selected: input["pages"]?.array?.compactMap { $0.number.map(Int.init) }, otherImages: examples.images.count,
                    textTokens: tokenizer!.encode(text: text + templateString + (input["instructions"]?.string ?? ""), addSpecialTokens: false).count,
                    outputTokens: input["max_tokens"]?.number.map { Int($0) } ?? 1, cancellation: cancellation)
            }
            guard !text.isEmpty || !images.isEmpty else { throw failure("Extraction requires text, image or document.") }
            guard examples.images.count + images.count <= 99 else { throw failure("NuExtract takes at most 99 images per request, examples included.") }
            let content: [[String: any Sendable]] = images.map { _ in ["type": "image"] } + [["type": "text", "text": text]]
            let result = try generate(
                messages: examples.messages + [["role": "user", "content": content]], images: examples.images + images,
                extra: ["mode": mode, "template": templateString, "instructions": input["instructions"]?.string ?? "", "enable_thinking": input["thinking"]?.bool ?? false],
                input: input, cancellation: cancellation,
                emit: emit)
            return try finished(result) { try parseOutput("extract", input: input, raw: result.text) }
        case "ground_screen":
            guard let path = input["image"]?.object?["path"]?.string, let instruction = input["instruction"]?.string, !instruction.isEmpty else {
                throw failure("Grounding requires image and instruction.")
            }
            let image = try image(path)
            // GroundGUI trained one tool schema per platform (desktop, web, mobile); others have a single prompt.
            let system = input["platform"]?.string.flatMap { spec["systems"]?.object?[$0]?.string } ?? spec["system"]?.string
            let result = try generate(
                messages: LanguagePrompts.messages(
                    prompt: spec["ground_prompt"]?.string?.replacingOccurrences(
                        of: "{instruction}", with: instruction.hasSuffix(".") ? String(instruction.dropLast()) : instruction) ?? instruction, system: system),
                images: [.ciImage(image)], extra: ["enable_thinking": false], input: input,
                cancellation: cancellation, emit: emit)
            return try finished(result) { try LanguageOutput.ground(id: id, text: result.text, label: instruction, size: image.extent.size) }
        default: throw failure("Unsupported operation.")
        }
    }
    /// TeleOCR page modes: layout, crops and recognition generations; the page is truncated when any generation was, with
    /// `repetition` reported over `length`.
    private func teleOCRPage(
        index: Int, image: CIImage, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void
    ) throws -> ArsenalWire.JSONValue {
        var reason = "stop"
        let distorted = input["mode"]?.string == "distorted_document"
        var page =
            try TeleDocument.page(index: index, image: image, distorted: distorted, cancellation: cancellation) { prompt, crop, kind in
                let result = try generate(
                    messages: LanguagePrompts.messages(prompt: prompt, system: spec["system"]?.string), images: [.ciImage(crop)], extra: [:], input: input, kind: kind,
                    cancellation: cancellation, emit: emit)
                if result.reason == "repetition" || (result.reason == "length" && reason == "stop") { reason = result.reason }
                return result.text
            }.object ?? [:]
        Self.mark(&page, reason: reason)
        return .object(page)
    }
    struct Generated { let text: String; var reasoning: String?; let count: Int; let prompt: Int; let reason: String; let first: Double; let rate: Double }
    /// A parsed result carries its finish_reason; a parse that fails after the output budget ran out or a loop was
    /// stopped says so.
    private func finished(_ output: Generated, _ parse: () throws -> JSONObject) throws -> JSONObject {
        do {
            var result = try parse()
            result["finish_reason"] = .string(output.reason)
            return result
        } catch let failure as WorkerFailure where output.reason == "length" {
            throw WorkerFailure(
                code: failure.code, message: failure.message + " Generation stopped at the max_tokens limit (\(output.count) tokens); raise max_tokens.")
        } catch let failure as WorkerFailure where output.reason == "repetition" {
            throw WorkerFailure(code: failure.code, message: failure.message + " Generation stopped after \(output.count) tokens because the output was repeating.")
        }
    }
    /// A document page stopped early says so: `truncated` and the generation's `finish_reason` (`length` or `repetition`).
    static func mark(_ page: inout JSONObject, reason: String) {
        guard reason != "stop" else { return }
        page["truncated"] = true; page["finish_reason"] = .string(reason)
    }
    private func generate(
        messages: [Message], images: [UserInput.Image], extra: [String: any Sendable], input: JSONObject, maxPixels: Int? = nil, kind: String? = nil,
        repetitionFloor: Float? = nil, splitThinking: Bool = false, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void
    ) throws -> Generated {
        guard let tokenizer else { throw failure("Model is not loaded.") }
        try cancellation.check()
        // The request's (schema-defaulted) budget; without one, the folder's max_tokens or the model's output limit, then
        // whatever context the prompt leaves.
        let requested = input["max_tokens"]?.number.map { Int($0) }
        let budget = requested ?? spec["max_tokens"]?.number.map { Int($0) } ?? maxOutput
        guard budget > 0, budget <= maxOutput, budget <= maximumContext else { throw failure("Invalid output token budget.") }
        let start = ProcessInfo.processInfo.systemUptime
        if let paddle { return try paddleGenerate(paddle, messages: messages, images: images, budget: budget, fixed: requested != nil, cancellation: cancellation) }
        guard let model else { throw failure("Model is not loaded.") }
        var user = UserInput(messages: messages, images: images, additionalContext: extra)
        if let min = spec["min_pixels"]?.number { user.processing.minPixels = Int(min) }
        if let max = spec["max_pixels"]?.number { user.processing.maxPixels = Int(max) }
        // A per-operation cap only lowers the budget for this call.
        if let maxPixels { user.processing.maxPixels = min(user.processing.maxPixels ?? maxPixels, maxPixels) }
        let prepared: LMInput
        if let processor {
            prepared = try localAwait { try await processor.prepare(input: user) }
        } else {
            prepared = LMInput(tokens: MLXArray(try tokenizer.applyChatTemplate(messages: messages, tools: nil, additionalContext: extra)))
        }
        let promptCount = prepared.text.tokens.size
        let maxTokens = requested != nil ? budget : min(budget, maximumContext - promptCount)
        guard maxTokens > 0, promptCount + maxTokens <= maximumContext else { throw failure("Prompt and output exceed the model context.") }
        try cancellation.check()
        let configuration = ModelConfiguration(directory: URL(fileURLWithPath: "/"), eosTokenIds: eos)
        let context = ModelContext(configuration: configuration, model: model, processor: processor ?? StandInUserInputProcessor(), tokenizer: tokenizer)
        var generated: [Int] = []; var first = 0.0; var detokenizer = StreamDetokenizer(); var emissionFailure: Error?
        // Every task but free chat stops a looping output early (`finish_reason` "repetition").
        var loop = operation == "chat" ? nil : RepetitionGuard(span: RepetitionGuard.span(operation: operation)), looped = false
        let thinking = extra["enable_thinking"] as? Bool ?? false
        let decoding = LanguageDecoding.make(spec: spec, input: input, thinking: thinking, kind: kind, repetitionFloor: repetitionFloor)
        // Chat-style answers separate the model's thinking (Qwen <think>, Gemma thought channel) into reasoning.
        let thinkingSplitter = splitThinking ? ThinkingSplitter.make(architecture: architecture, thinking: thinking) : nil
        var streamSplitter = thinkingSplitter
        let skipSpecial = !(thinkingSplitter != nil && ThinkingSplitter.keepsSpecialTokens(architecture))
        // PrefillState: decode steps keep the prefill's multimodal RoPE state, which the pinned TokenIterator drops.
        let iterator = try TokenIterator(
            input: prepared, model: PrefillState(model), processor: decoding.processor(promptTokens: promptCount, maxTokens: maxTokens), sampler: decoding.sampler(),
            maxTokens: maxTokens)
        let result = MLXLMCommon.generate(input: prepared, context: context, iterator: iterator) { (tokens: [Int]) -> GenerateDisposition in
            generated = tokens
            if first == 0 { first = ProcessInfo.processInfo.systemUptime - start }
            if input["stream"]?.bool == true {
                // Deltas end on complete characters; a split multi-byte character waits for its last token.
                let delta = detokenizer.next(tokens) { tokenizer.decode(tokenIds: $0, skipSpecialTokens: skipSpecial) }
                if !delta.isEmpty {
                    var parts = [ThinkingSplitter.Part(reasoning: false, text: delta)]
                    if streamSplitter != nil { parts = streamSplitter!.append(delta) }
                    do { for frame in ThinkingSplitter.frames(parts) { try emit(frame) } } catch { emissionFailure = error; return .stop }
                }
            }
            if let token = tokens.last, loop?.append(token) == true { looped = true; return .stop }
            do { try cancellation.check(); return .more } catch { return .stop }
        }
        if let emissionFailure { throw emissionFailure }
        if input["stream"]?.bool == true, streamSplitter != nil { for frame in ThinkingSplitter.frames(streamSplitter!.finish()) { try emit(frame) } }
        try cancellation.check()
        let elapsed = ProcessInfo.processInfo.systemUptime - start
        var text = tokenizer.decode(tokenIds: generated, skipSpecialTokens: skipSpecial), reasoning: String?
        if let thinkingSplitter { (text, reasoning) = ThinkingSplitter.split(text, splitter: thinkingSplitter) }
        let reason = looped ? "repetition" : generated.count >= maxTokens ? "length" : "stop"
        Self.labRawLog(kind: kind, prompt: promptCount, count: generated.count, reason: reason, text: text)
        // Role markers and control tokens never reach an operation's output text (OutputControl).
        text = OutputControl.strip(text, controlTokens: controlTokens)
        return Generated(
            text: text, reasoning: reasoning, count: result.generationTokenCount, prompt: promptCount,
            reason: reason, first: first, rate: elapsed > 0 ? Double(generated.count) / elapsed : 0)
    }
    /// Lab-only (`ARSENAL_NATIVE_LAB=1`, `ARSENAL_LAB_RAW_LOG=1`): one stderr line per generation with its raw text and
    /// MLX memory, for live trials that must see what each layout and block call produced.
    private static func labRawLog(kind: String?, prompt: Int, count: Int, reason: String, text: String) {
        let environment = ProcessInfo.processInfo.environment
        guard environment["ARSENAL_NATIVE_LAB"] == "1", environment["ARSENAL_LAB_RAW_LOG"] == "1" else { return }
        let line: JSONObject = [
            "kind": .string(kind ?? "default"), "prompt_tokens": .number(Double(prompt)), "tokens": .number(Double(count)), "reason": .string(reason),
            "text": .string(text), "mlx_active_mb": .number(Double(Memory.activeMemory) / 1e6), "mlx_cache_mb": .number(Double(Memory.cacheMemory) / 1e6),
            "mlx_peak_mb": .number(Double(Memory.peakMemory) / 1e6)
        ]
        if let data = try? WireJSON.encode(line) { FileHandle.standardError.write(Data("LANGUAGE_RAW ".utf8) + data + Data("\n".utf8)) }
    }
    private func paddleGenerate(
        _ model: PaddleOCRVLModel, messages: [Message], images: [UserInput.Image], budget: Int, fixed: Bool, cancellation: NativeCancellation
    ) throws -> Generated {
        guard images.count == 1, case .ciImage(let image) = images[0], let tokenizer else { throw failure("PaddleOCR requires one image.") }
        let prompt = (messages.last?["content"] as? [[String: String]])?.last?["text"] ?? "OCR:"
        let spotting = prompt == "Spotting:"
        var prepared = image
        if spotting && image.extent.width < 1500 && image.extent.height < 1500 {
            // Maker's spotting pre-upscale, before the processor's dynamic resolution selection.
            prepared = image.clampedToExtent().applyingFilter("CILanczosScaleTransform", parameters: ["inputScale": 2.0, "inputAspectRatio": 1.0])
                .cropped(to: CGRect(x: image.extent.minX * 2, y: image.extent.minY * 2, width: image.extent.width * 2, height: image.extent.height * 2))
        }
        let reduced = model.config.textConfig.maxPositionEmbeddings < 10000
        let processed = PaddleOCRVLImageProcessor(
            dynamicResolution: true, patchSize: model.config.visionConfig.patchSize,
            minPixels: reduced ? 3136 : 112896, maxPixels: reduced ? 3136 : spotting ? 1605632 : 1003520
        ).process(prepared)
        let cfg = model.config
        // Use the checkpoint's own chat template, including task prompt and assistant prefix.
        var ids = try tokenizer.applyChatTemplate(messages: LanguagePrompts.messages(prompt: prompt, system: nil), tools: nil, additionalContext: [:])
        let marker = tokenizer.convertTokenToId("<|IMAGE_PLACEHOLDER|>") ?? cfg.visionTokenId
        guard let imageStart = ids.firstIndex(of: marker) else { throw failure("Paddle template omitted its image placeholder.") }
        ids.replaceSubrange(imageStart...imageStart, with: Array(repeating: cfg.visionTokenId, count: processed.numImageTokens))
        let (positions, nextPosition) = PaddlePositions.imagePrompt(
            count: ids.count, imageStart: imageStart, height: processed.height / cfg.visionConfig.patchSize / 2, width: processed.width / cfg.visionConfig.patchSize / 2)
        let maxTokens = fixed ? budget : min(budget, maximumContext - ids.count)
        guard maxTokens > 0, ids.count + maxTokens <= maximumContext else { throw failure("Prompt exceeds context.") }
        let start = ProcessInfo.processInfo.systemUptime, cache = model.newCache()
        var logits = model.forward(
            inputIds: MLXArray(ids).reshaped(1, -1), pixelValues: processed.pixelValues, cache: cache, positions: MLXArray(positions.flatMap { $0 }).reshaped(3, -1))
        var output: [Int] = []; var first = 0.0
        for _ in 0..<maxTokens {
            try cancellation.check()
            let token = argMax(logits[0, -1], axis: -1).item(Int.self)
            if first == 0 { first = ProcessInfo.processInfo.systemUptime - start }
            if eos.contains(token) { break }
            output.append(token)
            logits = model.forwardGeneration(
                inputIds: MLXArray([token]).reshaped(1, 1), cache: cache, positions: MLX.full([3, 1], values: MLXArray(nextPosition + output.count - 1)))
        }
        let elapsed = ProcessInfo.processInfo.systemUptime - start
        return Generated(
            text: tokenizer.decode(tokenIds: output, skipSpecialTokens: true), count: output.count, prompt: ids.count, reason: output.count == maxTokens ? "length" : "stop",
            first: first, rate: Double(output.count) / max(elapsed, 0.000001))
    }
    /// Chandra's load_image (chandra/input.py) upscales image files, not PDF pages, whose short side is below
    /// MIN_IMAGE_DIM, with Lanczos and truncated sizes. Parsed coordinates still use the original page size.
    private func modelInput(_ image: CIImage, document path: String) -> CIImage {
        guard let side = spec["image_min_short_side"]?.number, URL(fileURLWithPath: path).pathExtension.lowercased() != "pdf" else { return image }
        let rgb = ImagePixels.rgb(image), short = Double(min(rgb.width, rgb.height))
        guard short < side else { return image }
        let scale = side / short
        return ImagePixels.ciImage(PILResample.resize(rgb, width: Int(Double(rgb.width) * scale), height: Int(Double(rgb.height) * scale), filter: .lanczos))
    }
    private func inlineImage(_ value: String) throws -> CIImage {
        guard value.hasPrefix("data:image/"), let delimiter = value.range(of: ";base64,"), let data = Data(base64Encoded: String(value[delimiter.upperBound...])),
            data.count <= 16 * 1024 * 1024,
            let image = CIImage(data: data, options: [.applyOrientationProperty: true]), image.extent.width > 0, image.extent.height > 0,
            image.extent.width * image.extent.height <= 16_000_000
        else { throw failure("Chat image_url requires a bounded inline base64 image.") }
        return image
    }
    private func image(_ path: String) throws -> CIImage {
        _ = try NativeMedia.image(URL(fileURLWithPath: path))
        guard let value = CIImage(contentsOf: URL(fileURLWithPath: path), options: [.applyOrientationProperty: true]) else { throw failure("Image cannot be decoded.") }
        return value
    }
    /// NuExtract document pages: at most 99 images per request (the card's vLLM limit), checked from the page count
    /// before rendering, and an estimate of the image tokens checked page by page before the processor runs.
    private func extractionPages(
        path: String, selected: [Int]?, otherImages: Int, textTokens: Int, outputTokens: Int, cancellation: NativeCancellation
    ) throws -> [UserInput.Image] {
        var count = selected?.count ?? 1
        if selected == nil, URL(fileURLWithPath: path).pathExtension.lowercased() == "pdf", let pdf = PDFDocument(url: URL(fileURLWithPath: path)) { count = pdf.pageCount }
        guard otherImages + count <= 99 else {
            throw failure("NuExtract takes at most 99 images per request; this document has \(count) pages. Choose pages with `pages`.")
        }
        var pages: [UserInput.Image] = [], estimate = textTokens
        try eachDocumentPage(path: path, selected: selected) { index, page in
            try cancellation.check()
            estimate += imageTokens(page.extent.size)
            guard estimate + outputTokens <= maximumContext else {
                throw failure("The selected pages need about \(estimate) prompt tokens by page \(index + 1); the model context is \(maximumContext). Choose fewer pages.")
            }
            pages.append(.ciImage(page))
        }
        return pages
    }
    /// Qwen-VL smart_resize estimate: the pixel count clamped to the processor's range, one token per merged patch.
    private func imageTokens(_ size: CGSize) -> Int {
        let config = spec["processor_config"]?.object ?? [:]
        let image = config["image_processor"]?.object ?? config
        let patch = image["patch_size"]?.number ?? 16, merge = image["merge_size"]?.number ?? 2
        let most = spec["max_pixels"]?.number ?? image["max_pixels"]?.number ?? image["size"]?.object?["longest_edge"]?.number ?? 16_777_216
        let least = spec["min_pixels"]?.number ?? image["min_pixels"]?.number ?? image["size"]?.object?["shortest_edge"]?.number ?? 65_536
        let pixels = min(max(Double(size.width * size.height), least), most)
        return Int((pixels / ((patch * merge) * (patch * merge))).rounded(.up)) + 2
    }
    private func document(path: String, selected: [Int]?) throws -> [(Int, CIImage)] {
        var result: [(Int, CIImage)] = []
        try eachDocumentPage(path: path, selected: selected) { result.append(($0, $1)) }
        return result
    }
    private func eachDocumentPage(path: String, selected: [Int]?, visit: (Int, CIImage) throws -> Void) throws {
        if URL(fileURLWithPath: path).pathExtension.lowercased() != "pdf" { try visit(0, image(path)); return }
        guard let pdf = PDFDocument(url: URL(fileURLWithPath: path)), pdf.pageCount > 0, pdf.pageCount <= 1024 else { throw failure("PDF is invalid or exceeds 1024 pages.") }
        for index in selected ?? Array(0..<pdf.pageCount) {
            try autoreleasepool {
                guard index >= 0, index < pdf.pageCount, let page = pdf.page(at: index) else { throw failure("Invalid PDF page index.") }
                let bounds = page.bounds(for: .mediaBox)
                guard bounds.width > 0, bounds.height > 0 else { throw failure("PDF page has no drawable extent.") }
                let scale = Self.pdfScale(page: bounds.size, spec: spec)
                let thumbnail = page.thumbnail(of: CGSize(width: bounds.width * scale, height: bounds.height * scale), for: .mediaBox)
                guard let tiff = thumbnail.tiffRepresentation, let ci = CIImage(data: tiff) else { throw failure("PDF page cannot be rendered.") }
                try visit(index, ci)
            }
        }
    }
    private func failure(_ message: String) -> WorkerFailure { WorkerFailure(code: "invalid_request", message: message) }
    /// Points-to-pixels factor for a PDF page: the folder's `pdf_dpi` (maker render dpi; 144 when the maker sets none),
    /// raised so the short side reaches `pdf_min_short_side` (Chandra's MIN_PDF_IMAGE_DIM), then bounded to a long side of
    /// 8192 px and 16,777,216 px, the largest input any language processor accepts.
    static func pdfScale(page: CGSize, spec: JSONObject) -> CGFloat {
        var scale = CGFloat(spec["pdf_dpi"]?.number ?? 144) / 72
        if let side = spec["pdf_min_short_side"]?.number { scale = max(scale, CGFloat(side) / min(page.width, page.height)) }
        return min(scale, 8192 / max(page.width, page.height), (16_777_216 / (page.width * page.height)).squareRoot())
    }

    /// TeleOCR's page modes chain layout, crop and recognition generations (TeleDocument).
    private func orchestrated(_ input: JSONObject) -> Bool { id == "teleocr" && ["ocr", "document", "distorted_document"].contains(input["mode"]?.string ?? "ocr") }
    /// One generated page through the model's document parser; `run` and `parse_output` share it.
    private func documentPage(
        _ raw: String, index: Int, image: CIImage, input: JSONObject, files: inout [ArsenalWire.JSONValue], reason: String = "stop",
        cancellation: NativeCancellation
    ) throws -> ArsenalWire.JSONValue {
        let mode = input["mode"]?.string ?? "ocr"
        let text = id == "teleocr" && ["table", "scientific_figure"].contains(mode) ? try OTSLOutput.html(raw) : raw
        guard var page = try LanguageOutput.document(id: id, index: index, text: text, size: image.extent.size, mode: mode).object else {
            throw failure("Document page result is not an object.")
        }
        // Truncation is never silent: a page cut off at max_tokens or at a loop says so.
        Self.mark(&page, reason: reason)
        if id == "ovisocr2", !(page["images"]?.array ?? []).isEmpty {
            guard let directory = input["out_dir"]?.string else { throw failure("Figure output requires the host job directory.") }
            files += try DocumentRegions.figures(page: &page, image: image, root: URL(fileURLWithPath: directory), cancellation: cancellation)
        }
        return .object(page)
    }
    private func documentResult(_ pages: [ArsenalWire.JSONValue], files: [ArsenalWire.JSONValue], input: JSONObject) throws -> JSONObject {
        guard id == "ovisocr2" else { return ["pages": .array(pages)] }
        guard let directory = input["out_dir"]?.string else { throw failure("Document files require the host job directory.") }
        let root = URL(fileURLWithPath: directory), file = root.appendingPathComponent("document.md")
        guard !FileManager.default.fileExists(atPath: file.path) else { throw failure("Document output already exists.") }
        try Data(pages.compactMap { $0.object?["markdown"]?.string }.joined(separator: "\n\n").utf8).write(to: file, options: .withoutOverwriting)
        let markdown = try NativeMedia.file(file, root: root, mime: "text/markdown", id: "document")
        return ["pages": .array(pages), "files": .array([ArsenalWire.JSONValue.object(markdown)] + files)]
    }
}

extension LanguageNative: NativeOutputParsing {
    /// The parsers `run` applies to generated text, exposed for maker-documented `gate/parser-fixtures.json` cases.
    func parseOutput(_ operation: String, input: JSONObject, raw: String) throws -> JSONObject {
        guard operations.contains(operation) else { throw WorkerFailure(code: "unsupported_op", message: "The loaded model does not support \(operation).") }
        switch operation {
        case "extract":
            return ["data": try LanguageOutput.json(raw)]
        case "ground_screen":
            guard let path = input["image"]?.object?["path"]?.string, let instruction = input["instruction"]?.string, !instruction.isEmpty else {
                throw failure("Grounding requires image and instruction.")
            }
            return try LanguageOutput.ground(id: id, text: raw, label: instruction, size: image(path).extent.size)
        case "tag":
            return ["tags": .array(raw.split(separator: ",").map { .string($0.trimmingCharacters(in: .whitespacesAndNewlines)) })]
        case "describe":
            return ["text": .string(raw)]
        case "read_document":
            guard let path = input["document"]?.object?["path"]?.string else { throw failure("A document path is required.") }
            guard !orchestrated(input) else { throw failure("TeleOCR page modes combine several generations; parser fixtures cover its single-generation modes.") }
            let selected = input["pages"]?.array?.first?.number.map { [Int($0)] }
            guard let page = try document(path: path, selected: selected).first else { throw failure("The document has no page to parse.") }
            var files: [ArsenalWire.JSONValue] = []
            let parsed = try documentPage(raw, index: page.0, image: page.1, input: input, files: &files, cancellation: NativeCancellation())
            return try documentResult([parsed], files: files, input: input)
        default:
            throw WorkerFailure(code: "unsupported_op", message: "\(operation) returns generated text without a structured parser.")
        }
    }
}
