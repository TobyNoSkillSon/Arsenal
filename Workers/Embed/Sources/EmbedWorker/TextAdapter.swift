// swiftlint:disable force_cast shorthand_operator
// SPDX-License-Identifier: MIT
import Foundation
import Hub
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire
import Tokenizers

final class TextRetrievalAdapter: NativeModel {
    var architecture = "qwen3"
    var operations: Set<String> { ranking ? ["rerank"] : ["embed"] }
    var peakBytes: Int { Memory.peakMemory }
    var parameters: [String: MLXArray] { graph.map { Dictionary(uniqueKeysWithValues: $0.parameters().flattened()) } ?? [:] }
    private var graph: Module?
    private var tokenizer: (any Tokenizers.Tokenizer)?
    private var ranking = false
    private var jina = false
    private var harrier = false
    /// Qwen3-Embedding: the maker's tokenizer appends <|endoftext|>, the token its last-token pooling reads.
    private var qwenEmbedding = false
    private var random = false
    private var vocabulary = 0
    private var width = 0
    private(set) var limit = 32768
    private var windows: [Bool] = []
    private var slidingWindow = 1024

    func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        try cancellation.check()
        let data = try Data(contentsOf: context.weights.appendingPathComponent("config.json"))
        let config = try WireJSON.decode(data)
        architecture = config["model_type"]?.string ?? ""
        random = context.randomWeights
        jina = config["architectures"]?.array?.contains(.string("JinaForRanking")) == true
        let manifest = try String(contentsOf: context.folder.appendingPathComponent("model.toml"), encoding: .utf8)
        let expression = try NSRegularExpression(pattern: #"(?m)^id\s*=\s*"([^"]+)"\s*$"#)
        guard let match = expression.firstMatch(in: manifest, range: NSRange(manifest.startIndex..., in: manifest)), let range = Range(match.range(at: 1), in: manifest) else {
            throw invalid("Model manifest has no retrieval model ID.")
        }
        let modelID = String(manifest[range])
        harrier = modelID.hasPrefix("harrier")
        qwenEmbedding = modelID.hasPrefix("qwen3-embedding")
        ranking = jina || modelID.contains("reranker")
        let args = try JSONDecoder().decode(EmbedQwenConfiguration.self, from: data)
        width = args.hiddenSize; vocabulary = args.vocabularySize
        // Qwen3-Reranker's card fixes `max_length = 8192` for the whole prompt (prefix, pair and suffix), below its
        // 40,960 position table; the embedders and Jina keep their own context caps.
        limit = min(args.maxPositionEmbeddings, jina ? 131072 : (ranking ? Self.qwenRerankLength : 32768))
        windows = config["layer_types"]?.array?.map { $0.string == "sliding_attention" } ?? []
        slidingWindow = Int(config["sliding_window"]?.number ?? 1024)
        guard !jina || windows.count == args.hiddenLayers, slidingWindow > 0 else { throw invalid("Jina requires its per-layer attention layout.") }
        let module: Module = RetrievalGraph(args, ranking: ranking, jina: jina)
        graph = module
        let half: DType = .float16
        if random {
            try NativeParameters.initializeRandom(module, half: half)
            if context.precision == "int8" || context.precision == "int4" {
                // Exercise the real streamed loader from a temporary synthetic half archive.
                let temp = context.weights.appendingPathComponent("random.safetensors")
                try MLX.save(arrays: Dictionary(uniqueKeysWithValues: module.parameters().flattened()), url: temp)
                defer { try? FileManager.default.removeItem(at: temp) }
                _ = try NativePrecisionTier(context.precision, half: half).load(module, archives: [temp], map: { ($0, $1) }, eligible: Self.eligible)
            }
        } else {
            let files = try FileManager.default.contentsOfDirectory(at: context.weights, includingPropertiesForKeys: nil).filter { $0.pathExtension == "safetensors" }
            _ = try NativePrecisionTier(context.precision, half: half).load(
                module, archives: files,
                map: { key, value in
                    try cancellation.check()
                    var name = key
                    if ["linear1", "linear2"].contains(key.components(separatedBy: ".")[0]) { name = "projector." + key }
                    name = name.replacingOccurrences(of: "projector.0.", with: "projector.linear1.").replacingOccurrences(of: "projector.2.", with: "projector.linear2.")
                    if !name.hasPrefix("model.") && !name.hasPrefix("projector.") && !name.hasPrefix("lm_head.") { name = "model." + name }
                    return (name, value)
                }, eligible: Self.eligible)
            let configJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: context.weights.appendingPathComponent("tokenizer_config.json"))) as? [NSString: Any]
            let tokenJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: context.weights.appendingPathComponent("tokenizer.json"))) as? [NSString: Any]
            guard let configJSON, let tokenJSON else { throw invalid("Missing local tokenizer files.") }
            tokenizer = try AutoTokenizer.from(tokenizerConfig: Config(configJSON), tokenizerData: Config(tokenJSON))
        }
        module.train(false)
        try NativeParameters.requireHalf(parameters)
    }

    static func eligible(_ key: String, _ shape: [Int]) -> Bool {
        key.hasSuffix(".weight") && shape.count == 2 && shape[1] % 64 == 0
    }

    private func encode(_ text: String, special: Bool = false) -> [Int] {
        if let tokenizer { return tokenizer.encode(text: text, addSpecialTokens: special) }
        // A small deterministic vocabulary for random-graph tests; never used on a serving checkpoint.
        precondition(random)
        let markers = ["<|embed_token|>", "<|rerank_token|>", "<|score_token|>"]
        var result: [Int] = []; var remaining = text[...]
        while !remaining.isEmpty {
            if let i = markers.firstIndex(where: { remaining.hasPrefix($0) }) {
                result.append(vocabulary - 1 - i); remaining = remaining.dropFirst(markers[i].count)
            } else {
                result.append(3 + Int(remaining.removeFirst().unicodeScalars.first?.value ?? 0) % (vocabulary - 8))
            }
        }
        return result.isEmpty ? [1] : result
    }
    private func token(_ text: String) throws -> Int {
        if let tokenizer, let id = tokenizer.convertTokenToId(text) { return id }
        if random {
            if text == "yes" { return 1 }; if text == "no" { return 2 }
            return vocabulary - 1 - (["<|embed_token|>", "<|rerank_token|>", "<|score_token|>"].firstIndex(of: text) ?? 0)
        }
        throw invalid("Tokenizer lacks required token: \(text)")
    }
    private func decode(_ ids: [Int]) -> String {
        tokenizer?.decode(tokens: ids, skipSpecialTokens: false) ?? String(repeating: "x", count: ids.count)
    }
    private func invalid(_ message: String) -> WorkerFailure { WorkerFailure(code: "invalid_request", message: message) }
    func unload() { graph = nil; tokenizer = nil; windows = [] }

    func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        try cancellation.check()
        guard let graph else { throw WorkerFailure(code: "not_loaded", message: "Load a model first.") }
        if operation == "embed", !ranking { return try embed(graph, input: input, cancellation: cancellation) }
        guard operation == "rerank", ranking else { throw invalid("Unsupported retrieval operation.") }
        guard let query = input["query"]?.string, !query.isEmpty, let values = input["documents"]?.array,
            !values.isEmpty, values.count <= 1024, values.allSatisfy({ $0.string != nil })
        else { throw invalid("Rerank requires a query and 1–1024 text documents.") }
        let documents = values.compactMap(\.string)
        let top = min(documents.count, try integer(input["top_n"], default: documents.count, minimum: 1, maximum: 1024))
        let maxDoc = try integer(input["max_tokens_per_doc"], default: jina ? 8192 : limit, minimum: 1, maximum: jina ? 8192 : limit)
        let scores: [Double]
        var projectedDocuments: [[Float]]?
        if jina {
            let ranked = try jinaScores(graph as! RetrievalGraph, query: query, documents: documents, maxDoc: maxDoc, cancellation: cancellation)
            scores = ranked.scores; projectedDocuments = ranked.embeddings
        } else {
            let instruction = input["instruction"]?.string ?? Self.defaultInstruction
            let prefix = encode(Self.qwenPrefix)
            let suffix = encode(Self.suffix)
            let yes = try token("yes"), no = try token("no")
            let body = "<Instruct>: \(instruction)\n<Query>: \(query)\n<Document>: "
            let leading = encode(body).count
            scores = try documents.map { document in
                try cancellation.check()
                let ids = try Self.qwenRerankIDs(
                    prefix: prefix, leading: leading, pair: encode(body + document), suffix: suffix, maxDocument: maxDoc, limit: limit)
                let model = graph as! RetrievalGraph
                let hidden = try model.backbone(MLXArray(ids).reshaped(1, -1), cancellation: cancellation)
                let logits = model.logits(hidden[0, ids.count - 1])[MLXArray([no, yes])].asType(.float32)
                let probability = softmax(logits, axis: -1)[1].item(Float.self)
                guard probability.isFinite else { throw invalid("Non-finite reranking score.") }
                return Double(probability)
            }
        }
        guard scores.count == documents.count, scores.allSatisfy(\.isFinite) else { throw invalid("Non-finite or incomplete reranking output.") }
        let indices = scores.indices.sorted { scores[$0] == scores[$1] ? $0 < $1 : scores[$0] > scores[$1] }.prefix(top)
        return [
            "results": .array(
                indices.map { index in
                    var item: JSONObject = ["index": .number(Double(index)), "relevance_score": .number(jina ? (1 + scores[index]) / 2 : scores[index])]
                    if input["return_embeddings"]?.bool == true, let projectedDocuments {
                        item["embedding"] = .array(projectedDocuments[index].map { .number(Double($0)) })
                    }
                    return .object(item)
                })
        ]
    }

    private func integer(_ value: JSONValue?, default fallback: Int, minimum: Int, maximum: Int) throws -> Int {
        guard let value else { return fallback }
        guard let number = value.number, number.isFinite, number.rounded() == number, number >= Double(minimum), number <= Double(maximum) else {
            throw invalid("Integer option outside its supported range.")
        }
        return Int(number)
    }
    private func embed(_ graph: Module, input: JSONObject, cancellation: NativeCancellation) throws -> JSONObject {
        guard let values = input["input"]?.array, !values.isEmpty, values.count <= 2048 else { throw invalid("Embed requires a nonempty input array.") }
        let task = input["task"]?.string ?? "document"
        guard ["query", "document", "similarity", "classification", "clustering"].contains(task) else { throw invalid("Unsupported embedding task.") }
        let dimension = try integer(input["dimensions"], default: width, minimum: harrier ? width : 32, maximum: width)
        let normalize = input["normalize"]?.bool ?? true
        var vectors: [JSONValue] = []; var tokens = 0
        for value in values {
            try cancellation.check()
            guard let text = value.string ?? (value["type"]?.string == "text" ? value["text"]?.string : nil) else { throw invalid("This model accepts text inputs only.") }
            guard !text.isEmpty else { throw invalid("Embedding text must be nonempty.") }
            let prompt = Self.embeddingPrompt(text, task: task, instruction: input["instruction"]?.string, qwen: qwenEmbedding)
            var ids = encode(prompt, special: qwenEmbedding)
            guard !ids.isEmpty else { throw invalid("Tokenizer produced no input tokens.") }
            if ids.count > limit {
                guard input["truncate"]?.bool == true else { throw invalid("Input exceeds the model's \(limit)-token context; enable truncate.") }
                ids = Self.truncated(ids, limit: limit, keepingLast: qwenEmbedding)
            }
            tokens += ids.count
            let model = graph as! RetrievalGraph
            let hidden = try model.backbone(MLXArray(ids).reshaped(1, -1), cancellation: cancellation)
            var vector = hidden[0, ids.count - 1]
            vector = vector[..<dimension].asType(.float32)
            if normalize { vector = vector / maximum(sqrt(sum(vector * vector)), MLXArray(1e-12)) }
            let row = vector.asArray(Float.self)
            guard row.count == dimension, row.allSatisfy(\.isFinite) else { throw invalid("Non-finite embedding.") }
            vectors.append(.array(row.map { .number(Double($0)) }))
        }
        return ["embeddings": .array(vectors), "input_tokens": .number(Double(tokens))]
    }

    /// An over-long input cut to `limit` ids. Qwen3-Embedding pools the last position, the <|endoftext|> its tokenizer
    /// appends after truncating, so that id stays at the end; the other embedders keep their first `limit` ids.
    static func truncated(_ ids: [Int], limit: Int, keepingLast: Bool) -> [Int] {
        guard ids.count > limit, let last = ids.last else { return ids }
        return keepingLast ? Array(ids.prefix(limit - 1)) + [last] : Array(ids.prefix(limit))
    }

    static let qwenRerankLength = 8192
    /// The card's `process_inputs`: the formatted pair is tokenized as one string (so the document's first token merges
    /// with the space after "<Document>:" exactly as in the reference), cut from the end to `limit` minus the prefix and
    /// suffix, then wrapped. `maxDocument` additionally bounds the document's share; an instruction and query that leave
    /// no room for the document fail rather than lose the query.
    static func qwenRerankIDs(prefix: [Int], leading: Int, pair: [Int], suffix: [Int], maxDocument: Int, limit: Int) throws -> [Int] {
        let room = limit - prefix.count - suffix.count
        guard leading < room else { throw WorkerFailure(code: "invalid_request", message: "Query and instruction exceed model context.") }
        return prefix + pair.prefix(min(room, leading + maxDocument)) + suffix
    }
    static let defaultInstruction = "Given a web search query, retrieve relevant passages that answer the query"
    static let qwenPrefix =
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    static let suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    /// `qwen`: Qwen3-Embedding's card writes "Query:" with no space before the query.
    static func embeddingPrompt(_ text: String, task: String, instruction: String?, qwen: Bool = false) -> String {
        if task == "document" { return text }
        let name =
            instruction ?? [
                "query": defaultInstruction, "similarity": "Retrieve semantically similar text", "classification": "Classify the text",
                "clustering": "Identify the topic of the text"
            ][task] ?? defaultInstruction
        return "Instruct: \(name)\nQuery:" + (qwen ? "" : " ") + text
    }
    private func jinaScores(_ graph: RetrievalGraph, query: String, documents: [String], maxDoc: Int, cancellation: NativeCancellation) throws -> (
        scores: [Double], embeddings: [[Float]]
    ) {
        func clean(_ text: String) -> String {
            ["<|embed_token|>", "<|rerank_token|>", "<|score_token|>"].reduce(text) { $0.replacingOccurrences(of: $1, with: "") }
        }
        let queryIDs = encode(clean(query))
        let query = queryIDs.count >= 1984 ? decode(Array(queryIDs.prefix(1984))) : clean(query)
        let docs = documents.map { text -> (String, Int) in
            let text = clean(text), ids = encode(text)
            return ids.count >= maxDoc ? (decode(Array(ids.prefix(maxDoc - 1))), maxDoc - 1) : (text, ids.count)
        }
        var blocks: [[String]] = [], block: [String] = [], count = encode(query).count
        for (text, length) in docs {
            block.append(text); count += length
            if block.count >= 125 || limit - count < maxDoc || count >= limit - maxDoc {
                blocks.append(block); block = []; count = encode(query).count
            }
        }
        if !block.isEmpty { blocks.append(block) }
        let docToken = try token("<|embed_token|>"), queryToken = try token("<|rerank_token|>")
        var allDocuments: [[Float]] = [], queries: [[Float]] = [], weights: [Double] = []
        for block in blocks {
            try cancellation.check()
            let ids = encode(Self.jinaPrompt(query: query, documents: block))
            guard ids.count <= limit else { throw invalid("Formatted Jina block exceeds model context.") }
            let docPositions = ids.indices.filter { ids[$0] == docToken }, queryPositions = ids.indices.filter { ids[$0] == queryToken }
            guard docPositions.count == block.count, queryPositions.count == 2, let projector = graph.projector else {
                throw invalid("Jina marker count does not match query/documents.")
            }
            let hidden = try graph.backbone(MLXArray(ids).reshaped(1, -1), cancellation: cancellation, windows: windows, window: slidingWindow)[0]
            let projected = projector(hidden[MLXArray(docPositions + [queryPositions[1]])]).asType(.float32)
            let rows = projected.asArray(Float.self).chunked(width: 512)
            guard rows.count == block.count + 1, rows.allSatisfy({ $0.count == 512 && $0.allSatisfy(\.isFinite) }) else {
                throw invalid("Non-finite or incomplete Jina projection.")
            }
            let queryRow = rows.last!
            let docRows = Array(rows.dropLast())
            let weight = docRows.map { (1 + Self.cosine($0, queryRow)) / 2 }.max() ?? 0
            allDocuments.append(contentsOf: docRows); queries.append(queryRow); weights.append(weight)
        }
        let total = weights.reduce(0, +)
        guard total > 0 else { throw invalid("Jina query fusion has zero weight.") }
        let fused = (0..<512).map { column in Float(zip(queries, weights).reduce(0.0) { $0 + Double($1.0[column]) * $1.1 } / total) }
        return (allDocuments.map { Self.cosine($0, fused) }, allDocuments)
    }
    static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        let dot = zip(a, b).reduce(0.0) { $0 + Double($1.0) * Double($1.1) }
        let aa = a.reduce(0.0) { $0 + Double($1) * Double($1) }, bb = b.reduce(0.0) { $0 + Double($1) * Double($1) }
        return max(-1, min(1, dot / max(1e-12, sqrt(aa * bb))))
    }
    static func jinaPrompt(query: String, documents: [String]) -> String {
        let system =
            "<|im_start|>system\nYou are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. If the query is a question, how relevant a passage is depends on how well it answers the question. If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. If an instruction is provided, you should follow the instruction when determining the ranking.<|im_end|>\n<|im_start|>user\n"
        var body =
            "I will provide you with \(documents.count) passages, each indicated by a numerical identifier. Rank the passages based on their relevance to query: \(query)<|rerank_token|>\n"
        body += documents.enumerated().map { "<passage id=\"\($0.offset)\">\n\($0.element)<|embed_token|>\n</passage>" }.joined(separator: "\n")
        body +=
            "\n<query>\n\(query)<|rerank_token|>\n</query>\nPlease provide the ranking of all passages based on their relevance to the search query, in descending order of relevance, with each label enclosed in square brackets (e.g., [2] > [1] > [3] > [0])."
        return system + body + suffix
    }
}

private extension Array where Element == Float {
    func chunked(width: Int) -> [[Float]] { stride(from: 0, to: count, by: width).map { Array(self[$0..<Swift.min($0 + width, count)]) } }
}

// swiftlint:enable force_cast shorthand_operator
