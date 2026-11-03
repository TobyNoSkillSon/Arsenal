// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import SwiftUI

// Search: one window for every model that declares `rerank` or `embed` — a working surface for judging a ranking
// model, not an index. A query and the passages (and, for an embedding model that takes images, pictures) to rank;
// the ranked list with each score. A reranker scores the passages itself; an embedding model embeds the query and the
// passages (as a query and as documents when it tells the two apart) and the window ranks them by cosine similarity.
// Raw vectors stay with the API (`embed`), where agents use them.

@MainActor final class SearchSession: CapabilitySession {
    /// The passage token limit and the result count are tuning: under Advanced, the query and passages stay in view.
    override func advancedKeys(_ op: String?) -> Set<String> { super.advancedKeys(op).union(["max_tokens_per_doc", "top_n"]) }
    private var embedding: Task<Void, Never>?

    init(models: [ModelChoice], selected: String, client: @escaping () -> ArsenalClient?, inbox: URL, outputDirectory: URL? = nil, scratch: URL? = nil) {
        super.init(tool: .search, models: models, selected: selected, client: client, inbox: inbox, outputDirectory: outputDirectory, scratch: scratch)
    }
    override func mainKeys(_ op: OperationDescriptor) -> [String: String] { ["query": "Search for"] }
    override func titles(_ op: OperationDescriptor) -> [String: String] { ["documents": "Passages", "top_n": "Results", "images": "Images"] }
    override var resultNoun: String { "ranking" }
    override var mainSingleLine: Bool { true }
    override var idleLine: String { "Ranking" }
    override func runTitle(_ op: OperationDescriptor?) -> String { "Rank" }
    /// The query is required whatever the schema says; so is something to rank.
    override var canRun: Bool {
        guard super.canRun else { return false }
        return !value("query").text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !(ToolForm.items(value("documents").text, separators: "\n").isEmpty && subject.isEmpty)
    }
    override var readyToRun: Bool { descriptor != nil }

    /// Embedding models get the window's own form: a query, passages one per line, pictures when their input takes
    /// images, and the options of theirs that change the ranking (the task is the window's: query against document).
    override func makeForm(_ op: OperationDescriptor) -> ToolForm {
        var form =
            op.name == "embed"
            ? ToolForm(schema: Self.embedForm(op.inputSchema), operation: op.name, output: op.output, main: mainKeys(op), titles: titles(op)) : super.makeForm(op)
        form.update("max_tokens_per_doc") { $0.unit = "tokens" }
        return form
    }
    static func embedForm(_ schema: JSONObject) -> JSONObject {
        let declared = schema["properties"]?.object ?? [:]
        var properties: JSONObject = ["query": ["type": "string"], "documents": ["type": "array", "items": ["type": "string"]]]
        if takesImages(schema) {
            properties["images"] = [
                "type": "array", "maxItems": 64, "items": ["type": "object", "x-arsenal-file": ["accept": ["image/png", "image/jpeg"]], "properties": ["path": ["type": "string"]]]
            ]
        }
        for key in ["instruction", "dimensions", "truncate"] { if let p = declared[key] { properties[key] = p } }
        return ["type": "object", "properties": .object(properties), "required": ["query"]]
    }
    /// The embed input's items include `{"type": "image", …}`.
    static func takesImages(_ schema: JSONObject) -> Bool {
        let items = schema["properties"]?["input"]?["items"]
        let options = items?["anyOf"]?.array ?? items?["oneOf"]?.array ?? [items].compactMap { $0 }
        return options.contains { $0["properties"]?["type"]?["enum"]?.array?.contains(.string("image")) == true || $0["properties"]?["image"] != nil }
    }

    // MARK: Embedding runs

    override func start() {
        guard descriptor?.name == "embed" else { super.start(); return }
        guard canRun, let op = descriptor, let model, let client = client() else { return }
        let query = value("query").text.trimmingCharacters(in: .whitespacesAndNewlines)
        let texts = ToolForm.items(value("documents").text, separators: "\n")
        let pictures = subject
        let declared = op.inputSchema["properties"]?.object ?? [:]
        var extras: JSONObject = [:]
        if let n = Int(value("dimensions").text.trimmingCharacters(in: .whitespaces)) { extras["dimensions"] = .number(Double(n)) }
        let instruction = value("instruction").text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !instruction.isEmpty { extras["instruction"] = .string(instruction) }
        if case .toggle(let on) = value("truncate"), declared["truncate"] != nil { extras["truncate"] = .bool(on) }
        if declared["normalize"] != nil { extras["normalize"] = true }
        let tasks = declared["task"]?["enum"]?.array?.compactMap(\.string) ?? []
        let common = extras
        let values = requestValues()
        problem = nil
        player.stop()
        run = ToolRun(model: model.id, operation: op.name, input: ["query": .string(query)])
        embedding = Task {
            var staged: [URL] = []
            defer { for folder in staged { try? FileManager.default.removeItem(at: folder) } }
            do {
                var documents: [JSONValue] = texts.map { .string($0) }
                for url in pictures {
                    let (folder, path) = try ToolFiles.stage(url, accept: FileAccept(mimes: ["image/png", "image/jpeg"]), inbox: inbox)
                    staged.append(folder)
                    documents.append(["type": "image", "image": ["path": .string(path)]])
                }
                func embed(_ input: [JSONValue], task: String?) async throws -> [[Double]] {
                    var body = common
                    body["input"] = .array(input)
                    if let task, tasks.contains(task) { body["task"] = .string(task) }
                    try OperationSchemaValidator.validate(.object(body), schema: .object(op.inputSchema))
                    let (data, _) = try await client.send(
                        "POST", "/arsenal/v1/models/\(model.id)/ops/embed", json: body, timeout: (op.timeout ?? 600) + 30, accept: "application/json")
                    guard let rows = try JSONDecoder().decode(JSONValue.self, from: data)["embeddings"]?.array else { throw ToolError("\(model.name) returned no embeddings.") }
                    return rows.map { $0.array?.compactMap(\.number) ?? [] }
                }
                let vectors: [[Double]]
                let q: [Double]
                if tasks.contains("query"), tasks.contains("document") {
                    q = try await embed([.string(query)], task: "query").first ?? []
                    vectors = try await embed(documents, task: "document")
                } else {
                    let all = try await embed([.string(query)] + documents, task: nil)
                    q = all.first ?? []; vectors = Array(all.dropFirst())
                }
                try Task.checkCancellation()
                guard vectors.count == documents.count, !q.isEmpty else { throw ToolError("\(model.name) returned \(vectors.count) embeddings for \(documents.count) items.") }
                let scores = vectors.map { Self.cosine(q, $0) }
                let ranked = scores.indices.sorted { scores[$0] > scores[$1] }
                let reply: JSONObject = [
                    "results": .array(ranked.map { ["index": .number(Double($0)), "relevance_score": .number(scores[$0])] }),
                    "dimensions": .number(Double(q.count))
                ]
                var input: JSONObject = ["query": .string(query), "documents": .array(texts.map { .string($0) })]
                input["images"] = .array(pictures.map { .string($0.path) })
                run?.finished = Date()
                let seconds = run?.seconds(now: Date()) ?? 0
                append(
                    Outcome(
                        model: model.id, modelName: model.name, operation: op.name, operationTitle: op.title, inputs: pictures, text: query, values: values, input: input,
                        reply: reply, files: [], seconds: seconds))
                finish(nil)
            } catch is CancellationError {
                return
            } catch let p as OperationForm.Problem {
                finish(.init(message: p.message, action: nil))
            } catch {
                finish(OperationProblem.from(error, model: model.name, task: stoppedNoun))
            }
        }
    }
    override func stop() {
        guard running else { return }
        if embedding != nil {
            embedding?.cancel(); embedding = nil
            run?.cancelled = true
            finish(nil)
            return
        }
        super.stop()
    }

    static func cosine(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in a.indices { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return na > 0 && nb > 0 ? dot / (na.squareRoot() * nb.squareRoot()) : 0
    }

    /// The ranked items of a result: its passages (and pictures) in the model's order with their scores.
    static func ranked(_ o: Outcome) -> [RankedList.Item] {
        let texts = o.input["documents"]?.array?.compactMap(\.string) ?? []
        return (o.reply["results"]?.array ?? []).compactMap { r in
            guard let i = r["index"]?.number.flatMap(\.roundedInt), let score = r["relevance_score"]?.number else { return nil }
            if i < texts.count { return RankedList.Item(id: i, text: texts[i], score: score) }
            let picture = i - texts.count
            guard o.inputs.indices.contains(picture) else { return nil }
            return RankedList.Item(id: i, text: o.inputs[picture].lastPathComponent, score: score, image: o.inputs[picture])
        }
    }
}

struct SearchView: View {
    @ObservedObject var session: SearchSession
    var body: some View {
        CapabilityView(session: session, subjectFills: false) {
            OutcomePanel(
                session: session, what: "ranking",
                copyText: { o in
                    SearchSession.ranked(o).enumerated().map { i, item in "\(i + 1). \(String(format: "%.3f", item.score))  \(item.text)" }.joined(separator: "\n")
                },
                content: { o in
                    RankedList(items: SearchSession.ranked(o))
                })
        }
    }
}
