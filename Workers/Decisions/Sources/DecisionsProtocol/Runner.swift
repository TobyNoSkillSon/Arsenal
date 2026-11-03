// The request -> result rules of Verdict's helper (Projects/Verdict/Sources/VerdictHelper/Service.swift @ f8177ed:
// judge, run, predict, finite, answer, item, questions), for one loaded model per process. Apache-2.0.
// Not ported: residency, eviction, micro-batching across clients, status.json (the Arsenal supervisor owns them).
import Foundation
import VerdictEngine

/// One loaded decision model and Verdict's request handling around it.
public final class DecisionsRunner {
    public let spec: DecisionsSpec
    public let catalog: DecisionsCatalog
    /// Helper bits the model was loaded at (0 = native).
    public let bits: Int
    public let recipe: KernelRecipe
    public let model: DecisionModel
    /// The optimized path failed during inference and the model serves on the stock path (Verdict's fallback).
    public private(set) var fallback: String?

    public init(model: DecisionModel, spec: DecisionsSpec, bits: Int, recipe: KernelRecipe, catalog: DecisionsCatalog = .verdict) {
        self.model = model; self.spec = spec; self.bits = bits; self.recipe = recipe; self.catalog = catalog
    }

    // MARK: System One (/v1/systemone)

    /// One System One request body (exact bytes; key order matters to the models). The reply is Verdict's: 200
    /// `{model, answers, usage}`, 422 FastAPI validation detail, 409/500 TypeSafe error bodies. A request for a model
    /// other than the loaded one (`auto` resolving elsewhere included) is `wrongModel` with the id it needs.
    public func decide(_ body: Data) -> DecideOutcome {
        switch SystemOne.validate(body, catalog: catalog) {
        case .failure(let issues):
            return .reply(SystemOneReply(SystemOne.unprocessable(issues)))
        case .success(let call):
            guard call.model == spec.id else { return .wrongModel(call.model) }
            if let requested = call.bits, let conflict = precisionConflict(requested) {
                return .reply(SystemOneReply(SystemOne.failure(409, "conflict_error", conflict)))
            }
            return .reply(run(call))
        }
    }

    public enum DecideOutcome {
        case reply(SystemOneReply)
        case wrongModel(String)
    }

    private func run(_ call: SystemOneCall) -> SystemOneReply {
        let id = spec.id
        let group = RequestGroup(items: [call.item], questions: call.questions)
        let outcome: Result<GroupResult, Error> = Result { try predict([group])[0] }
        let reply: (Int, [String: Any])
        switch outcome {
        case .failure(let error):
            let message = Self.message(error)
            if message.contains("Non-finite") || message.contains("wrong result count") {
                reply = SystemOne.failure(500, "api_error", message)
            } else {
                reply = SystemOne.unprocessable([ValidationIssue(loc: ["body", "questions"], msg: "Value error, " + message, type: "value_error")])
            }
        case .success(let result):
            switch result.results[0] {
            case .error(let message):
                reply = SystemOne.unprocessable([ValidationIssue(loc: ["body", "state"], msg: "Value error, " + message.replacingOccurrences(of: "Item ", with: "State "), type: "value_error")])
            case .answers(let answers):
                let byID = NSMutableDictionary()
                for (index, question) in call.questions.enumerated() {
                    guard let a = answers[question.id] else { continue }
                    byID[NSString(string: question.id)] = SystemOne.answer(a, question, legend: call.legends[index])
                }
                reply = (200, ["model": id, "answers": byID, "usage": ["input_tokens": result.inputTokens[0], "output_tokens": 0]])
            }
        }
        return SystemOneReply(reply, bits: spec.effectiveBits(bits))
    }

    // MARK: Batch judge (/v1/judge)

    /// Verdict's /v1/judge for this model: `{items, questions, model?, bits?}` -> `(200, {results: [...]})` or
    /// `(400 | 409, {error})`. Items that `auto` sends to another model make the whole call `wrongModel` with
    /// their indexes per model, so the supervisor can split the batch as Verdict's helper does.
    public func judge(_ data: Data) -> JudgeOutcome {
        do {
            var parser = OrderedJSONParser(data)
            let ordered = try parser.parse()
            guard let items = ordered["items"]?.arrayValues, !items.isEmpty else { throw ServiceError("items must be a nonempty list") }
            guard let rawQuestions = ordered["questions"], let fields = rawQuestions.fields, !fields.isEmpty else {
                throw ServiceError("questions must be a nonempty object")
            }
            let qs = try questions(rawQuestions)
            let requested = ordered["model"]?.text ?? "auto"
            var indexes: [Int] = [], elsewhere: [String: [Int]] = [:]
            var results = Array(repeating: [String: Any](), count: items.count)
            for (index, item) in items.enumerated() {
                if Self.containsMedia(item) {
                    results[index] = ["error": "Arsenal judges text; image, audio and video items are not supported.", "model": NSNull(), "ms": 0]
                    continue
                }
                let id = pick(item, requested: requested)
                if id == spec.id { indexes.append(index) } else { elsewhere[id, default: []].append(index) }
            }
            if !elsewhere.isEmpty { return .wrongModel(elsewhere) }
            if let raw = ordered["bits"], !raw.isNull {
                let bits = try Self.wholeNumber(raw, "bits")
                guard spec.acceptedBits.contains(bits) else { throw ServiceError(spec.bitsMessage) }
                if let conflict = precisionConflict(bits) { return .reply(409, ["error": conflict]) }
            }
            if !indexes.isEmpty {
                let start = Date()
                let answers = try predict([RequestGroup(items: indexes.map { Self.item(items[$0]) }, questions: qs)])[0].results
                guard answers.count == indexes.count else { throw ServiceError("Model returned wrong result count") }
                let accepted = answers.reduce(0) { count, result in
                    if case .answers = result { return count + 1 }
                    return count
                }
                let per = (Date().timeIntervalSince(start) * 10000 / Double(max(1, accepted))).rounded() / 10
                for (index, result) in zip(indexes, answers) { results[index] = resultObject(result, per) }
            }
            return .reply(200, ["results": results])
        } catch {
            return .reply(400, ["error": String(Self.message(error).prefix(500))])
        }
    }

    public enum JudgeOutcome {
        case reply(Int, [String: Any])
        case wrongModel([String: [Int]])
    }

    // MARK: Verdict's helper rules

    /// Runs the groups; if the optimized path throws or returns non-finite outputs, reruns on the stock path. When
    /// that succeeds the model stays on stock for its lifetime; when it fails too, the request was at fault: back to
    /// the optimized path and the stock run's error is reported.
    public func predict(_ groups: [RequestGroup]) throws -> [GroupResult] {
        func run() throws -> [GroupResult] {
            let out = try model.predict(groups: groups)
            guard out.count == groups.count, zip(out, groups).allSatisfy({ $0.results.count == $1.items.count }) else {
                throw ServiceError("Model returned wrong result count")
            }
            _ = try Self.finite(out.flatMap(\.results))
            return out
        }
        guard let paths = model as? InferencePathSwitching, paths.optimizedPathActive else { return try run() }
        let failure: Error
        do { return try run() } catch { failure = error }
        do { try paths.useStockPath(true) } catch {
            FileHandle.standardError.write(Data("{\"stock_fallback\":\"\(spec.id)\",\"unavailable\":\(Self.quoted(Self.message(error)))}\n".utf8))
            throw failure
        }
        do {
            let answers = try run()
            fallback = String(Self.message(failure).prefix(200))
            FileHandle.standardError.write(Data("{\"stock_fallback\":\"\(spec.id)\",\"reason\":\(Self.quoted(fallback!))}\n".utf8))
            return answers
        } catch {
            try? paths.useStockPath(false)
            throw error
        }
    }

    /// Every answer value must be finite: a NaN is a failed run, never a probability.
    public static func finite(_ results: [ItemResult]) throws -> [ItemResult] {
        for case .answers(let answers) in results {
            for (_, a) in answers {
                let values = [a.confidence, a.noul, a.score].compactMap { $0 } + (a.probabilities?.values ?? [])
                guard values.allSatisfy(\.isFinite) else { throw ServiceError("Non-finite model outputs") }
            }
        }
        return results
    }

    /// A request's bits are a requirement, never a switch (Verdict): nil when the model runs at `requested`.
    func precisionConflict(_ requested: Int) -> String? {
        let running = spec.effectiveBits(bits), wanted = spec.effectiveBits(requested)
        guard running != wanted else { return nil }
        return "\(spec.id) is loaded at \(running)-bit; this request asked for \(wanted)-bit. "
            + "A request's bits cannot change a model's precision: omit bits, or load it in Arsenal or through MCP model_load with dtype \(spec.dtype(bits: requested))."
    }

    /// The item's original JSON type travels with its text: Von formats a dict as key: value lines, but a string that
    /// happens to look like JSON stays the string (SDK _format_state).
    static func item(_ ordered: OrderedJSON) -> Item {
        switch ordered {
        case .string(let text): return Item(text: text, kind: .text)
        case .object: return Item(text: ordered.render(), kind: .object)
        default: return Item(text: ordered.render(), kind: .value)
        }
    }
    static func containsMedia(_ item: OrderedJSON) -> Bool {
        guard let fields = item.fields else { return false }
        return fields.contains { ["image", "images", "audio", "video", "videos"].contains($0.0) }
    }
    /// `auto`: letters >= 99.5 % ASCII -> laya-english, else laya-multilingual (Verdict's rule).
    func pick(_ item: OrderedJSON, requested: String) -> String {
        if requested != "auto" && !requested.isEmpty { return requested }
        return SystemOne.english(item.text ?? item.render()) ? "laya-english" : "laya-multilingual"
    }
    /// Questions from the order-preserving parse, by exact key bytes.
    func questions(_ raw: OrderedJSON) throws -> [Question] {
        guard let fields = raw.fields else { throw ServiceError("questions must be a nonempty object") }
        return try fields.map { id, q in
            guard let name = q["type"]?.text, let kind = QuestionKind(rawValue: name) else {
                throw ServiceError("Unknown question type '\(q["type"].map { $0.text ?? $0.render() } ?? "")'")
            }
            var criteria: [(String, String)] = []
            if let entries = q["criteria"]?.fields {
                // null means "no description" (the Von SDK and Laya's reference both use the bare label then).
                criteria = entries.map { ($0.0, SystemOne.text($0.1)) }
            } else if let values = q["criteria"]?.arrayValues {
                // Choice labels must be strings; score levels may be structured (their JSON text is the level).
                if kind == .score { criteria = values.map { (SystemOne.text($0), SystemOne.text($0)) } }
                else {
                    let labels = values.compactMap(\.text)
                    if labels.count == values.count { criteria = labels.map { ($0, $0) } }
                }
            }
            // Structured instructions (an object or array, as TypeSafe allows) reach the model as their JSON text.
            return Question(id: id, kind: kind, instructions: SystemOne.text(q["instructions"]), criteria: criteria, sourceJSON: q.render())
        }
    }
    static func answer(_ a: Answer) -> [String: Any] {
        var out: [String: Any] = [:]
        if let v = a.choice { out["choice"] = v }
        if let v = a.probabilities {
            // NSString keys compare literally, so byte-distinct labels ("é" / "e\u{301}") stay two JSON keys.
            let labelled = NSMutableDictionary()
            for (label, p) in v { labelled[NSString(string: label)] = p }
            out["probabilities"] = labelled
        }
        if let v = a.confidence { out["confidence"] = v }
        if let v = a.noul { out["noul"] = v }
        if let v = a.score { out["score"] = v }
        if let v = a.calibrated { out["calibrated"] = v }
        return out
    }
    private func resultObject(_ result: ItemResult, _ ms: Double) -> [String: Any] {
        switch result {
        case .error(let message): return ["error": message, "model": spec.id, "ms": 0]
        case .answers(let answers):
            let byID = NSMutableDictionary()
            for (question, a) in answers { byID[NSString(string: question)] = Self.answer(a) }
            return ["answers": byID, "model": spec.id, "ms": ms]
        }
    }
    /// A JSON whole number (4, 4.0 and 4e0 are the same) or its string form; fractions, booleans and others refused.
    static func wholeNumber(_ raw: OrderedJSON, _ name: String) throws -> Int {
        let literal = raw.text ?? raw.render()
        if case .scalar(let s) = raw, s != "true", s != "false", let value = Double(s), value.isFinite, value == value.rounded(), let exact = Int(exactly: value) {
            return exact
        }
        if let s = raw.text, let value = Int(s.trimmingCharacters(in: .whitespacesAndNewlines)) { return value }
        throw ServiceError("\(name) must be a whole number, not \(literal)")
    }
    public static func message(_ error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? String(describing: error)
    }
    static func quoted(_ s: String) -> String { OrderedJSON.string(s).render() }
}
