// Ported from Projects/Verdict/Sources/VerdictHelper/SystemOne.swift @ f8177ed (Apache-2.0). Changes: the catalog
// is DecisionsCatalog (Arsenal's five models), types are public, the HTTP batcher is not ported (one request per
// worker call; the runner answers requests in order).
import Foundation
import VerdictEngine

// POST /v1/systemone: compatible with the System One API (TypeSafe Jev; https://api.typesafe.ai/openapi.json), on
// local models. One request is one state and its named questions; the response is `{"model", "answers", "usage"}`.
// Request validation failures are 422 with FastAPI's `{"detail": [{"loc", "msg", "type", …}]}`; other failures use
// TypeSafe's `{"detail": {"error_type", "message"}}`. Concurrent requests for the same model are merged into one GPU
// pass (SystemOneBatcher); they all run at the model's one precision.

/// One pydantic-style validation error.
public struct ValidationIssue {
    let loc: [Any]          // "body", field names (String) and array indices (Int)
    let msg: String
    let type: String
    var input: OrderedJSON? = nil
    var ctx: [String: Any]? = nil
    public var object: [String: Any] {
        var out: [String: Any] = ["loc": loc, "msg": msg, "type": type]
        if let input { out["input"] = RawJSON(text: input.render()) }
        if let ctx { out["ctx"] = ctx }
        return out
    }
}

/// A validated request, ready to run.
public struct SystemOneCall {
    /// Resolved model id ("auto" already resolved).
    public let model: String
    /// Required precision (the `bits` extension), validated for `model`; nil = whatever the model runs at. It never
    /// changes the model's precision: a request asking for another one gets 409 (Service.precisionConflict).
    public let bits: Int?
    public let item: Item
    public let questions: [Question]
    /// Score questions' criteria as sent, by question index (the answer's `legend`).
    public let legends: [Int: [OrderedJSON]]
    /// The `merge` extension (default true): false runs this request in a GPU pass of its own, so its answers are
    /// exactly the ones it gets sent alone, whatever other clients send meanwhile.
    public var merge: Bool = true
    /// Requests with the same key share a GPU pass. A model runs at one precision for every client, so the model id is
    /// the (model, effective precision) pair; requests whose bits differ from it are answered 409 inside the batch.
    public var key: String { model }
}

/// A /v1/systemone reply: status, body and the effective precision that answered (the `x-verdict-bits` header).
public struct SystemOneReply {
    public let status: Int
    public let body: [String: Any]
    public var bits: Int? = nil
    public init(_ reply: (Int, [String: Any]), bits: Int? = nil) { status = reply.0; body = reply.1; self.bits = bits }
}

/// A validated request, or why not.
public enum Validated {
    case success(SystemOneCall)
    case failure([ValidationIssue])
}

public enum SystemOne {
    static let questionTypes = ["noul", "choice", "score"]
    static let expectedTags = "'noul', 'choice', 'score'"

    /// FastAPI's 422 body.
    public static func unprocessable(_ issues: [ValidationIssue]) -> (Int, [String: Any]) {
        (422, ["detail": issues.map(\.object)])
    }
    /// TypeSafe's error body for everything that is not request validation.
    public static func failure(_ status: Int, _ type: String, _ message: String) -> (Int, [String: Any]) {
        (status, ["detail": ["error_type": type, "message": message]])
    }

    /// `x-typesafe-request-id`, TypeSafe's format (`req_` + 32 hex digits).
    static func requestID() -> String {
        "req_" + (0..<16).map { _ in String(format: "%02x", UInt8.random(in: 0...255)) }.joined()
    }

    private static func isJSONContent(_ value: OrderedJSON) -> Bool {
        switch value { case .string, .object, .array: return true; case .scalar: return false }
    }
    private static func isJSONContentOrNull(_ value: OrderedJSON) -> Bool { isJSONContent(value) || value.isNull }
    /// Text the models read for a structured value: the string itself, else its JSON rendering (key order kept).
    public static func text(_ value: OrderedJSON?) -> String {
        guard let value, !value.isNull else { return "" }
        return value.text ?? value.render()
    }

    /// Validates a /v1/systemone body. `catalog` resolves model ids; `precisions` checks the `bits` extension.
    public static func validate(_ data: Data, catalog: DecisionsCatalog) -> Validated {
        // Strict syntax first (the order-keeping parser is lenient about scalars).
        do { _ = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) }
        catch {
            return .failure([ValidationIssue(loc: ["body", 0], msg: "JSON decode error", type: "json_invalid",
                                             ctx: ["error": (error as NSError).userInfo[NSDebugDescriptionErrorKey] as? String ?? "Invalid JSON"])])
        }
        var parser = OrderedJSONParser(data)
        guard let body = try? parser.parse() else {
            return .failure([ValidationIssue(loc: ["body", 0], msg: "JSON decode error", type: "json_invalid", ctx: ["error": "Invalid JSON"])])
        }
        guard body.fields != nil else {
            return .failure([ValidationIssue(loc: ["body"], msg: "Input should be a valid dictionary or object to extract fields from",
                                             type: "model_attributes_type", input: body)])
        }
        var issues: [ValidationIssue] = []
        func missing(_ field: String) { issues.append(ValidationIssue(loc: ["body", field], msg: "Field required", type: "missing", input: body)) }

        // state
        let state = body["state"]
        if let state {
            if !isJSONContent(state) {
                issues.append(ValidationIssue(loc: ["body", "state"], msg: "Input should be a valid string, object or array", type: "value_error", input: state))
            } else if let fields = state.fields, fields.contains(where: { ["image", "images", "audio", "video", "videos"].contains($0.0) }) {
                issues.append(ValidationIssue(loc: ["body", "state"], msg: "Arsenal judges text; image, audio and video state is not supported.", type: "value_error"))
            }
        } else { missing("state") }

        // model
        var model: String?
        if let raw = body["model"] {
            if let name = raw.text { model = name }
            else { issues.append(ValidationIssue(loc: ["body", "model"], msg: "Input should be a valid string", type: "string_type", input: raw)) }
        } else { missing("model") }

        // questions
        var questions: [Question] = [], legends: [Int: [OrderedJSON]] = [:]
        if let raw = body["questions"] {
            if let fields = raw.fields {
                if fields.isEmpty {
                    issues.append(ValidationIssue(loc: ["body", "questions"], msg: "Dictionary should have at least 1 item after validation, not 0",
                                                  type: "too_short", input: raw, ctx: ["field_type": "Dictionary", "min_length": 1, "actual_length": 0]))
                }
                for (id, q) in fields {
                    let before = issues.count
                    if let question = self.question(id, q, &issues) {
                        if issues.count == before {
                            if question.kind == .score, let levels = q["criteria"]?.arrayValues { legends[questions.count] = levels }
                            questions.append(question)
                        }
                    }
                }
            } else {
                issues.append(ValidationIssue(loc: ["body", "questions"], msg: "Input should be a valid dictionary", type: "dict_type", input: raw))
            }
        } else { missing("questions") }

        // model id (after state, so "auto" can look at it) and the bits extension
        var resolved: String?
        if let model {
            let local = catalog.local
            if model == "auto" {
                if let state, isJSONContent(state) { resolved = english(text(state)) ? "laya-english" : "laya-multilingual" }
            } else if local.contains(model) {
                resolved = model
            } else {
                let hosted = catalog.hosted.contains(model) || model.lowercased().contains("jev")
                let msg = hosted
                    ? "Value error, '\(model)' is TypeSafe's hosted model and does not run in Arsenal; use one of: auto, \(local.joined(separator: ", "))"
                    : "Value error, unknown model '\(model)'; Arsenal serves: auto, \(local.joined(separator: ", ")) (GET /v1/models)"
                issues.append(ValidationIssue(loc: ["body", "model"], msg: msg, type: "value_error", input: .string(model)))
            }
        }
        var bits: Int?
        if let raw = body["bits"], !raw.isNull {
            if case .scalar(let literal) = raw, let value = Double(literal), value.isFinite, value == value.rounded(), let whole = Int(exactly: value) {
                if let resolved, let spec = catalog.spec(resolved), !spec.acceptedBits.contains(whole) {
                    issues.append(ValidationIssue(loc: ["body", "bits"], msg: "Value error, \(spec.bitsMessage)", type: "value_error", input: raw))
                } else { bits = whole }
            } else {
                issues.append(ValidationIssue(loc: ["body", "bits"], msg: "Input should be a valid integer", type: "int_type", input: raw))
            }
        }
        var merge = true
        if let raw = body["merge"], !raw.isNull {
            if case .scalar(let literal) = raw, literal == "true" || literal == "false" { merge = literal == "true" }
            else { issues.append(ValidationIssue(loc: ["body", "merge"], msg: "Input should be a valid boolean", type: "bool_type", input: raw)) }
        }
        guard issues.isEmpty, let resolved, let state else { return .failure(issues) }
        let item: Item
        switch state {
        case .string(let text): item = Item(text: text, kind: .text)
        case .object: item = Item(text: state.render(), kind: .object)
        default: item = Item(text: state.render(), kind: .value)
        }
        return .success(SystemOneCall(model: resolved, bits: bits, item: item, questions: questions, legends: legends, merge: merge))
    }

    /// One question, OpenAPI `Question` (discriminated by `type`). Appends issues; returns the engine question.
    private static func question(_ id: String, _ q: OrderedJSON, _ issues: inout [ValidationIssue]) -> Question? {
        let base: [Any] = ["body", "questions", id]
        guard q.fields != nil else {
            issues.append(ValidationIssue(loc: base, msg: "Input should be a valid dictionary or object to extract fields from", type: "model_attributes_type", input: q))
            return nil
        }
        guard let tag = q["type"] else {
            issues.append(ValidationIssue(loc: base, msg: "Unable to extract tag using discriminator 'type'", type: "union_tag_not_found",
                                          input: q, ctx: ["discriminator": "'type'"]))
            return nil
        }
        guard let name = tag.text, let kind = QuestionKind(rawValue: name) else {
            let shown = tag.text ?? tag.render()
            issues.append(ValidationIssue(loc: base, msg: "Input tag '\(shown)' found using 'type' does not match any of the expected tags: \(expectedTags)",
                                          type: "union_tag_invalid", input: q, ctx: ["discriminator": "'type'", "tag": shown, "expected_tags": expectedTags]))
            return nil
        }
        let loc = base + [name]
        if let instructions = q["instructions"], !isJSONContentOrNull(instructions) {
            issues.append(ValidationIssue(loc: loc + ["instructions"], msg: "Input should be a valid string, object, array or null", type: "value_error", input: instructions))
        }
        var criteria: [(String, String)] = []
        let raw = q["criteria"]
        switch kind {
        case .choice:
            guard let raw else { issues.append(ValidationIssue(loc: loc + ["criteria"], msg: "Field required", type: "missing", input: q)); return nil }
            guard let options = raw.fields else {
                issues.append(ValidationIssue(loc: loc + ["criteria"], msg: "Input should be a valid dictionary", type: "dict_type", input: raw)); return nil
            }
            if options.isEmpty {
                issues.append(ValidationIssue(loc: loc + ["criteria"], msg: "Value error, a choice needs at least one option", type: "value_error", input: raw))
            }
            for (label, description) in options where !isJSONContentOrNull(description) {
                issues.append(ValidationIssue(loc: loc + ["criteria", label], msg: "Input should be a valid string, object, array or null", type: "value_error", input: description))
            }
            // null = the bare label (the Von SDK and Laya's reference both use the label then).
            criteria = options.map { ($0.0, text($0.1)) }
        case .score:
            guard let raw else { issues.append(ValidationIssue(loc: loc + ["criteria"], msg: "Field required", type: "missing", input: q)); return nil }
            guard let levels = raw.arrayValues else {
                issues.append(ValidationIssue(loc: loc + ["criteria"], msg: "Input should be a valid list", type: "list_type", input: raw)); return nil
            }
            if levels.isEmpty {
                issues.append(ValidationIssue(loc: loc + ["criteria"], msg: "List should have at least 1 item after validation, not 0", type: "too_short",
                                              input: raw, ctx: ["field_type": "List", "min_length": 1, "actual_length": 0]))
            }
            for (index, level) in levels.enumerated() where !isJSONContent(level) {
                issues.append(ValidationIssue(loc: loc + ["criteria", index], msg: "Input should be a valid string, object or array", type: "value_error", input: level))
            }
            criteria = levels.map { (text($0), text($0)) }
        case .noul:
            if let raw, !raw.isNull {
                guard let fields = raw.fields else {
                    issues.append(ValidationIssue(loc: loc + ["criteria"], msg: "Input should be a valid dictionary or object to extract fields from", type: "model_attributes_type", input: raw))
                    return nil
                }
                for key in ["true", "false"] {
                    if let value = raw[key], !isJSONContentOrNull(value) {
                        issues.append(ValidationIssue(loc: loc + ["criteria", key], msg: "Input should be a valid string, object, array or null", type: "value_error", input: value))
                    }
                }
                criteria = fields.filter { ["true", "false"].contains($0.0) }.map { ($0.0, text($0.1)) }
            }
        }
        return Question(id: id, kind: kind, instructions: text(q["instructions"]), criteria: criteria, sourceJSON: q.render())
    }

    /// `auto`: letters ≥ 99.5% ASCII → English.
    public static func english(_ text: String) -> Bool {
        let letters = text.unicodeScalars.filter { CharacterSet.letters.contains($0) }
        return letters.isEmpty || Double(letters.filter { $0.value < 128 }.count) / Double(letters.count) > 0.995
    }

    /// The answer object for one question: noul `{type, noul}`; choice `{type, choice, confidence, probabilities}`;
    /// score `{type, score, confidence, legend, probabilities}`.
    public static func answer(_ a: Answer, _ question: Question, legend: [OrderedJSON]?) -> [String: Any] {
        var out: [String: Any] = ["type": question.kind.rawValue]
        func probabilities(_ p: Probabilities?) -> NSMutableDictionary {
            // NSString keys compare literally: byte-distinct labels ("é" / "e\u{301}") stay two JSON keys.
            let labelled = NSMutableDictionary()
            for (label, value) in p ?? Probabilities(labels: [], values: []) { labelled[NSString(string: label)] = value }
            return labelled
        }
        switch question.kind {
        case .noul:
            out["noul"] = a.noul ?? 0
        case .choice:
            out["choice"] = a.choice ?? ""
            out["confidence"] = a.confidence ?? 0
            out["probabilities"] = probabilities(a.probabilities)
        case .score:
            out["score"] = a.score ?? 0
            out["confidence"] = a.confidence ?? 0
            out["probabilities"] = probabilities(a.probabilities)
            let levels = NSMutableDictionary()
            for (index, level) in (legend ?? []).enumerated() { levels[NSString(string: String(index))] = RawJSON(text: level.render()) }
            out["legend"] = levels
        }
        return out
    }
}
