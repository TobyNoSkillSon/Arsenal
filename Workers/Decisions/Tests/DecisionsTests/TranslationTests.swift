// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import DecisionsProtocol
import VerdictEngine

/// CPU-only: Verdict's request -> result translation with a model that returns fixed answers. The golden files in
/// Fixtures/systemone were recorded from the real worker on the oracle items it matched byte for byte
/// (Workers/Decisions/scripts/parity.py); here the "model" replays those answers, so what is tested is the System One
/// request parsing, answer objects, key order and errors — the part the API worker reuses.

/// Replays answers per state text; `error` per state text for over-context items.
final class ReplayModel: DecisionModel {
    let id: String
    var answers: [String: Answers] = [:]
    var errors: [String: String] = [:]
    var tokens: [String: Int] = [:]
    init(id: String) { self.id = id }
    var contextLimit: Int { 8192 }
    var residentBytes: Int { 0 }
    func tokenCount(_ item: Item, _ questions: [Question]) throws -> Int { tokens[item.text] ?? 1 }
    func predict(_ items: [Item], _ questions: [Question]) throws -> [ItemResult] {
        items.map { item in
            if let error = errors[item.text] { return .error(error) }
            return .answers(answers[item.text] ?? Answers())
        }
    }
    func predict(groups: [RequestGroup]) throws -> [GroupResult] {
        try groups.map { group in
            let results = try predict(group.items, group.questions)
            return GroupResult(results: results, inputTokens: group.items.map { tokens[$0.text] ?? 0 })
        }
    }
}

/// The state text the model sees (Verdict's Item rules), from the request JSON as sent.
func stateText(_ request: Data) throws -> String {
    var parser = OrderedJSONParser(request)
    let state = try parser.parse()["state"]!
    return state.text ?? state.render()
}

func answer(from object: [String: Any]) -> Answer {
    var a = Answer()
    switch object["type"] as? String {
    case "noul": a.noul = object["noul"] as? Double
    case "choice":
        a.choice = object["choice"] as? String; a.confidence = object["confidence"] as? Double
    default:
        a.score = object["score"] as? Double; a.confidence = object["confidence"] as? Double
    }
    if let p = object["probabilities"] as? [String: Double] {
        let labels = p.keys.sorted { (Double($0) ?? 0, $0) < (Double($1) ?? 0, $1) }
        a.probabilities = Probabilities(labels: labels, values: labels.map { p[$0]! })
    }
    return a
}

@Test(arguments: ["laya-english-fp16", "laya-multilingual-fp16", "laya-typed-decisions-fp16"])
func goldenSystemOne(_ name: String) throws {
    let url = Bundle.module.url(forResource: "Fixtures/systemone/\(name)", withExtension: "json")!
    // Parsed twice: in order (the request bytes as the parity driver sent them: json.dumps, ensure_ascii=False) and
    // as Foundation objects (the expected response; key order is irrelevant to JSON equality).
    var parser = OrderedJSONParser(try Data(contentsOf: url))
    let ordered = try parser.parse()
    let plain = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    let cases = try #require(plain["cases"] as? [[String: Any]])
    let orderedCases = ordered["cases"]!.arrayValues!
    #expect(cases.count >= 12)
    let id = try #require(plain["model"] as? String)
    let model = ReplayModel(id: id)
    let runner = DecisionsRunner(model: model, spec: DecisionsCatalog.verdict.spec(id)!, bits: 0, recipe: .exact)
    for (item, orderedItem) in zip(cases, orderedCases) {
        let request = Data(orderedItem["request"]!.render().utf8)
        let state = try stateText(request)
        let response = try #require(item["response"] as? [String: Any])
        if item["status"] as? Int == 200 {
            var replay = Answers()
            for (qid, value) in try #require(response["answers"] as? [String: Any]) { replay[qid] = answer(from: try #require(value as? [String: Any])) }
            model.answers[state] = replay
            model.tokens[state] = (try #require(response["usage"] as? [String: Any]))["input_tokens"] as? Int
        } else {
            let issue = (try #require(response["detail"] as? [[String: Any]]))[0]
            model.errors[state] = (try #require(issue["msg"] as? String)).replacingOccurrences(of: "Value error, ", with: "").replacingOccurrences(of: "State ", with: "Item ")
        }
        guard case .reply(let reply) = runner.decide(request) else { Issue.record("wrong model"); continue }
        #expect(reply.status == item["status"] as? Int)
        let got = try #require(JSONSerialization.jsonObject(with: ResponseJSON.data(reply.body as NSDictionary)) as? NSDictionary)
        #expect(got == response as NSDictionary, "\(name): \(state.prefix(60))")
    }
}

@Test func validationErrorsAreFastAPIShaped() throws {
    let model = ReplayModel(id: "laya-english")
    let runner = DecisionsRunner(model: model, spec: DecisionsCatalog.verdict.spec("laya-english")!, bits: 0, recipe: .fast)
    func reply(_ body: String) -> SystemOneReply? {
        if case .reply(let r) = runner.decide(Data(body.utf8)) { return r }
        return nil
    }
    let missing = try #require(reply(#"{"model": "laya-english", "questions": {"q": {"type": "noul"}}}"#))
    #expect(missing.status == 422)
    let detail = try #require((missing.body["detail"] as? [[String: Any]])?.first)
    #expect(detail["type"] as? String == "missing")
    #expect((detail["loc"] as? [Any])?.map { "\($0)" } == ["body", "state"])

    let tag = try #require(reply(#"{"model": "laya-english", "state": "x", "questions": {"q": {"type": "maybe"}}}"#))
    #expect(((tag.body["detail"] as? [[String: Any]])?.first?["type"] as? String) == "union_tag_invalid")

    let hosted = try #require(reply(#"{"model": "jev", "state": "x", "questions": {"q": {"type": "noul"}}}"#))
    #expect(((hosted.body["detail"] as? [[String: Any]])?.first?["msg"] as? String)?.contains("hosted model") == true)

    let bad = try #require(reply("{not json"))
    #expect(((bad.body["detail"] as? [[String: Any]])?.first?["type"] as? String) == "json_invalid")
}

@Test func routingAndPrecision() throws {
    let model = ReplayModel(id: "laya-english")
    let runner = DecisionsRunner(model: model, spec: DecisionsCatalog.verdict.spec("laya-english")!, bits: 0, recipe: .fast)
    // auto routes non-English text to laya-multilingual (Verdict's 99.5 % ASCII-letters rule): another runner's job.
    if case .wrongModel(let needed) = runner.decide(Data(#"{"model": "auto", "state": "Proszę o zwrot pieniędzy za zamówienie", "questions": {"q": {"type": "noul"}}}"#.utf8)) {
        #expect(needed == "laya-multilingual")
    } else {
        Issue.record("expected wrongModel")
    }
    if case .wrongModel(let needed) = runner.decide(Data(#"{"model": "von-1.2", "state": "x", "questions": {"q": {"type": "noul"}}}"#.utf8)) {
        #expect(needed == "von-1.2")
    } else {
        Issue.record("expected wrongModel")
    }
    // bits is a requirement, never a switch: 8 against a 16-bit model is a 409.
    guard case .reply(let conflict) = runner.decide(Data(#"{"model": "laya-english", "state": "x", "bits": 8, "questions": {"q": {"type": "noul"}}}"#.utf8)) else {
        Issue.record("expected reply"); return
    }
    #expect(conflict.status == 409)
    // judge: items auto-routed elsewhere are reported per model so the supervisor can split the batch.
    if case .wrongModel(let routes) = runner.judge(Data(#"{"items": ["hello there", "Dzień dobry, proszę o zwrot"], "questions": {"q": {"type": "noul"}}}"#.utf8)) {
        #expect(routes == ["laya-multilingual": [1]])
    } else {
        Issue.record("expected wrongModel")
    }
}

@Test func judgeKeepsByteDistinctLabelsAndOrder() throws {
    let model = ReplayModel(id: "von-1.2")
    var a = Answer()
    a.choice = "\u{e9}"; a.confidence = 0.5
    a.probabilities = Probabilities(labels: ["\u{e9}", "e\u{301}"], values: [0.6, 0.4])
    model.answers["caf\u{e9}"] = ["q": a]
    let runner = DecisionsRunner(model: model, spec: DecisionsCatalog.verdict.spec("von-1.2")!, bits: 16, recipe: .exact)
    guard
        case .reply(let status, let body) = runner.judge(
            Data(#"{"model": "von-1.2", "items": ["café"], "questions": {"q": {"type": "choice", "criteria": {"é": "a", "é": "b"}}}}"#.utf8))
    else {
        Issue.record("expected reply"); return
    }
    #expect(status == 200)
    let text = String(decoding: ResponseJSON.data(body as NSDictionary), as: UTF8.self)
    #expect(text.contains("\"\u{e9}\":0.6") && text.contains("\"e\u{301}\":0.4"))
}

@Test func wireFrameRoundTrip() throws {
    var pipe: [Int32] = [0, 0]
    #expect(Darwin.pipe(&pipe) == 0)
    let payload = Data(#"{"id": 1, "op": "decide", "body": "{\"state\": \"é\"}"}"#.utf8)
    Wire.write(payload, pipe[1])
    close(pipe[1])
    #expect(try Wire.read(pipe[0]) == payload)
    #expect(try Wire.read(pipe[0]) == nil)
    close(pipe[0])
}
