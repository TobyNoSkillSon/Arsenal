// SPDX-License-Identifier: MIT
// Weight-free checks of the maker's request path: prompts, answers and request rules against OneJev qev @81ce62f with
// the checkpoint's serve.patch (Fixtures/maker-prompts.json, made by Fixtures/maker_fixture.py from the maker's capture;
// Fixtures/maker-decide-mapping.json, made by Fixtures/maker_decide_mapping.py from the maker's own decide()).
import ElephantEngine
import Foundation
import Testing

struct MakerCase: Decodable {
    let id: String
    let request: String
    let prompts: [String]
    let suffixes: [String]
    let slot_logits: [[Double]]
    let answers: String
}

struct MakerFixture: Decodable { let cases: [MakerCase]; let slot_ids: [Int] }

func makerFixtureFile() throws -> MakerFixture {
    let url = try #require(Bundle.module.url(forResource: "maker-prompts", withExtension: "json", subdirectory: "Fixtures"))
    return try JSONDecoder().decode(MakerFixture.self, from: Data(contentsOf: url))
}

func makerFixture() throws -> [MakerCase] { try makerFixtureFile().cases }

/// Python compares strings by code points; Swift's == by canonical equivalence. Compare the bytes.
func same(_ a: String, _ b: String) -> Bool { Array(a.utf8) == Array(b.utf8) }

/// The pinned checkpoint's tokenizer files, when this Mac has them (no weights are read).
func makerTokenizerDirectory() -> URL? {
    let env = ProcessInfo.processInfo.environment["ELEPHANT_TOKENIZER"]
    let path = env ?? NSHomeDirectory() + "/.cache/huggingface/hub/models--baselabs--elephant-1/snapshots/b3f8a93462fb1051a9de031e94a7ff380e254072"
    let dir = URL(fileURLWithPath: path)
    return FileManager.default.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path) ? dir : nil
}

extension Trait where Self == ConditionTrait {
    /// Token-id checks need the pinned checkpoint's tokenizer files (no weights); without them the test is reported as
    /// skipped, never as a pass.
    static var makerTokenizer: Self { .enabled(if: makerTokenizerDirectory() != nil, "needs elephant-1's tokenizer (ELEPHANT_TOKENIZER or the Hugging Face cache)") }
}

func parsed(_ c: MakerCase) throws -> (PyJSON?, [(String, Question)]) {
    let request = try PyJSONParser.parse(Data(c.request.utf8))
    let state = request["state"].flatMap { $0.isNull ? nil : $0 }
    return (state, try Requests.questions(request["questions"]))
}

/// json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False).
func sortedCompact(_ v: PyJSON) -> String {
    switch v {
    case .object(let f):
        let s = f.sorted { $0.0.unicodeScalars.lexicographicallyPrecedes($1.0.unicodeScalars) { $0.value < $1.value } }
        return "{"
            + s.map { k, x in
                var key = ""
                PyJSON.jsonString(k, &key)
                return key + ":" + sortedCompact(x)
            }.joined(separator: ",") + "}"
    case .array(let a): return "[" + a.map(sortedCompact).joined(separator: ",") + "]"
    default: return v.compact()
    }
}

@Suite struct PromptTests {
    @Test func casesCountAndPromptsMatchTheMaker() throws {
        let cases = try makerFixture()
        #expect(cases.count == 9)
        for c in cases {
            let (state, questions) = try parsed(c)
            #expect(questions.count == c.prompts.count, "\(c.id)")
            for ((_, q), (prompt, suffix)) in zip(questions, zip(c.prompts, c.suffixes)) {
                let rendered = try Prompt.render(q)
                #expect(same(rendered.suffix, suffix), "\(c.id) suffix")
                #expect(same(try Prompt.chat(state: state, suffix: rendered.suffix), prompt), "\(c.id) prompt")
            }
        }
    }

    @Test func answersFromMakerLogitsMatchTheMaker() throws {
        for c in try makerFixture() {
            let (_, questions) = try parsed(c)
            var answers: [(String, PyJSON)] = []
            for ((name, q), logits) in zip(questions, c.slot_logits) {
                let r = try Prompt.render(q)
                let p = Answers.softmax(logits)
                switch q.kind {
                case .noul: answers.append((name, Answers.noul(p[0])))
                case .choice: answers.append((name, Answers.choice(r.labels, p)))
                case .score: answers.append((name, Answers.score(r.descriptions, p)))
                }
            }
            #expect(same(sortedCompact(.object(answers)), c.answers), "\(c.id)")
        }
    }

    @Test func slotLabels() {
        #expect(Prompt.slotLabels.count == 256)
        #expect(Prompt.slotLabels[25] == "Z" && Prompt.slotLabels[26] == "AA" && Prompt.slotLabels[255] == "JU")
        #expect(!Prompt.slotLabels.contains("BQ") && Set(Prompt.slotLabels).count == 256)
    }

    @Test func manyOptionsUseLabelsAndTheLabelAnswerLine() throws {
        let options = (0..<30).map { ("o\($0)", PyJSON?.none) }
        let r = try Prompt.render(Question(kind: .choice, instructions: nil, options: options))
        #expect(r.suffix.utf8.starts(with: "Question: Which option applies to the state?\n\nOptions:\nA. o0\n".utf8))
        #expect(r.suffix.contains("\nAD. o29\n\nAnswer with one label: A, B,"))
        let short = try Prompt.render(Question(kind: .score, instructions: .string("  How?  "), options: [("", nil), ("", .string(" high "))]))
        #expect(same(short.suffix, "Rate the state: How?\n\nOptions:\nA. level 0\nB. level 1: high\n\nAnswer with one letter: A, B."))
    }

    @Test func pythonStripAndRounding() {
        #expect(Prompt.pythonStrip("\u{3000} a b\u{85}\n") == "a b")
        #expect(Prompt.pythonStrip("\u{200B}x") == "\u{200B}x") // zero-width space is not whitespace to Python
        #expect(Prompt.pythonStrip("\u{1C}x\u{1F}") == "x")
        // Python: round(0.1234565, 6) == 0.123456, round(5e-7, 6) == 0.0, round(1.5e-6, 6) == 2e-6, round(0.6456785, 6) == 0.645679
        #expect(Answers.round6(0.1234565) == 0.123456 && Answers.round6(5e-7) == 0 && Answers.round6(1.5e-6) == 2e-6)
        #expect(Answers.round6(0.6456785) == 0.645679 && Answers.round6(0.9999996) == 1.0 && Answers.round6(2.5e-7) == 0.0)
        #expect(PyJSON.floatRepr(Answers.round6(0.000214)) == "0.000214")
    }

    @Test func questionRules() throws {
        func q(_ json: String) throws -> [(String, Question)] { try Requests.questions(try PyJSONParser.parse(Data(json.utf8))) }
        #expect(throws: InvalidRequest.self) { try q("{}") }
        #expect(throws: InvalidRequest.self) { try q(#"{"a":{"type":"maybe"}}"#) }
        #expect(throws: InvalidRequest.self) { try q(#"{"a":{"type":"choice","criteria":{}}}"#) }
        #expect(throws: InvalidRequest.self) { try q(#"{"a":{"type":"choice","criteria":{"x":5}}}"#) }
        #expect(throws: InvalidRequest.self) { try q(#"{"a":{"type":"score","criteria":[1,2,3,4,5,6,7,8,9,10,11]}}"#) }
        #expect(throws: InvalidRequest.self) { try q(#"{"a":{"type":"noul","instructions":3}}"#) }
        let many = "{" + (0..<256).map { "\"o\($0)\":null" }.joined(separator: ",") + "}"
        #expect(throws: InvalidRequest.self) { try q(#"{"a":{"type":"choice","criteria":"# + many + "}}") }
        let qs = (0..<65).map { "\"q\($0)\":{\"type\":\"noul\"}" }.joined(separator: ",")
        #expect(throws: InvalidRequest.self) { try q("{" + qs + "}") }
        // Extra fields are ignored; a repeated name keeps its first place and its last value, as a Python dict does.
        let d = try q(#"{"b":{"type":"noul","extra":1},"a":{"type":"choice","criteria":{"y":"1","x":null,"y":"2"}}}"#)
        #expect(d.map(\.0) == ["b", "a"])
        #expect(d[1].1.options.map(\.0) == ["y", "x"] && d[1].1.options[0].1?.string == "2")
        // judge's batch shorthand: a choice's labels as a list; refused by decide.
        let labels = #"{"a":{"type":"choice","criteria":["x","y","x"]}}"#
        #expect(throws: InvalidRequest.self) { try q(labels) }
        let b = try Requests.questions(try PyJSONParser.parse(Data(labels.utf8)), batch: true)
        #expect(b[0].1.options.map(\.0) == ["x", "y"] && b[0].1.options.allSatisfy { $0.1 == nil })
        // Noul criteria null or absent: default wording; "" stays empty.
        let n = try q(#"{"a":{"type":"noul","criteria":{"true":null,"false":""}}}"#)[0].1
        let r = try Prompt.render(n)
        #expect(r.suffix.contains("A. yes: the statement is true / the answer is yes\nB. no\n"))
    }

    @Test func decideMappingMatchesTheMakersDecide() throws {
        struct Mapping: Decodable { let request: String; let max_slots: Int; let suffixes: [String]; let logits: [[Double]]; let answers: String }
        let url = try #require(Bundle.module.url(forResource: "maker-decide-mapping", withExtension: "json", subdirectory: "Fixtures"))
        let m = try JSONDecoder().decode(Mapping.self, from: Data(contentsOf: url))
        let request = try PyJSONParser.parse(Data(m.request.utf8))
        let questions = try Requests.questions(request["questions"])
        let plan = try ElephantEngine.branches(questions.map(\.1), maxSlots: m.max_slots)
        // Counts first: the 7-option choice splits into three chunks (3, 3, 1 options plus "none"), six branches in all.
        #expect(plan.rendered.count == 6 && m.suffixes.count == 6 && plan.plans.count == 1)
        for (r, suffix) in zip(plan.rendered, m.suffixes) { #expect(same(r.suffix, suffix)) }
        let results = zip(plan.rendered, m.logits).map { Branch(kind: $0.kind, labels: $0.labels, descriptions: $0.descriptions, logits: $1) }
        #expect(same(sortedCompact(.object(ElephantEngine.assemble(questions, plan, results))), m.answers))
    }

    @Test(.makerTokenizer) func tokenIdsAndSlotsMatchTheMaker() throws {
        let dir = try #require(makerTokenizerDirectory())
        let tokenizer = try ElephantEngine.tokenizer(dir)
        let fixture = try makerFixtureFile()
        #expect(try ElephantEngine.verifySlots(tokenizer) == fixture.slot_ids)
        struct Ids: Decodable { let id: String; let prefix_ids: [Int]; let suffix_ids: [[Int]] }
        struct File: Decodable { let cases: [Ids] }
        let url = try #require(Bundle.module.url(forResource: "maker-prompts", withExtension: "json", subdirectory: "Fixtures"))
        let ids = try JSONDecoder().decode(File.self, from: Data(contentsOf: url)).cases
        #expect(ids.count == fixture.cases.count)
        for (c, expected) in zip(fixture.cases, ids) {
            let (state, questions) = try parsed(c)
            let plan = try ElephantEngine.branches(questions.map(\.1), maxSlots: 256)
            let layout = try ElephantEngine.layout(tokenizer, state, plan.rendered)
            #expect(layout.prefix == expected.prefix_ids, "\(c.id) prefix")
            #expect(layout.suffixes == expected.suffix_ids, "\(c.id) suffixes")
        }
    }

    @Test func pythonJSONWritingOfNonFiniteNumbers() throws {
        // json.dumps(json.loads("[NaN, Infinity, -Infinity, 1e999, -0.0, 1E+2]"), indent=2)
        let v = try PyJSONParser.parse(Data("[NaN, Infinity, -Infinity, 1e999, -0.0, 1E+2]".utf8))
        #expect(v.dumps(indent: 2) == "[\n  NaN,\n  Infinity,\n  -Infinity,\n  Infinity,\n  -0.0,\n  100.0\n]")
    }

    @Test func keysCompareByUTF8Bytes() throws {
        // "é" precomposed and "e" + combining acute are different keys to Python and stay two options.
        let json = "{\"a\":{\"type\":\"choice\",\"criteria\":{\"\u{E9}\":null,\"e\u{301}\":\"x\"}}}"
        let q = try Requests.questions(try PyJSONParser.parse(Data(json.utf8)))
        #expect(q[0].1.options.count == 2)
    }

    @Test func nonFiniteSlotLogitsAreRefused() {
        #expect(throws: NonFiniteError.self) { try ElephantEngine.finite([0.5, .nan]) }
        #expect(throws: NonFiniteError.self) { try ElephantEngine.finite([.infinity, 1]) }
        #expect((try? ElephantEngine.finite([1, -2])) == [1, -2])
    }
}
