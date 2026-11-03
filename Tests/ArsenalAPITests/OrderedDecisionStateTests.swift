// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore
@testable import ArsenalAPI

/// A decision model's state may be any JSON value (text, null, a list, an object) and its key order is the prompt's;
/// questions and their options are ordered too (elephant-1, review M3). Strings must stay text on every route.
private actor DecisionCaptureService: ModelsService {
    var requests: [JSONObject] = []
    let operation = ModelOperation(
        name: "decide", title: "Decide",
        inputSchema: [
            "type": "object", "required": ["state", "questions"],
            "properties": [
                "state": ["x-arsenal-ordered": true], "questions": ["type": "object", "minProperties": 1, "x-arsenal-ordered": true],
                "media": ["type": "array"]
            ]
        ], output: "json", stream: false, schemaVersion: 3)
    func rows() async -> [ModelRow] { [ModelRow(id: "decider", name: "Decider", kind: "decisions", states: ["fp16": .init(state: .loaded)])] }
    func kinds() async -> [KindSpec] { KindSpec.defaults }
    func operations(id: String) async -> [ModelOperation] { [operation] }
    func events() async -> AsyncStream<ServiceEvent> { AsyncStream { $0.finish() } }
    func get(id: String, precision: String) async throws {}
    func cancelGet(id: String) async {}
    func load(id: String, precision: String?, fast: Bool?) async throws {}
    func unload(id: String) async throws {}
    func setKeepHot(id: String, on: Bool) async throws {}
    func delete(id: String) async throws {}
    func menuActions() async -> [LoadedAction] { [] }
    func memory() async -> MemorySnapshot { .init() }
    func catalogStatus() async -> CatalogStatus { .init() }
    /// RealModelsService.validatedRequest's two shapes: a raw decisions body as is; otherwise split, validate, attach.
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject {
        if params.count == 1, params["body"]?.string != nil {
            requests.append(params)
        } else {
            let (request, ordered) = try OrderedJSONMembers.split(params, schema: operation.inputSchema)
            try OperationSchemaValidator.validate(.object(request), schema: .object(operation.inputSchema))
            requests.append(OrderedJSONMembers.attach(ordered, to: request))
        }
        return ["answers": ["q": ["noul": 0.5]]]
    }
}

struct OrderedDecisionStateTests {
    static let state = #"{"screen": "<image:1>", "after": "<image:2>"}"#
    static let questions = #"{"z": {"type": "choice", "criteria": {"stop": null, "go": null}}, "a": {"type": "noul"}}"#
    static let schema: JSONObject = [
        "type": "object", "properties": ["state": ["x-arsenal-ordered": true], "questions": ["type": "object", "x-arsenal-ordered": true]]
    ]

    func post(_ port: Int, _ body: String, async: Bool, session: URLSession) async throws -> (Int, String) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/models/decider/ops/decide")!)
        request.httpMethod = "POST"; request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if async { request.setValue("respond-async", forHTTPHeaderField: "Prefer") }
        request.httpBody = Data(body.utf8)
        let (data, response) = try await session.data(for: request)
        return ((response as? HTTPURLResponse)?.statusCode ?? 0, String(decoding: data, as: UTF8.self))
    }

    @Test func httpStatesKeepTheirOrderAndStringsStayText() async throws {
        let service = DecisionCaptureService()
        let root = URL(fileURLWithPath: "/tmp/arsenal-ordered-decisions/\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let server = ArsenalAPIServer(service: service, supportDirectory: root, port: 0)
        let port = try await server.start(); defer { server.stop() }
        let config = URLSessionConfiguration.ephemeral; config.connectionProxyDictionary = [:]
        let session = URLSession(configuration: config); defer { session.invalidateAndCancel() }
        func captured(_ count: Int) async throws -> JSONObject {
            for _ in 0..<400 where await service.requests.count < count { try await Task.sleep(for: .milliseconds(10)) }
            return try #require(await service.requests.dropFirst(count - 1).first)
        }
        let object = #"{"state": "# + Self.state + #", "questions": "# + Self.questions + "}"
        // Synchronous: the decisions route forwards the body's own bytes.
        var (status, text) = try await post(port, object, async: false, session: session)
        #expect(status == 200, "\(text)")
        // (The route adds the model from the URL; the state's and questions' bytes are the ones sent.)
        let raw = try #require(try await captured(1)["body"]?.string)
        #expect(raw.contains(Self.state) && raw.contains(Self.questions))
        // Questions sent as their JSON text reach the worker as the object they stand for, as on the async route.
        let asText = try String(decoding: JSONSerialization.data(withJSONObject: Self.questions, options: .fragmentsAllowed), as: UTF8.self)
        (status, text) = try await post(port, #"{"state": "s", "questions": "# + asText + "}", async: false, session: session)
        #expect(status == 200, "\(text)")
        #expect(try #require(try await captured(2)["body"]?.string).contains(#""questions":"# + Self.questions))
        // Asynchronous job: the decoded members plus their exact text.
        (status, text) = try await post(port, object, async: true, session: session)
        #expect(status == 202, "\(text)")
        let job = try await captured(3)
        #expect(job[OrderedJSONMembers.transport]?["state"]?.string == Self.state)
        #expect(job[OrderedJSONMembers.transport]?["questions"]?.string == Self.questions)
        #expect(job["state"]?["screen"]?.string == "<image:1>" && job["questions"]?["a"]?["type"]?.string == "noul")
        // A text state that looks like JSON is still text; null is null; neither gets a transport text.
        for (body, expected) in [
            (#"{"state": "{\"a\": 1} <image:1>", "questions": {"q": {"type": "noul"}}}"#, JSONValue.string(#"{"a": 1} <image:1>"#)),
            (#"{"state": null, "questions": {"q": {"type": "noul"}}}"#, .null)
        ] {
            let before = await service.requests.count
            (status, text) = try await post(port, body, async: true, session: session)
            #expect(status == 202, "\(text)")
            let request = try await captured(before + 1)
            #expect(request["state"] == expected)
            #expect(request[OrderedJSONMembers.transport]?["state"] == nil)
        }
    }

    /// The shipped op schemas: state, questions and items are ordered, and MCP input staging leaves them (and the
    /// transport) alone, so a state holding a "path" member keeps matching its text; media is staged as before.
    @Test func elephantSchemasMarkOrderedTextMembers() throws {
        let ops = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("models/elephant-1/ops")
        for (op, names) in [("decide", ["questions", "state"]), ("judge", ["items", "questions"])] {
            let schema = try WireJSON.decode(Data(contentsOf: ops.appendingPathComponent("\(op).input.json")))
            #expect(OrderedJSONMembers.names(schema) == names)
            #expect(OrderedJSONMembers.verbatimNames(schema) == Set(names + [OrderedJSONMembers.transport]))
        }
        // An ordered property that can carry a file is staged.
        let filed: JSONObject = ["properties": ["doc": ["x-arsenal-ordered": true, "properties": ["f": ["x-arsenal-file": ["accept": ["image/png"]]]]]]]
        #expect(OrderedJSONMembers.verbatimNames(filed) == [OrderedJSONMembers.transport])
        let referenced: JSONObject = [
            "$defs": ["f": ["x-arsenal-file": ["accept": ["image/png"]]], "loop": ["items": ["$ref": "#/$defs/loop"]]],
            "properties": ["a": ["x-arsenal-ordered": true, "$ref": "#/$defs/f"], "b": ["x-arsenal-ordered": true, "$ref": "#/$defs/loop"]]
        ]
        #expect(OrderedJSONMembers.verbatimNames(referenced) == ["b", OrderedJSONMembers.transport])
    }

    /// A repeated key is read as the worker reads it (the last one): text that would show the worker another
    /// question than the one validated is refused.
    @Test func transportTextWithRepeatedKeysMustMatchAsTheWorkerReadsIt() throws {
        let validated: JSONValue = ["q": ["type": "noul"]]
        #expect(throws: WorkerFailure.self) {
            _ = try OrderedJSONMembers.split(
                ["questions": validated, OrderedJSONMembers.transport: ["questions": #"{"q":{"type":"noul"},"q":{"type":"bogus"}}"#]], schema: Self.schema)
        }
        let kept = try OrderedJSONMembers.split(
            ["questions": validated, OrderedJSONMembers.transport: ["questions": #"{"q":{"type":"bogus"},"q":{"type":"noul"}}"#]], schema: Self.schema)
        #expect(kept.ordered["questions"] != nil)
    }

    /// MCP's arguments arrive as a dictionary: questions (object-typed) may come as their JSON text; the state keeps a
    /// string as text and takes its ordered text only from the transport member, which must match the state sent.
    @Test func mcpShapedInputsKeepStringStatesAndCarryOrderedTexts() throws {
        let text = try OrderedJSONMembers.split(["state": "look at <image:1>", "questions": .string(Self.questions)], schema: Self.schema)
        #expect(text.request["state"] == .string("look at <image:1>"))
        #expect(text.request["questions"]?.object?.keys.sorted() == ["a", "z"] && text.ordered == ["questions": Self.questions])
        // JSON-looking text is not decoded into an object; null stays null.
        #expect(try OrderedJSONMembers.split(["state": #"{"a":1}"#], schema: Self.schema).request["state"] == .string(#"{"a":1}"#))
        #expect(try OrderedJSONMembers.split(["state": .null], schema: Self.schema).request["state"] == .null)
        // An object state with its exact text in the transport member keeps that text.
        let decoded = try JSONDecoder().decode(JSONValue.self, from: Data(Self.state.utf8))
        let ordered = try OrderedJSONMembers.split(["state": decoded, OrderedJSONMembers.transport: ["state": .string(Self.state)]], schema: Self.schema)
        #expect(ordered.ordered == ["state": Self.state] && ordered.request["state"] == decoded)
        // Lists and strings may carry text too, but it must decode to the value sent.
        #expect(try OrderedJSONMembers.split(["state": "hi", OrderedJSONMembers.transport: ["state": #""hi""#]], schema: Self.schema).ordered == ["state": #""hi""#])
        #expect(throws: WorkerFailure.self) {
            _ = try OrderedJSONMembers.split(["state": "hi", OrderedJSONMembers.transport: ["state": .string(Self.state)]], schema: Self.schema)
        }
        #expect(throws: WorkerFailure.self) {
            _ = try OrderedJSONMembers.split(["state": ["screen": "x"], OrderedJSONMembers.transport: ["state": .string(Self.state)]], schema: Self.schema)
        }
    }
}
