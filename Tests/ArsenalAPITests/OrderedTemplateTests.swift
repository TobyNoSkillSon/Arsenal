// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore
@testable import ArsenalAPI

/// An extraction template's key order is the order NuExtract writes its answer in; a Swift dictionary loses it.
private actor OrderedCaptureService: ModelsService {
    var requests: [JSONObject] = []
    let operation = ModelOperation(
        name: "extract", title: "Extract",
        inputSchema: [
            "type": "object", "required": ["text", "template"], "additionalProperties": false,
            "properties": ["text": ["type": "string"], "template": ["type": "object", "x-arsenal-ordered": true]]
        ], output: "json", stream: false)
    func rows() async -> [ModelRow] { [ModelRow(id: "extractor", name: "Extractor", kind: "chat", states: ["fp16": .init(state: .loaded)])] }
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
    /// The same split/validate/attach sequence RealModelsService.validatedRequest applies before the worker.
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject {
        let (request, ordered) = try OrderedJSONMembers.split(params, schema: operation.inputSchema)
        try OperationSchemaValidator.validate(.object(request), schema: .object(operation.inputSchema))
        requests.append(OrderedJSONMembers.attach(ordered, to: request))
        return ["data": request["template"] ?? .null]
    }
}

struct OrderedTemplateTests {
    static let template = #"{"zeta":"string","alpha":["string"],"mid":{"b":"verbatim-string","a":"integer"}}"#

    @Test func httpTemplateReachesTheWorkerInItsOwnKeyOrder() async throws {
        let service = OrderedCaptureService()
        let root = URL(fileURLWithPath: "/tmp/arsenal-ordered-template/\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let server = ArsenalAPIServer(service: service, supportDirectory: root, port: 0)
        let port = try await server.start(); defer { server.stop() }
        let config = URLSessionConfiguration.ephemeral; config.connectionProxyDictionary = [:]
        let session = URLSession(configuration: config); defer { session.invalidateAndCancel() }
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/models/extractor/ops/extract")!)
        request.httpMethod = "POST"; request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.httpBody = Data((#"{"text": "Ada paid 12 dollars.", "template": "# + Self.template + "}").utf8)
        let (body, response) = try await session.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200, "\(String(decoding: body, as: UTF8.self))")
        let captured = try #require(await service.requests.first)
        #expect(captured[OrderedJSONMembers.transport]?["template"]?.string == Self.template)
        #expect(captured["template"]?["mid"]?["b"]?.string == "verbatim-string")
    }

    @Test func templateTextIsValidatedAsItsObjectAndForwardedExactly() throws {
        let schema: JSONObject = ["type": "object", "properties": ["template": ["type": "object", "x-arsenal-ordered": true], "text": ["type": "string"]]]
        #expect(OrderedJSONMembers.names(schema) == ["template"])
        // MCP clients send the text; key order and bytes survive, the object is what the schema checks.
        let (request, ordered) = try OrderedJSONMembers.split(["text": "Ada", "template": .string(Self.template)], schema: schema)
        #expect(request["template"]?.object?.keys.sorted() == ["alpha", "mid", "zeta"])
        #expect(ordered == ["template": Self.template])
        let forwarded = OrderedJSONMembers.attach(ordered, to: request)
        // A second validation pass (image preparation re-validates) accepts the forwarded request unchanged.
        let again = try OrderedJSONMembers.split(forwarded, schema: schema)
        #expect(again.request == request && again.ordered == ordered)
        #expect(throws: WorkerFailure.self) { _ = try OrderedJSONMembers.split(["template": "[1, 2]"], schema: schema) }
        // An object given directly carries no order of its own: nothing is invented for it.
        #expect(try OrderedJSONMembers.split(["template": ["a": 1]], schema: schema).ordered.isEmpty)
    }

    /// A client cannot slip text past validation: transport texts are accepted only for declared ordered names and
    /// only when they decode to the very object the request validated.
    @Test func transportTextMustBeDeclaredAndMatchTheValidatedObject() throws {
        let schema: JSONObject = ["type": "object", "properties": ["template": ["type": "object", "x-arsenal-ordered": true], "text": ["type": "string"]]]
        let object = try WireJSON.decode(Data(Self.template.utf8))
        // The matching text of a validated object is kept (a re-validated, already split request).
        let kept = try OrderedJSONMembers.split(
            ["template": .object(object), OrderedJSONMembers.transport: ["template": .string(Self.template)]], schema: schema)
        #expect(kept.ordered == ["template": Self.template])
        // Text that decodes to something else than the validated object is refused.
        #expect(throws: WorkerFailure.self) {
            _ = try OrderedJSONMembers.split(["template": ["a": 1], OrderedJSONMembers.transport: ["template": #"{"a":2}"#]], schema: schema)
        }
        // A name the schema does not declare ordered is refused, even when the request has that member.
        #expect(throws: WorkerFailure.self) {
            _ = try OrderedJSONMembers.split(["text": "Ada", OrderedJSONMembers.transport: ["text": #""Ada""#]], schema: schema)
        }
        // A transport text without its member forwards nothing.
        #expect(try OrderedJSONMembers.split(["text": "Ada", OrderedJSONMembers.transport: ["template": #"{"a":1}"#]], schema: schema).ordered.isEmpty)
        #expect(throws: WorkerFailure.self) { _ = try OrderedJSONMembers.split([OrderedJSONMembers.transport: "{}"], schema: schema) }
    }

    @Test func orderedKeywordIsABooleanSchemaAnnotation() throws {
        try ModelSchemaValidation.checkSchema(schemaData: Data(#"{"type":"object","properties":{"template":{"type":"object","x-arsenal-ordered":true}}}"#.utf8))
        #expect(throws: (any Error).self) {
            try ModelSchemaValidation.checkSchema(schemaData: Data(#"{"type":"object","properties":{"template":{"type":"object","x-arsenal-ordered":"yes"}}}"#.utf8))
        }
    }
}
