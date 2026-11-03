// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore
@testable import ArsenalAPI

/// A decisions service with elephant-1's shipped decide schema that checks what reaches it as RealModelsService does
/// (`checkSchema3Input`): the raw body the HTTP route forwards, or the members an async job sends.
private actor ConfiningDecisionService: ModelsService {
    let support: URL
    let operation: ModelOperation
    var forwarded: [JSONObject] = []
    init(support: URL, schema: JSONObject) {
        self.support = support
        operation = ModelOperation(name: "decide", title: "Decide", inputSchema: schema, output: "json", stream: false, schemaVersion: 3)
    }
    func rows() async -> [ModelRow] { [ModelRow(id: "elephant-1", name: "elephant-1", kind: "decisions", states: ["bf16": .init(state: .loaded)])] }
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
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject {
        if params.count == 1, let body = params["body"]?.string {
            try RealModelsService.checkSchema3Input(try LastKeyJSON.parse(Data(body.utf8)), schema: operation.inputSchema, support: support)
        } else {
            let (request, _) = try OrderedJSONMembers.split(params, schema: operation.inputSchema)
            try RealModelsService.checkSchema3Input(.object(request), schema: operation.inputSchema, support: support)
        }
        forwarded.append(params)
        return ["answers": ["q": ["noul": 0.5]]]
    }
}

/// Release review E (L-E1): the raw-body path confinement through the HTTP route with elephant-1's real ops schema.
/// A picture or a video frame outside Arsenal's inbox is refused with 422, sync or as a job, and never reaches the worker.
struct DecisionRouteConfinementTests {
    static let ops = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("models/elephant-1/ops")

    func post(_ port: Int, _ path: String, _ body: String, async: Bool = false, session: URLSession) async throws -> (Int, JSONObject) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
        request.httpMethod = "POST"; request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if async { request.setValue("respond-async", forHTTPHeaderField: "Prefer") }
        request.httpBody = Data(body.utf8)
        let (data, response) = try await session.data(for: request)
        return ((response as? HTTPURLResponse)?.statusCode ?? 0, (try? WireJSON.decode(data)) ?? [:])
    }

    @Test func outsidePicturesAndFramesAreRefusedOnTheRoute() async throws {
        let schema = try WireJSON.decode(Data(contentsOf: Self.ops.appendingPathComponent("decide.input.json")))
        // The shipped schema marks both a picture's path and each video frame's path as files.
        let text = String(decoding: try Data(contentsOf: Self.ops.appendingPathComponent("decide.input.json")), as: UTF8.self)
        #expect(text.components(separatedBy: "x-arsenal-file").count == 3, "picture and frame paths are both files")
        let root = URL(fileURLWithPath: "/private/tmp/arsenal-route-confinement/\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let png = Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLttAAAAABJRU5ErkJggg==")!
        let inbox = root.appendingPathComponent("inbox/pixel.png"), outside = root.appendingPathComponent("outside/pixel.png")
        for url in [inbox, outside] {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try png.write(to: url)
        }
        let service = ConfiningDecisionService(support: root, schema: schema)
        let server = ArsenalAPIServer(service: service, supportDirectory: root, port: 0)
        let port = try await server.start(); defer { server.stop() }
        let config = URLSessionConfiguration.ephemeral; config.connectionProxyDictionary = [:]
        let session = URLSession(configuration: config); defer { session.invalidateAndCancel() }

        func body(_ media: String) -> String { #"{"state":"look <image:1>","questions":{"q":{"type":"noul"}},"media":"# + media + "}" }
        func video(_ frame: URL) -> String {
            let frames = #"[{"path":"\#(inbox.path)"},{"path":"\#(frame.path)"}]"#
            return #"{"state":"watch <video:1>","questions":{"q":{"type":"noul"}},"media":[{"type":"video","frames":"# + frames + #","fps":1}]}"#
        }
        let route = "/arsenal/v1/models/elephant-1/ops/decide"
        // Inside the inbox, a picture and a video both reach the worker.
        var (status, reply) = try await post(port, route, body(#"[{"type":"image","path":"\#(inbox.path)"}]"#), session: session)
        #expect(status == 200, "\(reply)")
        (status, reply) = try await post(port, route, video(inbox), session: session)
        #expect(status == 200, "\(reply)")
        #expect(await service.forwarded.count == 2)
        // Outside it: refused with 422 on the sync route, a picture or a frame, and nothing more is forwarded.
        for refused in [body(#"[{"type":"image","path":"\#(outside.path)"}]"#), video(outside)] {
            (status, reply) = try await post(port, route, refused, session: session)
            #expect(status == 422, "\(reply)")
        }
        #expect(await service.forwarded.count == 2)
        // As a job: the member path refuses the outside picture and frame; the job fails as invalid_request (422).
        for refused in [body(#"[{"type":"image","path":"\#(outside.path)"}]"#), video(outside)] {
            (status, reply) = try await post(port, route, refused, async: true, session: session)
            #expect(status == 202, "\(reply)")
            let url = try #require(reply["urls"]?["get"]?.string.flatMap(URL.init(string:)))
            var job: JSONObject = [:]
            for _ in 0..<400 {
                job = (try? WireJSON.decode(try await session.data(from: url).0)) ?? [:]
                if job["status"]?.string == "failed" || job["status"]?.string == "succeeded" { break }
                try await Task.sleep(for: .milliseconds(10))
            }
            #expect(job["status"]?.string == "failed" && job["failure"]?["status"]?.number == 422, "\(job)")
        }
        #expect(await service.forwarded.count == 2)
    }
}
