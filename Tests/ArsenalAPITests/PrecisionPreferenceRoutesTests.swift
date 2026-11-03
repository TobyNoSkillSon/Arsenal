// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
@testable import ArsenalAPI

@Test func precisionPreferenceIsConsistentInHTTPAndMCPRowsAndDefaultActions() async throws {
    let support = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(".build/precision-api-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: support) }
    let row = ModelRow(id: "chosen", name: "Chosen fixture", kind: "chat", precisions: ["bf16", "int8"], states: ["bf16": .init(state: .loaded)])
    let service = StubModelsService(rows: [row], supportDirectory: support)
    let router = APIRouter(service: service, support: support)
    let server = ArsenalAPIServer(service: service, supportDirectory: support, port: 0)
    let port = try await server.start(); defer { server.stop() }
    try await service.setPrecision(id: row.id, precision: "int8")
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/models")!)
    request.timeoutInterval = 10
    let (bytes, response) = try await URLSession.shared.data(for: request)
    #expect((response as? HTTPURLResponse)?.statusCode == 200)
    let http = try JSONDecoder().decode(JSONValue.self, from: bytes)
    #expect(http.array?.first?["selectedPrecision"] == "int8")
    func call(_ name: String, _ arguments: JSONObject) async throws -> String {
        mcpResultText(try await router.callMCPTool(MCPNames.tool(name), arguments: arguments, requester: "Precision test", server: nil, progress: nil))
    }
    let found = try await call("find", ["format": "json"])
    #expect(try WireJSON.decode(Data(found.utf8))["models"]?.array?.first?["dtype"] == "int8")
    let inspected = try await call("model", ["model": .string(row.id)])
    #expect(inspected.contains("int8 not downloaded") && inspected.contains("selected"), "\(inspected)")
    #expect(await service.rows().first?.residentDType == "bf16")
    _ = try await call("download", ["model": .string(row.id), "allow_download": true])
    for _ in 0..<200 where await service.rows().first?.states["int8"]?.state != .ready { try await Task.sleep(for: .milliseconds(10)) }
    _ = try await call("load", ["model": .string(row.id)])
    #expect(await service.rows().first?.residentDType == "int8")
}

@Test func warnedInt4CanBeSelectedDownloadedAndLoadedThroughHTTPAndMCP() async throws {
    let support = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("arsenal-int4-api-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: support) }
    // bf16 is on this Mac, so the model is on the table (/models) while int4 is not downloaded yet.
    var row = ModelRow(id: "int4-fixture", name: "Fixture", kind: "images", precisions: ["bf16", "int8", "int4"], states: ["bf16": .init(state: .ready)])
    for dtype in ["int8", "int4"] {
        row.precisionStats[dtype] = PrecisionStatSet(kind: .images, dtype: dtype, runner: "native", unavailableReason: "Did not pass the quality check")
    }
    let service = StubModelsService(rows: [row], supportDirectory: support)
    let router = APIRouter(service: service, support: support)
    let server = ArsenalAPIServer(service: service, supportDirectory: support, port: 0)
    let port = try await server.start(); defer { server.stop() }
    func request(_ suffix: String, method: String = "GET", body: JSONObject = [:]) async throws -> (JSONValue, Int?) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/" + suffix)!)
        request.httpMethod = method; request.timeoutInterval = 10
        if method != "GET" { request.httpBody = try WireJSON.encode(body); request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        let (data, response) = try await URLSession.shared.data(for: request)
        return (try JSONDecoder().decode(JSONValue.self, from: data), (response as? HTTPURLResponse)?.statusCode)
    }
    let models = try await request("models")
    #expect(models.0.array?.first?["precisionQualityWarnings"]?["int4"]?.string == PrecisionStatSet.int4QualityWarning)
    // The same warning reaches the agent.
    let model = mcpResultText(try await router.callMCPTool(MCPNames.tool("model"), arguments: ["model": .string(row.id)], requester: "Int4 test", server: nil, progress: nil))
    #expect(model.contains("int4 not downloaded, quality may be lower"), "\(model)")
    try await service.setPrecision(id: row.id, precision: "int4")
    _ = try await router.callMCPTool(
        MCPNames.tool("download"), arguments: ["model": .string(row.id), "dtype": "int4", "allow_download": true], requester: "Int4 test", server: nil, progress: nil)
    for _ in 0..<200 where await service.rows().first?.states["int4"]?.state != .ready { try await Task.sleep(for: .milliseconds(10)) }
    _ = try await router.callMCPTool(MCPNames.tool("load"), arguments: ["model": .string(row.id), "dtype": "int4"], requester: "Int4 test", server: nil, progress: nil)
    #expect(await service.rows().first?.residentDType == "int4")
    do {
        try await service.setPrecision(id: row.id, precision: "int8")
        Issue.record("Failed int8 was offered")
    } catch let error as WorkerFailure { #expect(error.code == "unsupported_format") }
}
