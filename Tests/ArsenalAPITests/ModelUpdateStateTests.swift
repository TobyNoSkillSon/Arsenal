// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore
@testable import ArsenalAPI

@Test(arguments: [Int64(0), Int64(3_000_000)])
func pendingUpdateSurvivesHTTPAndMCP(networkBytes: Int64) async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-update-api-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let row = ModelRow(
        id: "update", name: "Pending update", kind: "sound", precisions: ["fp16"],
        states: ["fp16": .init(state: .updateAvailable, phase: "updateAvailable")], selectedPrecision: "fp16", downloadBytes: ["fp16": networkBytes])
    let service = StubModelsService(rows: [row], supportDirectory: root)
    let server = ArsenalAPIServer(service: service, supportDirectory: root, port: 0)
    let port = try await server.start(); defer { server.stop() }
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/models")!)
    request.timeoutInterval = 10
    let (data, response) = try await URLSession.shared.data(for: request)
    #expect((response as? HTTPURLResponse)?.statusCode == 200)
    let rows = try JSONDecoder().decode([JSONValue].self, from: data)
    #expect(rows.count == 1)
    #expect(rows[0]["states"]?["fp16"]?["state"] == "updateAvailable")
    #expect(rows[0]["states"]?["fp16"]?["phase"] == "updateAvailable")
    #expect(rows[0]["states"]?["fp16"]?["error"] == nil)
    let router = APIRouter(service: service, support: root)
    let model = mcpResultText(try await router.callMCPTool(MCPNames.tool("model"), arguments: ["model": "update"], requester: "Update test", server: nil, progress: nil))
    #expect(
        model.contains("fp16 downloaded (update pending)") && model.contains("An update must be downloaded before it runs: \(MCPNames.tool("download"))(model:\"update\")"),
        "\(model)")
    let listed = try await router.callMCPTool(MCPNames.tool("find"), arguments: ["format": "json"], requester: "Update test", server: nil, progress: nil)
    let models = try WireJSON.decode(JSONEncoder().encode(try #require(listed.structuredContent)))["models"]?.array
    #expect(models?.count == 1)
    #expect(models?.first?["state"] == "updateAvailable")
    let run = mcpResultText(try await router.callMCPTool(MCPNames.tool("download"), arguments: ["model": "update"], requester: "Update test", server: nil, progress: nil))
    #expect(
        networkBytes == 0 ? !run.hasPrefix("Not downloaded") : run.hasPrefix("Not downloaded: Pending update has an update that must be downloaded before it runs — 3 MB"), "\(run)"
    )
}
