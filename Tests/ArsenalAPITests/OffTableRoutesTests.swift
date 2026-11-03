// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
@testable import ArsenalAPI

/// A downloaded model runs its operations over HTTP; a model that is not downloaded is refused. The table holds exactly the
/// models with weights on this Mac: taking a downloaded model off without deleting its weights leaves it on
/// the table, and the one without weights is never there.
@Test func installedModelRunsItsOperationOverHTTPAndAMissingOneIsRefused() async throws {
    let installed = ModelRow(id: "off-installed", name: "Off Installed", kind: "images", states: ["bf16": .init(state: .ready)])
    let notInstalled = ModelRow(id: "off-missing", name: "Off Missing", kind: "images", states: ["bf16": .init(state: .notInstalled)])
    let op = ModelOperation(name: "generate", title: "Generate", inputSchema: ["type": "object"], output: "image", stream: false)
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-off-table-api-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let service = StubModelsService(
        rows: [installed, notInstalled], operations: [installed.id: [op], notInstalled.id: [op]], supportDirectory: root)
    for id in [installed.id, notInstalled.id] { try await service.remove(id: id) }
    #expect(await service.rows().map { [$0.id: $0.onTable] } == [[installed.id: true]])
    let server = ArsenalAPIServer(service: service, supportDirectory: root, port: 0)
    let port = try await server.start(); defer { server.stop() }
    func post(_ id: String) async throws -> (Int, Data) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/models/\(id)/ops/generate")!)
        request.httpMethod = "POST"; request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try WireJSON.encode(["prompt": "a red apple"])
        let (data, response) = try await URLSession.shared.data(for: request)
        return ((response as? HTTPURLResponse)?.statusCode ?? -1, data)
    }
    let (status, body) = try await post(installed.id)
    #expect(status == 200, "\(String(decoding: body, as: UTF8.self))")
    // Not downloaded: the operation is refused by request admission, never run.
    let (missing, _) = try await post(notInstalled.id)
    #expect(missing != 200)
}
