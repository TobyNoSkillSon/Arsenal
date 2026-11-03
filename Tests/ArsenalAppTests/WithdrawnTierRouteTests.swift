// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
import ArsenalAPI
@testable import Arsenal

/// Agents remove a model's withdrawn precisions over HTTP, as the table's Remove does: `POST /models/{id}/delete-withdrawn`.
@Suite struct WithdrawnTierRouteTests {
    private func post(_ client: ArsenalClient, _ path: String) async throws -> (status: Int, body: [String: Any]) {
        let (data, response) = try await URLSession.shared.data(for: try client.request("POST", "/arsenal/v1/models/" + path, json: [String: Any]()))
        return ((response as? HTTPURLResponse)?.statusCode ?? 0, try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:])
    }

    @Test func deleteWithdrawnRemovesOnlyTheWithdrawnPrecisionsAndNamesThem() async throws {
        var row = ModelRow(id: "model", name: "Model", kind: "chat", precisions: ["bf16"])
        row.withdrawnTiers = ["int8": 1_200_000_000, "int4": 600_000_000]
        let service = FixtureService(rows: [row])
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-withdrawn-route-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let server = ArsenalAPIServer(service: service, supportDirectory: root, port: 0)
        let client = ArsenalClient(port: try await server.start())
        defer { server.stop() }

        // A refusal (a Get running, or a withdrawn precision loaded) keeps the files and answers 409.
        await service.fail("deleteWithdrawnTiers", with: WorkerFailure(code: "busy", message: "Finish Get or Load before removing files."))
        let refused = try await post(client, "model/delete-withdrawn")
        #expect(refused.status == 409)
        #expect(try #require(await service.rows().first).withdrawnTiers.count == 2)
        await service.fail("deleteWithdrawnTiers", with: nil)

        let removed = try await post(client, "model/delete-withdrawn")
        #expect(removed.status == 200 && removed.body["ok"] as? Bool == true)
        #expect(removed.body["removed"] as? [String] == ["int4", "int8"])
        #expect(try #require(await service.rows().first).withdrawnTiers.isEmpty)
        #expect(try #require(await service.rows().first).precisions == ["bf16"])

        // Nothing left: still a success, with nothing removed.
        let again = try await post(client, "model/delete-withdrawn")
        #expect(again.status == 200 && again.body["removed"] as? [String] == [])
        #expect(await service.calls.filter { $0 == "delete withdrawn model" }.count == 3)

        #expect(try await post(client, "missing/delete-withdrawn").status == 404)
    }
}
