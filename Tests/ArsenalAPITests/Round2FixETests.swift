// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalAPI
import ArsenalCore

@Test func brokenLocalResidentRemainsProviderVisibleAndErrorsAre409() async throws {
    let row = ModelRow(id: "local", name: "Local", kind: "voices", states: ["bf16": .init(state: .failed, error: "Bad folder", phase: "localFolderErrorLoaded")])
    let router = APIRouter(service: StubModelsService(rows: [row]), support: URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-e/api-\(UUID().uuidString)"))
    #expect(await router.round2ContainsResident(row.id))
    #expect(try await router.round2ResolveResident(row.id) == row.id)
    #expect(workerHTTPStatus(WorkerFailure(code: "local_folder", message: "Bad folder")) == 409)
}

private extension APIRouter {
    func round2ContainsResident(_ id: String) async -> Bool { await installedModels().contains { $0["id"] as? String == id } }
    func round2ResolveResident(_ id: String) async throws -> String? { try await resolve(id, kind: "voices")["id"] as? String }
}
