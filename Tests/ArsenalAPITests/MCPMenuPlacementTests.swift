// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import Testing
@testable import ArsenalAPI

@Test func mcpMenuPlacementValidatedAndAtomic() async throws {
    let support = URL(fileURLWithPath: "/private/tmp/arsenal-r12-menu-stability/mcp-\(UUID())")
    defer { try? FileManager.default.removeItem(at: support) }
    let declaration: JSONObject = ["id": "speed", "type": "number", "title": "Speed", "default": 1, "min": 0.5, "max": 2, "applies": "request", "maps_to": "request.speed"]
    let service = StubModelsService(
        // Show in menu belongs to a table model: one with weights on this Mac.
        rows: [ModelRow(id: "voice", name: "Voice", kind: "voices", states: ["bf16": .init(state: .ready)])], metadata: ["voice": ["setting": [.object(declaration)]]],
        supportDirectory: support)
    let router = APIRouter(service: service, support: support)
    func set(_ values: JSONObject) async throws -> Bool {
        let result = try await router.callMCPTool(
            MCPNames.tool("settings"), arguments: ["model": "voice", "set": .object(values)], requester: "Regression", server: nil, progress: nil)
        return result.isError != true
    }
    #expect(try await set(["show_in_menu": false, "speed": 1.5]))
    #expect(await service.rows().first?.toolsShown == false)
    #expect(try await service.settings(id: "voice").first?.value == 1.5)
    let savedBytes = try Data(contentsOf: support.appendingPathComponent("config.json"))
    let config = try ArsenalConfiguration.read(support.appendingPathComponent("config.json"))
    #expect(config.selections["voice"]?.toolsShown == false && config.settings["voice"]?["speed"] == 1.5)
    for invalid: JSONObject in [["show_in_menu": "false", "speed": 1], ["show_in_menu": true, "speed": 99], ["show_in_menu": true, "unknown": 1]] {
        #expect(try await set(invalid) == false)
        #expect(await service.rows().first?.toolsShown == false)
        #expect(try await service.settings(id: "voice").first?.value == 1.5)
        #expect(try Data(contentsOf: support.appendingPathComponent("config.json")) == savedBytes)
    }
    #expect(try await set(["show_in_menu": true]))
    #expect(await service.rows().first?.toolsShown == true)
}
