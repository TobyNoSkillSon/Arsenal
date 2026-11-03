// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore
import Testing

@testable import Arsenal

/// The menu's repair of a row (ModelsMenu.prepareTool, the tool item's Get or Load) uses the precision the table shows: a
/// saved 8-bit preference is the 16-bit copy. A second repair while one runs does nothing.
@MainActor @Suite(.menuState, .serialized) final class TableRepairTests {
    private let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-table-repair-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: support) }

    /// A voice model saved at int8 (no cell any more), with its 16-bit copy in the given state. The table holds only models
    /// with weights or a Get, so a repair that Gets starts from a failed Get's row.
    private func row(_ bf16: ModelState) -> ModelRow {
        var r = ModelRow(
            id: "voice", name: "Voice", kind: "voices", params: "82M", precisions: ["bf16", "int8", "int4"],
            states: ["bf16": PrecisionState(state: bf16)], selectedPrecision: "int8", sizeBytes: 1_000_000_000)
        r.downloadBytes = ["bf16": 1_000_000_000]
        return r
    }

    private func menu(_ row: ModelRow) async -> (ModelsMenu, TableController, FixtureService) {
        let service = FixtureService(rows: [row], supportDirectory: support)
        let controller = TableController(service: service, chip: "M5 Max")
        let menu = ModelsMenu(controller: controller)
        menu.presentAlert = { _ in .alertFirstButtonReturn }
        controller.confirmGet = { _, _ in true }
        await controller.refresh()
        return (menu, controller, service)
    }

    @Test func aRepairOfASavedEightBitRowLoadsTheSixteenBitCopy() async throws {
        let (menu, controller, service) = await menu(row(.ready))
        menu.prepareTool(try #require(controller.rows.first), action: "Load")
        try await waitUntil { await service.calls.contains("load voice bf16") }
        #expect(await service.calls.allSatisfy { !$0.contains("int8") })
    }

    @Test func aRepairOfASavedEightBitRowGetsTheSixteenBitCopy() async throws {
        let (menu, controller, service) = await menu(row(.failed))
        menu.prepareTool(try #require(controller.rows.first), action: "Get")
        try await waitUntil { await service.approvedGets.count == 1 }
        #expect(await service.approvedGets.first?.dtype == "bf16")
    }

    @Test func aSecondRepairWhileOneRunsDoesNothing() async throws {
        let (menu, controller, service) = await menu(row(.failed))
        let voice = try #require(controller.rows.first)
        menu.prepareTool(voice, action: "Get")
        menu.prepareTool(voice, action: "Get")
        #expect(controller.working == ["voice"])
        try await waitUntil {
            let gets = await service.approvedGets.count
            return controller.working.isEmpty && gets == 1
        }
        #expect(await service.approvedGets.count == 1)
    }
}
