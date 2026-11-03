// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// The 25 Oct requests: added models stay out of the menu, a deleted chosen precision switches to a copy on disk, and the
/// owner's yes re-approves a changed local runner (with Revoke to forget it).
@Suite(.serialized) struct DayPass26CoreTests {
    @Test func aModelAddedToTheTableStartsWithShowInMenuOff() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"], addToTable: false); defer { f.stop() }
        let s = f.service, id = "fixture-voices"
        try await s.add(id: id)
        #expect(await catalogRow(s, id)?.toolsShown == false)
        #expect(try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json")).selections[id]?.toolsShown == false)
        // Turned on, then removed and added again: a new add starts off again.
        try await s.setToolsShown(id: id, on: true)
        #expect(await catalogRow(s, id)?.toolsShown == true)
        try await s.remove(id: id); try await s.add(id: id)
        #expect(await catalogRow(s, id)?.toolsShown == false)
        // Its download puts it on the table, still out of the menu.
        try await s.get(id: id, precision: "bf16")
        #expect(await s.rows().first { $0.id == id }?.toolsShown == false)
        await s.shutdown()
    }

    @Test func deletingTheChosenPrecisionSwitchesToACopyStillOnDisk() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
        let s = f.service, id = "fixture-chat"
        try await s.get(id: id, precision: "int4")
        try await s.setPrecision(id: id, precision: "int4")
        #expect(await s.rows().first?.selectedPrecision == "int4")
        _ = try await s.deletePrecision(id: id, precision: "int4")
        let row = try #require(await s.rows().first)
        #expect(row.states["int4"]?.state == .notInstalled)
        #expect(row.selectedPrecision == "bf16")
        #expect(try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json")).selections[id]?.precision == "bf16")
        await s.shutdown()
    }

    @Test func deletingAnotherPrecisionKeepsTheChoice() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
        let s = f.service, id = "fixture-chat"
        try await s.get(id: id, precision: "int4")
        try await s.setPrecision(id: id, precision: "bf16")
        _ = try await s.deletePrecision(id: id, precision: "int4")
        #expect(await s.rows().first?.selectedPrecision == "bf16")
        await s.shutdown()
    }

    /// Bug note 25 Oct 20:40: an approved local runner whose code changed was refused without parity and bench proof.
    @Test func theOwnersYesReapprovesAChangedLocalRunnerAndRevokeForgetsIt() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"], addToTable: false); defer { f.stop() }
        let s = f.service, id = "local-voices", local = f.paths.support.appendingPathComponent("local-models/" + id)
        for (path, data) in f.source.files {
            let relative = String(path.dropFirst("models/fixture-voices/".count)); let target = local.appendingPathComponent(relative)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            let bytes =
                relative == "model.toml"
                ? Data(
                    String(decoding: data, as: UTF8.self).replacingOccurrences(of: "fixture-voices", with: id).replacingOccurrences(
                        of: "language = \"swift\"", with: "language = \"rust\""
                    ).utf8) : data
            try bytes.write(to: target)
        }
        let first = try #require(await s.catalog().first { $0.id == id }?.row.contentDigest)
        try await s.add(id: id)
        try await s.approveLocalCode(id: id, contentDigest: first, approvedBy: .app)
        try await s.get(id: id, precision: "bf16"); try await s.load(id: id); try await s.unload(id: id)
        // The runner's code changes: its implementation fingerprint moves, and no results record any parity or speed check.
        let manifest = local.appendingPathComponent("model.toml")
        let changed = try String(contentsOf: manifest, encoding: .utf8).replacingOccurrences(of: "[[op]]", with: "[[op]]\n# runner code changed\n", options: [], range: nil)
        try changed.write(to: manifest, atomically: true, encoding: .utf8)
        try Data("// changed runner".utf8).write(to: local.appendingPathComponent("runner-patch.txt"))
        let entry = try #require(await s.catalog().first { $0.id == id })
        let second = try #require(entry.row.contentDigest)
        #expect(second != first && entry.row.needsCodeConsent && entry.row.approval?.digest == first)
        try await s.approveLocalCode(id: id, contentDigest: second, approvedBy: .mcp)
        let approved = try #require(await s.rows().first { $0.id == id })
        #expect(!approved.needsCodeConsent && approved.approval?.digest == second && approved.approval?.by == .mcp)
        try await s.load(id: id)
        // Revoke unloads it, forgets the approval, and the next Load asks again.
        try await s.revokeLocalCode(id: id)
        let revoked = try #require(await s.rows().first { $0.id == id })
        #expect(revoked.residentDType == nil && revoked.approval == nil && revoked.needsCodeConsent)
        let config = try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json"))
        #expect(config.localCodeConsent[id] == nil && config.localCodeApprovals[id] == nil)
        do { try await s.load(id: id); Issue.record("Revoked code loaded") } catch let e as WorkerFailure { #expect(e.code == "code_consent") }
        // No rollback brings back code without an approval.
        let previous = await s.previousRevisions[id], dormant = await s.dormantRevisions[id]
        #expect(previous == nil && dormant == nil)
        await #expect(throws: WorkerFailure.self) { try await s.rollbackRevision(id: id) }
        await #expect(throws: WorkerFailure.self) { try await s.revokeLocalCode(id: "fixture-voices") }
        await s.shutdown()
    }
}
