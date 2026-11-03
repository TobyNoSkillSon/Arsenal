// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore
import MCP
@testable import ArsenalAPI

/// The 25 Oct requests over MCP: delete one precision (bug note 19:51), revoke and re-approve local code (bug note 20:40),
/// and the model summary's parameters, languages and figures.
struct MCPDayPass26Tests {
    @Test func deleteTakesOnePrecisionAndSwitchesTheChoiceToTheCopyLeft() async throws {
        var row = ModelRow(
            id: "voice", name: "Voice", kind: "voices", params: "82M", precisions: ["fp16", "int4"],
            states: ["fp16": .init(state: .ready), "int4": .init(state: .ready)], selectedPrecision: "int4", downloadBytes: ["fp16": 170_000_000, "int4": 48_000_000])
        row.runner = "native"
        let h = try await MCPHarness(rows: [row]); defer { Task { await h.close() } }
        #expect(
            try await h.text("delete", ["model": "voice", "dtype": "int4"])
                == "Not deleted: Voice (voice) has int4 48 MB on disk. Ask the user. With their yes, call \(MCPNames.prefix)delete(model:\"voice\", dtype:\"int4\", confirm:true).")
        #expect(await h.service.rows().first?.states["int4"]?.state == .ready)
        let deleted = try await h.text("delete", ["model": "voice", "dtype": "int4", "confirm": true])
        #expect(deleted == "Deleted the int4 files of Voice (voice), about 48 MB freed. It now uses fp16, the copy still on disk.", "\(deleted)")
        let after = try #require(await h.service.rows().first)
        #expect(after.states["int4"]?.state != .ready && after.states["fp16"]?.state == .ready && after.selectedPrecision == "fp16")
        #expect(try await h.text("delete", ["model": "voice", "dtype": "int4", "confirm": true]) == "Voice has no downloaded int4 files; nothing deleted.")
        #expect(try await h.call("delete", ["model": "voice", "dtype": "q3"]).isError == true)
        #expect(try await h.call("delete", ["model": "voice", "dtype": "fp16", "remove": true, "confirm": true]).isError == true)
        #expect(await h.service.rows().first?.states["fp16"]?.state == .ready)
        let schema = try #require(MCPTools.all.first { $0.name == MCPNames.tool("delete") })
        #expect(String(decoding: try JSONEncoder().encode(schema.inputSchema), as: UTF8.self).contains("\"dtype\""))
    }

    @Test func settingsRevokeForgetsALocalApprovalAndApproveTakesTheNewDigest() async throws {
        var row = ModelRow(id: "mine", name: "Mine", kind: "voices", precisions: ["fp16"], states: ["fp16": .init(state: .ready)], selectedPrecision: "fp16")
        row.runner = "native"; row.local = true; row.contentDigest = String(repeating: "b", count: 64)
        row.approval = CodeApproval(by: .app, digest: String(repeating: "b", count: 64))
        let h = try await MCPHarness(rows: [row]); defer { Task { await h.close() } }
        let revoked = try await h.text("settings", ["model": "mine", "action": "revoke_code"])
        #expect(revoked.hasPrefix("Revoked the approval of Mine's local code (digest bbbbbbbbbbbb…). It runs again only after the user approves it"), "\(revoked)")
        let now = try #require(await h.service.rows().first)
        #expect(now.approval == nil && now.needsCodeConsent)
        #expect(try await h.text("settings", ["model": "mine", "action": "revoke_code"]) == "Mine's code is not approved; nothing changed.")
        #expect(try await h.text("settings", ["model": "mine", "action": "approve_code"]).hasPrefix("Approved Mine's local code (digest bbbbbbbbbbbb…)"))
        #expect(await h.service.rows().first?.needsCodeConsent == false)
        let bad = try await h.call("settings", ["action": "nonsense"])
        #expect(mcpResultText(bad).contains("refresh, approve_code, revoke_code or rollback"))
    }

    @Test func theModelSummaryPrintsParametersLanguagesAndFigures() async throws {
        var row = ModelRow(id: "voice", name: "Voice", kind: "voices", params: "82M", precisions: ["fp16"], states: ["fp16": .init(state: .ready)], selectedPrecision: "fp16")
        row.runner = "native"
        var stats = PrecisionStatSet(kind: .voices, dtype: "fp16", runner: "native")
        let provenance = StatProvenance(chip: "m5-max", date: "2026-10-20", method: "intake")
        stats.metrics[StatMetric.realTimeFactor.rawValue] = StatValue(state: .measured, value: 0.042, provenance: provenance)
        stats.metrics[StatMetric.peakMemoryGB.rawValue] = StatValue(state: .quickCheck, value: 1.3, provenance: provenance)
        row.precisionStats = ["fp16": stats]
        let h = try await MCPHarness(rows: [row], metadata: ["voice": ["languages": ["en", "pl"]]]); defer { Task { await h.close() } }
        let model = try await h.text("model", ["model": "voice"])
        let lines = model.components(separatedBy: "\n")
        #expect(lines.contains("Parameters: 82M. Languages: en, pl."), "\(model)")
        #expect(lines.contains("Figures: fp16 on m5-max: real-time factor 0.04, peak memory 1.21 GB (quick check)."), "\(model)")
        let find = try await h.text("find")
        #expect(find.contains("• voice — Voice:") && find.contains("82M parameters") && find.contains("languages: en, pl"), "\(find)")
        #expect(APIRouter.mcpLanguages(["und"]) == nil && APIRouter.mcpLanguages(["mul"]) == "multilingual" && APIRouter.mcpLanguages(["en"]) == "language: en")
    }
}

/// Core review 11: every MCP tool parameter says what it does (docs, confirm, remove, history had no words), and the
/// structured memory figures are in the GB the text says: the 100 GB limit is 100, never 107.37.
@Test func everyMCPParameterIsDescribedAndMemoryFiguresUseTheTextsGB() async throws {
    for tool in MCPTools.all {
        let schema = try WireJSON.decode(JSONEncoder().encode(tool.inputSchema))
        for (key, property) in schema["properties"]?.object ?? [:] {
            let words = property["description"]?.string ?? ""
            #expect(words.count > 0, "\(tool.name).\(key)")
        }
    }
    let h = try await MCPHarness(rows: [decisionRow(state: .ready)])
    defer { Task { await h.close() } }
    let status = try await h.call("status", ["format": "json"])
    let object = try WireJSON.decode(JSONEncoder().encode(try #require(status.structuredContent)))
    #expect(object["memory"]?["limit_gb"] == 100 && object["memory"]?["total_gb"] == 128 && object["memory"]?["in_use_gb"] == 40)
}
