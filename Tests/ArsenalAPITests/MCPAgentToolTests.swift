// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore
import MCP
@testable import ArsenalAPI

/// The 15-tool surface: listing, text-first results, the consent rule and the one-call run.
@Suite struct MCPAgentToolTests {
    @Test func toolListIsFifteenStableToolsWithoutOutputSchemas() async throws {
        let names = MCPTools.all.map(\.name)
        #expect(
            names == [
                "\(MCPNames.prefix)find", "\(MCPNames.prefix)model", "\(MCPNames.prefix)status", "\(MCPNames.prefix)add", "\(MCPNames.prefix)remove", "\(MCPNames.prefix)delete",
                "\(MCPNames.prefix)download", "\(MCPNames.prefix)load", "\(MCPNames.prefix)unload", "\(MCPNames.prefix)run",
                "\(MCPNames.prefix)wait", "\(MCPNames.prefix)cancel", "\(MCPNames.prefix)upload", "\(MCPNames.prefix)settings", "\(MCPNames.prefix)playback"
            ])
        #expect(MCPTools.all.allSatisfy { $0.outputSchema == nil && $0.annotations.openWorldHint != nil })
        #expect(Set(MCPTools.all.filter { $0.annotations.destructiveHint == true }.map(\.name)) == ["\(MCPNames.prefix)delete", "\(MCPNames.prefix)cancel"])
        #expect(
            Set(MCPTools.all.filter { $0.annotations.readOnlyHint == true }.map(\.name)) == [
                "\(MCPNames.prefix)find", "\(MCPNames.prefix)model", "\(MCPNames.prefix)status", "\(MCPNames.prefix)wait"
            ])
        for name in ["\(MCPNames.prefix)run", "\(MCPNames.prefix)wait"] {
            #expect(MCPTools.all.first { $0.name == name }?._meta?["anthropic/maxResultSizeChars"] == .int(500_000))
        }
        #expect(MCPTools.all.first { $0.name == "\(MCPNames.prefix)run" }?.description?.hasPrefix("Run a local model on this Mac: transcribe") == true)
        #expect(MCPTools.instructions.contains("allow_download:true") && MCPTools.instructions.count < 1400)
    }

    @Test func findAndModelAnswerInTextAndJSONOnRequest() async throws {
        var row = decisionRow()
        row.summary = "Typed yes/no, choice and score answers"; row.purpose = "Typed decisions over text"
        let large = ModelRow(id: "giant", name: "Giant", kind: "decisions", precisions: ["fp16"], selectedPrecision: "fp16", downloadBytes: ["fp16": 150_000_000_000])
        let h = try await MCPHarness(
            rows: [row, large, ModelRow(id: "plain", name: "Plain Voice", kind: "voices", precisions: ["fp16"], selectedPrecision: "fp16")],
            metadata: [
                "laya": [
                    "good_for": ["Bulk triage of text"], "limits": ["Text only.", "Checked live on 20 Oct at FP16: a short triage list sorted correctly."],
                    "tasks": ["decide"]
                ]
            ], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        let found = try await h.call("find", ["query": "bulk triage"])
        let text = mcpResultText(found)
        #expect(found.structuredContent == nil && found.isError != true && (try? WireJSON.decode(Data(text.utf8))) == nil)
        // Not downloaded, so not on the user's table (it holds exactly the models with files on this Mac).
        #expect(text.hasPrefix("1 model for \"bulk triage\":"), "\(text)")
        #expect(text.contains("• laya — Laya Decisions: Typed decisions over text. 0.6B parameters, 2.5 GB, needs ~3 GB RAM. Not on the user's table; not downloaded (2.5 GB)."))
        #expect(text.contains("  Good for: Bulk triage of text. Limits: Text only.\n  Checked: live on 20 Oct at FP16: a short triage list sorted correctly.\n"), "\(text)")
        #expect(!text.contains("Can't"), "\(text)")
        let all = try await h.text("find")
        #expect(all.contains("No good-for/limits notes yet.") && all.contains("1 more model doesn't fit this Mac (over the 100 GB limit)"), "\(all)")
        #expect(try await h.text("find", ["runs_here": false]).contains("giant"))
        let json = try await h.call("find", ["task": "decide", "format": "json"])
        let structured = try WireJSON.decode(JSONEncoder().encode(try #require(json.structuredContent)))
        #expect(mcpResultText(json) == String(decoding: try WireJSON.encode(structured), as: UTF8.self))
        #expect(structured["models"]?.array?.first?["good_for"] == ["Bulk triage of text"] && structured["too_large"] == 1)
        let model = try await h.text("model", ["model": "laya"])
        let lines = model.components(separatedBy: "\n")
        #expect(lines[0] == "Laya Decisions (laya) — Typed decisions over text. Apache-2.0, commercial use allowed.", "\(model)")
        #expect(lines[1].hasPrefix("Not on the user's table; fp16 not downloaded (2.5 GB), selected. Not loaded. Needs ~3 GB RAM at fp16; 60 GB free under the 100 GB limit now."))
        #expect(lines.contains("Parameters: 0.6B.") && lines.contains("Figures: none recorded yet."), "\(model)")
        #expect(model.contains("  decide (default) — Decide; returns data\n    questions: object (required) — Named questions"))
        #expect(model.contains("Run: \(MCPNames.prefix)run(model:\"laya\", input:{\"questions\":{}})"))
        let checked = "Checked: live on 20 Oct at FP16: a short triage list sorted correctly."
        #expect(lines.contains("Good for: Bulk triage of text. Limits: Text only.") && lines.contains(checked), "\(model)")
        #expect(model.contains("Full input schema: arsenal://models/laya/operations/decide/schema"))
        #expect(lines.last == "What it is for, exact input formats, limits and licence: \(MCPNames.prefix)model(model:\"laya\", docs:true).", "\(model)")
        let docs = try await h.text("model", ["model": "laya", "docs": true])
        #expect(docs.components(separatedBy: "\n").last == "README: unavailable for this model.", "\(docs)")
        let unknown = try await h.call("model", ["model": "layaa"])
        #expect(
            unknown.isError == true
                && mcpResultText(unknown) == "[unknown_model] No model \"layaa\". Did you mean laya? \(MCPNames.prefix)find(query:\"layaa\") searches the catalog.")
        let card = try await h.router.readMCPResource("arsenal://models/laya")
        let body = try #require(card.text)
        #expect(try WireJSON.decode(Data(body.utf8))["goodFor"] == ["Bulk triage of text"])
    }

    /// Core review 7: a download larger than its copy on disk (RIFE: an FP32 source cast to FP16) names both, so the
    /// consent line agrees with arsenal_model's size; a download the size of its copy says one figure.
    @Test func aDownloadLargerThanItsCopyOnDiskNamesBoth() async throws {
        let h = try await MCPHarness(rows: [decisionRow()])
        defer { Task { await h.close() } }
        let row = decisionRow(bytes: 23_000_000)
        func text(disk: Int64) async -> String {
            let entry = CatalogEntry(row: row, sizesByDtype: ["fp16": disk])
            let plan = GetPlan(id: row.id, dtype: "fp16", runner: "native", items: [.init(id: "w", name: "weights", kind: "weights", downloadBytes: 23_000_000)])
            return await h.router.mcpSizeAndTime(.init(entry: entry, dtype: "fp16", state: .notInstalled, needsGet: true, inProgress: false, update: false, plan: plan))
        }
        #expect(await text(disk: 11_000_000) == "a 23 MB download (11 MB on disk at fp16), time unknown until the download starts")
        #expect(await text(disk: 23_000_000) == "23 MB, time unknown until the download starts")
    }

    @Test func notDownloadedRunAsksWithSizeAndTimeAndDownloadsNothing() async throws {
        let h = try await MCPHarness(rows: [decisionRow()], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        try await h.service.remove(id: "laya")
        for force in [false, true] {
            let result = try await h.call("run", ["model": "laya", "input": .object(decideInput), "force": .bool(force)])
            let text = mcpResultText(result)
            #expect(result.isError != true)
            #expect(
                text == """
                    Not run: Laya Decisions isn't downloaded yet — 2.5 GB, time unknown until the download starts. Once downloaded it needs ~3 GB RAM (60 GB free now).
                    Ask the user. With their yes, call \(MCPNames.prefix)run again with the same input and allow_download:true; it downloads and runs.
                    """, "\(text)")
        }
        let calls = await h.service.getCalls, table = await h.service.rows()
        #expect(calls.isEmpty && table.isEmpty)
        // A measured rate on this Mac gives the estimate its basis.
        await h.router.mcpRates.record(host: "other", bytes: 1_000_000_000, seconds: 100)
        let measured = try await h.text("download", ["model": "laya"])
        #expect(measured.hasPrefix("Not downloaded: Laya Decisions isn't downloaded yet — 2.5 GB, about 4 min at this Mac's last measured 10 MB/s."), "\(measured)")
        #expect(measured.hasSuffix("call \(MCPNames.prefix)download again with allow_download:true; it downloads."))
        #expect(await h.service.getCalls.isEmpty)
    }

    @Test func consentedRunDownloadsOnceRunsAndPutsTheModelOnTheTable() async throws {
        // A speed needs 3 s of samples inside the first answer's 5 s window. At 2× that window is 2.5 s of real time: at 50×
        // (100 ms) and 10× a late first progress sample on a busy Mac left under 3 s, and the answer said "speed not measured
        // yet" (2 Nov, about one run in three).
        let h = try await MCPHarness(rows: [decisionRow(bytes: 100_000_000)], operations: ["laya": [decideOperation]], factor: 2)
        defer { Task { await h.close() } }
        try await h.service.remove(id: "laya")
        await h.service.setGetScript(mcpScript(h.environment, total: 100_000_000, steps: mcpSteady(from: 0, to: 100_000_000, rate: 10_000_000)))
        let started = h.environment.now()
        let first = try await h.text("run", ["model": "laya", "input": .object(decideInput), "allow_download": true])
        #expect(h.environment.now().timeIntervalSince(started) < 8)
        #expect(
            first.hasPrefix("Downloading Laya Decisions: 100 MB, under a minute at ")
                && first.contains(" MB/s. The decide runs as soon as it finishes."), "\(first)")
        let job = try #require(MCPHarness.job(first))
        #expect(first.hasSuffix("Job \(job). Tell the user, then call \(MCPNames.prefix)wait(job:\"\(job)\") to receive the result."))
        let done = try await h.text("wait", ["job": .string(job)])
        let lines = done.components(separatedBy: "\n")
        #expect(lines[0].hasPrefix("Downloaded Laya Decisions (100 MB in "), "\(done)")
        #expect(lines[1].hasPrefix("Decisions from laya (took ") && lines[2] == "bug: yes 0.77", "\(done)")
        // The user's table holds exactly the models with files on this Mac: the downloaded model is on it.
        let calls = await h.service.getCalls, table = await h.service.rows().filter(\.onTable).map(\.id)
        #expect(calls == ["laya fp16"] && table == ["laya"])
        #expect(await h.service.lastRunForce == false, "allow_download never implies force")
        // A second run uses the downloaded model directly.
        let again = try await h.text("run", ["model": "laya", "input": .object(decideInput)])
        let count = await h.service.getCalls.count
        #expect(again.hasPrefix("Decisions from laya") && count == 1)
        // Like the real service, an installed model's plan has nothing left to transfer.
        try await h.service.updateRow("laya") { $0.downloadBytes["fp16"] = 0 }
        #expect(try await h.text("download", ["model": "laya"]) == "Laya Decisions is already downloaded at fp16. Nothing to do.")
    }

    /// The table holds exactly the models with files on this Mac or a download under way. Using a
    /// downloaded model keeps it there; remove never hides files (delete does, with the user's yes); a download puts a
    /// model on the table, and add of one not downloaded waits for its download.
    @Test func downloadedModelsStayOnTheTableAndADownloadPutsOneThere() async throws {
        let h = try await MCPHarness(
            rows: [decisionRow(bytes: 100_000_000, state: .ready), decisionRow("other", bytes: 100_000_000), decisionRow("third", bytes: 100_000_000)],
            operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        func onTable(_ id: String) async -> Bool { await h.service.catalog(query: .init()).first { $0.id == id }?.added == true }
        // Remove refuses a model with files: taking it off the table deletes them, which needs the user's yes.
        let removed = try await h.text("remove", ["model": "laya"])
        #expect(
            removed == "Not removed: Laya Decisions (laya) is on the user's table because its files are on this Mac, and removing it deletes them. "
                + "Ask the user. With their yes, call \(MCPNames.prefix)delete(model:\"laya\", confirm:true).", "\(removed)")
        #expect(await h.service.catalog(query: .init()).first { $0.id == "laya" }?.row.states["fp16"]?.state == .ready)
        #expect(try await h.text("find", ["query": "laya decisions"]).contains("On the user's table; downloaded."))
        let ran = try await h.text("run", ["model": "laya", "input": .object(decideInput)])
        #expect(ran.hasPrefix("Decisions from laya") && ran.contains("bug: yes 0.77"), "\(ran)")
        #expect(await onTable("laya"))
        // Only the downloaded model is on the table; the others are in the catalog alone.
        let rows = await h.service.rows()
        #expect(rows.map(\.id) == ["laya"] && rows.map(\.onTable) == [true])
        #expect(try await h.text("unload", ["model": "laya"]).hasPrefix("Unloaded"))
        #expect(try await h.text("load", ["model": "laya"]).hasPrefix("Loaded Laya Decisions at fp16"))
        #expect(await onTable("laya"))
        #expect(try await h.text("settings", ["model": "laya", "set": ["keep_hot": true]]).contains("keep_hot"))
        #expect(await onTable("laya"))
        // Download with the user's yes puts the model on the table, even one never added.
        try await h.service.remove(id: "other")
        #expect(await !onTable("other"))
        await h.service.setGetScript(mcpScript(h.environment, total: 100_000_000, steps: mcpSteady(from: 0, to: 100_000_000, rate: 10_000_000)))
        let started = try await h.text("download", ["model": "other", "allow_download": true])
        let job = try #require(MCPHarness.job(started), "\(started)")
        #expect(try await h.text("wait", ["job": .string(job)]).hasPrefix("Downloaded Laya Decisions (100 MB in "))
        #expect(await h.service.getCalls == ["other fp16"])
        #expect(await onTable("other"))
        // Add of a model with no files records it but it joins the table only when its download starts.
        #expect(
            try await h.text("add", ["model": "third"])
                == "Nothing was downloaded, so Laya Decisions (third) is not on the user's table yet; it joins it when its download starts. "
                + "\(MCPNames.prefix)run downloads it once the user agrees.")
        #expect(await !onTable("third"))
        #expect(try await h.text("add", ["model": "laya"]) == "Laya Decisions (laya) is already on the user's models; nothing changed.")
        #expect(await h.service.rows().filter(\.onTable).map(\.id) == ["laya", "other"])
    }

    @Test func concurrentConsentedRunsShareOneDownload() async throws {
        let h = try await MCPHarness(rows: [decisionRow(bytes: 100_000_000)], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        await h.service.setGetScript(mcpScript(h.environment, total: 100_000_000, steps: mcpSteady(from: 0, to: 100_000_000, rate: 5_000_000)))
        async let a = h.text("run", ["model": "laya", "input": .object(decideInput), "allow_download": true])
        async let b = h.text("run", ["model": "laya", "input": .object(decideInput), "allow_download": true])
        let (first, second) = try await (a, b)
        let jobs = [MCPHarness.job(first), MCPHarness.job(second)].compactMap { $0 }
        #expect(jobs.count == 2 && Set(jobs).count == 2)
        for job in jobs { #expect(try await h.text("wait", ["job": .string(job), "timeout_s": 600]).contains("Decisions from laya")) }
        #expect(await h.service.getCalls == ["laya fp16"])
        try await h.service.updateRow("laya") { $0.downloadBytes["fp16"] = 0 }
        let joined = try await h.text("download", ["model": "laya"])
        #expect(joined.hasPrefix("Laya Decisions is already downloaded"))
    }

    @Test func diskShortageAndModelsThatNeverFitAreRefusedBeforeAnyGet() async throws {
        let h = try await MCPHarness(
            rows: [decisionRow(), decisionRow("giant", bytes: 150_000_000_000)], operations: ["laya": [decideOperation], "giant": [decideOperation]], freeDisk: 1_100_000_000)
        defer { Task { await h.close() } }
        let disk = try await h.call("run", ["model": "laya", "input": .object(decideInput), "allow_download": true])
        #expect(disk.isError == true && mcpResultText(disk) == "[disk] Not started: Laya Decisions needs 2.5 GB, 1.1 GB free on the startup disk.")
        #expect(try await h.text("run", ["model": "laya", "input": .object(decideInput)]).hasPrefix("[disk] Not started"))
        for arguments: JSONObject in [["allow_download": true], ["force": true], ["allow_download": true, "force": true]] {
            let never = try await h.call("run", arguments.merging(["model": "giant", "input": .object(decideInput)]) { a, _ in a })
            #expect(never.isError == true && mcpResultText(never) == "[memory] Won't run on this Mac: Laya Decisions needs ~140 GB RAM at fp16; the limit is 100 GB.")
        }
        #expect(try await h.text("load", ["model": "giant", "force": true]).hasPrefix("[memory] Won't run on this Mac"))
        #expect(await h.service.getCalls.isEmpty)
    }

    @Test func aPlanThatGrewMoreThanTenPercentAsksAgain() async throws {
        let h = try await MCPHarness(rows: [decisionRow(bytes: 1_000_000_000)], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        _ = try await h.text("download", ["model": "laya"])
        try await h.service.updateRow("laya") { $0.downloadBytes["fp16"] = 1_200_000_000 }
        let again = try await h.call("download", ["model": "laya", "allow_download": true])
        #expect(again.isError != true)
        #expect(mcpResultText(again).hasPrefix("The download changed since it was offered: now 1.2 GB, was 1 GB.\nNot downloaded: Laya Decisions isn't downloaded yet — 1.2 GB"))
        #expect(await h.service.getCalls.isEmpty)
        try await h.service.updateRow("laya") { $0.downloadBytes["fp16"] = 1_300_000_000 }
        let proceeds = try await h.text("download", ["model": "laya", "allow_download": true])
        #expect(!proceeds.hasPrefix("Not downloaded") && proceeds.contains("Laya Decisions"), "\(proceeds)")
        #expect(await h.service.getCalls == ["laya fp16"])
    }

    @Test func localImportsCloneWithoutConsentAndCopiesAsk() async throws {
        var row = decisionRow(bytes: 0); row.localCopyBytes = ["fp16": 3_000_000_000]
        for clones in [true, false] {
            let h = try await MCPHarness(rows: [row], operations: ["laya": [decideOperation]])
            // The core plan says per donor file whether it clones (same volume) or copies (another volume).
            if clones { await h.service.setCloneBytes("laya", "fp16", 3_000_000_000) }
            let text = try await h.text("run", ["model": "laya", "input": .object(decideInput)])
            if clones {
                let calls = await h.service.getCalls
                #expect(!text.hasPrefix("Not run") && calls == ["laya fp16"], "\(text)")
                // Core review, 2 Nov: a clone from the user's own files is not a download, and says so.
                let job = try #require(MCPHarness.job(text))
                let done = try await h.text("wait", ["job": .string(job)])
                #expect(!done.contains("Downloaded") && done.contains("Restored Laya Decisions from a copy already on this Mac"), "\(done)")
            } else {
                #expect(
                    text.hasPrefix(
                        "Not run: Laya Decisions isn't downloaded yet — no download needed: copies 3 GB from files already on this Mac."),
                    "\(text)")
                #expect(await h.service.getCalls.isEmpty)
            }
            await h.close()
        }
    }

    @Test func dependenciesAndUpdatesAreNamedInTheConsentLine() async throws {
        var row = decisionRow(bytes: 2_000_000_000, state: .updateAvailable)
        row.states["fp16"] = .init(state: .updateAvailable, phase: "updateAvailable")
        let h = try await MCPHarness(rows: [row], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        await h.service.setExtraPlanItems("laya", [GetPlanItem(id: "python", name: "the Python runtime for this model", kind: "runtime", downloadBytes: 180_000_000)])
        let text = try await h.text("run", ["model": "laya", "input": .object(decideInput)])
        #expect(
            text.hasPrefix(
                "Not run: Laya Decisions has an update that must be downloaded before it runs — 2.2 GB, time unknown until the download starts, including the Python runtime for this model (180 MB)."
            ), "\(text)")
        #expect(text.contains("it downloads and runs."))
    }

    @Test func invalidInputFailsBeforeConsentAndNamesTheFields() async throws {
        let h = try await MCPHarness(rows: [decisionRow()], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        let missing = try await h.call("run", ["model": "laya", "input": ["items": []]])
        #expect(missing.isError == true && mcpResultText(missing).hasPrefix("[invalid_input] Not run: the input doesn't match decide:"))
        #expect(mcpResultText(missing).hasSuffix(". \(MCPNames.prefix)model(model:\"laya\") lists the fields."), "\(mcpResultText(missing))")
        let wrong = try await h.call("run", ["model": "laya", "operation": "speak", "input": [:]])
        #expect(
            mcpResultText(wrong)
                == "[unsupported] Not run: Laya Decisions has no operation \"speak\". It offers: decide. \(MCPNames.prefix)model(model:\"laya\") shows their inputs.")
        let schema = try await h.call("run", ["model": "laya", "input": 3] as JSONObject)
        #expect(schema.isError == true && mcpResultText(schema).hasPrefix("[invalid_input]"))
        #expect(await h.service.getCalls.isEmpty)
    }
}
