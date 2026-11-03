// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import Testing
@testable import ArsenalCore
import MCP
@testable import ArsenalAPI

/// One test per finding of the independent code review (MCP-CODE-REVIEW.md, 21 Oct).
@Suite struct MCPAgentReviewTests {
    // 1. A local import from another volume is a copy: it needs the user's yes and counts against free disk.
    @Test func copiesFromAnotherVolumeAskAndCountAgainstDisk() async throws {
        var row = decisionRow(bytes: 0); row.localCopyBytes = ["fp16": 30_000_000_000]
        let h = try await MCPHarness(rows: [row], operations: ["laya": [decideOperation]], freeDisk: 10_000_000_000)
        defer { Task { await h.close() } }
        await h.service.setCloneBytes("laya", "fp16", 5_000_000_000)
        let disk = try await h.call("run", ["model": "laya", "input": .object(decideInput)])
        #expect(mcpResultText(disk) == "[disk] Not started: Laya Decisions needs 25 GB, 10 GB free on the startup disk.", "\(mcpResultText(disk))")
        let roomy = try await MCPHarness(rows: [row], operations: ["laya": [decideOperation]])
        defer { Task { await roomy.close() } }
        await roomy.service.setCloneBytes("laya", "fp16", 5_000_000_000)
        let asked = try await roomy.text("run", ["model": "laya", "input": .object(decideInput)])
        #expect(asked.hasPrefix("Not run: Laya Decisions isn't downloaded yet — no download needed: copies 25 GB from files already on this Mac."), "\(asked)")
        #expect(await roomy.service.getCalls.isEmpty)
    }

    // 2. Making an int8 or int4 tier on this Mac writes gigabytes: it asks and is disk-checked like a download.
    @Test func makingATierOnThisMacAsksAndIsDiskChecked() async throws {
        var row = ModelRow(
            id: "laya", name: "Laya Decisions", kind: "decisions", precisions: ["fp16", "int8"], states: ["fp16": .init(state: .ready)], selectedPrecision: "fp16",
            downloadBytes: ["fp16": 0, "int8": 0])
        row.runner = "native"
        row.formats = [ModelFormat(id: "int8", bitsPerWeight: 8.5, source: .derived, runners: ["native"], sizeBytes: 1_400_000_000)]
        let metadata: JSONObject = ["derive": ["int8": "native:laya"], "weights": ["dtype": "fp16"]]
        let h = try await MCPHarness(rows: [row], metadata: ["laya": metadata], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        let load = try await h.call("load", ["model": "laya", "dtype": "int8"])
        let text = mcpResultText(load)
        #expect(load.isError != true)
        #expect(
            text.hasPrefix(
                "Not loaded: Laya Decisions isn't made at int8 yet — nothing to download: Arsenal makes int8 from the installed weights on this Mac, writing about 1.4 GB"),
            "\(text)")
        #expect(await h.service.getCalls.isEmpty)
        let tight = try await MCPHarness(rows: [row], metadata: ["laya": metadata], operations: ["laya": [decideOperation]], freeDisk: 1_000_000)
        defer { Task { await tight.close() } }
        let refused = try await tight.text("load", ["model": "laya", "dtype": "int8", "allow_download": true])
        #expect(refused == "[disk] Not started: Laya Decisions needs 1.4 GB, 1 MB free on the startup disk.", "\(refused)")
        #expect(await tight.service.getCalls.isEmpty)
    }

    // 3. Cancelling one job never stops a download someone else is waiting for, one started in the app, or one that is over.
    @Test func cancellingAJobLeavesDownloadsOthersNeed() async throws {
        let h = try await MCPHarness(rows: [decisionRow(bytes: 100_000_000)], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        await h.service.setGetScript(mcpScript(h.environment, total: 100_000_000, steps: mcpSteady(from: 0, to: 100_000_000, rate: 2_000_000)))
        let a = try #require(MCPHarness.job(try await h.text("run", ["model": "laya", "input": .object(decideInput), "allow_download": true])))
        let b = try #require(MCPHarness.job(try await h.text("run", ["model": "laya", "input": .object(decideInput), "allow_download": true])))
        #expect(try await h.text("cancel", ["job": .string(a)]).hasPrefix("Cancelled job \(a)"))
        let shared = try await h.text("wait", ["job": .string(b), "timeout_s": 600]) { _, _, _ in }
        #expect(shared.contains("Decisions from laya"), "\(shared)")
        let cancels = await h.service.cancelGetCalls, gets = await h.service.getCalls
        #expect(cancels.isEmpty && gets == ["laya fp16"])

        // A download the user started in the app: the agent's cancel stops its own job only.
        let app = try await MCPHarness(rows: [decisionRow(bytes: 40_000_000)], operations: ["laya": [decideOperation]])
        defer { Task { await app.close() } }
        await app.service.setGetScript(mcpScript(app.environment, total: 40_000_000, steps: mcpSteady(from: 0, to: 40_000_000, rate: 2_000_000)))
        let started = Task { try await app.service.get(id: "laya", precision: "fp16") }
        for _ in 0..<200 where (await app.service.rows().first?.states["fp16"]?.bytes ?? 0) == 0 { try await Task.sleep(for: .milliseconds(5)) }
        let joined = try #require(MCPHarness.job(try await app.text("run", ["model": "laya", "input": .object(decideInput)])))
        _ = try await app.text("cancel", ["job": .string(joined)])
        try await started.value
        let appCancels = await app.service.cancelGetCalls, appState = await app.service.rows().first?.states["fp16"]?.state
        #expect(appCancels.isEmpty && appState == .ready)

        // A run cancelled after its download finished does not touch Get at all.
        let late = try await MCPHarness(rows: [decisionRow(bytes: 10_000_000)], operations: ["laya": [decideOperation]])
        defer { Task { await late.close() } }
        let environment = late.environment
        await late.service.setRunHook { _, _ in try await environment.sleep(600) }
        let run = try #require(MCPHarness.job(try await late.text("run", ["model": "laya", "input": .object(decideInput), "allow_download": true, "wait_seconds": 30])))
        let job = try #require(await late.router.mcpJobs[run])
        for _ in 0..<400 where await job.phase != .running { try await Task.sleep(for: .milliseconds(5)) }
        _ = try await late.text("cancel", ["job": .string(run)])
        #expect(await late.service.cancelGetCalls.isEmpty)

        // The only waiter on an MCP download does stop it.
        let alone = try await MCPHarness(rows: [decisionRow(bytes: 100_000_000)], operations: ["laya": [decideOperation]])
        defer { Task { await alone.close() } }
        await alone.service.setGetScript(mcpScript(alone.environment, total: 100_000_000, steps: mcpSteady(from: 0, to: 100_000_000, rate: 2_000_000)))
        let only = try #require(MCPHarness.job(try await alone.text("download", ["model": "laya", "allow_download": true])))
        _ = try await alone.text("cancel", ["job": .string(only)])
        #expect(await alone.service.cancelGetCalls == ["laya"])
    }

    // 5. Path work runs off the router actor with a deadline, and a cross-volume copy is a job phase with progress.
    @Test func pathWorkNeverBlocksTheRouterAndCopiesShowProgress() async throws {
        let outside = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-mcp-copy-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: outside) }
        let clip = outside.appendingPathComponent("clip.wav"); try Data(repeating: 7, count: 20_000_000).write(to: clip)
        let voice = ModelRow(id: "voice", name: "Voice", kind: "voices", states: ["bf16": .init(state: .ready)])
        let slow = try await MCPHarness(
            rows: [voice],
            cloneFile: { _, _ in
                Thread.sleep(forTimeInterval: 1.5); return EXDEV
            }, pathDeadline: 1_000_000)
        defer { Task { await slow.close() } }
        let running = Task { try await slow.call("run", ["model": "voice", "input": ["text": "Hi", "reference": ["path": .string(clip.path)]], "play": false]) }
        try await Task.sleep(for: .milliseconds(200))
        let clock = ContinuousClock(), before = clock.now
        _ = try await slow.text("status")
        #expect(before.duration(to: clock.now) < .milliseconds(700), "status answered while a path was being cloned")
        let ran = try await running.value
        #expect(ran.isError != true, "\(mcpResultText(ran))")
        let job = try #require(await slow.router.mcpJobs.values.first)
        let copied = await job.copied, total = await job.copyTotal
        #expect(copied == 20_000_000 && total == 20_000_000)
        let inbox = try FileManager.default.subpathsOfDirectory(atPath: slow.root.appendingPathComponent("inbox").path).filter { $0.hasSuffix("clip.wav") }
        #expect(inbox.count == 1)
        // A check that outlasts the deadline (a privacy prompt waiting for the user) is refused instead of waiting.
        let stuck = try await MCPHarness(
            rows: [voice],
            cloneFile: { _, _ in
                Thread.sleep(forTimeInterval: 0.5); return 0
            }, pathDeadline: 20)
        defer { Task { await stuck.close() } }
        let refused = try await stuck.text("run", ["model": "voice", "input": ["text": "Hi", "reference": ["path": .string(clip.path)]]])
        #expect(refused.hasPrefix("[invalid_input] Not run: reading the input files took longer than 20 s."), "\(refused)")
    }

    // 6. Hard links into the support folder are refused and inbox paths are used by their real path.
    @Test func hardLinksAndSymlinkedInboxPathsAreHandled() async throws {
        let h = try await MCPHarness(rows: [])
        defer { Task { await h.close() } }
        try await h.router.prepareMCP()
        let outside = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-mcp-links-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: outside) }
        // Same volume as the support folder (both in the temporary directory), so the hard link can be made.
        let alias = outside.appendingPathComponent("token.wav")
        #expect(link(h.root.appendingPathComponent("mcp-token").path, alias.path) == 0)
        await #expect(throws: MCPFailure(.invalidInput, "Not run: input.audio.path points into Arsenal's own support folder, which is not an input location.")) {
            _ = try await h.router.mcpPrepareInputs(["audio": ["path": .string(alias.path)]], environment: h.environment, importing: true)
        }
        let folder = h.root.appendingPathComponent("inbox/fixture"); try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let inside = folder.appendingPathComponent("clip.wav"); try Data("clip".utf8).write(to: inside)
        let pointer = outside.appendingPathComponent("pointer.wav"); try FileManager.default.createSymbolicLink(at: pointer, withDestinationURL: inside)
        let (rewritten, copies) = try await h.router.mcpPrepareInputs(["audio": ["path": .string(pointer.path)]], environment: h.environment, importing: true)
        #expect(rewritten["audio"]?["path"]?.string == mcpReal(inside) && copies.isEmpty, "the real inbox path, never the symlink")
        let hard = folder.appendingPathComponent("hard.wav"); #expect(link(inside.path, hard.path) == 0)
        await #expect(throws: MCPFailure(.invalidInput, "Not run: \(hard.path) is a hard link; pass the file itself.")) {
            _ = try await h.router.mcpPrepareInputs(["audio": ["path": .string(hard.path)]], environment: h.environment, importing: true)
        }
    }

    // 7. Adding a model to the table keeps the precision the call chose.
    @Test func addingKeepsTheChosenPrecision() async throws {
        let row = ModelRow(
            id: "laya", name: "Laya Decisions", kind: "decisions", precisions: ["fp16", "int8"], states: ["fp16": .init(state: .ready)], selectedPrecision: "fp16",
            runner: "native", downloadBytes: ["fp16": 0, "int8": 0])
        let h = try await MCPHarness(rows: [row], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        try await h.service.remove(id: "laya")
        await h.service.setAddSelects("laya", "int8")
        let ran = try await h.text("run", ["model": "laya", "input": .object(decideInput)])
        #expect(ran.hasPrefix("Decisions from laya"), "\(ran)")
        let selected = await h.service.rows().first?.selectedPrecision, gets = await h.service.getCalls
        #expect(selected == "fp16" && gets.isEmpty)
    }

    // 8. Only the oldest finished jobs leave memory; their result stays readable in full; a cancel before attach sticks.
    @Test func evictionKeepsUnreadResultsAndCancelBeforeAttachStopsTheTask() async throws {
        let h = try await MCPHarness(rows: [decisionRow("ready", bytes: 0, state: .ready)], operations: ["ready": [decideOperation]])
        defer { Task { await h.close() } }
        let first = try #require(MCPHarness.job(try await h.text("run", ["model": "ready", "input": .object(decideInput), "wait_seconds": 0])))
        for _ in 0..<400 where await h.router.mcpJobs[first]?.status == .running { try await Task.sleep(for: .milliseconds(5)) }
        let entry = try await h.router.mcpEntry("ready")
        for _ in 0..<(APIRouter.mcpJobLimit - 1) {
            _ = try await h.router.startMCPJob(kind: .run, entry: entry, operation: "decide", dtype: "fp16", requester: "Test") { _ in mcpText("quick") }
        }
        let running = try await h.router.startMCPJob(kind: .run, entry: entry, operation: "decide", dtype: "fp16", requester: "Test") { _ in
            try await h.environment.sleep(100_000); return mcpText("never")
        }
        _ = try await h.router.startMCPJob(kind: .run, entry: entry, operation: "decide", dtype: "fp16", requester: "Test") { _ in mcpText("one more") }
        #expect(await h.router.mcpJobs[first] == nil, "the oldest finished job left")
        let count = await h.router.mcpJobs.count, kept = await h.router.mcpJobs[running.id] != nil
        #expect(count == APIRouter.mcpJobLimit && kept)
        let read = try await h.text("wait", ["job": .string(first)])
        #expect(read.hasPrefix("Job \(first) (decide with Laya Decisions) finished.\nDecisions from ready") && read.hasSuffix("bug: yes 0.77"), "\(read)")
        await running.cancel()
        let job = MCPJob(kind: .run, model: "m", name: "M", operation: nil, dtype: "fp16", requester: "Test", environment: h.environment)
        await job.cancel()
        let task = Task<Void, Never> { _ = try? await Task.sleep(for: .seconds(30)) }
        await job.attach(task)
        #expect(task.isCancelled)
    }

    // 9. A refusal after a finished download says the download is kept.
    @Test func aRefusalAfterADownloadSaysTheFilesAreKept() async throws {
        let h = try await MCPHarness(rows: [decisionRow(bytes: 100_000_000)], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        await h.service.setGetScript(mcpScript(h.environment, total: 100_000_000, steps: mcpSteady(from: 0, to: 100_000_000, rate: 5_000_000)))
        let refusal = LoadMemoryRefusal(model: "laya", name: "Laya Decisions", precision: "fp16", neededGB: 20.4, freeGB: 11.2, limitGB: 107.4, totalGB: 137.4)
        await h.service.setRunHook { _, _ in throw refusal.failure }
        let id = try #require(MCPHarness.job(try await h.text("run", ["model": "laya", "input": .object(decideInput), "allow_download": true])))
        let text = try await h.text("wait", ["job": .string(id)])
        #expect(text.hasPrefix("[memory] Downloaded Laya Decisions (100 MB in ") && text.contains("), kept on this Mac. Not run: Laya Decisions needs ~19 GB"), "\(text)")
    }

    // 10. Without a progress token nothing keeps the stdio bridge's request alive, so a wait answers within 50 s.
    @Test func waitsWithoutProgressTokensAreCappedAtFiftySeconds() async throws {
        let h = try await MCPHarness(rows: [decisionRow(bytes: 600_000_000)], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        await h.service.setGetScript(mcpScript(h.environment, total: 600_000_000, steps: mcpSteady(from: 0, to: 600_000_000, rate: 1_000_000)))
        let id = try #require(MCPHarness.job(try await h.text("download", ["model": "laya", "allow_download": true])))
        let before = h.environment.now()
        let still = try await h.text("wait", ["job": .string(id), "timeout_s": 600])
        let elapsed = h.environment.now().timeIntervalSince(before)
        #expect(still.hasPrefix("Still downloading") && elapsed < 58, "\(elapsed) s")
        let before2 = h.environment.now()
        _ = try await h.text("wait", ["job": .string(id), "timeout_s": 100]) { _, _, _ in }
        #expect(h.environment.now().timeIntervalSince(before2) >= 95, "a client with a progress token may wait longer")
    }

    // 11. find filters by language and by state (a broken download).
    @Test func findFiltersByLanguageAndState() async throws {
        var polish = decisionRow("laya-multilingual"); polish.name = "Laya Multilingual"
        polish.states["fp16"] = .init(state: .failed, error: "connection reset")
        let h = try await MCPHarness(
            rows: [polish, decisionRow("laya-english")], metadata: ["laya-multilingual": ["languages": ["en", "pl"]], "laya-english": ["languages": ["en"]]])
        defer { Task { await h.close() } }
        let pl = try await h.text("find", ["language": "pl"])
        #expect(pl.hasPrefix("1 model in language \"pl\"") && pl.contains("• laya-multilingual"), "\(pl)")
        let failed = try await h.text("find", ["state": "failed"])
        #expect(failed.contains("• laya-multilingual") && !failed.contains("• laya-english") && failed.contains("the last download failed: connection reset"), "\(failed)")
        #expect(try await h.text("find", ["query": "polish pl"]).contains("• laya-multilingual"))
    }
}
