// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore
import MCP
@testable import ArsenalAPI

/// Jobs in simulated time: a 2.5 GB download with a speed change, a six-minute stall and a conversion step, waits that
/// time out, cancellation, restarts and failures.
@Suite struct MCPAgentJobTests {
    @Test func slowDownloadReportsETAHeartbeatsAcrossAStallAndEndsInTheRunResult() async throws {
        let h = try await MCPHarness(rows: [decisionRow()], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        let total: Int64 = 2_500_000_000
        // Speeds follow the simulated clock, so a loaded Mac cannot make 10 MB/s read as 7.3.
        let parts: [MCPDownloadPart] = [.steady(rate: 10_000_000, to: 1_000_000_000), .stall(360), .steady(rate: 20_000_000, to: total)]
        await h.service.setGetScript(mcpRatedScript(h.environment, total: total, parts: parts, derive: 10))
        let started = h.environment.now()
        let first = try await h.text("run", ["model": "laya", "input": .object(decideInput), "allow_download": true])
        #expect(h.environment.now().timeIntervalSince(started) < 8, "the first call returns within the confirm budget")
        #expect(
            first.range(
                of: #"^Downloading Laya Decisions: 2\.5 GB, about [45] min at (9(\.\d)?|10(\.\d)?) MB/s\. The decide runs as soon as it finishes\."#, options: .regularExpression)
                != nil, "\(first)")
        let id = try #require(MCPHarness.job(first))
        let job = try #require(await h.router.mcpJobs[id])
        // The default wait ends at 50 s with the next step, and progress messages carry the figures.
        let notes = MCPNotes(), environment = h.environment
        let notify: MCPNotify = { value, total, message in await notes.add(environment.now(), value, total, message) }
        let still = try await h.text("wait", ["job": .string(id)], notify: notify)
        #expect(
            still.hasPrefix("Still downloading Laya Decisions: 0.") && still.range(of: #"left at (8|9|10)(\.\d)? MB/s"#, options: .regularExpression) != nil
                && still.hasSuffix(". Call \(MCPNames.prefix)wait(job:\"\(id)\") again."),
            "\(still)")
        // One long wait across the stall, the faster second half and the conversion.
        let watcher = Task {
            var sawStall = false, sawFaster = false, sawConverting = false
            while await job.status == .running {
                let line = await job.statusLine()
                if line.contains("no data for") { sawStall = true }
                if let range = line.range(of: #"at \d+(\.\d)? MB/s"#, options: .regularExpression), let rate = Double(line[range].dropFirst(3).dropLast(5)), rate > 15 {
                    sawFaster = true
                }
                if line.hasPrefix("Converting Laya Decisions (time not measured yet)") { sawConverting = true }
                try? await environment.sleep(2)
            }
            return (sawStall, sawFaster, sawConverting)
        }
        let long = MCPNotes()
        let done = try await h.text("wait", ["job": .string(id), "timeout_s": 3600]) { value, total, message in await long.add(environment.now(), value, total, message) }
        let seen = await watcher.value
        #expect(seen.0 && seen.1 && seen.2, "stall \(seen.0), faster \(seen.1), converting \(seen.2)")
        #expect(done.hasPrefix("Downloaded Laya Decisions (2.5 GB in ") && done.contains("\nDecisions from laya (took ") && done.hasSuffix("bug: yes 0.77"), "\(done)")
        #expect(await notes.values.contains { $0.message.hasPrefix("Downloading Laya Decisions: ") })
        let values = await long.values
        #expect(values.count >= 10)
        for (previous, next) in zip(values, values.dropFirst()) {
            #expect(next.progress > previous.progress, "progress must strictly increase")
            #expect(next.at.timeIntervalSince(previous.at) <= 60, "a heartbeat at least every 60 s, also across the stall")
        }
        #expect(values.contains { $0.message.contains("no data for") } && values.allSatisfy { !$0.message.isEmpty })
        #expect(values.contains { $0.total == Double(total) } && values.contains { $0.total == nil })
        let count = values.count
        try await h.environment.sleep(120)
        #expect(await long.values.count == count, "nothing is sent after the job completed")
        // The measured rate is kept for the next estimate.
        #expect(await h.router.mcpRates.median(host: "other") != nil)
    }

    @Test func waitsTimeOutAtFiftySecondsOrTheGivenTimeout() async throws {
        let h = try await MCPHarness(rows: [decisionRow(bytes: 600_000_000)], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        await h.service.setGetScript(mcpScript(h.environment, total: 600_000_000, steps: mcpSteady(from: 0, to: 600_000_000, rate: 5_000_000)))
        let id = try #require(MCPHarness.job(try await h.text("download", ["model": "laya", "allow_download": true])))
        for timeout: JSONValue? in [nil, 5] {
            let before = h.environment.now()
            var arguments: JSONObject = ["job": .string(id)]
            if let timeout { arguments["timeout_s"] = timeout }
            let still = try await h.text("wait", arguments)
            let elapsed = h.environment.now().timeIntervalSince(before)
            #expect(still.hasPrefix("Still downloading Laya Decisions"), "\(still)")
            #expect(timeout == nil ? (elapsed >= 49 && elapsed < 58) : (elapsed >= 4.5 && elapsed < 10), "waited \(elapsed) s")
        }
        var answer = ""
        for _ in 0..<10 {
            answer = try await h.text("wait", ["job": .string(id)])
            if !answer.hasPrefix("Still") { break }
        }
        #expect(answer.hasPrefix("Downloaded Laya Decisions (600 MB in 2 min"), "\(answer)")
        let json = try await h.call("wait", ["job": .string(id), "format": "json"])
        #expect(json.structuredContent == nil, "a download has no output data; its text stays the answer")
    }

    @Test func cancellingAWaitLeavesTheJobAndCancellingTheRunStopsIt() async throws {
        let h = try await MCPHarness(rows: [decisionRow(bytes: 600_000_000)], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        await h.service.setGetScript(mcpScript(h.environment, total: 600_000_000, steps: mcpSteady(from: 0, to: 600_000_000, rate: 1_000_000)))
        let id = try #require(MCPHarness.job(try await h.text("run", ["model": "laya", "input": .object(decideInput), "allow_download": true])))
        let job = try #require(await h.router.mcpJobs[id])
        let waiter = Task { try await h.call("wait", ["job": .string(id), "timeout_s": 3600]) }
        for _ in 0..<200 where await !job.isWaitedOn { try await Task.sleep(for: .milliseconds(5)) }
        waiter.cancel()
        await #expect(throws: CancellationError.self) { try await waiter.value }
        #expect(await job.status == .running)
        let cancelled = try await h.text("cancel", ["job": .string(id)])
        #expect(cancelled == "Cancelled job \(id) (decide with Laya Decisions). Partial download kept; it resumes next time.")
        #expect(await job.status == .cancelled)
        let after = try await h.call("wait", ["job": .string(id)])
        #expect(after.isError == true && mcpResultText(after) == "[cancelled] Job \(id) (decide with Laya Decisions) was cancelled. Partial download kept; it resumes next time.")
        #expect(try await h.text("cancel", ["job": .string(id)]) == "Job \(id) already ended (cancelled); nothing to cancel.")
        // Cancelling the run call itself, inside its own budget, cancels its job.
        let second = try await MCPHarness(rows: [decisionRow()], operations: ["laya": [decideOperation]])
        defer { Task { await second.close() } }
        await second.service.setGetScript(mcpScript(second.environment, total: 2_500_000_000, steps: mcpSteady(from: 0, to: 2_500_000_000, rate: 100_000)))
        let call = Task { try await second.call("run", ["model": "laya", "input": .object(decideInput), "allow_download": true]) }
        for _ in 0..<400 where await second.router.mcpJobs.isEmpty { try await Task.sleep(for: .milliseconds(1)) }
        call.cancel()
        _ = try? await call.value
        let jobs = await second.router.mcpJobs.values
        #expect(jobs.count == 1)
        for job in jobs { #expect(await job.status == .cancelled) }
    }

    @Test func aRestartAnswersInterruptedAndFinishedJobsStayReadable() async throws {
        let h = try await MCPHarness(
            rows: [decisionRow(bytes: 600_000_000), decisionRow("ready", bytes: 1, state: .ready)], operations: ["laya": [decideOperation], "ready": [decideOperation]])
        defer { Task { await h.close() } }
        await h.service.setGetScript(mcpScript(h.environment, total: 600_000_000, steps: [(100_000, 1)]))
        let running = try #require(MCPHarness.job(try await h.text("download", ["model": "laya", "allow_download": true])))
        let finished = try await h.text("run", ["model": "ready", "input": .object(decideInput), "wait_seconds": 0])
        let finishedID = try #require(MCPHarness.job(finished))
        #expect(try await h.text("wait", ["job": .string(finishedID)]).hasPrefix("Decisions from ready"))
        // A crash: the second router reads the file the first one left.
        let restarted = APIRouter(service: h.service, support: h.root)
        try await restarted.prepareMCP()
        let interrupted = try await restarted.callMCPTool("\(MCPNames.prefix)wait", arguments: ["job": .string(running)], requester: "Restart", notify: nil)
        #expect(
            interrupted.isError == true
                && mcpResultText(interrupted)
                    == "[failed] Job \(running) (download of Laya Decisions) was interrupted by an Arsenal restart. Call the same tool again; a download resumes where it stopped."
        )
        let kept = mcpResultText(try await restarted.callMCPTool("\(MCPNames.prefix)wait", arguments: ["job": .string(finishedID)], requester: "Restart", notify: nil))
        #expect(kept.hasPrefix("Job \(finishedID) (decide with Laya Decisions) finished.\nDecisions from ready") && kept.hasSuffix("bug: yes 0.77"), "\(kept)")
        let history = try await restarted.callMCPTool("\(MCPNames.prefix)status", arguments: ["history": true], requester: "Restart", notify: nil)
        #expect(mcpResultText(history).contains("\(running) interrupted — download of Laya Decisions"))
        let missing = try await restarted.callMCPTool("\(MCPNames.prefix)wait", arguments: ["job": "0000000000"], requester: "Restart", notify: nil)
        #expect(mcpResultText(missing) == "[not_found] No job \"0000000000\". \(MCPNames.prefix)status lists running and recent jobs.")
        // A clean quit interrupts its own running jobs too.
        await h.router.stopMCP()
        let records = try JSONDecoder.mcp.decode([MCPJobRecord].self, from: Data(contentsOf: h.root.appendingPathComponent("mcp-jobs.json")))
        #expect(records.first { $0.id == running }?.status == .interrupted)
    }

    @Test func aFailedDownloadSaysHowFarItGotAndHowToResume() async throws {
        let h = try await MCPHarness(rows: [decisionRow()], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        let steps = mcpSteady(from: 0, to: 1_100_000_000, rate: 50_000_000) + [(1, nil)]
        await h.service.setGetScript(mcpScript(h.environment, total: 2_500_000_000, steps: steps, fail: URLError(.networkConnectionLost)))
        let id = try #require(MCPHarness.job(try await h.text("run", ["model": "laya", "input": .object(decideInput), "allow_download": true])))
        let failed = try await h.call("wait", ["job": .string(id), "timeout_s": 600])
        let text = mcpResultText(failed)
        #expect(failed.isError == true)
        #expect(text.hasPrefix("[download] Download of Laya Decisions failed at 1.1 of 2.5 GB: "), "\(text)")
        #expect(text.hasSuffix("Call \(MCPNames.prefix)download(model:\"laya\", allow_download:true) to resume; the user already agreed."))
    }

    @Test func aDownloadStartedInTheAppIsListedJoinedAndCancellable() async throws {
        let h = try await MCPHarness(rows: [decisionRow(bytes: 600_000_000)], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        await h.service.setGetScript(mcpScript(h.environment, total: 600_000_000, steps: mcpSteady(from: 0, to: 600_000_000, rate: 1_000_000)))
        let app = Task { try await h.service.get(id: "laya", precision: "fp16") }
        for _ in 0..<200 where (await h.service.rows().first?.states["fp16"]?.bytes ?? 0) == 0 { try await Task.sleep(for: .milliseconds(5)) }
        let status = try await h.text("status")
        #expect(status.contains("laya download started in the app: ") && status.contains("(\(MCPNames.prefix)cancel(model:\"laya\") stops it)"), "\(status)")
        // Joining needs no consent: the user started it.
        let joined = try await h.text("run", ["model": "laya", "input": .object(decideInput)])
        #expect(joined.hasPrefix("Downloading Laya Decisions"), "\(joined)")
        let stopped = try await h.text("cancel", ["model": "laya"])
        #expect(stopped == "Stopped the download of Laya Decisions. Partial download kept; it resumes next time.")
        app.cancel()
        _ = try? await app.value
        #expect(try await h.text("cancel", ["model": "laya"]) == "No download of Laya Decisions is running; nothing changed.")
        #expect(await h.service.getCalls == ["laya fp16"])
    }

    @Test func progressNotificationsRiseStrictlyEvenWhenFiguresFall() async throws {
        let h = try await MCPHarness(rows: [decisionRow("ready", bytes: 1, state: .ready)], operations: ["ready": [decideOperation]])
        defer { Task { await h.close() } }
        let entry = try await h.router.mcpEntry("ready")
        let job = try await h.router.startMCPJob(kind: .run, entry: entry, operation: "decide", dtype: "fp16", requester: "Test") { job in
            await job.download(PrecisionState(state: .downloading, bytes: 5_000_000, totalBytes: 10_000_000))
            try await h.environment.sleep(12)
            await job.runProgress(["step": 2, "total": 9])
            try await h.environment.sleep(12)
            await job.runProgress(["step": 3, "total": 9])
            try await h.environment.sleep(12)
            return mcpText("done")
        }
        let notes = MCPNotes(), environment = h.environment
        let result = try await h.router.mcpWaitResult(
            job, seconds: 600, notify: { value, total, message in await notes.add(environment.now(), value, total, message) }, cancelJob: false, json: false, again: true)
        #expect(mcpResultText(result) == "done")
        let values = await notes.values.map(\.progress)
        #expect(values.contains(5_000_000) && zip(values, values.dropFirst()).allSatisfy { $1 > $0 }, "\(values)")
        #expect(await notes.values.contains { $0.message.hasPrefix("Running Laya Decisions: step 3 of 9, about") })
    }
}
