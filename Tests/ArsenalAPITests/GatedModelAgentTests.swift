// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore
import MCP
@testable import ArsenalAPI

/// A Get refused by a gated Hugging Face model reaches agents as `[waiting_on_person]` (MCP) and `waiting_on_person`
/// (HTTP), with the page, what the user must do and that the app retries; never as an opaque download failure.
@Suite struct GatedModelAgentTests {
    static let withToken = HFGateError(repo: "org/laya", hasToken: true, mode: "auto")
    static let withoutToken = HFGateError(repo: "org/laya", hasToken: false, mode: "manual")

    func finished(_ h: MCPHarness, _ started: String) async throws -> String {
        // A refusal that comes at once is the call's own answer; otherwise the job's.
        guard let id = MCPHarness.job(started) else { return started }
        var answer = ""
        for _ in 0..<10 {
            answer = try await h.text("wait", ["job": .string(id)])
            if !answer.hasPrefix("Still") { break }
        }
        return answer
    }

    @Test(arguments: ["download", "run"]) func downloadAndRunSayAPersonMustAcceptTheTerms(tool: String) async throws {
        let h = try await MCPHarness(rows: [decisionRow(bytes: 600_000_000)], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        await h.service.setGetScript { _, _, _ in throw Self.withToken }
        var arguments: JSONObject = ["model": "laya", "allow_download": true]
        if tool == "run" { arguments["input"] = .object(decideInput) }
        let answer = try await finished(h, try await h.text(tool, arguments))
        #expect(
            answer
                == "[waiting_on_person] Laya Decisions is gated on Hugging Face: the user must accept its terms at https://huggingface.co/org/laya, signed in to "
                + "Hugging Face. Arsenal opened that page and retries every 30 s, up to 10 times. Tell the user; once they accept, Arsenal's next retry "
                + "downloads it (calling again meanwhile only reports this wait).", "\(answer)")
    }

    static func waitingRow(gaveUp: Bool) -> ModelRow {
        var row = decisionRow(bytes: 600_000_000)
        let wait = GateWait(
            repo: "org/laya", page: "https://huggingface.co/org/laya", hasToken: true, mode: "auto", tries: gaveUp ? 10 : 2, limit: 10,
            retryAt: gaveUp ? nil : Date().addingTimeInterval(20))
        row.states["fp16"] = PrecisionState(state: .failed, error: "gated", phase: PrecisionState.gatedPhase, gate: wait)
        return row
    }

    /// Once Arsenal stopped retrying, a consented download is a fresh try, as a click on Get is: the token or access may
    /// have changed since (1 Nov: a token saved after the tenth try left every agent call answering "no token").
    @Test func aConsentedDownloadAfterArsenalGaveUpTriesAgain() async throws {
        let h = try await MCPHarness(rows: [Self.waitingRow(gaveUp: true)], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        await h.service.setGetScript { _, _, _ in throw Self.withToken }
        for tool in ["download", "load", "run"] {
            var arguments: JSONObject = ["model": "laya"]
            if tool == "run" { arguments["input"] = .object(decideInput) }
            let asked = try await h.text(tool, arguments)
            #expect(asked.hasPrefix("[waiting_on_person]"), "\(tool) without consent only reports the wait: \(asked)")
        }
        #expect(await h.service.getCalls.isEmpty)
        let answer = try await finished(h, try await h.text("download", ["model": "laya", "allow_download": true]))
        #expect(await h.service.getCalls.count == 1, "one fresh Get")
        #expect(answer.hasPrefix("[waiting_on_person]"), "\(answer)")
    }

    @Test(arguments: [false]) func pollingAWaitAnswersFromItWithoutAGet(gaveUp: Bool) async throws {
        let h = try await MCPHarness(rows: [Self.waitingRow(gaveUp: gaveUp)], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        for tool in ["download", "load", "run"] {
            var arguments: JSONObject = ["model": "laya", "allow_download": true]
            if tool == "run" { arguments["input"] = .object(decideInput) }
            let answer = try await h.text(tool, arguments)
            #expect(answer.hasPrefix("[waiting_on_person] Laya Decisions is gated on Hugging Face"), "\(tool): \(answer)")
            #expect(answer.contains(gaveUp ? "stopped retrying after 10 tries" : "retries every 30 s"), "\(answer)")
        }
        #expect(await h.service.getCalls.isEmpty, "no try spent and no page reopened")
        // HTTP's POST get answers the same way.
        let server = ArsenalAPIServer(service: h.service, supportDirectory: h.root.appendingPathComponent("http"), port: 0)
        let port = try await server.start(); defer { server.stop() }
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/get")!)
        request.httpMethod = "POST"; request.httpBody = try WireJSON.encode(["id": "laya", "dtype": "fp16"])
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 403)
        #expect(String(decoding: data, as: UTF8.self).contains("waiting_on_person"))
        #expect(await h.service.getCalls.isEmpty)
    }

    @Test func withoutATokenTheyNameTheTokenPageAndThatAgentsCannotSetIt() async throws {
        let h = try await MCPHarness(rows: [decisionRow(bytes: 600_000_000)], operations: ["laya": [decideOperation]])
        defer { Task { await h.close() } }
        await h.service.setGetScript { _, _, _ in throw Self.withoutToken }
        let answer = try await finished(h, try await h.text("download", ["model": "laya", "allow_download": true]))
        #expect(answer.hasPrefix("[waiting_on_person] Laya Decisions is gated on Hugging Face and Arsenal has no Hugging Face token"), "\(answer)")
        #expect(answer.contains("https://huggingface.co/settings/tokens") && answer.contains("agents cannot set it"))
        #expect(answer.contains("accept the model's terms at https://huggingface.co/org/laya") && answer.contains("Arsenal opened both pages"))
        #expect(answer.contains("Its authors approve each request, which can take days."))
    }

    @Test func httpFailureCarriesTheCodeAndThePages() {
        let failure = serviceHTTPFailure(Self.withoutToken, modelID: "laya")
        #expect(failure.status == 403 && failure.code == "waiting_on_person")
        #expect(failure.details["page"]?.string == "https://huggingface.co/org/laya" && failure.details["has_token"]?.bool == false)
        #expect(failure.details["token_page"]?.string == "https://huggingface.co/settings/tokens" && failure.details["gate"]?.string == "manual")
        #expect(failure.message.hasPrefix("laya is gated on Hugging Face"))
        #expect(MCPCategory(rawValue: "waiting_on_person") == .waitingOnPerson)
    }

    @Test func noRouteSetsAToken() {
        // Token entry is app-only: neither the MCP tools nor the HTTP API mention one.
        let tools = MCPTools.all.map(\.name).joined(separator: " ").lowercased()
        #expect(!tools.contains("token"))
    }
}
