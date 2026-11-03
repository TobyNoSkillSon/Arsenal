// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore
import MCP
@testable import ArsenalAPI

/// A router over the stub service with simulated time (200× the wall clock), a fixed Mac (40 of 100 GB in use, 128 GB
/// RAM, 500 GB free disk) and a muted playback queue. Nothing downloads, plays or touches the installed app.
struct MCPHarness {
    static let gib = 1.073741824
    let root: URL
    let service: StubModelsService
    let router: APIRouter
    let environment: MCPEnvironment
    init(
        rows: [ModelRow], metadata: [String: JSONObject] = [:], operations: [String: [ModelOperation]] = [:], factor: Double = 200, freeDisk: Int64? = 500_000_000_000,
        inUseGB: Double = 40 * gib, cloneFile: (@Sendable (Int32, String) -> Int32)? = nil, pathDeadline: Double = 2000
    ) async throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-mcp-agent-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        service = StubModelsService(rows: rows, metadata: metadata, operations: operations, supportDirectory: root)
        let queue = await MainActor.run { PlaybackQueue(muted: true) }
        router = APIRouter(service: service, support: root, playbackQueue: queue)
        var environment = MCPEnvironment.scaled(factor)
        environment.freeDiskBytes = { _ in freeDisk }
        environment.macMemory = { MCPMacMemory(inUseGB: inUseGB, limitGB: 100 * Self.gib, totalGB: 128 * Self.gib) }
        environment.pathDeadlineSeconds = pathDeadline
        if let cloneFile { environment.cloneFile = cloneFile }
        self.environment = environment
        await router.configureMCP(environment)
    }
    func close() async { await router.stopMCP(); try? FileManager.default.removeItem(at: root) }
    func call(_ name: String, _ arguments: JSONObject = [:], notify: MCPNotify? = nil) async throws -> CallTool.Result {
        try await router.callMCPTool(MCPNames.tool(name), arguments: arguments, requester: "MCP agent test", notify: notify)
    }
    func text(_ name: String, _ arguments: JSONObject = [:], notify: MCPNotify? = nil) async throws -> String {
        mcpResultText(try await call(name, arguments, notify: notify))
    }
    /// The job id a result names ("Job 1a2b3c4d5e").
    static func job(_ text: String) -> String? {
        guard let range = text.range(of: #"Job ([0-9a-f]{10})"#, options: .regularExpression) else { return nil }
        return String(text[range].dropFirst(4))
    }
}

/// A path's real form (`/private/var/…`), as Arsenal reports inbox and job paths.
func mcpReal(_ url: URL) -> String { APIRouter.mcpRealPath(url) }

/// A download script: `chunks` of (simulated seconds to wait, bytes after it); a nil byte count stalls without a report.
func mcpScript(_ environment: MCPEnvironment, total: Int64, steps: [(Double, Int64?)], derive: Double = 0, fail: (any Error)? = nil) -> StubModelsService.GetScript {
    { _, _, report in
        await report(PrecisionState(state: .downloading, progress: 0, bytes: 0, totalBytes: total))
        for (seconds, bytes) in steps {
            try await environment.sleep(seconds)
            if let bytes { await report(PrecisionState(state: .downloading, progress: Double(bytes) / Double(total), bytes: bytes, totalBytes: total)) }
        }
        if let fail { throw fail }
        if derive > 0 {
            await report(PrecisionState(state: .deriving, phase: "deriving"))
            try await environment.sleep(derive)
        }
    }
}
/// One part of a download whose speed is fixed by the simulated clock: `steady` reports `rate` bytes per simulated second
/// up to `to` bytes, `stall` reports nothing for that many seconds.
enum MCPDownloadPart { case steady(rate: Double, to: Int64), stall(Double) }

/// A download script whose byte counts follow the simulated clock, not the number of sleeps: a sleep that wakes late on a
/// busy Mac moves the bytes on by the time that really passed, so every measured speed is exactly the scripted rate.
func mcpRatedScript(_ environment: MCPEnvironment, total: Int64, parts: [MCPDownloadPart], derive: Double = 0) -> StubModelsService.GetScript {
    { _, _, report in
        await report(PrecisionState(state: .downloading, progress: 0, bytes: 0, totalBytes: total))
        var bytes: Int64 = 0
        for part in parts {
            switch part {
            case .stall(let seconds):
                try await environment.sleep(seconds)
            case .steady(let rate, let end):
                let start = environment.now(), base = bytes
                while bytes < end {
                    try await environment.sleep(1)
                    bytes = min(end, base + Int64(rate * environment.now().timeIntervalSince(start)))
                    await report(PrecisionState(state: .downloading, progress: Double(bytes) / Double(total), bytes: bytes, totalBytes: total))
                }
            }
        }
        if derive > 0 {
            await report(PrecisionState(state: .deriving, phase: "deriving"))
            try await environment.sleep(derive)
        }
    }
}
/// Evenly spaced reports: `seconds` per step at `rate` bytes per second, from `start` up to `end` bytes.
func mcpSteady(from start: Int64, to end: Int64, rate: Double, every seconds: Double = 2) -> [(Double, Int64?)] {
    var steps: [(Double, Int64?)] = [], bytes = start
    while bytes < end {
        bytes = min(end, bytes + Int64(rate * seconds)); steps.append((seconds, bytes))
    }
    return steps
}
actor MCPNotes {
    var values: [(at: Date, progress: Double, total: Double?, message: String)] = []
    func add(_ at: Date, _ progress: Double, _ total: Double?, _ message: String) { values.append((at, progress, total, message)) }
}
let decideOperation = ModelOperation(
    name: "decide", title: "Decide",
    inputSchema: [
        "type": "object", "required": ["questions"],
        "properties": ["questions": ["type": "object", "description": "Named questions. Each has a type."], "items": ["type": "array", "items": ["type": "object"]]]
    ], output: "json", stream: false, annotations: ["primary": true])
func decisionRow(_ id: String = "laya", bytes: Int64 = 2_500_000_000, state: ModelState = .notInstalled) -> ModelRow {
    ModelRow(
        id: id, name: "Laya Decisions", kind: "decisions", params: "0.6B", licence: "Apache-2.0", precisions: ["fp16"], states: ["fp16": .init(state: state)],
        selectedPrecision: "fp16", downloadBytes: ["fp16": bytes])
}
let decideInput: JSONObject = ["questions": ["bug": ["type": "noul", "instructions": "Is this a bug?"]]]
