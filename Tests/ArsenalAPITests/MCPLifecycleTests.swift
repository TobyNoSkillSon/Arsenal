// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore
import MCP
@testable import ArsenalAPI

extension APIRouter {
    func installAgedSessionForTest(_ id: String) async throws {
        var session = try await makeMCPSession(requester: "Lifecycle test")
        session.touched = Date().addingTimeInterval(-7200); mcpSessions[id] = session
    }
}
@Test func mcpSessionActivityRefreshAndIdleExpiry() async throws {
    let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-mcp-sessions-" + UUID().uuidString)
    let router = APIRouter(service: StubModelsService(), support: support)
    try await router.installAgedSessionForTest("active"); try await router.installAgedSessionForTest("idle")
    let touched = try await router.touchMCPSession("active")
    #expect(touched.touched.timeIntervalSinceNow > -1)
    await router.purgeMCPSessions()
    #expect(await router.mcpSessions.count == 1)
    await #expect(throws: HTTPFailure.self) { try await router.touchMCPSession("idle") }
    await router.stopMCP()
}
@Test func mcpSDKReplayBufferIsBounded() async throws {
    let transport = StatefulHTTPServerTransport()
    let message = Data(#"{"jsonrpc":"2.0","method":"notifications/progress","params":{"progressToken":"test","progress":7}}"#.utf8)
    for _ in 0..<1000 { try await transport.send(message) }
    #expect(await transport.retainedEventCount <= 256)
    let large = try JSONSerialization.data(withJSONObject: ["jsonrpc": "2.0", "method": "test", "params": ["text": String(repeating: "x", count: 1024 * 1024)]])
    for _ in 0..<8 { try await transport.send(large) }
    #expect(await transport.retainedEventBytes <= 4 * 1024 * 1024)
    await transport.disconnect()
}
@Test func mcpBridgeAnswersMalformedAndFailedRequests() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-mcp-bridge-errors-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let serverPath = ProcessInfo.processInfo.environment["ARSENAL_TEST_SERVER"] ?? FileManager.default.currentDirectoryPath + "/.build/debug/arsenal-testserver"
    let process = Process(), input = Pipe(), output = Pipe()
    process.executableURL = URL(fileURLWithPath: serverPath).deletingLastPathComponent().appendingPathComponent("arsenal-mcp")
    process.environment = ProcessInfo.processInfo.environment.merging(["ARSENAL_SUPPORT_DIR": root.path]) { _, b in b }
    process.standardInput = input; process.standardOutput = output; process.standardError = FileHandle.nullDevice
    try process.run()
    defer { if process.isRunning { process.terminate() } }
    try input.fileHandleForWriting.write(contentsOf: Data("bad JSON\n{\"jsonrpc\":\"2.0\",\"id\":42,\"method\":\"ping\"}\n".utf8))
    try input.fileHandleForWriting.close()
    let bytes = try #require(try output.fileHandleForReading.readToEnd())
    let responses = try String(decoding: bytes, as: UTF8.self).split(separator: "\n").map { try WireJSON.decode(Data($0.utf8)) }
    #expect(responses.count == 2)
    #expect(responses[0]["error"]?["code"] == -32700)
    #expect(responses[1]["id"] == 42 && responses[1]["error"]?["code"] == -32002)
}

@Test func mcpSDKStandaloneReconnectKeepsTheNewStream() async throws {
    let transport = StatefulHTTPServerTransport()
    let initialize = Data(#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25"}}"#.utf8)
    let initial = await transport.handleRequest(
        .init(method: "POST", headers: ["Content-Type": "application/json", "Accept": "application/json, text/event-stream"], body: initialize, path: "/mcp"))
    let session = try #require(initial.headers["Mcp-Session-Id"] ?? initial.headers["MCP-Session-Id"])
    let headers = ["Accept": "text/event-stream", "MCP-Session-Id": session, "MCP-Protocol-Version": "2025-11-25"]
    let (first, oldLease) = await transport.handleArsenalRequest(.init(method: "GET", headers: headers, path: "/mcp"))
    #expect(first.statusCode == 200)
    let old = try #require(oldLease)
    let (conflict, conflictLease) = await transport.handleArsenalRequest(.init(method: "GET", headers: headers, path: "/mcp"))
    #expect(conflict.statusCode == 409 && conflictLease == nil)
    guard case .stream(let stream, _) = first else { Issue.record("Expected SSE stream"); return }
    var iterator = stream.makeAsyncIterator()
    let priming = String(decoding: try #require(try await iterator.next()), as: UTF8.self)
    let anchor = try #require(priming.split(separator: "\n").first { $0.hasPrefix("id:") }).dropFirst(3).trimmingCharacters(in: .whitespaces)
    try await transport.send(Data(#"{"jsonrpc":"2.0","method":"notifications/progress","params":{"progressToken":"test","progress":7}}"#.utf8))
    var resumedHeaders = headers; resumedHeaders["Last-Event-ID"] = anchor
    let (resumed, newLease) = await transport.handleArsenalRequest(.init(method: "GET", headers: resumedHeaders, path: "/mcp"))
    #expect(resumed.statusCode == 200 && newLease != oldLease)
    await transport.closeStandalone(ifLease: old)
    let (stillConnected, _) = await transport.handleArsenalRequest(.init(method: "GET", headers: headers, path: "/mcp"))
    #expect(stillConnected.statusCode == 409)
    guard case .stream(let replay, _) = resumed else { Issue.record("Expected replay stream"); return }
    var replayIterator = replay.makeAsyncIterator()
    let event = String(decoding: try #require(try await replayIterator.next()), as: UTF8.self)
    #expect(event.contains("notifications/progress") && event.contains("7"))
    await transport.closeStandalone(ifLease: try #require(newLease))
    let (reconnected, _) = await transport.handleArsenalRequest(.init(method: "GET", headers: headers, path: "/mcp"))
    #expect(reconnected.statusCode == 200)
    await transport.disconnect()
}

@Test func mcpSettingsPathsAreImportedLikeRunInputs() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-mcp-setting-paths-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let declaration: JSONObject = ["id": "reference_path", "title": "Reference clip", "type": "text", "default": "", "applies": "request", "maps_to": "request.reference.path"]
    let service = StubModelsService(rows: [ModelRow(id: "voice", name: "Voice", kind: "voices")], metadata: ["voice": ["setting": [.object(declaration)]]], supportDirectory: root)
    let router = APIRouter(service: service, support: root)
    let refused = try await router.callMCPTool("\(MCPNames.prefix)settings", arguments: ["model": "voice", "set": ["reference_path": "/etc"]], requester: "Test", notify: nil)
    #expect(refused.isError == true && mcpResultText(refused) == "[invalid_input] Not run: /etc is not a regular file (it is a folder). Pass a file.", "\(mcpResultText(refused))")
    let outside = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-mcp-setting-clip-" + UUID().uuidString + ".wav")
    try Data("fixture".utf8).write(to: outside); defer { try? FileManager.default.removeItem(at: outside) }
    let changed = try await router.callMCPTool(
        "\(MCPNames.prefix)settings", arguments: ["model": "voice", "set": ["reference_path": .string(outside.path)]], requester: "Test", notify: nil)
    #expect(mcpResultText(changed) == "Changed reference_path for Voice.")
    let saved = try #require(try await service.settings(id: "voice").first?.value.string)
    #expect(saved.hasPrefix(APIRouter.mcpRealPath(root.appendingPathComponent("inbox")) + "/"), "\(saved)")
    #expect(try Data(contentsOf: URL(fileURLWithPath: saved)) == Data("fixture".utf8))
    try await service.setSettings(id: "voice", values: ["reference_path": "/etc/hosts"])
    #expect(try await service.settings(id: "voice").first?.value == "/etc/hosts") // Native UI/service remains unrestricted.
}
@Test func mcpRejectedInitializationStopsItsTransport() async throws {
    let router = APIRouter(service: StubModelsService(), support: FileManager.default.temporaryDirectory)
    let session = try await router.makeMCPSession(requester: "Test")
    let response = await session.transport.handleRequest(
        .init(method: "POST", headers: [:], body: Data(#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#.utf8), path: "/mcp"))
    #expect(response.statusCode != 200)
    await router.finishMCPSessionResponse(session, response: response, initializing: true, method: "POST")
    #expect(await router.mcpSessions.isEmpty)
    #expect(await session.transport.handleRequest(.init(method: "GET", headers: [:], path: "/mcp")).statusCode == 404)
}
