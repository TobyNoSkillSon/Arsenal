// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import Testing
import ArsenalCore
import MCP
@testable import ArsenalAPI

private actor ProgressEvidence {
    var values: [Double] = []
    private var step: AsyncStream<Bool>.Continuation?
    func receive(_ value: Double) {
        values.append(value)
        if value == 7 || value == 8 { step?.yield(true); step?.finish() }
    }
    func waitForStep() async -> Bool {
        if values.contains(where: { $0 == 7 || $0 == 8 }) { return true }
        let stream = AsyncStream<Bool> { step = $0 }
        defer { step?.finish(); step = nil }
        return await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                for await value in stream { return value }; return false
            }
            group.addTask {
                try? await Task.sleep(for: .seconds(5)); return false
            }
            let received = await group.next() ?? false
            group.cancelAll(); return received
        }
    }
}

@Test func mcpOfficialSDKClientConformance() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-mcp-sdk-" + UUID().uuidString)
    let support = root.appendingPathComponent("support"), cache = root.appendingPathComponent("cache")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let server = Process()
    let serverPath = ProcessInfo.processInfo.environment["ARSENAL_TEST_SERVER"] ?? FileManager.default.currentDirectoryPath + "/.build/debug/arsenal-testserver"
    server.executableURL = URL(fileURLWithPath: serverPath)
    server.environment = ProcessInfo.processInfo.environment.merging([
        "ARSENAL_SUPPORT_DIR": support.path, "ARSENAL_CACHE_DIR": cache.path,
        "ARSENAL_API_PORT": "0", "ARSENAL_PLAYBACK_MUTED": "1"
    ]) { _, b in b }
    let log = root.appendingPathComponent("server.log"); FileManager.default.createFile(atPath: log.path, contents: nil)
    let logHandle = try FileHandle(forWritingTo: log); server.standardOutput = logHandle; server.standardError = logHandle
    try server.run()
    defer { if server.isRunning { server.terminate(); server.waitUntilExit() }; try? logHandle.close(); try? FileManager.default.removeItem(at: root) }
    let statusFile = support.appendingPathComponent("status.json")
    for _ in 0..<50 {
        if FileManager.default.fileExists(atPath: statusFile.path) { break }
        try await Task.sleep(for: .milliseconds(100))
    }
    let status = try WireJSON.decode(Data(contentsOf: statusFile)), port = try #require(status["port"]?.number)
    let token = try String(contentsOf: support.appendingPathComponent("mcp-token"), encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
    let endpoint = try #require(URL(string: "http://127.0.0.1:\(Int(port))/mcp"))
    let connectURL = endpoint.deletingLastPathComponent().appendingPathComponent("arsenal/v1/mcp/connect")
    let (connectData, _) = try await URLSession.shared.data(from: connectURL)
    let connect = try WireJSON.decode(connectData)
    #expect(connect["token_path"]?.string == support.appendingPathComponent("mcp-token").path)
    #expect(connect["token"] == nil && !String(decoding: connectData, as: UTF8.self).contains(token))
    let transport = HTTPClientTransport(
        endpoint: endpoint,
        requestModifier: { request in
            var request = request; request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization"); return request
        })
    let client = Client(name: "Arsenal official SDK conformance", version: "1", configuration: .init(strict: true))
    let evidence = ProgressEvidence()
    await client.onNotification(ProgressNotification.self) { notification in await evidence.receive(notification.params.progress) }
    let initialized = try await client.connect(transport: transport)
    #expect(initialized.serverInfo.name == "Arsenal")
    try await client.ping()
    #expect(initialized.instructions == MCPTools.instructions)
    let tools = try await client.listTools().tools
    #expect(tools.map(\.name) == MCPTools.all.map(\.name) && tools.count == 15)
    #expect(tools.allSatisfy { $0.annotations.openWorldHint != nil && $0.outputSchema == nil })
    func call(_ name: String, _ arguments: JSONObject, progress: Bool = false) async throws -> CallTool.Result {
        let values = try JSONDecoder().decode([String: MCP.Value].self, from: WireJSON.encode(arguments))
        let context: RequestContext<CallTool.Result> = try await client.send(
            CallTool.request(
                .init(
                    name: name, arguments: values,
                    meta: progress ? .init(progressToken: .string(UUID().uuidString)) : nil)))
        return try await context.value
    }
    func text(_ result: CallTool.Result) -> String { mcpResultText(result) }
    let found = try await call("\(MCPNames.prefix)find", ["limit": 50])
    #expect(found.isError == false && found.structuredContent == nil && text(found).contains("• stub-voices — Stub voices"))
    let model = try await call("\(MCPNames.prefix)model", ["model": "stub-voices"])
    #expect(text(model).contains("Operations:\n  speak (default)"), "\(text(model))")
    let schema = try await client.readResource(uri: "arsenal://models/stub-voices/operations/speak/schema")
    #expect(schema.count == 1)
    let speech = try await call("\(MCPNames.prefix)run", ["model": "stub-voices", "input": ["text": "hello"]], progress: true)
    #expect(speech.isError == false && text(speech).hasPrefix("Created 1 audio file with stub-voices"), "\(text(speech))")
    let path = String(text(speech).components(separatedBy: "\n")[1].prefix { $0 != " " })
    #expect(FileManager.default.fileExists(atPath: path))
    let link = try #require(
        speech.content.compactMap { content -> String? in
            if case .resourceLink(let uri, _, _, _, _, _) = content { return uri }; return nil
        }.first)
    #expect(try await client.readResource(uri: link).count == 1)
    let image = try await call("\(MCPNames.prefix)run", ["model": "stub-images", "input": ["prompt": "fixture"]])
    #expect(
        image.isError == false
            && image.content.contains {
                if case .image = $0 { return true }; return false
            })
    let slow = try await call("\(MCPNames.prefix)run", ["model": "stub-sound", "input": ["prompt": "slow progress"], "wait_seconds": 0])
    let id = try #require(text(slow).range(of: #"Job ([0-9a-f]{10})"#, options: .regularExpression).map { String(text(slow)[$0].dropFirst(4)) })
    #expect(try await client.readResource(uri: "arsenal://jobs/" + id).count == 1)
    #expect(try await call("\(MCPNames.prefix)wait", ["job": .string(id)], progress: true).isError == false)
    let invalid = try await call("\(MCPNames.prefix)run", ["model": "stub-voices", "input": ["text": 12]])
    #expect(invalid.isError == true && text(invalid).hasPrefix("[invalid_input] Not run: the input doesn't match speak:"), "\(text(invalid))")
    let badArguments = try await call("\(MCPNames.prefix)run", ["model": "stub-voices", "input": "text"])
    #expect(badArguments.isError == true && text(badArguments).hasPrefix("[invalid_input] \(MCPNames.prefix)run arguments:"))
    let cancelled = try await call("\(MCPNames.prefix)run", ["model": "stub-images", "input": ["prompt": "slow"], "wait_seconds": 0])
    let cancelledID = try #require(text(cancelled).range(of: #"Job ([0-9a-f]{10})"#, options: .regularExpression).map { String(text(cancelled)[$0].dropFirst(4)) })
    #expect(text(try await call("\(MCPNames.prefix)cancel", ["job": .string(cancelledID)])) == "Cancelled job \(cancelledID) (generate with Stub images).")
    let cancelledRequest: RequestContext<CallTool.Result> = try await client.send(
        CallTool.request(.init(name: "\(MCPNames.prefix)run", arguments: ["model": "stub-images", "input": ["prompt": "restart"], "wait_seconds": 40])))
    try await Task.sleep(for: .milliseconds(100))
    try await client.cancelRequest(cancelledRequest.requestID)
    try await Task.sleep(for: .milliseconds(100))
    let history = text(try await call("\(MCPNames.prefix)status", ["history": true]))
    #expect(history.components(separatedBy: " cancelled — generate with Stub images").count - 1 == 2, "\(history)")
    let receivedStep = await evidence.waitForStep(), observedProgress = await evidence.values
    #expect(receivedStep, "SDK progress values: \(observedProgress)")
    // A zero-byte Get needs no consent; cancelling the call stops it.
    let getRequest: RequestContext<CallTool.Result> = try await client.send(CallTool.request(.init(name: "\(MCPNames.prefix)download", arguments: ["model": "stub-absent"])))
    try await Task.sleep(for: .milliseconds(100)); try await client.cancelRequest(getRequest.requestID)
    try await Task.sleep(for: .milliseconds(150))
    let (stateBytes, _) = try await URLSession.shared.data(from: URL(string: "http://127.0.0.1:\(Int(port))/arsenal/v1/models")!)
    let stateRows = try JSONDecoder().decode([JSONValue].self, from: stateBytes)
    #expect(stateRows.first?["metrics"]?["canceled_runs"]?.number == 2)
    #expect(stateRows.first?["metrics"]?["cancelled_gets"]?.number == 1)
    #expect(stateRows.first?["metrics"]?["get_active"] == false)
    await client.disconnect()

    let bridge = Process(), bridgeInput = Pipe(), bridgeOutput = Pipe()
    bridge.executableURL = URL(fileURLWithPath: serverPath).deletingLastPathComponent().appendingPathComponent("arsenal-mcp")
    bridge.environment = server.environment; bridge.standardInput = bridgeInput; bridge.standardOutput = bridgeOutput; bridge.standardError = logHandle
    try bridge.run()
    defer { if bridge.isRunning { bridge.terminate(); bridge.waitUntilExit() } }
    let stdio = StdioTransport(input: .init(rawValue: bridgeOutput.fileHandleForReading.fileDescriptor), output: .init(rawValue: bridgeInput.fileHandleForWriting.fileDescriptor))
    let desktop = Client(name: "Desktop bridge conformance", version: "1")
    try await desktop.connect(transport: stdio)
    try await desktop.ping()
    #expect(try await desktop.listTools().tools.count == 15)
    let bridgeSpeech = try await desktop.callTool(name: "\(MCPNames.prefix)run", arguments: ["model": "stub-voices", "input": ["text": "bridge hello"], "play": false])
    #expect(bridgeSpeech.isError == false)
    func stop(_ process: Process) async throws {
        if process.isRunning { process.terminate() }
        for _ in 0..<50 { if !process.isRunning { return }; try await Task.sleep(for: .milliseconds(100)) }
        throw MCPError.internalError("Fixture server did not stop.")
    }
    func restart(_ old: Process, samePort: Bool) async throws -> Process {
        try await stop(old)
        let fresh = Process(); fresh.executableURL = server.executableURL; fresh.environment = server.environment
        fresh.environment?["ARSENAL_API_PORT"] = samePort ? String(Int(port)) : "0"
        fresh.standardOutput = logHandle; fresh.standardError = logHandle; try fresh.run()
        for _ in 0..<50 {
            if let bytes = try? Data(contentsOf: statusFile), let state = try? WireJSON.decode(bytes), state["pid"]?.number == Double(fresh.processIdentifier) { return fresh }
            try await Task.sleep(for: .milliseconds(100))
        }
        if fresh.isRunning { fresh.terminate() }
        throw MCPError.internalError("Restart fixture startup deadline.")
    }
    let samePort = try await restart(server, samePort: true)
    defer { if samePort.isRunning { samePort.terminate() } }
    try await desktop.ping(); #expect(try await desktop.listTools().tools.count == 15)
    let newPort = try await restart(samePort, samePort: false)
    defer { if newPort.isRunning { newPort.terminate() } }
    try await desktop.ping(); #expect(try await desktop.listTools().tools.count == 15)
    try await stop(newPort)
    let beforeError = ContinuousClock.now
    var receivedError = false
    do { try await desktop.ping() } catch { receivedError = String(describing: error).contains("Start Arsenal") }
    #expect(receivedError && beforeError.duration(to: .now) < .seconds(5))
    await desktop.disconnect()
    try bridgeInput.fileHandleForWriting.close()

}

@Test func mcpTokenPermissionsPersistenceAndAuthentication() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-mcp-security-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let router = APIRouter(service: StubModelsService(), support: root)
    try await router.prepareMCP(); await router.setPort(1234)
    let token = try #require(await router.mcpToken)
    let attrs = try FileManager.default.attributesOfItem(atPath: root.appendingPathComponent("mcp-token").path)
    #expect((attrs[.posixPermissions] as? NSNumber)?.intValue == 0o600)
    try await router.prepareMCP(); #expect(await router.mcpToken == token)
    func request(_ auth: String?, origin: String? = nil, host: String = "127.0.0.1:1234") -> ArsenalAPI.HTTPRequest {
        var headers = ["host": host]; headers["authorization"] = auth; headers["origin"] = origin
        return .init(method: "POST", target: "/mcp", headers: headers, body: Data())
    }
    try await router.authorizeMCP(request("Bearer " + token))
    for request in [request(nil), request("Bearer wrong"), request("Bearer " + token, origin: "https://example.com"), request("Bearer " + token, host: "example.com:1234")] {
        await #expect(throws: HTTPFailure.self) { try await router.authorizeMCP(request) }
    }
    let outside = root.appendingPathComponent("outside"); try Data("secret".utf8).write(to: outside)
    try FileManager.default.removeItem(at: root.appendingPathComponent("mcp-token"))
    try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("mcp-token"), withDestinationURL: outside)
    await #expect(throws: HTTPFailure.self) { try await router.prepareMCP() }
    #expect(try Data(contentsOf: outside) == Data("secret".utf8))
}

@Test func mcpZeroBytePlansFollowTheInstalledState() async throws {
    for state in [ModelState.notInstalled, .ready, .loaded] {
        let row = ModelRow(id: "plan", name: "Get plan fixture", kind: "voices", states: ["bf16": .init(state: state)], downloadBytes: ["bf16": 0])
        let h = try await MCPHarness(rows: [row])
        let answer = try await h.text("download", ["model": "plan"])
        if state == .notInstalled {
            #expect(answer.hasPrefix("Preparing Get plan fixture from files already on this Mac.") || answer.hasPrefix("Downloaded Get plan fixture"), "\(answer)")
            #expect(await h.service.getCalls == ["plan bf16"])
        } else {
            #expect(answer == "Get plan fixture is already downloaded at bf16. Nothing to do.")
            #expect(await h.service.getCalls.isEmpty)
        }
        await h.close()
    }
}
