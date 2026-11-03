// SPDX-License-Identifier: MIT
import Foundation
import Testing
import MCP

private actor BridgeProgressProbe {
    var values: [Double] = []
    func append(_ value: Double) { values.append(value) }
}
@Test func mcpBridgeStreamsPostProgressAndKeepsReconnectingGet() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-mcp-live-bridge-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = root.appendingPathComponent("server.py")
    try bridgeHTTPFixture.write(to: fixture, atomically: true, encoding: .utf8)
    let server = Process(); server.executableURL = URL(fileURLWithPath: "/usr/bin/python3"); server.arguments = [fixture.path, root.path]
    server.standardOutput = FileHandle.nullDevice; server.standardError = FileHandle.nullDevice
    try server.run()
    defer { if server.isRunning { server.terminate() } }
    for _ in 0..<100 { if FileManager.default.fileExists(atPath: root.appendingPathComponent("status.json").path) { break }; try await Task.sleep(for: .milliseconds(20)) }
    let serverPath = ProcessInfo.processInfo.environment["ARSENAL_TEST_SERVER"] ?? FileManager.default.currentDirectoryPath + "/.build/debug/arsenal-testserver"
    let bridge = Process(), input = Pipe(), output = Pipe()
    bridge.executableURL = URL(fileURLWithPath: serverPath).deletingLastPathComponent().appendingPathComponent("arsenal-mcp")
    bridge.environment = ProcessInfo.processInfo.environment.merging(["ARSENAL_SUPPORT_DIR": root.path]) { _, b in b }
    bridge.standardInput = input; bridge.standardOutput = output; bridge.standardError = FileHandle.nullDevice
    try bridge.run()
    defer { if bridge.isRunning { bridge.terminate() } }
    let transport = StdioTransport(input: .init(rawValue: output.fileHandleForReading.fileDescriptor), output: .init(rawValue: input.fileHandleForWriting.fileDescriptor))
    let client = Client(name: "Live bridge", version: "1"), probe = BridgeProgressProbe()
    await client.onNotification(ProgressNotification.self) { value in await probe.append(value.params.progress) }
    try await client.connect(transport: transport)
    let started = ContinuousClock.now
    let call: RequestContext<CallTool.Result> = try await client.send(CallTool.request(.init(name: "fixture", meta: .init(progressToken: .string("bridge")))))
    for _ in 0..<50 { if await probe.values.contains(7) { break }; try await Task.sleep(for: .milliseconds(10)) }
    #expect(await probe.values.contains(7))
    #expect(started.duration(to: .now) < .seconds(1)) // Final response is delayed 1.2 s.
    #expect(try await call.value.isError == false)
    let countFile = root.appendingPathComponent("get-count")
    for _ in 0..<200 {
        if let value = try? String(contentsOf: countFile, encoding: .utf8), Int(value) ?? 0 >= 6 { break }
        try await Task.sleep(for: .milliseconds(20))
    }
    #expect(Int(try String(contentsOf: countFile, encoding: .utf8)) ?? 0 >= 6)
    await client.disconnect(); try input.fileHandleForWriting.close()
    for _ in 0..<100 { if !bridge.isRunning { break }; try await Task.sleep(for: .milliseconds(20)) }
    #expect(!bridge.isRunning && bridge.terminationStatus == 0)
}

private let bridgeHTTPFixture = #"""
    import http.server, json, os, pathlib, sys, threading, time
    root = pathlib.Path(sys.argv[1])
    count = 0
    lock = threading.Lock()
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *args): pass
        def headers_for(self, kind, length=None):
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('MCP-Session-Id', 'fixture-session')
            if length is not None: self.send_header('Content-Length', str(length))
            else: self.send_header('Connection', 'close')
            self.end_headers()
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if 'id' not in request:
                self.send_response(202); self.send_header('Content-Length', '0'); self.end_headers(); return
            if request['method'] == 'initialize':
                body = json.dumps({'jsonrpc':'2.0', 'id':request['id'], 'result': {'protocolVersion':'2025-11-25', 'capabilities':{}, 'serverInfo':{'name':'Fixture', 'version':'1'}}}).encode()
                self.headers_for('application/json', len(body)); self.wfile.write(body); return
            self.headers_for('text/event-stream')
            progress = {'jsonrpc':'2.0', 'method':'notifications/progress', 'params':{'progressToken':'bridge', 'progress':7}}
            self.wfile.write(('data: ' + json.dumps(progress) + '\n\n').encode()); self.wfile.flush()
            time.sleep(1.2)
            result = {'jsonrpc':'2.0', 'id':request['id'], 'result':{'content':[{'type':'text', 'text':'{}'}], 'structuredContent':{}, 'isError':False}}
            self.wfile.write(('data: ' + json.dumps(result) + '\n\n').encode()); self.wfile.flush(); self.close_connection = True
        def do_GET(self):
            global count
            with lock:
                count += 1; current = count; (root/'get-count').write_text(str(count))
            self.headers_for('text/event-stream')
            self.wfile.write(('id: get_' + str(current) + '\n\n').encode()); self.wfile.flush()
            if current >= 7: time.sleep(15)
            self.close_connection = True
        def do_DELETE(self):
            self.send_response(200); self.send_header('Content-Length', '0'); self.end_headers()
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    (root/'status.json').write_text(json.dumps({'pid':os.getpid(), 'port':server.server_port}))
    (root/'mcp-token').write_text('f' * 64 + '\n')
    server.serve_forever()
    """#
