// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import MCP

struct MCPSession: Sendable {
    let server: MCP.Server
    let transport: StatefulHTTPServerTransport
    let requester: String
    var touched: Date
}
extension APIRouter {
    func mcp(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        try authorizeMCP(request)
        var session: MCPSession
        if let id = request.headers["mcp-session-id"] {
            session = try touchMCPSession(id)
        } else {
            guard request.method == "POST", let body = try? WireJSON.decode(request.body), body["method"]?.string == "initialize" else {
                throw HTTPFailure(400, "Initialize MCP first, then send MCP-Session-Id.")
            }
            await purgeMCPSessions()
            guard mcpSessions.count < 32 else { throw HTTPFailure(429, "Too many MCP sessions; DELETE an idle session before connecting again.") }
            let name = body["params"]?["clientInfo"]?["name"]?.string ?? "MCP"
            session = try await makeMCPSession(requester: ActivityRequestContext.displayName(String(name.prefix(128))))
        }
        let (response, lease) = await session.transport.handleArsenalRequest(
            MCP.HTTPRequest(method: request.method, headers: request.headers, body: request.body, path: request.path))
        await finishMCPSessionResponse(session, response: response, initializing: request.headers["mcp-session-id"] == nil, method: request.method)
        var headers = response.headers
        let type = headers.first(where: { $0.key.lowercased() == "content-type" })?.value ?? "application/json"
        headers = headers.filter { !["content-type", "connection", "transfer-encoding", "cache-control"].contains($0.key.lowercased()) }
        switch response {
        case .stream(let stream, _):
            do {
                try await connection.beginStream(headers: headers)
                for try await data in stream { try await connection.chunk(data) }
                try await connection.finishStream()
            } catch { if let lease { await session.transport.closeStandalone(ifLease: lease) }; throw error }
            if let lease { await session.transport.closeStandalone(ifLease: lease) }
        default: try await connection.respondBytes(response.bodyData ?? Data(), type: type, status: response.statusCode, headers: headers)
        }
    }
    func finishMCPSessionResponse(_ session: MCPSession, response: MCP.HTTPResponse, initializing: Bool, method: String) async {
        if let id = response.headers.first(where: { $0.key.lowercased() == "mcp-session-id" })?.value {
            if method == "DELETE" { mcpSessions.removeValue(forKey: id); await session.server.stop() } else if response.statusCode == 200 { mcpSessions[id] = session }
        }
        if initializing && response.statusCode != 200 { await session.server.stop() }
    }
    func touchMCPSession(_ id: String) throws -> MCPSession {
        guard var session = mcpSessions[id] else { throw HTTPFailure(404, "MCP session expired; initialize a new session.") }
        session.touched = Date(); mcpSessions[id] = session; return session
    }
    func purgeMCPSessions(now: Date = Date()) async {
        for (id, old) in mcpSessions where old.touched < now.addingTimeInterval(-3600) {
            await old.server.stop(); mcpSessions.removeValue(forKey: id)
        }
    }
    func makeMCPSession(requester: String) async throws -> MCPSession {
        let transport = StatefulHTTPServerTransport()
        let server = MCP.Server(
            name: MCPNames.server, version: Self.releaseVersion(), instructions: MCPTools.instructions, capabilities: .init(resources: .init(), tools: .init()))
        await server.withMethodHandler(ListTools.self) { _ in .init(tools: MCPTools.all) }
        await server.withMethodHandler(CallTool.self) { [self, weak server] params in
            guard let definition = MCPTools.all.first(where: { $0.name == params.name }) else { throw MCPError.invalidParams("Unknown tool; call tools/list.") }
            let arguments: JSONObject
            do {
                arguments = try wireObject(params.arguments ?? [:])
                let schema = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(definition.inputSchema))
                try OperationSchemaValidator.validate(.object(arguments), schema: schema)
            } catch {
                let message = (error as? WorkerFailure)?.message ?? error.localizedDescription
                return MCPFailure(.invalidInput, "\(params.name) arguments: \(message)").result
            }
            return try await ActivityRequestContext.$requester.withValue(requester) {
                try await LoadMemoryContext.$force.withValue(arguments["force"]?.bool == true) {
                    try await self.callMCPTool(params.name, arguments: arguments, requester: requester, server: server, progress: params._meta?.progressToken)
                }
            }
        }
        await server.withMethodHandler(ListResources.self) { [self] _ in .init(resources: await mcpResources()) }
        await server.withMethodHandler(ReadResource.self) { [self] params in .init(contents: [try await readMCPResource(params.uri)]) }
        try await server.start(transport: transport)
        return .init(server: server, transport: transport, requester: requester, touched: Date())
    }
    func mcpConnect(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        guard request.method == "GET" else { throw HTTPFailure(405, "Use GET.") }
        let endpoint = origin + "/mcp"
        try await connection.respond([
            "url": endpoint, "token_path": support.appendingPathComponent("mcp-token").path,
            "claude": "claude mcp add --transport http arsenal \(endpoint) --header \"Authorization: Bearer $ARSENAL_MCP_TOKEN\"",
            "codex": "codex mcp add arsenal --url \(endpoint) --bearer-token-env-var ARSENAL_MCP_TOKEN"
        ])
    }
    func playbackControl(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        if request.method == "POST" {
            let body = try object(request)
            if let value = body["muted"] { guard let muted = value.bool else { throw HTTPFailure(422, "muted must be boolean.") }; try await playback.setMuted(muted) }
            if body["stop"]?.bool == true { await playback.stop() }
        } else if request.method != "GET" {
            throw HTTPFailure(405, "Use GET or POST.")
        }
        try await connection.respondBytes(try JSONEncoder().encode(await playback.snapshot()), type: "application/json")
    }
}

extension APIRouter {
    func stopMCP() async {
        for task in operationJobTasks.values { task.cancel() }
        operationJobTasks.removeAll()
        let sessions = mcpSessions.values; mcpSessions.removeAll()
        for session in sessions { await session.server.stop() }
        let jobs = mcpJobs.values; mcpJobs.removeAll()
        for job in jobs {
            await job.interrupt()
            if let record = mcpJobRecords[job.id], record.status == .running { mcpJobRecords[job.id] = await job.record() }
        }
        for get in mcpGets.values { get.task.cancel() }
        mcpGets.removeAll()
        saveMCPJobRecords()
    }
}
