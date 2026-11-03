// SPDX-License-Identifier: MIT
import Foundation
import Darwin

actor Bridge {
    let support: URL
    var endpoint: URL?
    var token = ""
    var hostPID: Int32?
    var session: String?
    var protocolVersion = "2025-11-25"
    var initialize: Data?
    var initialized = Data(#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#.utf8)
    var notifications: Task<Void, Never>?
    var recovering: Task<Void, Error>?
    var streamGeneration = 0
    var lastEventID: String?
    var active = 0
    var waiters: [CheckedContinuation<Void, Never>] = []
    init(support: URL) { self.support = support }
    func refresh() throws {
        guard let data = try? Data(contentsOf: support.appendingPathComponent("status.json")),
            let status = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let port = status["port"] as? Int, (1...65535).contains(port), let pid = status["pid"] as? Int32, kill(pid, 0) == 0,
            let url = URL(string: "http://127.0.0.1:\(port)/mcp")
        else { throw BridgeError.message("Start Arsenal, then reconnect.") }
        endpoint = url; hostPID = pid
        token = try String(contentsOf: support.appendingPathComponent("mcp-token"), encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
    }
    func request(_ method: String, body: Data? = nil) throws -> URLRequest {
        if endpoint == nil { try refresh() }
        guard let endpoint else { throw BridgeError.message("Start Arsenal, then reconnect.") }
        var request = URLRequest(url: endpoint); request.httpMethod = method; request.httpBody = body
        request.timeoutInterval = method == "GET" ? 365 * 86400 : 60
        request.setValue("application/json", forHTTPHeaderField: "Content-Type"); request.setValue("application/json, text/event-stream", forHTTPHeaderField: "Accept")
        request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization"); request.setValue(protocolVersion, forHTTPHeaderField: "MCP-Protocol-Version")
        if let session { request.setValue(session, forHTTPHeaderField: "MCP-Session-Id") }
        if method == "GET", let lastEventID { request.setValue(lastEventID, forHTTPHeaderField: "Last-Event-ID") }
        return request
    }
    func accept(_ data: Data) async {
        guard !data.isEmpty else { return }
        guard data.count <= 1024 * 1024 else { error(data, code: -32600, message: "Request exceeds 1 MiB."); return }
        guard let value = try? JSONSerialization.jsonObject(with: data) else { error(data, code: -32700, message: "Invalid JSON; send a JSON-RPC object."); return }
        guard let object = value as? [String: Any], object["jsonrpc"] as? String == "2.0",
            object["method"] is String || (object["id"] != nil && (object["result"] != nil || object["error"] != nil))
        else {
            error(data, code: -32600, message: "Invalid request; send a JSON-RPC 2.0 object."); return
        }
        if object["method"] as? String == "initialize" {
            initialize = data
            do { try await post(data); listen() } catch { self.error(data, message: error.localizedDescription) }
        } else if object["id"] == nil {
            if object["method"] as? String == "notifications/initialized" { initialized = data }
            do { try await send(data) } catch { diagnostic(error.localizedDescription) }
        } else {
            guard active < 64 else { error(data, message: "Too many concurrent calls; wait for a result."); return }
            active += 1
            Task {
                do { try await send(data) } catch { self.error(data, message: error.localizedDescription) }; finished()
            }
        }
    }
    func send(_ data: Data) async throws {
        if let recovering { try await recovering.value }
        let failedSession = session
        do { try await post(data) } catch {
            if let failure = error as? BridgeError {
                guard case .http(404) = failure else { throw error }
            }
            guard initialize != nil else { throw error }
            if failedSession != session { try await post(data); return }
            if !(error is BridgeError) {
                let previousURL = endpoint, previousPID = hostPID
                try refresh()
                guard endpoint != previousURL || hostPID != previousPID else { throw error }
            }
            try await recover(); try await post(data)
        }
    }
    func post(_ body: Data, output: Bool = true) async throws {
        let outbound = try request("POST", body: body)
        let usedSession = outbound.value(forHTTPHeaderField: "MCP-Session-Id")
        let (bytes, response) = try await BridgeHTTP.session.bytes(for: outbound)
        guard let http = response as? HTTPURLResponse else { throw BridgeError.message("Invalid HTTP response; reconnect.") }
        guard (200...299).contains(http.statusCode) else { throw BridgeError.http(http.statusCode) }
        if session == usedSession { session = http.value(forHTTPHeaderField: "MCP-Session-Id") ?? session }
        if http.statusCode == 202 { return }
        if http.value(forHTTPHeaderField: "Content-Type")?.contains("text/event-stream") == true {
            for try await line in bytes.lines {
                if line.hasPrefix("data: "), line.count > 6 { try responseData(Data(line.dropFirst(6).utf8), output: output) }
            }
        } else {
            var data = Data()
            for try await byte in bytes {
                guard data.count < 90 * 1024 * 1024 else { throw BridgeError.message("MCP response exceeds 90 MiB; reconnect.") }
                data.append(byte)
            }
            try responseData(data, output: output)
        }
    }

    func responseData(_ bytes: Data, output: Bool) throws {
        if let object = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any],
            let result = object["result"] as? [String: Any], let version = result["protocolVersion"] as? String
        {
            protocolVersion = version
        }
        if output { try emit(bytes) }
    }
    func recover() async throws {
        if let recovering { try await recovering.value; return }
        let task = Task {
            guard let initialize else { throw BridgeError.message("Initialize MCP first.") }
            notifications?.cancel(); notifications = nil; streamGeneration += 1
            session = nil; lastEventID = nil; try refresh()
            try await post(initialize, output: false); try await post(initialized, output: false)
            listen()
        }
        recovering = task
        defer { recovering = nil }
        try await task.value
    }
    func listen() {
        guard notifications == nil else { return }
        streamGeneration += 1; let generation = streamGeneration
        notifications = Task {
            var attempt = 0
            while attempt < 3 {
                guard !Task.isCancelled, generation == streamGeneration else { return }
                do {
                    let (bytes, response) = try await BridgeHTTP.session.bytes(for: request("GET"))
                    guard let http = response as? HTTPURLResponse else { break }
                    if http.statusCode == 400 {
                        lastEventID = nil
                    } else if http.statusCode != 200 {
                        break
                    } else {
                        attempt = 0
                        for try await line in bytes.lines {
                            guard !Task.isCancelled, generation == streamGeneration else { return }
                            if line.hasPrefix("id: ") { lastEventID = String(line.dropFirst(4)) }
                            if line.hasPrefix("data: "), line.count > 6 { try emit(Data(line.dropFirst(6).utf8)) }
                        }
                    }
                } catch { if Task.isCancelled { return } }
                attempt += 1
                try? await Task.sleep(for: .milliseconds(250 * attempt))
            }
            if generation == streamGeneration { notifications = nil }
        }
    }
    func error(_ request: Data, code: Int = -32002, message: String) {
        let object = (try? JSONSerialization.jsonObject(with: request)) as? [String: Any]
        let result: [String: Any] = ["jsonrpc": "2.0", "id": object?["id"] ?? NSNull(), "error": ["code": code, "message": message]]
        if let data = try? JSONSerialization.data(withJSONObject: result, options: .sortedKeys) { try? emit(data) }
    }
    func diagnostic(_ text: String) { try? FileHandle.standardError.write(contentsOf: Data("arsenal-mcp: \(text)\n".utf8)) }
    func emit(_ data: Data) throws { try FileHandle.standardOutput.write(contentsOf: data + Data("\n".utf8)) }
    func finished() { active -= 1; if active == 0 { let saved = waiters; waiters.removeAll(); for waiter in saved { waiter.resume() } } }
    func drain() async { if active > 0 { await withCheckedContinuation { waiters.append($0) } } }
    func close() async { notifications?.cancel(); if let request = try? request("DELETE") { _ = try? await BridgeHTTP.session.data(for: request) } }
}
enum BridgeError: Error, LocalizedError {
    case message(String)
    case http(Int)
    var errorDescription: String? {
        switch self {
        case .message(let text): return text
        case .http(let code): return "MCP HTTP \(code); check Arsenal and reconnect."
        }
    }
}

/// The bridge talks only to the local app: an ephemeral session keeps no disk cache or cookie store, so uninstalling leaves
/// nothing behind (VM run, 2 Nov: URLSession.shared left ~/Library/Caches/arsenal-mcp and ~/Library/HTTPStorages/arsenal-mcp).
enum BridgeHTTP {
    static let session = URLSession(configuration: .ephemeral)
}
