// SPDX-License-Identifier: MIT
import Foundation
import Network
import ArsenalCore

/// Service errors keep their useful one-line diagnosis rather than becoming opaque HTTP 500s.
func serviceHTTPFailure(_ error: any Error, modelID: String? = nil) -> HTTPFailure {
    if let failure = error as? HTTPFailure { return failure }
    func line(_ text: String) -> String { text.components(separatedBy: .newlines).filter { !$0.isEmpty }.joined(separator: " ") }
    if error is GetPlanChanged { return HTTPFailure(409, error.localizedDescription, code: "get_plan_changed") }
    // A gated Hugging Face model: a person must accept its terms; the app opened the page and retries by itself.
    if let gate = error as? HFGateError {
        return HTTPFailure(403, GatedModelText.line(gate, name: modelID ?? gate.repo), code: "waiting_on_person", details: GatedModelText.details(gate))
    }
    // A pinned file the repository lacks at its revision: final, never a retryable download failure.
    if let missing = error as? HFMissingFileError { return HTTPFailure(422, missing.localizedDescription, code: "pinned_file_missing") }
    if let failure = error as? WorkerFailure {
        let statuses = [
            "invalid_request": 422, "invalid_output": 502, "download_size": 422, "not_found": 404, "not_installed": 404, "precision": 400, "settings": 422, "setting": 422,
            "kind_order": 400,
            "pinned_file_missing": 422, "insufficient_memory": 409, "memory": 507, "busy": 409, "code_consent": 409, "mode": 400, "integrity": 409, "update_available": 409,
            "config_changed": 409, "get_plan_changed": 409, "catalog_key": 409,
            "not_running": 409,
            "unsupported_format": 422, "runner": 503, "updating": 503, "shutting_down": 503, "timeout": 504, "quarantined": 503,
            "hello_timeout": 504, "worker_exit": 502, "protocol": 502
        ]
        return HTTPFailure(statuses[failure.code] ?? workerHTTPStatus(failure), line(failure.message), code: failure.code, details: failure.details)
    }
    var message = line(error.localizedDescription)
    if error is ManifestError {
        let lower = message.lowercased()
        let status =
            lower.contains("disk full") ? 507 : lower.contains("precision is not offered") ? 400 : lower.contains("not installed") || lower.contains("asset is missing") ? 404 : 409
        if status == 409, let modelID, !lower.contains("\(MCPNames.prefix)download") { message += "; call \(MCPNames.prefix)download(model: \(modelID)) to repair." }
        return HTTPFailure(status, message)
    }
    if let error = error as? URLError { return HTTPFailure(error.code == .timedOut ? 504 : 503, message) }
    let ns = error as NSError
    if ns.domain == NSCocoaErrorDomain {
        if ns.code == NSFileWriteOutOfSpaceError { return HTTPFailure(507, message) }
        if ns.code == NSFileNoSuchFileError || ns.code == NSFileReadNoSuchFileError { return HTTPFailure(404, message) }
    }
    return HTTPFailure(500, message)
}

/// One request per connection. Network.framework handles TCP backpressure; every send awaits its completion.
final class HTTPConnection: @unchecked Sendable {
    let connection: NWConnection
    let requestID = "req_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    let queue = DispatchQueue(label: "arsenal.http.connection")
    private let lock = NSLock()
    private var opened = false
    private var ended = false
    private var task: Task<Void, Never>?
    private var watchdog: DispatchWorkItem?
    private var path = ""
    private let onEnd: @Sendable () -> Void
    init(_ connection: NWConnection, onEnd: @escaping @Sendable () -> Void) { self.connection = connection; self.onEnd = onEnd }
    func begin(port: Int, handler: @escaping @Sendable (HTTPRequest, HTTPConnection) async throws -> Void) {
        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed, .cancelled: self?.close()
            default: break
            }
        }
        connection.start(queue: queue)
        let work = Task {
            do {
                let request = try await readRequest(port: port)
                if request.headers["upgrade"]?.lowercased() != "websocket" { watchDisconnect() }
                let heartbeats = Task {
                    while !Task.isCancelled {
                        try? await Task.sleep(for: .seconds(10))
                        guard !Task.isCancelled else { return }
                        if isStreaming { try? await chunk(Data(": keepalive\n\n".utf8)) }
                    }
                }
                defer { heartbeats.cancel() }
                try await handler(request, self)
                if isStreaming { try await finishStream() }
            } catch {
                if !Task.isCancelled { await fail(serviceHTTPFailure(error)) }
            }
            close()
        }
        lock.withLock {
            task = work; if ended { work.cancel() }
        }
    }
    private var streamFinished = false
    private var streaming = false
    var isStreaming: Bool { lock.withLock { streaming && !streamFinished && !ended } }
    func close() {
        let changed = lock.withLock { () -> Bool in
            guard !ended else { return false }; ended = true; task?.cancel(); task = nil; return true
        }
        guard changed else { return }
        connection.cancel()
        queue.async { [self] in
            watchdog?.cancel(); watchdog = nil
        }
        onEnd()
    }
    private func arm(_ seconds: Double) {
        queue.async { [self] in
            watchdog?.cancel()
            let item = DispatchWorkItem { [weak self] in self?.close() }
            watchdog = item; queue.asyncAfter(deadline: .now() + seconds, execute: item)
        }
    }
    func receive() async throws -> Data {
        try await withCheckedThrowingContinuation { continuation in
            connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { data, _, complete, error in
                if let error {
                    continuation.resume(throwing: error)
                } else if let data, !data.isEmpty {
                    continuation.resume(returning: data)
                } else if complete {
                    continuation.resume(throwing: CancellationError())
                } else {
                    continuation.resume(throwing: HTTPFailure(400, "Incomplete request."))
                }
            }
        }
    }
    private func watchDisconnect() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1) { [weak self] _, _, done, error in
            if done || error != nil { self?.close() } else { self?.close() } // No HTTP pipelining.
        }
    }
    private func readRequest(port: Int) async throws -> HTTPRequest {
        arm(10)
        var buffer = Data()
        let separator = Data("\r\n\r\n".utf8)
        while buffer.range(of: separator) == nil {
            buffer += try await receive()
            if buffer.count > 65536 && buffer.range(of: separator) == nil { throw HTTPFailure(431, "Headers exceed 64 KiB.") }
        }
        let range = buffer.range(of: separator)!
        guard range.lowerBound <= 65536, let text = String(data: buffer[..<range.lowerBound], encoding: .utf8) else { throw HTTPFailure(400, "Malformed request head.") }
        let lines = text.components(separatedBy: "\r\n")
        let first = lines[0].split(separator: " ", omittingEmptySubsequences: false)
        guard first.count == 3, first[1].hasPrefix("/"), first[2] == "HTTP/1.1" else { throw HTTPFailure(400, "Use origin-form HTTP/1.1.") }
        path = String(first[1]).components(separatedBy: "?")[0]
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { throw HTTPFailure(400, "Malformed header.") }
            let key = String(line[..<colon]).lowercased()
            guard !key.isEmpty, key.utf8.allSatisfy({ (33...126).contains($0) && !"()<>@,;:\\\"/[]?={} \t".utf8.contains($0) }), headers[key] == nil else {
                throw HTTPFailure(400, "Invalid or repeated header.")
            }
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            guard !value.contains("\0") else { throw HTTPFailure(400, "Invalid header value.") }
            headers[key] = value
        }
        guard headers["origin"] == nil else { throw HTTPFailure(403, "Browser origins are not accepted.") }
        guard ["127.0.0.1:\(port)", "localhost:\(port)"].contains(headers["host"]?.lowercased() ?? "") else {
            throw HTTPFailure(403, "Host must be 127.0.0.1:\(port) or localhost:\(port).")
        }
        let type = headers["content-type"]?.components(separatedBy: ";")[0].lowercased()
        let maxBody = type == "multipart/form-data" ? 16 * 1024 * 1024 : 1024 * 1024
        let length = headers["content-length"].flatMap(Int.init)
        if headers["content-length"] != nil && (length == nil || length! < 0) { throw HTTPFailure(400, "Invalid Content-Length.") }
        if (length ?? 0) > maxBody { throw HTTPFailure(413, "Request body exceeds \(maxBody) bytes.") }
        let chunked = headers["transfer-encoding"] != nil
        if chunked && (headers["transfer-encoding"]?.lowercased() != "chunked" || length != nil) { throw HTTPFailure(400, "Ambiguous body framing.") }
        let emptyCancel =
            (path.hasPrefix("/providers/replicate/v1/predictions/") || path.hasPrefix("/arsenal/v1/jobs/")) && path.hasSuffix("/cancel") && (length ?? 0) == 0 && !chunked
            && type == nil
        let emptyRefresh = path == "/arsenal/v1/refresh" && (length ?? 0) == 0 && !chunked && type == nil
        if first[0] == "POST" && type != "application/json" && type != "multipart/form-data" && type != "application/x-www-form-urlencoded" && !emptyCancel && !emptyRefresh {
            throw HTTPFailure(415, "Send application/json or multipart/form-data.")
        }
        if let expect = headers["expect"] {
            guard expect.lowercased() == "100-continue" else { throw HTTPFailure(417, "Unsupported Expect header.") }
            try await send(Data("HTTP/1.1 100 Continue\r\n\r\n".utf8))
        }
        buffer = Data(buffer[range.upperBound...]); arm(30)
        func take(_ count: Int) async throws -> Data {
            while buffer.count < count { buffer += try await receive(); arm(30) }
            let bytes = Data(buffer.prefix(count)); buffer = Data(buffer.dropFirst(count)); return bytes
        }
        var body = Data()
        if chunked {
            while true {
                while buffer.range(of: Data("\r\n".utf8)) == nil {
                    guard buffer.count < 1024 else { throw HTTPFailure(400, "Chunk header too long.") }
                    buffer += try await receive(); arm(30)
                }
                let end = buffer.range(of: Data("\r\n".utf8))!
                let line = String(decoding: buffer[..<end.lowerBound], as: UTF8.self).components(separatedBy: ";")[0]
                guard !line.isEmpty, line.utf8.allSatisfy({ "0123456789abcdefABCDEF".utf8.contains($0) }), let size = Int(line, radix: 16), size <= maxBody - body.count else {
                    throw HTTPFailure(413, "Invalid or oversized chunk.")
                }
                buffer = Data(buffer.dropFirst(end.upperBound))
                if size == 0 {
                    // Trailer fields are not needed by our endpoints; reject instead of letting them override security headers.
                    guard try await take(2) == Data("\r\n".utf8) else { throw HTTPFailure(400, "Chunk trailers are unsupported.") }
                    break
                }
                body += try await take(size)
                guard try await take(2) == Data("\r\n".utf8) else { throw HTTPFailure(400, "Malformed chunk framing.") }
            }
        } else {
            body = try await take(length ?? 0)
        }
        guard buffer.isEmpty else { throw HTTPFailure(400, "Bytes after request body are unsupported.") }
        queue.async { [self] in
            watchdog?.cancel(); watchdog = nil
        }
        if type == "multipart/form-data" { _ = try Multipart.parse(body, contentType: headers["content-type"] ?? "") }
        return HTTPRequest(method: String(first[0]), target: String(first[1]), headers: headers, body: body)
    }
    func send(_ data: Data) async throws {
        try Task.checkCancellation()
        do {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                connection.send(
                    content: data,
                    completion: .contentProcessed { error in
                        if let error { continuation.resume(throwing: error) } else { continuation.resume() }
                    })
            }
        } catch {
            // Cancel the handler (and its ModelsService request) before a sink
            // failure reaches the worker adapter. A lost client is not bad wire.
            close()
            throw CancellationError()
        }
    }
    func respond(_ object: Any, status: Int = 200, headers: [String: String] = [:], isolation: isolated (any Actor)? = #isolation) async throws {
        try await respondBytes(jsonData(object), type: "application/json", status: status, headers: headers)
    }
    func respondBytes(_ data: Data, type: String, status: Int = 200, headers: [String: String] = [:]) async throws {
        guard markOpened() else { throw HTTPFailure(500, "Response already started.") }
        try await send(head(status: status, type: type, framing: "Content-Length: \(data.count)\r\n", headers: headers) + data)
    }
    /// A file sent from disk in 1 MiB pieces after its exact length, so large artifacts never sit in memory whole.
    func respondFile(_ url: URL, size: Int64, type: String, headers: [String: String] = [:]) async throws {
        guard markOpened() else { throw HTTPFailure(500, "Response already started.") }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        try await send(head(status: 200, type: type, framing: "Content-Length: \(size)\r\n", headers: headers))
        var sent: Int64 = 0
        while let piece = try handle.read(upToCount: 1 << 20), !piece.isEmpty {
            try Task.checkCancellation()
            try await send(piece); sent += Int64(piece.count)
        }
        guard sent == size else { close(); throw HTTPFailure(500, "Output file changed while it was sent.") }
    }
    func markOpened() -> Bool {
        lock.withLock {
            if opened { return false }; opened = true; return true
        }
    }
    func beginStream(type: String = "text/event-stream", headers: [String: String] = [:]) async throws {
        guard markOpened() else { throw HTTPFailure(500, "Response already started.") }
        lock.withLock { streaming = type == "text/event-stream" }
        try await send(head(status: 200, type: type, framing: "Transfer-Encoding: chunked\r\n", headers: headers))
    }
    func chunk(_ data: Data) async throws {
        guard !data.isEmpty else { return }
        try await send(Data("\(String(data.count, radix: 16))\r\n".utf8) + data + Data("\r\n".utf8))
    }
    func event(_ object: Any, name: String? = nil, isolation: isolated (any Actor)? = #isolation) async throws {
        var text = name.map { "event: \($0)\n" } ?? ""
        text += "data: \(String(decoding: jsonData(object), as: UTF8.self))\n\n"
        try await chunk(Data(text.utf8))
    }
    func finishStream() async throws {
        let finish = lock.withLock {
            if streamFinished { return false }; streamFinished = true; return true
        }
        if finish { try await send(Data("0\r\n\r\n".utf8)) }
    }
    private func head(status: Int, type: String, framing: String, headers: [String: String]) -> Data {
        let extra = headers.map { "\($0.key): \($0.value)\r\n" }.joined()
        let reason =
            [
                200: "OK", 201: "Created", 202: "Accepted", 400: "Bad Request", 404: "Not Found", 422: "Unprocessable Content", 429: "Too Many Requests",
                500: "Internal Server Error"
            ][status] ?? "Error"
        return Data(
            "HTTP/1.1 \(status) \(reason)\r\nContent-Type: \(type)\r\n\(framing)Connection: close\r\nx-request-id: \(requestID)\r\nx-typesafe-request-id: \(requestID)\r\nCache-Control: no-store\r\n\(extra)\r\n"
                .utf8)
    }
    private func fail(_ error: HTTPFailure) async {
        do {
            if isStreaming {
                let body = errorBody(error, path: path, requestID: requestID)
                let prefix = path.contains("/anthropic/") ? "event: error\n" : ""
                try await chunk(Data("\(prefix)data: \(String(decoding: body, as: UTF8.self))\n\n".utf8)); try await finishStream()
            } else if !lock.withLock({ opened }) {
                try await respondBytes(errorBody(error, path: path, requestID: requestID), type: "application/json", status: error.status)
            }
        } catch {}
    }
}
