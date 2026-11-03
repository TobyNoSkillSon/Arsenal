// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import ArsenalWire

struct CheckFailure: Error, CustomStringConvertible {
    let description: String
    init(_ text: String) { description = text }
}

/// Bounded pipe reads: a silent or partially written frame cannot hang the checker.
final class ProcessSession {
    let process = Process()
    let input = Pipe()
    let output = Pipe()
    let exited = DispatchSemaphore(value: 0)
    let timeout: Double
    let log: URL
    private let logHandle: FileHandle
    private var reaped = false
    private(set) var memories: [JSONObject] = []

    init(command: [String], support: URL, timeout: Double, label: String) throws {
        signal(SIGPIPE, SIG_IGN)
        self.timeout = timeout
        log = support.appendingPathComponent("\(label)-stderr.txt")
        FileManager.default.createFile(atPath: log.path, contents: nil)
        logHandle = try FileHandle(forWritingTo: log)
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = command
        var environment = ProcessInfo.processInfo.environment
        environment["ARSENAL_SUPPORT_DIR"] = support.path
        environment["ARSENAL_CACHE_DIR"] = support.appendingPathComponent("cache").path
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment
        process.standardInput = input
        process.standardOutput = output
        process.standardError = logHandle
        let semaphore = exited
        process.terminationHandler = { _ in semaphore.signal() }
        try process.run()
        // Foundation's Process creates its own process group on macOS. The explicit
        // attempt also covers implementations where that is not already done.
        _ = setpgid(process.processIdentifier, process.processIdentifier)
        try input.fileHandleForReading.close()
        try output.fileHandleForWriting.close()
        let fd = input.fileHandleForWriting.fileDescriptor
        _ = fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK)
    }
    deinit { stop(); try? logHandle.close() }

    func stop() {
        guard !reaped else { return }
        if process.isRunning {
            if getpgid(process.processIdentifier) == process.processIdentifier {
                _ = kill(-process.processIdentifier, SIGKILL)
            } else {
                _ = kill(process.processIdentifier, SIGKILL)
            }
        }
        _ = exited.wait(timeout: .now() + 2)
        reaped = true
        try? input.fileHandleForWriting.close()
        try? output.fileHandleForReading.close()
    }
    func send(_ object: JSONObject) throws { try raw(WireFrame.json(object).encoded()) }
    func raw(_ bytes: Data) throws {
        let fd = input.fileHandleForWriting.fileDescriptor
        let deadline = Date().addingTimeInterval(timeout)
        try bytes.withUnsafeBytes { buffer in
            var offset = 0
            while offset < buffer.count {
                let remaining = deadline.timeIntervalSinceNow
                guard remaining > 0 else { throw CheckFailure("runner input timed out") }
                var descriptor = pollfd(fd: fd, events: Int16(POLLOUT), revents: 0)
                let ready = poll(&descriptor, 1, Int32(min(remaining * 1000 + 1, Double(Int32.max))))
                if ready < 0 && errno == EINTR { continue }
                guard ready > 0 else { throw CheckFailure("runner input timed out or pipe failed") }
                let n = Darwin.write(fd, buffer.baseAddress!.advanced(by: offset), buffer.count - offset)
                if n < 0 && [EINTR, EAGAIN].contains(errno) { continue }
                guard n > 0 else { throw CheckFailure("runner stopped accepting input") }
                offset += n
            }
        }
    }
    func closeInput() throws { try input.fileHandleForWriting.close() }

    private func exact(_ count: Int, deadline: Date, allowEOF: Bool = false) throws -> Data? {
        var data = Data()
        let fd = output.fileHandleForReading.fileDescriptor
        while data.count < count {
            let remaining = deadline.timeIntervalSinceNow
            guard remaining > 0 else { throw CheckFailure("runner timed out; stderr: \(log.path)") }
            var descriptor = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
            let ready = poll(&descriptor, 1, Int32(min(remaining * 1000 + 1, Double(Int32.max))))
            if ready < 0 && errno == EINTR { continue }
            guard ready > 0 else { throw CheckFailure("runner timed out or pipe failed; stderr: \(log.path)") }
            var bytes = [UInt8](repeating: 0, count: min(count - data.count, 65536))
            let n = Darwin.read(fd, &bytes, bytes.count)
            if n < 0 && errno == EINTR { continue }
            if n == 0 && data.isEmpty && allowEOF { return nil }
            guard n > 0 else { throw CheckFailure("truncated frame; stderr: \(log.path)") }
            data.append(contentsOf: bytes.prefix(n))
        }
        return data
    }
    func read(deadline: Date? = nil) throws -> WireFrame? {
        let deadline = deadline ?? Date().addingTimeInterval(timeout)
        guard let header = try exact(4, deadline: deadline, allowEOF: true) else { return nil }
        let size = header.reduce(UInt32(0)) { $0 << 8 | UInt32($1) }
        guard size > 0 && size <= WireFrame.maximumLength else { throw CheckFailure("invalid frame length \(size)") }
        return try WireFrame.decodeBody(exact(Int(size), deadline: deadline)!)
    }
    func hello(kind: String, ops: [String]) throws {
        guard case .json(let value) = try read(), value["event"]?.string == "hello",
            value["proto"]?.number == 1, value["pid"]?.number == Double(process.processIdentifier),
            !(value["worker"]?.string ?? "").isEmpty, !(value["runner"]?.string ?? "").isEmpty,
            !(value["build"]?.string ?? "").isEmpty,
            value["kinds"]?.array?.contains(.string(kind)) == true
        else {
            throw CheckFailure("invalid first-frame handshake")
        }
        let advertised = Set(value["ops"]?.array?.compactMap(\.string) ?? [])
        guard Set(ops + ["load", "status", "ping", "cancel", "unload"]).isSubset(of: advertised) else {
            throw CheckFailure("hello.ops omits a required operation")
        }
    }
    func observe(_ object: JSONObject) throws {
        if let memory = object["memory"]?.object {
            for key in ["footprint_mb", "mlx_active_mb", "mlx_cache_mb"] {
                guard let number = memory[key]?.number, number.isFinite, number >= 0 else {
                    throw CheckFailure("memory.\(key) must be a finite non-negative number")
                }
            }
            memories.append(memory)
        }
    }
    func response(id: Int, success: Bool = true) throws -> (JSONObject, [WireFrame]) {
        let deadline = Date().addingTimeInterval(timeout)
        var events: [WireFrame] = []
        var received = 0
        while let frame = try read(deadline: deadline) {
            received += try frame.encoded().count
            guard received <= 64 * 1024 * 1024, events.count < 100000 else { throw CheckFailure("smoke response exceeds checker retention limit") }
            if case .json(let object) = frame {
                try observe(object)
                if object["event"]?.string == "status" && object["id"] == nil { events.append(frame); continue }
                guard object["id"]?.number == Double(id) else { throw CheckFailure("unexpected reply id") }
                if object["event"] != nil { events.append(frame); continue }
                guard object["ok"]?.bool == success else { throw CheckFailure("op returned \(object["error"]?["message"]?.string ?? "invalid ok flag")") }
                if !success { try Self.error(object) }
                return (object, events)
            }
            events.append(frame)
        }
        throw CheckFailure("runner exited before reply \(id)")
    }
    static func error(_ object: JSONObject) throws {
        guard let error = object["error"]?.object, !(error["code"]?.string ?? "").isEmpty,
            let message = error["message"]?.string, !message.isEmpty,
            !message.contains("\n"), !message.contains("\r")
        else { throw CheckFailure("invalid error envelope") }
    }
    func exitStatus() throws -> Int32 {
        guard exited.wait(timeout: .now() + timeout) == .success else { throw CheckFailure("runner did not exit") }
        reaped = true
        process.waitUntilExit()
        return process.terminationStatus
    }
}
