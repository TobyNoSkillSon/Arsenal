// SPDX-License-Identifier: MIT
import Foundation
import CoreFoundation
import Darwin

struct InvalidFrame: Error {}
final class Job: @unchecked Sendable {
    let id: Int
    private let lock = NSLock()
    private var cancelled = false
    init(_ id: Int) { self.id = id }
    func cancel() { lock.lock(); cancelled = true; lock.unlock() }
    func isCancelled() -> Bool { lock.lock(); defer { lock.unlock() }; return cancelled }
}

/// Locks cover state and whole output frames; work alone runs off the reader.
final class ToyRunner: @unchecked Sendable {
    private let output = NSLock()
    private let state = NSLock()
    private let jobs = DispatchGroup()
    private var active: Job?
    private var loaded = false
    private var scale = 1.0
    private let maximum = 16 * 1024 * 1024

    func exact(_ count: Int, eof: Bool = false) throws -> Data? {
        var data = Data()
        while data.count < count {
            guard let next = try FileHandle.standardInput.read(upToCount: count - data.count), !next.isEmpty else {
                if eof && data.isEmpty { return nil }; throw InvalidFrame()
            }
            data.append(next)
        }
        return data
    }
    func send(_ value: [String: Any]) {
        do {
            var body = Data([1]); body.append(try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]))
            guard body.count <= maximum else { throw InvalidFrame() }
            let n = UInt32(body.count)
            var frame = Data([UInt8(n >> 24), UInt8((n >> 16) & 255), UInt8((n >> 8) & 255), UInt8(n & 255)])
            frame.append(body)
            output.lock(); defer { output.unlock() }
            try FileHandle.standardOutput.write(contentsOf: frame)
        } catch { exit(2) }
    }
    func memory() -> [String: Double] {
        var info = mach_task_basic_info(); var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count) }
        }
        guard status == KERN_SUCCESS else { exit(2) }
        return ["footprint_mb": Double(info.resident_size) / 1_000_000, "mlx_active_mb": 0, "mlx_cache_mb": 0]
    }
    func reply(_ id: Int, _ fields: [String: Any] = [:]) {
        send(["event": "status", "memory": memory()])
        var result = fields; result["id"] = id; result["ok"] = true; send(result)
    }
    func error(_ id: Int, _ code: String, _ message: String) { send(["id": id, "ok": false, "error": ["code": code, "message": message]]) }
    func number(_ value: Any?) -> Double? {
        guard let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(), n.doubleValue.isFinite else { return nil }; return n.doubleValue
    }
    func finishJobs() {
        state.lock(); active?.cancel(); state.unlock()
        jobs.wait()
    }
    func evaluate(_ id: Int, _ request: [String: Any]) {
        let allowed: Set<String> = ["id", "op", "values", "iterations"]
        guard Set(request.keys).isSubset(of: allowed), let raw = request["values"] as? [Any], (1...64).contains(raw.count) else {
            error(id, "invalid_request", "Values must match the input schema."); return
        }
        let values = raw.compactMap(number)
        let n = request["iterations"].flatMap(number) ?? (request["iterations"] == nil ? 1 : 0)
        guard values.count == raw.count, values.allSatisfy({ abs($0) <= 1000000 }), n.rounded() == n, n >= 1, n <= 1000000000 else {
            error(id, "invalid_request", "Values and iterations must match the input schema."); return
        }
        state.lock()
        guard loaded, active == nil else {
            let code = loaded ? "busy" : "not_loaded"; state.unlock()
            error(id, code, "Load the model and wait for the active request."); return
        }
        let job = Job(id); active = job; let factor = scale
        jobs.enter(); state.unlock()
        DispatchQueue.global().async { [self] in
            defer {
                state.lock(); if active === job { active = nil }; state.unlock()
                jobs.leave()
            }
            var result = 0.0
            for i in 0..<Int(n) {
                if i % 1024 == 0 && job.isCancelled() { error(id, "cancelled", "Request cancelled."); return }
                result = values.reduce(0, +) * factor
            }
            if job.isCancelled() { error(id, "cancelled", "Request cancelled.") } else { reply(id, ["result": result]) }
        }
    }
    func run() throws {
        send([
            "event": "hello", "proto": 1, "worker": "toy", "runner": "swift", "pid": getpid(), "build": "template-1", "kinds": ["toy"],
            "ops": ["load", "ping", "status", "cancel", "unload", "evaluate"]
        ])
        defer { finishJobs() }
        while let header = try exact(4, eof: true) {
            let count = Int(header.reduce(UInt32(0)) { $0 << 8 | UInt32($1) })
            guard count > 0, count <= maximum, let body = try exact(count), body.first == 1,
                let request = try JSONSerialization.jsonObject(with: body.dropFirst()) as? [String: Any],
                let value = number(request["id"]), value >= 1, value <= Double(Int32.max), value.rounded() == value,
                let op = request["op"] as? String
            else { throw InvalidFrame() }
            let id = Int(value)
            switch op {
            case "load":
                state.lock(); let busy = active != nil; state.unlock()
                if busy { error(id, "busy", "Runner is busy."); continue }
                let config = request["config"] as? [String: Any] ?? [:]
                let settings = config["settings"] as? [String: Any] ?? [:]
                let rawScale = settings["scale"] ?? config["scale"] ?? 1
                guard let factor = number(rawScale), abs(factor) <= 100 else { error(id, "invalid_request", "Scale must be a finite number from -100 to 100."); continue }
                state.lock(); scale = factor; loaded = true; state.unlock()
                reply(id, ["load_s": 0, "engine": "cpu-toy"])
            case "ping", "status": reply(id)
            case "cancel":
                state.lock(); if number(request["target"]) == active.map({ Double($0.id) }) { active?.cancel() }; state.unlock()
                send(["id": id, "ok": true])
            case "evaluate": evaluate(id, request)
            case "unload": finishJobs(); state.lock(); loaded = false; state.unlock(); reply(id); return
            default: error(id, "unsupported_op", "Unsupported operation.")
            }
        }
    }
}
signal(SIGPIPE, SIG_IGN)
do { try ToyRunner().run() } catch { fputs("Invalid or truncated runner input.\n", stderr); exit(2) }
