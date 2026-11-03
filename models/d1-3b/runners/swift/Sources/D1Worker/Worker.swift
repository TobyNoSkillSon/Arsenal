// SPDX-License-Identifier: MIT
// d1-3B's ArsenalWire runner (docs/RUNNERS.md): the request loop of python/arsenal_worker (one op at a time on a work
// thread, the reader free to take `cancel`), with the d1 ops in Backend.
import D1Engine
import Darwin
import Foundation
import MLX

public enum D1Worker {
    public static let build = "d1-hybrid/mlx-swift-9019419"

    public static func run() -> Never {
        // The fp32 option readout stays fp32: MLX would otherwise run fp32 GEMMs as TF32. Set before MLX is touched.
        setenv("MLX_ENABLE_TF32", "0", 1)
        let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
        // Metal library: an explicit override, else a signed app's Contents/Resources/mlx.metallib; else MLX finds
        // mlx-swift_Cmlx.bundle next to the binary (build.sh output, the app's Helpers layout).
        if let path = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] {
            GPU.metallib = URL(fileURLWithPath: path)
        } else {
            let bundled = executable.deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Resources/mlx.metallib")
            if FileManager.default.fileExists(atPath: bundled.path) { GPU.metallib = bundled }
        }
        // stdout carries frames only: keep a private descriptor for them and point fd 1 at stderr.
        let frames = dup(1)
        dup2(2, 1)
        let worker = Loop(out: frames)
        let code = worker.serve()
        exit(code)
    }
}

final class Loop: @unchecked Sendable {
    private let out: Int32
    private let backend = Backend()
    private let lock = NSLock()
    private var jobs: [Int: Bool] = [:] // request id -> cancelled
    private var accepted: [Int: Double] = [:]
    private var queue: [(Int, String, PyJSON)] = []
    private let available = DispatchSemaphore(value: 0)
    private let unloaded = DispatchSemaphore(value: 0)

    init(out: Int32) { self.out = out }

    private func send(_ fields: [(String, PyJSON)]) { Wire.write(.object(fields), out) }

    private func now() -> Double { ProcessInfo.processInfo.systemUptime }

    func serve() -> Int32 {
        let ops = ["cancel", "measurement_components", "convert", "decide", "judge", "load", "ping", "status", "unload"]
        send([
            ("event", .string("hello")), ("proto", .int(1)), ("worker", .string("d1")), ("runner", .string("native")), ("pid", .int(Int(getpid()))),
            ("build", .string(D1Worker.build)), ("kinds", .array([.string("decisions")])), ("ops", .array(ops.map { .string($0) }))
        ])
        let thread = Thread { [self] in work() }
        thread.stackSize = 16 << 20
        thread.start()
        do {
            while let payload = try Wire.read(0) {
                let request = try PyJSONParser.parse(payload)
                guard request.isObject, let id = request["id"]?.int, id >= 1, let op = request["op"]?.string else {
                    throw Wire.Malformed(description: "Invalid request.")
                }
                if op == "cancel" {
                    lock.lock()
                    if let target = request["target"]?.int, jobs[target] != nil { jobs[target] = true }
                    lock.unlock()
                    send([("id", .int(id)), ("ok", .bool(true))])
                    continue
                }
                lock.lock()
                if jobs[id] != nil {
                    lock.unlock()
                    throw Wire.Malformed(description: "Duplicate request id.")
                }
                if jobs.count >= 64 {
                    lock.unlock()
                    send([("id", .int(id)), ("ok", .bool(false)), ("error", .object([("code", .string("busy")), ("message", .string("Runner queue is full."))]))])
                    continue
                }
                jobs[id] = false
                accepted[id] = now()
                queue.append((id, op, request))
                lock.unlock()
                if !["ping", "status", "unload"].contains(op) { send([("id", .int(id)), ("event", .string("progress")), ("phase", .string("queued"))]) }
                available.signal()
                if op == "unload" {
                    unloaded.wait()
                    return 0
                }
            }
        } catch {
            FileHandle.standardError.write(Data("arsenal-worker-d1: \(error)\n".utf8))
            return 2
        }
        // EOF: cancel what is queued or running, then leave.
        lock.lock()
        for id in jobs.keys { jobs[id] = true }
        lock.unlock()
        return 0
    }

    private func cancelled(_ id: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return jobs[id] ?? true
    }

    private func work() {
        while true {
            available.wait()
            lock.lock()
            let (id, op, request) = queue.removeFirst()
            lock.unlock()
            dispatch(id, op, request)
            lock.lock()
            jobs[id] = nil
            accepted[id] = nil
            lock.unlock()
            if op == "unload" {
                unloaded.signal()
                return
            }
        }
    }

    private func dispatch(_ id: Int, _ op: String, _ request: PyJSON) {
        let scheduled = now()
        lock.lock()
        let acceptedAt = accepted[id] ?? scheduled
        lock.unlock()
        let instrumented = !["ping", "status", "unload"].contains(op)
        if instrumented {
            Memory.peakMemory = 0
            send([("id", .int(id)), ("event", .string("progress")), ("phase", .string(op == "load" ? "loading" : "step"))])
        }
        do {
            var result: [(String, PyJSON)]
            switch op {
            case "ping": result = []
            case "status": result = [("memory", backend.memory())]
            case "unload": result = backend.unload()
            case "load": result = try backend.load(request)
            case "measurement_components": result = try backend.intakeComponents(request)
            case "convert": result = try backend.convert(request)
            case "decide": result = try backend.decide(request) { self.cancelled(id) }
            case "judge": result = try backend.judge(request) { self.cancelled(id) }
            default: throw WorkerError("unsupported_op", "Unsupported operation.")
            }
            if cancelled(id) { throw WorkerError("cancelled", "Request cancelled.") }
            if instrumented {
                var timing: [(String, PyJSON)] = [("queue_s", .number(max(0, scheduled - acceptedAt)))]
                if op == "load" { timing.append(("load_s", .number(now() - scheduled))) }
                timing.append(("peak_mb", .number(Double(Memory.peakMemory) / 1e6)))
                result.append(("timing", .object(timing)))
                send([("id", .int(id)), ("event", .string("progress")), ("phase", .string("done")), ("timing", .object(timing))])
            }
            send([("event", .string("status")), ("memory", backend.memory())])
            send(result + [("id", .int(id)), ("ok", .bool(true))])
        } catch let error as WorkerError {
            var detail: [(String, PyJSON)] = [("code", .string(error.code)), ("message", .string(error.message))]
            if let status = error.status { detail.append(("status", .int(status))) }
            send([("id", .int(id)), ("ok", .bool(false)), ("error", .object(detail))])
        } catch {
            FileHandle.standardError.write(Data("arsenal-worker-d1: \(op) \(id): \(error)\n".utf8))
            send([
                ("id", .int(id)), ("ok", .bool(false)),
                ("error", .object([("code", .string("runner")), ("message", .string("Runner request failed; see its log."))]))
            ])
        }
    }
}
