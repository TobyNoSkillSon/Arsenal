// SPDX-License-Identifier: MIT
// ArsenalWire v1 runner for FLUX.2 Klein 4B and FLUX 3 Image Klein (folder architecture flux3-image-klein): hello, then framed requests on stdin served one at a time in arrival order
// (queued requests wait; `cancel` reaches a queued or running one). Ops: load, generate, edit (FLUX 3), status, unload, convert.
import Darwin
import Foundation
import MLX

/// Cancellation flags for every admitted request (Python runner semantics: up to 64 queued).
final class Jobs: @unchecked Sendable {
    private let lock = NSLock()
    private var flags: [Double: Bool] = [:]
    func admit(_ id: Double) -> Bool {
        lock.withLock {
            guard flags.count < 64, flags[id] == nil else { return false }; flags[id] = false; return true
        }
    }
    func cancel(_ id: Double) -> Bool {
        lock.withLock {
            guard flags[id] != nil else { return false }; flags[id] = true; return true
        }
    }
    func cancelled(_ id: Double) -> Bool { lock.withLock { flags[id] ?? true } }
    func finish(_ id: Double) { _ = lock.withLock { flags.removeValue(forKey: id) } }
    func stopAll() { lock.withLock { for key in flags.keys { flags[key] = true } } }
}

public enum FluxWorker {
    public static let runnerName = "native"
    public static let build = "flux2-klein-swift/arsenal-v1"

    public static func run() async {
        // Library logs cannot corrupt framed stdout: the wire keeps the original stdout, fd 1 becomes stderr.
        let fd = dup(STDOUT_FILENO); guard fd >= 0 else { exit(1) }; dup2(STDERR_FILENO, STDOUT_FILENO)
        let wire = WorkerChannel(output: FileHandle(fileDescriptor: fd, closeOnDealloc: true))
        let jobs = Jobs()
        do {
            try wire.send([
                "event": "hello", "proto": 1, "worker": "flux", "runner": .string(runnerName), "pid": .number(Double(getpid())),
                "build": .string(build), "kinds": ["images"], "ops": ["cancel", "convert", "edit", "generate", "measurement_components"]
            ])
            let (stream, continuation) = AsyncStream<JSONObject>.makeStream()
            let reader = Thread {
                do {
                    while let request = try Frames.read(from: .standardInput) {
                        guard let id = request["id"]?.number, id.isFinite, id.rounded() == id, id >= 1, let op = request["op"]?.string
                        else { throw WorkerFailure(code: "malformed_frame", message: "Invalid request.") }
                        if op == "cancel" {
                            try wire.reply(.number(id), ["cancelled": .bool(request["target"]?.number.map(jobs.cancel) ?? false)]); continue
                        }
                        if op == "ping" { try wire.reply(.number(id)); continue }
                        guard jobs.admit(id) else { try wire.failure(.number(id), code: "busy", message: "Runner queue is full."); continue }
                        if !["status", "unload"].contains(op) { try wire.accepted(.number(id)) }
                        continuation.yield(request)
                    }
                    jobs.stopAll(); continuation.finish()
                } catch {
                    FileHandle.standardError.write(Data("flux: malformed or closed wire; exiting\n".utf8)); exit(2)
                }
            }
            // A process-lifetime blocking reader stays off GCD and the cooperative pool.
            reader.name = "arsenal.flux.stdin"; reader.start()
            let runner = Runner(wire: wire, jobs: jobs)
            for await request in stream {
                let id = request["id"] ?? .null
                do {
                    if try runner.run(request) { exit(0) }
                } catch let failure as WorkerFailure {
                    try wire.failure(id, code: failure.code, message: failure.message)
                } catch is KleinRequestCancelled {
                    try wire.failure(id, code: "cancelled", message: "Image request cancelled.")
                } catch {
                    try wire.failure(id, code: "engine_error", message: error.localizedDescription)
                }
                jobs.finish(id.number ?? -1)
            }
        } catch { FileHandle.standardError.write(Data("flux: \(error.localizedDescription)\n".utf8)); exit(1) }
    }
}

struct KleinRequestCancelled: Error {}
