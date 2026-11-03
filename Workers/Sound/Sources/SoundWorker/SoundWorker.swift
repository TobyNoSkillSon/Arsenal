// SPDX-License-Identifier: MIT
import Darwin
import Foundation
import ArsenalWire
import WorkerTransport
import VireoCore
import SoundEngine

/// The sound family's entry point: the `arsenal-worker-sound` executable here and the shipped combined native worker
/// (Workers/Native, one MLX copy for every family) both call it.
public enum SoundWorker {
    static func progressFrame(_ id: JSONValue, fraction: Double, step: Int? = nil, total: Int? = nil) -> JSONObject {
        var value: JSONObject = ["id": id, "event": "progress", "phase": "step", "fraction": .number(fraction)]
        value["step"] = step.map { .number(Double($0)) }; value["total"] = total.map { .number(Double($0)) }; return value
    }

    @MainActor public static func run() async {
        if let code = KernelGate.runChildIfRequested() { exit(code) }
        // Third-party tokenizer logs cannot corrupt framed stdout, even outside our vendor subset.
        let fd = dup(STDOUT_FILENO); guard fd >= 0 else { exit(1) }; dup2(STDERR_FILENO, STDOUT_FILENO)
        let wire = WorkerChannel(output: FileHandle(fileDescriptor: fd, closeOnDealloc: true))
        let requests = Requests()
        do {
            try wire.send(
                try WorkerHello(
                    worker: "sound", kinds: ["sound", "enhancement"],
                    ops: ["generate_sound", "generate_music", "enhance", "convert", "measurement_components", "measurement_source", "parameter_inventory"]
                ).json)
            let (stream, continuation) = AsyncStream<(JSONObject, GenerationControl)>.makeStream()
            WorkerThread.start(name: "arsenal.sound.stdin") {
                do {
                    while let frame = try FrameIO.read(from: .standardInput) {
                        guard case .json(let request) = frame, let id = request["id"], let op = request["op"]?.string,
                            let number = id.number, number.isFinite, number.rounded() == number, number >= 0
                        else { throw WireError.malformedFrame }
                        if op == "ping" { try wire.reply(id); continue }
                        if op == "cancel" { try wire.reply(id, ["cancelled": .bool(requests.cancel(request["target"] ?? .null))]); continue }
                        guard
                            let control = requests.admit(
                                id,
                                progress: { fraction in
                                    try? wire.send(progressFrame(id, fraction: fraction))
                                },
                                step: { fraction, step, total in
                                    try? wire.send(progressFrame(id, fraction: fraction, step: step, total: total))
                                })
                        else { try wire.failure(id, code: "busy", message: "Worker is busy; cancel the active request or wait"); continue }
                        if !["status", "unload", "voices"].contains(op) { try wire.accepted(id) }
                        continuation.yield((request, control))
                    }
                    requests.stop(); continuation.finish()
                } catch {
                    FileHandle.standardError.write(Data(("sound: malformed or closed wire; exiting\n").utf8)); exit(2)
                }
            }
            let runner = Runner(wire: wire)
            for await (request, control) in stream {
                do { if try await runner.run(request, control: control) { exit(0) } } catch let failure as WorkerFailure {
                    try wire.failure(request["id"] ?? .null, code: failure.code, message: failure.message)
                } catch is CancellationError { try wire.failure(request["id"] ?? .null, code: "cancelled", message: "Audio request cancelled") } catch {
                    try wire.failure(request["id"] ?? .null, code: "engine_error", message: error.localizedDescription)
                }
                requests.release()
            }
        } catch { FileHandle.standardError.write(Data(("sound: \(error.localizedDescription)\n").utf8)); exit(1) }
    }
}
