// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import MLX
import ArsenalWire

/// Framed lifecycle for newly integrated families. The stdin reader remains responsive to cancel while the serial engine queue works.
public enum NativeFamilyWorker {
    public static func run(family: String, kinds: [String], operations: [String], registrations: [NativeRegistration]) {
        NativeRuntime.prepare()
        signal(SIGPIPE, SIG_IGN)
        let host = NativeFamilyHost(registrations: registrations)
        host.send(
            (try? WorkerHello(
                worker: family, kinds: kinds,
                ops: ["load", "ping", "status", "cancel", "unload", "parameter_inventory", "convert"] + operations
            ).json) ?? [:])
        do {
            while let frame = try autoreleasepool(invoking: { try FrameIO.read(from: .standardInput) }) {
                guard case .json(let request) = frame, let number = request["id"]?.number, number > 0,
                    number <= Double(Int32.max), number.rounded() == number, let op = request["op"]?.string
                else { throw WireError.malformedFrame }
                let id = Int(number)
                if op == "cancel" {
                    guard let target = request["target"]?.number, target.isFinite, target > 0, target <= Double(Int32.max), target.rounded() == target else {
                        host.failure(id, WorkerFailure(code: "invalid_request", message: "cancel requires a positive target request ID.")); continue
                    }
                    host.cancel(Int(target)); host.send(["id": .number(number), "ok": true]); continue
                }
                if op == "ping" { host.send(["id": .number(number), "ok": true]); continue }
                if op == "unload" { host.cancel(nil); host.enqueue(request, id: id, op: op, unloading: true); continue }
                host.enqueue(request, id: id, op: op)
            }
            host.cancel(nil)
            host.queue.sync {
                host.model?.unload(); host.model = nil
            }
        } catch { host.cancel(nil); FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8)) }
    }
}

private final class NativeFamilyHost: @unchecked Sendable {
    let queue = DispatchQueue(label: "arsenal.native.family", qos: .utility)
    let registrations: [NativeRegistration]
    private let state = NSLock(), output = NSLock()
    private var active: (Int, NativeCancellation)?
    var model: (any NativeModel)?
    init(registrations: [NativeRegistration]) { self.registrations = registrations }
    func sendFrame(_ frame: WireFrame) throws { try output.withLock { try FrameIO.write(frame, to: .standardOutput) } }
    func send(_ value: JSONObject) { try? sendFrame(.json(value)) }
    func cancel(_ target: Int?) { state.withLock { if target == nil || target == active?.0 { active?.1.cancel() } } }
    func enqueue(_ request: JSONObject, id: Int, op: String, unloading: Bool = false) {
        let token = NativeCancellation()
        let accepted = state.withLock { () -> Bool in
            if active != nil && !unloading { return false }
            if !unloading { active = (id, token) }; return true
        }
        guard accepted else { failure(id, WorkerFailure(code: "busy", message: "Wait for or cancel the active request.")); return }
        queue.async { [self] in
            autoreleasepool {
                do {
                    try token.check()
                    var result: JSONObject = [:]
                    switch op {
                    case "convert":
                        guard model == nil, let recipe = request["recipe"]?.string else {
                            throw WorkerFailure(code: "unsupported_recipe", message: "Conversion requires an unloaded model and recipe.")
                        }
                        if recipe == NativeConversion.normalizationRecipe {
                            result = try NativeConversion.normalize(request, cancellation: token)
                        } else if let converter = registrations.compactMap(\.conversion).first(where: { $0.recipe == recipe }) {
                            result = try converter.run(request, cancellation: token)
                        } else {
                            throw WorkerFailure(code: "unsupported_recipe", message: "No registered converter for this recipe.")
                        }
                    case "load":
                        guard model == nil else { throw WorkerFailure(code: "busy", message: "Unload before replacing a model.") }
                        let context = try NativeLoadContext(request)
                        let configFile = context.weights.appendingPathComponent("config.json")
                        let config = FileManager.default.fileExists(atPath: configFile.path) ? try WireJSON.decode(Data(contentsOf: configFile)) : [:]
                        let registration = try NativeRegistration.resolve(checkpoint: config, architecture: context.architecture, registrations: registrations)
                        let instance = registration.make(), start = ProcessInfo.processInfo.systemUptime
                        do { try instance.load(context, cancellation: token); try NativeParameters.requireHalf(instance.parameters); try token.check() } catch {
                            instance.unload(); Memory.clearCache(); throw error
                        }
                        model = instance
                        result = ["engine": "mlx-swift", "dtype": .string(context.precision), "load_s": .number(ProcessInfo.processInfo.systemUptime - start)]
                    case "unload": model?.unload(); model = nil; Memory.clearCache()
                    case "status": result = ["loaded": .bool(model != nil), "memory": .object(memory()), "peak_bytes": .number(Double(model?.peakBytes ?? Memory.peakMemory))]
                    case "parameter_inventory":
                        guard ProcessInfo.processInfo.environment["ARSENAL_NATIVE_LAB"] == "1", let model else {
                            throw WorkerFailure(code: "invalid_request", message: "Parameter inventory requires a loaded isolated lab worker.")
                        }
                        result = ["parameters": .object(NativeParameters.inventory(model.parameters))]
                    case "parse_output":
                        // Lab-only: a maker-documented model output through the adapter's own serving parser.
                        guard ProcessInfo.processInfo.environment["ARSENAL_NATIVE_LAB"] == "1", let model else {
                            throw WorkerFailure(code: "invalid_request", message: "Parser fixtures require a loaded isolated lab worker.")
                        }
                        guard let parser = model as? any NativeOutputParsing else {
                            throw WorkerFailure(code: "unsupported_op", message: "The loaded model does not expose its output parser.")
                        }
                        guard let operation = request["operation"]?.string, model.operations.contains(operation), let raw = request["raw"]?.string else {
                            throw WorkerFailure(code: "invalid_request", message: "parse_output requires a declared operation and raw model output text.")
                        }
                        result = try parser.parseOutput(operation, input: request["input"]?.object ?? [:], raw: raw)
                    default:
                        guard let model else { throw WorkerFailure(code: "not_loaded", message: "Load a model first.") }
                        guard model.operations.contains(op) else { throw WorkerFailure(code: "unsupported_op", message: "The loaded model does not support \(op).") }
                        var input = request; input.removeValue(forKey: "id"); input.removeValue(forKey: "op")
                        result = try model.run(
                            op, input: input, cancellation: token,
                            emit: { frame in
                                try token.check()
                                switch frame {
                                case .json(var event): event["id"] = .number(Double(id)); try self.sendFrame(.json(event))
                                case .binary: try self.sendFrame(frame)
                                }
                            })
                        try NativeParameters.requireHalf(model.parameters)
                    }
                    try token.check()
                    send(["event": "status", "memory": .object(memory()), "peak_bytes": .number(Double(model?.peakBytes ?? Memory.peakMemory))])
                    result["id"] = .number(Double(id)); result["ok"] = true; complete(id, result)
                    if unloading { exit(0) }
                } catch { complete(id, failurePayload(id, error)) }
            }
        }
    }
    /// Release admission before making the terminal reply observable. The old turn performs no further model work.
    private func complete(_ id: Int, _ result: JSONObject) {
        state.withLock { if active?.0 == id { active = nil } }
        send(result)
    }
    func failure(_ id: Int, _ error: Error) { send(failurePayload(id, error)) }
    private func failurePayload(_ id: Int, _ error: Error) -> JSONObject {
        let failure = error as? WorkerFailure ?? WorkerFailure(code: "runner", message: error.localizedDescription)
        return ["id": .number(Double(id)), "ok": false, "error": ["code": .string(failure.code), "message": .string(failure.message)]]
    }
    func memory() -> JSONObject {
        var info = task_vm_info_data_t(); var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return [
            "footprint_mb": .number(status == KERN_SUCCESS ? Double(info.phys_footprint) / 1e6 : Double(Memory.activeMemory) / 1e6),
            "mlx_active_mb": .number(Double(Memory.activeMemory) / 1e6), "mlx_cache_mb": .number(Double(Memory.cacheMemory) / 1e6)
        ]
    }
}
