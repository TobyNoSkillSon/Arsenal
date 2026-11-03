// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire
import ArsenalCore
import CryptoKit

final class RunnerChecks {
    let command: [String]
    let model: CheckModel
    let support: URL
    let weights: URL
    let timeout: Double
    let fast: Bool
    var passed = 0
    var failed = 0

    init(command: [String], model: CheckModel, support: URL, weights: URL, timeout: Double, fast: Bool) {
        self.command = command; self.model = model; self.support = support
        self.weights = weights; self.timeout = timeout; self.fast = fast
    }
    func check(_ name: String, _ body: () throws -> Void) {
        do { try body(); passed += 1; print("PASS \(name)") } catch {
            failed += 1; print("FAIL \(name): \(String(describing: error).components(separatedBy: .newlines).joined(separator: " "))")
        }
    }
    func session(_ label: String) throws -> ProcessSession { try ProcessSession(command: command, support: support, timeout: timeout, label: label) }
    func run() throws {
        let runner = try session("main")
        defer { runner.stop() }
        // A bad handshake or load makes subsequent op checks meaningless.
        try runner.hello(kind: model.kind, ops: model.ops.map(\.name)); print("PASS handshake"); passed += 1
        var request: JSONObject = [
            "id": 1, "op": "load", "model_dir": .string(model.folder.path), "weights_dir": .string(weights.path), "dtype": .string(model.dtype), "fast": .bool(fast), "assets": [:],
            "config": ["settings": [:], "kernels": []]
        ]
        for (key, value) in model.load where !["id", "op", "model_dir", "weights_dir", "fast"].contains(key) { request[key] = value }
        var config = request["config"]?.object ?? [:]
        let assets = model.load["assets"]?.object ?? config["assets"]?.object ?? [:]
        request["assets"] = .object(assets); config["assets"] = .object(assets)
        if !model.kernelOverride {
            let kernels = try KernelManifest.loadIfPresent(folder: model.folder).resolved(chip: HostChip.current, fast: fast)
            config["kernels"] = .array(
                try kernels.map { kernel in
                    var value = try WireJSON.decode(JSONEncoder().encode(kernel))
                    value["files"] = .array(try kernel.files.map { .string(try SafePath.resolve($0, inside: model.folder.appendingPathComponent("kernels")).path) })
                    return .object(value)
                })
        }
        request["config"] = .object(config)
        try runner.send(request)
        let (loaded, _) = try runner.response(id: 1)
        guard let seconds = loaded["load_s"]?.number, seconds >= 0, seconds.isFinite,
            !(loaded["engine"]?.string ?? "").isEmpty, !runner.memories.isEmpty
        else { throw CheckFailure("load needs load_s, engine, and memory") }
        print("PASS load \(fast ? "fast" : "exact")"); passed += 1
        check("ping") {
            try runner.send(["id": 2, "op": "ping"]); _ = try runner.response(id: 2)
        }
        check("status memory") {
            let before = runner.memories.count
            try runner.send(["id": 3, "op": "status"]); _ = try runner.response(id: 3)
            guard runner.memories.count > before else { throw CheckFailure("status omitted memory") }
        }
        check("unsupported op") {
            try runner.send(["id": 4, "op": "__arsenal_unknown_op__"])
            let (error, _) = try runner.response(id: 4, success: false)
            guard error["error"]?["code"]?.string == "unsupported_op" else { throw CheckFailure("unknown op must return unsupported_op") }
        }
        var identifier = 10
        for op in model.ops {
            let id = identifier; identifier += 2
            check("op \(op.name)") {
                var params = op.params; params["id"] = .number(Double(id)); params["op"] = .string(op.name)
                let before = runner.memories.count
                try runner.send(params)
                let (reply, events) = try runner.response(id: id)
                try validateOutput(op, reply: reply, events: events, root: support.appendingPathComponent("jobs"))
                guard runner.memories.count > before else { throw CheckFailure("op omitted memory update") }
            }
            if op.stream && op.output == "audio" && op.schema["properties"]?["stream"] != nil {
                let streamID = identifier; identifier += 1
                check("op \(op.name) audio streaming") {
                    var params = op.params
                    params.removeValue(forKey: "out_dir"); params["stream"] = true
                    try CheckSchema.validate(.object(params), schema: op.schema)
                    params["id"] = .number(Double(streamID)); params["op"] = .string(op.name)
                    try runner.send(params)
                    let (reply, events) = try runner.response(id: streamID)
                    guard
                        events.contains(where: {
                            if case .binary = $0 { return true }; return false
                        })
                    else { throw CheckFailure("streaming audio produced no chunks") }
                    try validateOutput(op, reply: reply, events: events, root: support.appendingPathComponent("jobs"))
                }
            }
            check("op \(op.name) invalid input") {
                var invalid = op.params
                if let key = op.schema["required"]?.array?.first?.string {
                    invalid.removeValue(forKey: key)
                } else if let key = op.schema["properties"]?.object?.keys.sorted().first {
                    invalid[key] = .array([.null])
                } else if op.schema["additionalProperties"]?.bool == false {
                    invalid["__invalid__"] = true
                } else {
                    throw CheckFailure("unconstrained op schema needs an invalid-input constraint")
                }
                guard (try? CheckSchema.validate(.object(invalid), schema: op.schema)) == nil else { throw CheckFailure("cannot synthesize invalid input for \(op.name)") }
                invalid["id"] = .number(Double(id + 1)); invalid["op"] = .string(op.name)
                try runner.send(invalid)
                let (reply, _) = try runner.response(id: id + 1, success: false)
                guard reply["error"]?["code"]?.string != "unsupported_op" else { throw CheckFailure("declared op rejected as unsupported") }
            }
        }
        check("cancel active request") { try cancel(runner, id: identifier) }
        check("cancel unknown request") {
            try runner.send(["id": .number(Double(identifier + 2)), "op": "cancel", "target": 2147483647])
            _ = try runner.response(id: identifier + 2)
        }
        check("unload and exit") {
            try runner.send(["id": .number(Double(identifier + 3)), "op": "unload"])
            _ = try runner.response(id: identifier + 3)
            guard try runner.read() == nil, try runner.exitStatus() == 0 else { throw CheckFailure("unload must close stdout and exit zero") }
        }
        for round in 0..<2 {
            check("repeated load/unload \(round + 1)") {
                let probe = try session("lifecycle-\(round)")
                defer { probe.stop() }
                try probe.hello(kind: model.kind, ops: model.ops.map(\.name))
                try probe.send(request)
                _ = try probe.response(id: 1)
                try probe.send(["id": 2, "op": "unload"])
                _ = try probe.response(id: 2)
                guard try probe.read() == nil, try probe.exitStatus() == 0 else { throw CheckFailure("repeated lifecycle did not exit cleanly") }
            }
        }
        let malformed: [(String, Data)] = [
            ("zero length", Data([0, 0, 0, 0])), ("oversized length", Data([1, 0, 0, 1])),
            ("unknown frame kind", Data([0, 0, 0, 1, 99])), ("invalid JSON", Data([0, 0, 0, 2, 1, 123])),
            ("invalid UTF-8", Data([0, 0, 0, 2, 1, 255])),
            ("non-object JSON", Data([0, 0, 0, 3, 1, 91, 93])),
            ("truncated binary header", Data([0, 0, 0, 2, 2, 0])),
            ("truncated frame", Data([0, 0, 0, 3, 1])),
            ("invalid envelope", try WireFrame.json(["id": "bad", "op": "ping"]).encoded())
        ]
        for (label, bytes) in malformed {
            check("reject \(label)") {
                let probe = try session(label.replacingOccurrences(of: " ", with: "-"))
                defer { probe.stop() }
                try probe.hello(kind: model.kind, ops: model.ops.map(\.name))
                try probe.raw(bytes); try probe.closeInput()
                // A valid framed error followed by EOF is also safe, but replying
                // successfully or waiting forever on corrupt framing is not.
                if let frame = try probe.read() {
                    guard case .json(let reply) = frame, reply["ok"]?.bool == false else { throw CheckFailure("malformed input accepted") }
                    try ProcessSession.error(reply)
                    guard try probe.read() == nil else { throw CheckFailure("runner kept corrupt connection open") }
                }
                _ = try probe.exitStatus()
            }
        }
    }
    func retainReport() throws {
        let report: JSONObject = [
            "protocol": .string(failed == 0 ? "pass" : "fail"), "parity": "not-run", "task": "not-run",
            "cases": ["passed": .number(Double(passed)), "failed": .number(Double(failed)), "operations": .number(Double(model.ops.count))],
            "command": .array(command.map(JSONValue.string)), "model": .string(model.folder.path), "weights": .string(weights.path),
            "date": .string(ISO8601DateFormatter().string(from: Date())), "environment": ["os": .string(ProcessInfo.processInfo.operatingSystemVersionString)],
            "note": "Protocol evidence only. Independent numerical parity and task qualification require the model gate."
        ]
        try WireJSON.encode(report).write(to: support.appendingPathComponent("runner-check.json"), options: .atomic)
    }
    func cancel(_ runner: ProcessSession, id: Int) throws {
        let fixture = model.cancel
        let opName = fixture?["op"]?.string ?? model.ops.first!.name
        guard let op = model.ops.first(where: { $0.name == opName }) else { throw CheckFailure("cancel fixture names undeclared op") }
        var params = fixture?["params"]?.object ?? op.params
        if op.schema["properties"]?["out_dir"] != nil || ["file", "image", "audio"].contains(op.output) {
            let out = support.appendingPathComponent("jobs/cancel")
            try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
            params["out_dir"] = .string(out.path)
        }
        var validated = params
        if op.schema["properties"]?["out_dir"] == nil { validated["out_dir"] = nil }
        try CheckSchema.validate(.object(validated), schema: op.schema)
        params["id"] = .number(Double(id)); params["op"] = .string(op.name)
        try runner.send(params)
        try runner.send(["id": .number(Double(id + 1)), "op": "cancel", "target": .number(Double(id))])
        let deadline = Date().addingTimeInterval(timeout)
        var ack = false; var terminal = false
        while !ack || !terminal {
            guard let frame = try runner.read(deadline: deadline) else { throw CheckFailure("runner exited during cancel") }
            guard case .json(let object) = frame else { continue }
            try runner.observe(object)
            if object["event"] != nil { continue }
            if object["id"]?.number == Double(id + 1) {
                guard object["ok"]?.bool == true else { throw CheckFailure("cancel must acknowledge") }; ack = true
            } else if object["id"]?.number == Double(id) {
                if object["ok"]?.bool == false {
                    try ProcessSession.error(object)
                    guard object["error"]?["code"]?.string == "cancelled" else { throw CheckFailure("cancelled target needs cancelled error") }
                } else {
                    guard object["ok"]?.bool == true, fixture == nil else { throw CheckFailure("cancel fixture completed without observing cancellation") }
                }
                terminal = true
            } else {
                throw CheckFailure("unexpected id during cancel")
            }
        }
    }
}
