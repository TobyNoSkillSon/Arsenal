// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import ArsenalWire
import ChatContract
import ChatEngine

final class WorkerIO: @unchecked Sendable {
    private let outputLock = NSLock()
    private let stateLock = NSLock()
    private var active: (Int, Cancellation)?
    func send(_ object: JSONObject) {
        outputLock.withLock {
            do { try FrameIO.write(.json(object), to: .standardOutput) } catch { exit(1) }
        }
    }
    func begin(_ id: Int) throws -> Cancellation {
        try stateLock.withLock {
            guard active == nil else { throw ChatFailure("busy", "The chat worker is busy; cancel the active request or wait for it to finish.") }
            let c = Cancellation(); active = (id, c); return c
        }
    }
    func finish(_ id: Int) { stateLock.withLock { if active?.0 == id { active = nil } } }
    func cancel(_ id: Int?) -> Bool {
        stateLock.withLock {
            guard let active, id == nil || id == active.0 else { return false }
            active.1.cancel(); return true
        }
    }
    func failure(_ id: JSONValue, _ error: any Error) {
        let failure = error as? ChatFailure ?? ChatFailure("runner_error", error.localizedDescription.replacingOccurrences(of: "\n", with: " "))
        send(["id": id, "ok": false, "error": ["code": .string(failure.code), "message": .string(failure.message)]])
    }
    func status() { send(["event": "status", "memory": encoded(ChatMemory.snapshot())]) }
    func encoded<T: Encodable>(_ value: T) -> JSONValue {
        do { return try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(value)) } catch { exit(1) }
    }
}

/// The chat family's entry point: the `arsenal-worker-chat` executable here and the shipped combined native worker
/// (Workers/Native, one MLX copy for every family) both call it.
public enum ChatWorker {
    /// Shared production preparation boundary; tests inject generation without loading weights.
    static func chat(_ params: [String: Any], generate: (ChatRequest) async throws -> ChatOutput) async throws -> ChatOutput {
        try await generate(ChatRequest(params))
    }
    public static func run() {
        signal(SIGPIPE, SIG_IGN)
        let io = WorkerIO(), engine = ChatEngine()
        do {
            io.send(
                try WorkerHello(worker: "chat", kinds: ["chat"], ops: ["load", "chat", "count_tokens", "status", "cancel", "unload", "convert", "ping", "measurement_logits"]).json)
            while let frame = try FrameIO.read(from: .standardInput) {
                guard case .json(let request) = frame else { throw WireError.malformedFrame }
                guard let number = request["id"]?.number, number.isFinite, number >= 0, number <= Double(Int32.max), number.rounded() == number,
                    let op = request["op"]?.string
                else { throw WireError.malformedFrame }
                let id = Int(number), wireID = JSONValue.number(number)
                switch op {
                case "ping": io.send(["id": wireID, "ok": true])
                case "cancel":
                    guard let target = request["target"]?.number, target.isFinite, target >= 0, target <= Double(Int32.max), target.rounded() == target else {
                        io.failure(wireID, ChatFailure("cancel requires a nonnegative target request ID.")); continue
                    }
                    io.send(["id": wireID, "ok": true, "cancelled": .bool(io.cancel(Int(target)))])
                case "status":
                    Task.detached {
                        let s = await engine.status()
                        io.send([
                            "id": wireID, "ok": true, "loaded": .bool(s.loaded), "dtype": s.dtype.map(JSONValue.string) ?? .null,
                            "context": .number(Double(s.context)), "memory": io.encoded(ChatMemory.snapshot())
                        ])
                    }
                case "unload":
                    _ = io.cancel(nil)
                    Task.detached {
                        await engine.unload()
                        io.send(["id": wireID, "ok": true]); io.status(); exit(0)
                    }
                case "load", "chat", "count_tokens", "convert", "measurement_logits":
                    do {
                        let cancellation = try io.begin(id)
                        let accepted = ProcessInfo.processInfo.systemUptime
                        io.send(["id": wireID, "event": "progress", "phase": "queued"])
                        Task.detached {
                            defer { io.finish(id) }
                            do {
                                let data = try WireJSON.encode(request)
                                guard let params = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw ChatFailure("request must be a JSON object.") }
                                let scheduled = op == "chat" ? ProcessInfo.processInfo.systemUptime : await engine.resetTimingPeak()
                                if op == "count_tokens" || op == "convert" { io.send(["id": wireID, "event": "progress", "phase": "step"]) }
                                var reply: JSONObject = ["id": wireID, "ok": true]
                                switch op {
                                case "measurement_logits":
                                    guard let directory = request["output_dir"]?.string, let prefix = request["prefix"]?.string,
                                        let continuation = request["continuation"]?.string
                                    else { throw ChatFailure("measurement", "Fixed text and an isolated output directory are required.") }
                                    let support = ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"] ?? ""
                                    let root = URL(fileURLWithPath: support).appendingPathComponent("benchmarks").resolvingSymlinksInPath()
                                    let folder = URL(fileURLWithPath: directory).resolvingSymlinksInPath()
                                    guard !support.isEmpty, folder.path.hasPrefix(root.path + "/") else {
                                        throw ChatFailure("measurement", "Logits output must stay in the supervisor's benchmark directory.")
                                    }
                                    let logits = try await engine.measurementLogits(prefix: prefix, continuation: continuation, steps: 32)
                                    let file = folder.appendingPathComponent("logits-" + UUID().uuidString + ".json")
                                    try JSONEncoder().encode(logits).write(to: file, options: .atomic)
                                    reply["output_file"] = .string(file.lastPathComponent)
                                case "load":
                                    guard let directory = request["weights_dir"]?.string, let dtype = request["dtype"]?.string else {
                                        throw ChatFailure("load requires weights_dir and dtype.")
                                    }
                                    let config = params["config"] as? [String: Any] ?? [:]
                                    let limit = try ChatRequest.contextLimit(config)
                                    io.send(["id": wireID, "event": "progress", "phase": "loading"])
                                    let seconds = try await engine.load(directory: directory, dtype: dtype, contextLimit: limit)
                                    reply["load_s"] = .number(seconds); reply["timing"] = .object(RequestTiming(load_s: seconds).json); reply["context"] = .number(Double(limit))
                                    reply["memory"] = io.encoded(ChatMemory.snapshot()); reply["engine"] = "mlx-swift"
                                    reply["engine_reason"] = "Fast/Exact is unavailable; this runner uses the pinned upstream kernels."
                                case "chat":
                                    let result = try await chat(params) { chat in
                                        try await engine.chat(
                                            chat, cancellation: cancellation, accepted: accepted,
                                            progress: { fields in
                                                var event = fields; event["id"] = wireID; io.send(event)
                                            },
                                            emit: { delta in
                                                io.send(["id": wireID, "event": .string(delta.event.rawValue), "text": .string(delta.text)])
                                            })
                                    }
                                    if case .object(let fields) = io.encoded(result) { reply.merge(fields) { _, value in value } }
                                case "count_tokens":
                                    guard (request["text"] != nil) != (request["messages"] != nil) else {
                                        throw ChatFailure("count_tokens requires either text or messages, not both.")
                                    }
                                    let text = request["text"]?.string
                                    if request["text"] != nil && text == nil { throw ChatFailure("text must be a string.") }
                                    let messages = request["messages"] == nil ? nil : try ChatRequest.messages(params["messages"], systemPrompt: params["system_prompt"])
                                    let thinking = try ChatRequest.boolean(params, "thinking", default: false)
                                    let count = try await engine.count(text: text, messages: messages, thinking: thinking)
                                    reply["tokens"] = .number(Double(count)); reply["input_tokens"] = .number(Double(count))
                                default:
                                    guard let src = request["src_dir"]?.string, let dst = request["dst_dir"]?.string,
                                        let dtype = request["dtype"]?.string, let recipe = request["recipe"]?.string
                                    else {
                                        throw ChatFailure("convert requires src_dir, dst_dir, dtype, group_size, and recipe.")
                                    }
                                    let group = try ChatRequest.integer(params, "group_size", default: 64, range: 64...64)
                                    let files = try await engine.convert(source: src, destination: dst, dtype: dtype, groupSize: group, recipe: recipe)
                                    reply["files"] = io.encoded(files)
                                }
                                var timing = RequestTiming(queue_s: max(0, scheduled - accepted), peak_mb: await engine.timingPeakMB()).json
                                timing.merge(reply["timing"]?.object ?? [:]) { _, value in value }; reply["timing"] = .object(timing)
                                var done: JSONObject = ["id": wireID, "event": "progress", "phase": "done"]
                                done["timing"] = reply["timing"]
                                done["_arsenal_first_token_at"] = reply["_arsenal_first_token_at"]
                                done["prompt_tokens"] = reply["usage"]?["prompt_tokens"]
                                done["cached_tokens"] = reply["cached_tokens"]
                                done["generated_tokens"] = reply["generated_tokens"]
                                io.send(done)
                                io.finish(id); io.send(reply); io.status()
                            } catch { io.finish(id); io.failure(wireID, error) }
                        }
                    } catch { io.failure(wireID, error) }
                default: io.failure(wireID, ChatFailure("unsupported_op", "The chat runner does not implement \(op)."))
                }
            }
            _ = io.cancel(nil)
        } catch {
            try? FileHandle.standardError.write(contentsOf: Data("chat: malformed or disconnected runner transport.\n".utf8))
            exit(1)
        }
    }
}
