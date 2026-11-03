// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import MLX
import DecisionsProtocol
import ArsenalKernels
import ArsenalWire
import VerdictEngine

/// Op dispatch for one runner process.
final class Worker {
    enum Next { case `continue`, exit, malformed }
    private let out: Int32
    private let catalog = DecisionsCatalog.verdict
    private var runner: DecisionsRunner?
    private var loadSeconds: Double = 0
    private var telemetryStart: Double?
    private var telemetryOp: String?
    /// load.config.kernels (docs/KERNELS.md); nil when the supervisor sent none.
    private var kernels: FolderKernels?

    /// MLX buffer cache bound (Verdict's VERDICT_CACHE_LIMIT_MB default, 1024 MB): without it the cache keeps every
    /// activation shape it has seen (measured 9 GB after 202 single items).
    private let cacheLimitMB = Int(ProcessInfo.processInfo.environment["ARSENAL_DECISIONS_CACHE_LIMIT_MB"] ?? "") ?? 1024

    init(out: Int32) {
        self.out = out
        Memory.cacheLimit = cacheLimitMB * 1024 * 1024
    }
    /// After a request, as Verdict's helper: return the cache when it grew past the limit.
    private func trim() { if Memory.cacheMemory > cacheLimitMB * 1_000_000 { Memory.clearCache() } }

    func hello() {
        let hello = WorkerHello(
            worker: "decisions", runner: "native", pid: Int(getpid()), build: Self.build, kinds: ["decisions"],
            ops: ["load", "status", "unload", "ping", "cancel", "convert", "decide", "judge", "measurement_components", "measurement_source"])
        send([
            "event": hello.event, "proto": hello.proto, "worker": hello.worker, "runner": hello.runner, "pid": hello.pid,
            "build": hello.build, "kinds": hello.kinds, "ops": hello.ops
        ])
    }

    func handle(_ payload: Data) -> Next {
        var parser = OrderedJSONParser(payload)
        // A kind-1 frame that is not a JSON object is a malformed frame: the runner exits (CONTRACTS §4).
        guard let request = try? parser.parse(), request.fields != nil else { return .malformed }
        let id: Any =
            request["id"].flatMap { value -> Any? in
                if case .scalar(let s) = value, let n = Int(s) { return n }
                return value.text
            } ?? NSNull()
        let op = request["op"]?.text ?? ""
        if ["load", "convert", "decide", "judge"].contains(op) {
            telemetryStart = ProcessInfo.processInfo.systemUptime; telemetryOp = op; Memory.peakMemory = 0
            send(["id": id, "event": "progress", "phase": op == "load" ? "loading" : "prefill"])
        }
        defer { telemetryStart = nil; telemetryOp = nil }
        do {
            switch op {
            case "ping": reply(id, [:])
            case "status": reply(id, status())
            case "cancel":
                // Ops run one at a time on this thread: a cancel only arrives after its target finished.
                reply(id, ["cancelled": false])
            case "load": reply(id, try load(request)); pushStatus()
            case "unload":
                reply(id, [:])
                return .exit
            case "convert": reply(id, try Convert.run(request))
            case "measurement_source":
                guard runner == nil, let source = request["src_dir"]?.text, let target = request["dst_dir"]?.text else {
                    throw DecisionsError("bad_request", "Source cast requires unloaded worker, src_dir and dst_dir")
                }
                _ = try VonStore.convert(
                    snapshot: URL(fileURLWithPath: source), to: URL(fileURLWithPath: target), model: "source-calibration", repository: "wfzyx/von", revision: nil, format: "BF16",
                    truncate: request["truncate"]?.render() == "true")
                reply(id, ["output_dir": target])
            case "decide": try decide(id, request); trim(); pushStatus()
            case "judge": try judge(id, request); trim(); pushStatus()
            case "measurement_components": reply(id, try components(request)); trim()
            default: fail(id, "unsupported_op", "Unknown op '\(op)'.")
            }
        } catch let error as DecisionsError {
            fail(id, error.code, error.message)
        } catch {
            fail(id, "failed", DecisionsRunner.message(error))
        }
        return .continue
    }

    // MARK: ops

    private var requestedHalf: String?

    private func load(_ request: OrderedJSON) throws -> [String: Any] {
        if let runner { throw DecisionsError("already_loaded", "This runner already serves \(runner.spec.id); start another runner for another model.") }
        guard let weights = request["weights_dir"]?.text else { throw DecisionsError("bad_request", "load needs weights_dir.") }
        let config = request["config"]
        let modelID = config?["id"]?.text ?? request["model_dir"]?.text.flatMap(ModelFolder.id) ?? ""
        guard let spec = catalog.spec(modelID) else {
            throw DecisionsError("unknown_model", "Unknown decision model '\(modelID)'; this worker runs \(catalog.local.joined(separator: ", ")).")
        }
        let dtype = request["dtype"]?.text ?? "fp16"
        guard let bits = spec.helperBits(dtype: dtype) else {
            throw DecisionsError("unsupported_dtype", "\(spec.id) runs at fp16, int8 or int4, not \(dtype).")
        }
        if let derived = Convert.derivedDtype(URL(fileURLWithPath: weights)), derived != dtype {
            throw DecisionsError("dtype_mismatch", "\(weights) holds the \(derived) view; load it with dtype \(derived).")
        }
        // Fast = Verdict's Optimized · Fast (chip kernels that passed their gate); Exact = Optimized · Exact (output
        // identical to the stock path). The folder's load setting stock_mlx (config.settings) selects the stock MLX path
        // (diagnosis); an explicit config.recipe wins over both.
        let fast = request["fast"].map { $0.render() == "true" } ?? true
        var recipe: KernelRecipe = fast ? .fast : .exact
        if config?["settings"]?["stock_mlx"]?.render() == "true" { recipe = .standard }
        if let name = config?["recipe"]?.text {
            guard let value = KernelRecipe(rawValue: name) else { throw DecisionsError("bad_request", "config.recipe must be standard, exact or fast.") }
            recipe = value
        }
        let start = Date()
        let snapshot = URL(fileURLWithPath: weights, isDirectory: true)
        let model: DecisionModel
        switch spec.runtime {
        case "von": model = try VonLoader.load(id: spec.id, snapshot: snapshot, bits: bits, recipe: recipe)
        default:
            // Folder kernels (models/<id>/kernels, resolved by the supervisor for this chip and Exact/Fast).
            let folder = try config?["kernels"].map { value -> FolderKernels in
                do { return try FolderKernels(config: try JSONSerialization.jsonObject(with: Data(value.render().utf8))) } catch {
                    throw DecisionsError("bad_request", "config.kernels: \(error)")
                }
            }
            kernels = folder?.configured == true ? folder : nil
            model = try LayaLoader.load(id: spec.id, snapshot: snapshot, bits: bits, recipe: recipe, kernels: kernels)
        }
        if recipe == .standard, let paths = model as? InferencePathSwitching { try paths.useStockPath(true) }
        requestedHalf = dtype
        runner = DecisionsRunner(model: model, spec: spec, bits: bits, recipe: recipe, catalog: catalog)
        Memory.clearCache()
        loadSeconds = (Date().timeIntervalSince(start) * 100).rounded() / 100
        var out = status()
        out["load_s"] = loadSeconds
        return out
    }

    private func status() -> [String: Any] {
        var out: [String: Any] = ["memory": memory()]
        guard let runner else { out["model"] = NSNull(); return out }
        let engine = Engine.describe(runner)
        out["model"] = runner.spec.id
        out["dtype"] = requestedHalf ?? runner.spec.dtype(bits: runner.bits)
        out["fast"] = runner.recipe == .fast
        out["recipe"] = runner.recipe.rawValue
        out["context"] = runner.model.contextLimit
        out["engine"] = engine.engine
        out["engine_reason"] = engine.reason ?? NSNull()
        out["optimizations"] = engine.optimizations
        out["load_s"] = loadSeconds
        if let kernels { out["kernels"] = kernels.wire }
        return out
    }

    private func loaded() throws -> DecisionsRunner {
        guard let runner else { throw DecisionsError("not_loaded", "No model is loaded; send load first.") }
        return runner
    }

    /// `decide`: either `body` (the System One request as a JSON string, byte-exact) or the fields inline
    /// (`state`, `questions`, `model?`, `bits?`), read in their sent order. A missing model means the loaded one.
    private func decide(_ id: Any, _ request: OrderedJSON) throws {
        let runner = try loaded()
        let body = try Self.body(request, fields: ["state", "questions", "model", "bits", "merge"], defaultModel: runner.spec.id)
        switch runner.decide(body) {
        case .wrongModel(let needed):
            fail(id, "wrong_model", "This runner serves \(runner.spec.id); the request needs \(needed).", extra: ["model": needed])
        case .reply(let reply):
            if reply.status == 200 {
                var out = reply.body
                out["bits"] = reply.bits ?? NSNull()
                self.reply(id, out)
            } else {
                fail(id, Self.code(reply), Self.sentence(reply), extra: ["status": reply.status, "body": reply.body])
            }
        }
    }

    /// `judge`: `body` (JSON string) or inline `items`, `questions`, `model?`, `bits?`. Reply `{results: [...]}`.
    private func judge(_ id: Any, _ request: OrderedJSON) throws {
        let runner = try loaded()
        let body = try Self.body(request, fields: ["items", "questions", "model", "bits"], defaultModel: runner.spec.id)
        switch runner.judge(body) {
        case .wrongModel(let routes):
            fail(
                id, "wrong_model", "This runner serves \(runner.spec.id); some items route to \(routes.keys.sorted().joined(separator: ", ")).",
                extra: ["routes": routes])
        case .reply(let status, let object):
            if status == 200 {
                reply(id, object)
            } else {
                fail(id, status == 409 ? "conflict" : "bad_request", object["error"] as? String ?? "Request failed.", extra: ["status": status, "body": object])
            }
        }
    }

    /// Arsenal intake (lab only): `{fixture_dir, output_dir, capture, states?}`. Laya models only; Von has no intake adapter yet.
    private func components(_ request: OrderedJSON) throws -> [String: Any] {
        let runner = try loaded()
        guard let fixture = request["fixture_dir"]?.text, let output = request["output_dir"]?.text else {
            throw DecisionsError("bad_request", "measurement_components needs fixture_dir and output_dir.")
        }
        let f = URL(fileURLWithPath: fixture), o = URL(fileURLWithPath: output), capture = request["capture"]?.render() == "true"
        let states = request["states"]?.arrayValues?.compactMap(\.text)
        if let laya = runner.model as? LayaModel {
            try laya.intakeComponents(fixture: f, output: o, capture: capture, states: states)
        } else if let von = runner.model as? VonModel {
            try von.intakeComponents(fixture: f, output: o, capture: capture, states: states)
        } else {
            throw DecisionsError("unsupported_op", "No component adapter")
        }
        return ["output_dir": output]
    }

    // MARK: helpers

    static func body(_ request: OrderedJSON, fields: [String], defaultModel: String) throws -> Data {
        if let raw = request["body"] {
            guard let text = raw.text else { throw DecisionsError("bad_request", "body must be the request JSON as a string.") }
            return Data(text.utf8)
        }
        var entries: [(String, OrderedJSON)] = []
        for name in fields { if let value = request[name] { entries.append((name, value)) } }
        if request["model"] == nil { entries.append(("model", .string(defaultModel))) }
        return Data(OrderedJSON.object(entries).render().utf8)
    }

    private static func code(_ reply: SystemOneReply) -> String {
        if reply.status == 422 { return "validation_error" }
        return ((reply.body["detail"] as? [String: Any])?["error_type"] as? String) ?? "failed"
    }
    private static func sentence(_ reply: SystemOneReply) -> String {
        if let detail = reply.body["detail"] as? [String: Any], let message = detail["message"] as? String { return message }
        if let issues = reply.body["detail"] as? [[String: Any]], let first = issues.first, let msg = first["msg"] as? String {
            let loc = (first["loc"] as? [Any])?.map { "\($0)" }.joined(separator: ".") ?? ""
            return "\(loc): \(msg)"
        }
        return "Request failed with status \(reply.status)."
    }

    private func memory() -> [String: Any] {
        [
            "footprint_mb": (Double(Self.footprint()) / 1e6).rounded(),
            "footprint_peak_mb": Double(Self.footprint(peak: true)) / 1e6,
            "mlx_active_mb": (Double(Memory.activeMemory) / 1e6).rounded(),
            "mlx_cache_mb": (Double(Memory.cacheMemory) / 1e6).rounded(),
            "mlx_peak_mb": (Double(Memory.peakMemory) / 1e6).rounded()
        ]
    }
    /// phys_footprint: what Activity Monitor's Memory column shows for this process.
    /// `peak`: the process's lifetime phys_footprint peak (load and every request so far).
    static func footprint(peak: Bool = false) -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return kr == KERN_SUCCESS ? (peak ? UInt64(info.ledger_phys_footprint_peak) : info.phys_footprint) : 0
    }

    private func pushStatus() { send(["event": "status", "memory": memory()]) }
    private func reply(_ id: Any, _ fields: [String: Any]) {
        var out = fields
        if let start = telemetryStart {
            var timing: [String: Any] = ["peak_mb": Double(Memory.peakMemory) / 1e6]
            if telemetryOp == "load" { timing["load_s"] = ProcessInfo.processInfo.systemUptime - start }
            out["timing"] = timing
            send(["id": id, "event": "progress", "phase": "done", "timing": timing])
        }
        out["id"] = id; out["ok"] = true; send(out)
    }
    private func fail(_ id: Any, _ code: String, _ message: String, extra: [String: Any] = [:]) {
        var error: [String: Any] = ["code": code, "message": message]
        for (k, v) in extra { error[k] = v }
        send(["id": id, "ok": false, "error": error])
    }
    private func send(_ object: [String: Any]) { Wire.write(ResponseJSON.data(object as NSDictionary), out) }

    static let build: String = {
        let attrs = try? FileManager.default.attributesOfItem(atPath: CommandLine.arguments[0])
        let mtime = Int((attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
        return "verdict-f8177ed+arsenal mlx-swift 9019419 \(mtime)"
    }()
}

/// Reads `id = "…"` from an Arsenal model folder's model.toml (the rest of the folder is the supervisor's business).
enum ModelFolder {
    static func id(_ dir: String) -> String? {
        let url = URL(fileURLWithPath: dir).appendingPathComponent("model.toml")
        if let text = try? String(contentsOf: url, encoding: .utf8) {
            for line in text.split(separator: "\n") {
                let parts = line.split(separator: "=", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
                if parts.count == 2, parts[0] == "id", parts[1].hasPrefix("\""), let end = parts[1].dropFirst().firstIndex(of: "\"") {
                    return String(parts[1][parts[1].index(after: parts[1].startIndex)..<end])
                }
            }
        }
        return URL(fileURLWithPath: dir).lastPathComponent
    }
}

/// Verdict's engine label (Service.engine / optimizations), per loaded model.
enum Engine {
    static func describe(_ runner: DecisionsRunner) -> (engine: String, reason: String?, optimizations: [String: Any]) {
        var optimizations: [String: Any] = [:]
        let model = runner.model
        let tokenizer = (model as? TokenizerPathReporting)?.tokenizerPath
        let kernel = (model as? KernelPathReporting)?.kernelPath
        if let tokenizer { optimizations["tokenizer"] = tokenizer }
        if let kernel { optimizations["attention"] = kernel.hasPrefix("windowed") ? "windowed" : "stock" }
        if let s = (model as? SmallMReporting)?.smallMPath { optimizations["smallm"] = s }
        if let fallback = runner.fallback {
            return ("mlx", "the optimized path failed during inference (\(fallback)); switched to the stock MLX path", optimizations)
        }
        if runner.recipe == .standard { return ("mlx", "Standard selected", optimizations) }
        var why: [String] = []
        if let tokenizer, tokenizer != "fast" { why.append("tokenizer format not recognised") }
        if let kernel, !kernel.hasPrefix("windowed") {
            why.append(kernel.contains("self-test failed") ? "kernel self-test did not pass on this chip" : "windowed attention disabled")
        }
        return why.isEmpty ? ("optimized", nil, optimizations) : ("mlx", why.joined(separator: "; "), optimizations)
    }
}
