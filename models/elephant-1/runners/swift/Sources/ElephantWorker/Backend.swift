// SPDX-License-Identifier: MIT
// elephant-1's ops: load, convert (FP16 cast at Get), decide and judge, with d1-3B's request conventions (model,
// bits, per-item judge errors) and the maker's schema checks and length limits.
import ArsenalWire
import Darwin
import ElephantEngine
import Foundation
import IntegrationKit
import Metal
import MLX

struct WorkerError: Error {
    let code: String
    let message: String
    let status: Int?
    init(_ code: String, _ message: String, status: Int? = nil) {
        self.code = code
        self.message = message.split(whereSeparator: \.isNewline).joined(separator: " ")
        self.status = status
    }
    /// A request stopped by `cancel`, an unload or the host closing the pipe: Arsenal's status for `cancelled` on every
    /// route (workerHTTPStatus), never a runner fault (500).
    static let cancelled = WorkerError("cancelled", "Request cancelled.", status: 400)
}

final class Backend: @unchecked Sendable {
    static let model = "elephant-1"
    /// The Get-time cast: every tensor of the BF16 checkpoint to FP16, file by file.
    static let recipe = "cast:elephant-1-v1"
    static let cacheLimit = 512 * 1024 * 1024

    private(set) var engine: ElephantEngine?
    private var precision = "fp16"

    func load(_ request: PyJSON, cancelled: () -> Bool) throws -> [(String, PyJSON)] {
        if engine != nil { throw WorkerError("already_loaded", "elephant-1 is already loaded.", status: 409) }
        let dtype = request["dtype"]?.string ?? "fp16"
        guard dtype == "fp16" else { throw WorkerError("unsupported_dtype", "elephant-1 runs at FP16.", status: 422) }
        if let kernels = request["config"]?["kernels"], kernels.truthy { throw WorkerError("unsupported_kernel", "This runner does not accept folder kernels.") }
        guard let path = request["weights_dir"]?.string else { throw WorkerError("missing_weights", "Get the verified elephant-1 checkpoint first.") }
        let weights = URL(fileURLWithPath: path).resolvingSymlinksInPath()
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: weights.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw WorkerError("missing_weights", "Get the verified elephant-1 checkpoint first.")
        }
        guard MTLCreateSystemDefaultDevice() != nil else { throw WorkerError("unsupported_device", "elephant-1 requires Apple Silicon Metal.") }
        let started = Date()
        do {
            Memory.cacheLimit = Self.cacheLimit
            let engine = try ElephantEngine(weights: weights, cancelled: cancelled)
            guard engine.model.dtype == .float16 else {
                throw WorkerError("unsupported_dtype", "Weights are not the FP16 snapshot Get makes.", status: 422)
            }
            try engine.warm()
            self.engine = engine
        } catch let error as WorkerError {
            engine = nil; Memory.clearCache(); throw error
        } catch is Cancelled {
            engine = nil; Memory.clearCache(); throw WorkerError.cancelled
        } catch {
            engine = nil; Memory.clearCache(); throw WorkerError("load_failed", "\(error)")
        }
        Memory.clearCache()
        return [("load_s", .number(Date().timeIntervalSince(started))), ("engine", .string("mlx-swift/elephant-qwen35")), ("memory", memory())]
    }

    func memory() -> PyJSON {
        .object([
            ("footprint_peak_mb", .number(peakFootprintMB())), ("footprint_mb", .number(footprintMB())),
            ("mlx_active_mb", .number(Double(Memory.activeMemory) / 1e6)), ("mlx_cache_mb", .number(Double(Memory.cacheMemory) / 1e6))
        ])
    }

    func unload() -> [(String, PyJSON)] {
        engine = nil
        Memory.clearCache()
        return [("memory", memory())]
    }

    /// Get's FP16 cast through IntegrationKit's streamed safetensors converter (no tensor is quantized).
    func convert(_ request: PyJSON, cancelled: @escaping () -> Bool) throws -> [(String, PyJSON)] {
        guard engine == nil else { throw WorkerError("busy", "Unload elephant-1 before converting.", status: 409) }
        guard request["recipe"]?.string == Self.recipe, request["dtype"]?.string == "fp16" else {
            throw WorkerError("unsupported_recipe", "elephant-1 converts only \(Self.recipe) to fp16.")
        }
        let wire = try JSONDecoder().decode(JSONValue.self, from: Data(request.compact().utf8))
        guard let object = wire.object else { throw WorkerError("invalid_request", "Send a JSON object.") }
        let token = NativeCancellation()
        let watcher = Thread {
            while !Thread.current.isCancelled {
                if cancelled() { token.cancel(); return }
                Thread.sleep(forTimeInterval: 0.2)
            }
        }
        watcher.start()
        defer { watcher.cancel() }
        do {
            let result = try NativeConversion(recipe: Self.recipe, eligible: { _, _ in false }).run(object, cancellation: token)
            let back = try PyJSONParser.parse(JSONEncoder().encode(JSONValue.object(result)))
            return back.fields ?? []
        } catch let failure as WorkerFailure {
            throw WorkerError(failure.code, failure.message)
        }
    }

    // MARK: request checks

    /// The request body in the caller's own order: a raw body (the HTTP decisions routes) as sent; otherwise the
    /// frame's members, which the host writes with sorted keys, with every member it also forwards as exact JSON text
    /// under `ordered_json` (the schema's `x-arsenal-ordered` state, items and questions; OrderedJSONMembers) read from
    /// that text instead. A raw body's own `ordered_json` member means nothing.
    static func body(_ request: PyJSON) throws -> PyJSON {
        if let raw = request["body"] {
            // A raw body comes alone; beside request members it would be a second request the host never checked.
            guard !(request.fields ?? []).contains(where: { ["state", "questions", "items", "media", orderedTransport].contains($0.0) }) else {
                throw WorkerError("invalid_request", "body is a whole raw request; send it alone or send its members.", status: 422)
            }
            guard let text = raw.string, let parsed = try? PyJSONParser.parse(Data(text.utf8)) else {
                throw WorkerError("invalid_request", "Send a JSON object.", status: 422)
            }
            return parsed
        }
        var fields = (request.fields ?? []).filter { !["id", "op", orderedTransport].contains($0.0) }
        if let ordered = request[orderedTransport] {
            guard let members = ordered.fields else { throw WorkerError("invalid_request", "\(orderedTransport) must be an object of JSON texts.", status: 422) }
            // Only the members the op schemas mark ordered; anything else in the transport is ignored, never read.
            for (name, value) in members where orderedNames.contains(name) {
                guard let text = value.string, let parsed = try? PyJSONParser.parse(Data(text.utf8)) else {
                    throw WorkerError("invalid_request", "\(orderedTransport).\(name) must be JSON text.", status: 422)
                }
                if let index = fields.firstIndex(where: { $0.0.utf8.elementsEqual(name.utf8) }) { fields[index].1 = parsed } else { fields.append((name, parsed)) }
            }
        }
        return .object(fields)
    }

    static let orderedTransport = "ordered_json"
    static let orderedNames: Set<String> = ["state", "questions", "items"]

    private func body(_ request: PyJSON, judge: Bool = false) throws -> PyJSON {
        let body = try Self.body(request)
        guard body.isObject else { throw WorkerError("invalid_request", "Send a JSON object.", status: 422) }
        if let model = body["model"], model.string != Self.model { throw WorkerError("wrong_model", "Select model elephant-1 explicitly.", status: 422) }
        if let merge = body["merge"], merge.bool != true {
            throw WorkerError("unsupported_merge", "elephant-1 always shares the state prefix across questions; merge must be true.", status: 422)
        }
        if let bits = body["bits"], !bits.isNull, bits.int != 0 && bits.int != 16 {
            throw WorkerError("precision_conflict", "elephant-1 is loaded at FP16; this request cannot change it.", status: 409)
        }
        if let images = body["images"], images.array?.isEmpty != true, !images.isNull {
            throw WorkerError(
                "unsupported_media", "elephant-1 takes pictures as media: [{\"type\": \"image\", \"path\": …}], referenced in the state by <image:1>, <image:2>, …",
                status: 422)
        }
        // judge reads text and JSON states only: a media list would be answered about pictures the model never saw.
        if judge, let media = body["media"], !media.isNull, media.array?.isEmpty != true {
            throw WorkerError("unsupported_media", "judge reads text and JSON states; send pictures and videos with decide's media list.", status: 422)
        }
        if engine == nil { throw WorkerError("not_loaded", "Load elephant-1 before deciding.", status: 409) }
        return body
    }

    private func questions(_ raw: PyJSON?, batch: Bool = false) throws -> [(String, Question)] {
        do { return try Requests.questions(raw, batch: batch) } catch let e as InvalidRequest { throw WorkerError("invalid_request", e.description, status: 422) }
    }

    private func media(_ raw: PyJSON?) throws -> [MediaItem] {
        do { return try Requests.media(raw, maxItems: ElephantEngine.maxImages) } catch let e as Requests.MediaError {
            throw WorkerError(e.code, e.description, status: 422)
        }
    }

    private func answer(_ state: PyJSON?, _ qs: [(String, Question)], media items: [MediaItem] = [], cancelled: () -> Bool) throws -> [(String, PyJSON)] {
        let outcome: (answers: [(String, PyJSON)], tokens: Int)
        do {
            outcome = try engine!.decide(state: state, questions: qs, media: items, cancelled: cancelled)
        } catch is Cancelled {
            throw WorkerError.cancelled
        } catch is NonFiniteError {
            throw WorkerError("non_finite", "elephant-1 returned a non-finite answer.")
        } catch let e as LimitError {
            throw WorkerError("context_limit", e.description, status: 422)
        } catch let e as PromptError {
            throw WorkerError("invalid_request", e.description, status: 422)
        }
        guard outcome.answers.allSatisfy({ Self.finite($0.1) }) else { throw WorkerError("non_finite", "elephant-1 returned a non-finite answer.") }
        return [("answers", .object(outcome.answers)), ("usage", .object([("input_tokens", .int(outcome.tokens)), ("output_tokens", .int(0))]))]
    }

    static func finite(_ v: PyJSON) -> Bool {
        switch v {
        case .object(let f): return f.allSatisfy { finite($0.1) }
        case .array(let a): return a.allSatisfy(finite)
        case .scalar(let s): return !["nan", "inf", "-inf"].contains(s)
        case .string: return true
        }
    }

    // MARK: ops

    func decide(_ request: PyJSON, cancelled: () -> Bool) throws -> [(String, PyJSON)] {
        let body = try self.body(request)
        guard let stateValue = body["state"] else { throw WorkerError("invalid_request", "state is required.", status: 422) }
        let state: PyJSON? = stateValue.isNull ? nil : stateValue
        let qs = try questions(body["questions"])
        let items = try media(body["media"])
        if cancelled() { throw WorkerError.cancelled }
        return try answer(state, qs, media: items, cancelled: cancelled) + [("model", .string(Self.model)), ("bits", .int(16))]
    }

    func judge(_ request: PyJSON, cancelled: () -> Bool) throws -> [(String, PyJSON)] {
        let body = try self.body(request, judge: true)
        guard let items = body["items"]?.array, !items.isEmpty, items.count <= 256 else {
            throw WorkerError("invalid_request", "items must contain 1–256 states.", status: 400)
        }
        let qs = try questions(body["questions"], batch: true)
        var results: [PyJSON] = []
        for item in items {
            if cancelled() { throw WorkerError.cancelled }
            let start = Date()
            do {
                if let f = item.fields, f.contains(where: { ["image", "images", "audio", "video", "videos"].contains($0.0) }) {
                    throw WorkerError("unsupported_media", "judge reads text and JSON states; send pictures with decide's media list.")
                }
                let answers = try answer(item.isNull ? nil : item, qs, cancelled: cancelled)
                results.append(.object(answers + [("model", .string(Self.model)), ("ms", .number(Date().timeIntervalSince(start) * 1000))]))
            } catch let e as WorkerError where e.code != "cancelled" {
                results.append(.object([("error", .string(e.message)), ("model", .null), ("ms", .int(0))]))
            }
        }
        if cancelled() { throw WorkerError.cancelled }
        return [("results", .array(results)), ("bits", .int(16))]
    }
}

/// The process's physical footprint (task_vm_info.phys_footprint), decimal MB.
func footprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return result == KERN_SUCCESS ? Double(info.phys_footprint) / 1e6 : 0
}

private func peakFootprintMB() -> Double {
    var info = rusage_info_v4()
    let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
    }
    return status == 0 ? Double(info.ri_lifetime_max_phys_footprint) / 1e6 : 0
}
